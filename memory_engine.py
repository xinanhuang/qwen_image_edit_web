#!/usr/bin/env python3
"""
Persistent Intelligence & Memory Engine
========================================
Self-learning system for the Qwen Image Edit server.

Features:
- Pattern recognition across all edit jobs
- Smart prompt suggestions based on history
- Self-improving default settings from successful jobs
- Trending / popular prompt detection
- User preference learning (per IP)
- Feedback loop: learns from successful vs failed jobs
- Semantic prompt categorization
- Knowledge base of "what works" for image editing
"""

import os
import json
import time
import sqlite3
import hashlib
import threading
from collections import Counter, defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")
MEMORY_LOCK = threading.Lock()

# Prompt categories the system recognizes (for pattern grouping)
PROMPT_CATEGORIES = {
    # Style transfers
    "style_watercolor": ["watercolor", "水彩", "watercolour"],
    "style_oil": ["oil painting", "油画", "painting"],
    "style_anime": ["anime", "动漫", "manga", "cartoon"],
    "style_cyberpunk": ["cyberpunk", "赛博朋克", "neon", "霓虹"],
    "style_vintage": ["vintage", "复古", "retro", "怀旧", "old photo"],
    "style_sketch": ["sketch", "素描", "pencil", "drawing"],
    "style_pixel": ["pixel art", "像素", "pixel"],
    "style_3d": ["3d render", "3d", "render", "blender"],
    "style_pop_art": ["pop art", "波普", "comic", "漫画风格"],
    "style_infrared": ["infrared", "红外", "thermal"],

    # Scene / background changes
    "scene_sunset": ["sunset", "日落", "sunset beach"],
    "scene_winter": ["winter", "冬季", "snow", "雪景", "snowy"],
    "scene_beach": ["beach", "海滩", "ocean", "sea", "海洋"],
    "scene_forest": ["forest", "森林", "woods", "tree"],
    "scene_city": ["city", "城市", "urban", "metropolis", "都市"],
    "scene_space": ["space", "太空", "galaxy", "星空", "cosmos"],
    "scene_fantasy": ["fantasy", "奇幻", "magical", "magic"],
    "scene_storm": ["storm", "风暴", "thunder", "lightning"],
    "scene_autumn": ["autumn", "秋季", "fall", "autumn leaves"],
    "scene_spring": ["spring", "春季", "flower", "blossom", "花"],

    # Subject edits
    "edit_hair": ["hair", "发型", "haircut", "hairstyle"],
    "edit_age_younger": ["younger", "年轻", "young"],
    "edit_age_older": ["older", "年长", "old", "aging"],
    "edit_skin": ["skin", "皮肤", "clear skin", "smooth"],
    "edit_eyes": ["eyes", "眼睛", "eyelids", "眼皮"],
    "edit_smile": ["smile", "微笑", "happy"],
    "edit_clothes": ["clothes", "衣服", "outfit", "服装"],
    "edit_background": ["background", "背景", "change background"],

    # Lighting / atmosphere
    "light_dramatic": ["dramatic", "戏剧性", "dramatic lighting"],
    "light_soft": ["soft light", "柔和", "soft lighting", "gentle"],
    "light_night": ["night", "夜晚", "night scene", "dark"],
    "light_golden": ["golden hour", "金色", "golden light"],
    "light_moody": ["moody", "忧郁", "atmospheric", "氛围"],
    "light_cinematic": ["cinematic", "电影", "cinema"],

    # Quality / enhancement
    "enhance_quality": ["high quality", "高清", "sharp", "detailed", "4k"],
    "enhance_color": ["vibrant", "鲜艳", "colorful", "saturated"],
    "enhance_bw": ["black and white", "黑白", "monochrome"],
}

# Flatten categories for fast lookup
_CATEGORY_KEYWORDS = {}
for cat, keywords in PROMPT_CATEGORIES.items():
    for kw in keywords:
        _CATEGORY_KEYWORDS[kw.lower()] = cat


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_memory_db():
    """Get a thread-local SQLite connection to the memory database."""
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_memory_db():
    """Create all memory tables if they don't exist."""
    conn = _get_memory_db()
    try:
        # Job outcomes table — tracks every job with its outcome
        conn.execute('''
            CREATE TABLE IF NOT EXISTS job_outcomes (
                job_id TEXT PRIMARY KEY,
                ip TEXT,
                prompt TEXT,
                categories TEXT,
                settings TEXT,
                status TEXT,
                elapsed REAL,
                timestamp REAL
            )
        ''')

        # Pattern counts — aggregated prompt patterns
        conn.execute('''
            CREATE TABLE IF NOT EXISTS pattern_counts (
                pattern_hash TEXT PRIMARY KEY,
                pattern_text TEXT,
                category TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                total_elapsed REAL DEFAULT 0,
                last_seen REAL DEFAULT 0
            )
        ''')

        # User preferences — per-IP learned preferences
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                ip TEXT PRIMARY KEY,
                preferred_lightning INTEGER DEFAULT 1,
                preferred_steps INTEGER DEFAULT 4,
                preferred_cfg REAL DEFAULT 1.0,
                preferred_guidance REAL DEFAULT 1.0,
                category_counts TEXT DEFAULT '{}',
                total_jobs INTEGER DEFAULT 0,
                successful_jobs INTEGER DEFAULT 0,
                last_active REAL DEFAULT 0
            )
        ''')

        # Knowledge base — "what works" entries
        conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                prompt_template TEXT,
                success_rate REAL DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                avg_elapsed REAL DEFAULT 0,
                last_updated REAL DEFAULT 0
            )
        ''')

        # Self-learning insights — system-generated insights
        conn.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT,
                content TEXT,
                confidence REAL DEFAULT 0,
                timestamp REAL DEFAULT 0
            )
        ''')

        conn.commit()
    finally:
        conn.close()


_ensure_memory_db()


# ---------------------------------------------------------------------------
# Prompt categorization
# ---------------------------------------------------------------------------

def categorize_prompt(prompt: str) -> list:
    """Categorize a prompt into known editing categories."""
    if not prompt:
        return []
    lower = prompt.lower()
    categories = set()
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in lower:
            categories.add(cat)
    return sorted(categories)


# ---------------------------------------------------------------------------
# Pattern hashing (normalize prompts for pattern matching)
# ---------------------------------------------------------------------------

def _normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for pattern matching (lowercase, trimmed)."""
    return prompt.strip().lower()


def _pattern_hash(prompt: str) -> str:
    """Create a hash for a normalized prompt pattern."""
    return hashlib.md5(_normalize_prompt(prompt).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core learning functions
# ---------------------------------------------------------------------------

def record_job_outcome(job_id: str, ip: str, prompt: str, settings: dict,
                       status: str, elapsed: float, timestamp: float):
    """
    Record a job outcome and update all memory tables.
    Called after every job completes (success or failure).
    """
    categories = categorize_prompt(prompt)
    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            # 1. Record raw outcome
            conn.execute('''
                INSERT OR REPLACE INTO job_outcomes
                (job_id, ip, prompt, categories, settings, status, elapsed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id, ip, prompt[:500],
                json.dumps(categories),
                json.dumps(settings),
                status, elapsed or 0, timestamp
            ))

            # 2. Update pattern counts
            is_success = (status == 'complete')
            phash = _pattern_hash(prompt)
            primary_cat = categories[0] if categories else "general"

            conn.execute('''
                INSERT INTO pattern_counts (pattern_hash, pattern_text, category,
                                           success_count, fail_count, total_elapsed, last_seen)
                VALUES (?, ?, ?, 0, 0, 0, 0)
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    success_count = success_count + CASE WHEN ? THEN 1 ELSE 0 END,
                    fail_count = fail_count + CASE WHEN ? THEN 1 ELSE 0 END,
                    total_elapsed = total_elapsed + ?,
                    last_seen = ?
            ''', (phash, prompt[:200], primary_cat,
                  is_success, not is_success, elapsed or 0, timestamp))

            # 3. Update user preferences
            lightning = settings.get('use_lightning', True)
            steps = settings.get('num_inference_steps', 4)
            cfg = settings.get('true_cfg_scale', 1.0)
            guidance = settings.get('guidance_scale', 1.0)

            # Get or create user preference record
            row = conn.execute(
                'SELECT * FROM user_preferences WHERE ip = ?', (ip,)
            ).fetchone()

            if row:
                cat_counts = json.loads(row['category_counts'] or '{}')
                succ = row['successful_jobs'] + (1 if is_success else 0)
                total = row['total_jobs'] + 1
            else:
                cat_counts = {}
                succ = 1 if is_success else 0
                total = 1

            # Track category usage
            for cat in categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            # Learn preferences from successful jobs (EMA)
            if is_success:
                alpha = 0.3  # learning rate for successful jobs
                pref_lightning = 1 if lightning else 0
                new_lightning = (row['preferred_lightning'] if row else 1) * (1 - alpha) + pref_lightning * alpha
                new_steps = (row['preferred_steps'] if row else 4) * (1 - alpha) + steps * alpha
                new_cfg = (row['preferred_cfg'] if row else 1.0) * (1 - alpha) + cfg * alpha
                new_guidance = (row['preferred_guidance'] if row else 1.0) * (1 - alpha) + guidance * alpha
            else:
                new_lightning = row['preferred_lightning'] if row else 1
                new_steps = row['preferred_steps'] if row else 4
                new_cfg = row['preferred_cfg'] if row else 1.0
                new_guidance = row['preferred_guidance'] if row else 1.0

            conn.execute('''
                INSERT INTO user_preferences
                (ip, preferred_lightning, preferred_steps, preferred_cfg,
                 preferred_guidance, category_counts, total_jobs, successful_jobs, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ip) DO UPDATE SET
                    preferred_lightning = ?,
                    preferred_steps = ROUND(?, 0),
                    preferred_cfg = ?,
                    preferred_guidance = ?,
                    category_counts = ?,
                    total_jobs = ?,
                    successful_jobs = ?,
                    last_active = ?
            ''', (
                ip, new_lightning, round(new_steps), new_cfg, new_guidance,
                json.dumps(cat_counts), total, succ, timestamp,
                # ON CONFLICT values
                new_lightning, new_steps, new_cfg, new_guidance,
                json.dumps(cat_counts), total, succ, timestamp
            ))

            # 4. Update knowledge base for categories
            for cat in categories:
                kb_row = conn.execute(
                    'SELECT * FROM knowledge_base WHERE category = ? AND prompt_template IS NOT NULL',
                    (cat,)
                ).fetchone()

                if kb_row:
                    new_count = kb_row['usage_count'] + 1
                    new_success_rate = kb_row['success_rate'] * (1 - 0.1) + (0.1 if is_success else 0)
                    new_avg_elapsed = kb_row['avg_elapsed'] * 0.85 + (elapsed or 0) * 0.15
                    conn.execute('''
                        UPDATE knowledge_base SET
                            success_rate = ?, usage_count = ?, avg_elapsed = ?, last_updated = ?
                        WHERE id = ?
                    ''', (new_success_rate, new_count, new_avg_elapsed, timestamp, kb_row['id']))
                else:
                    # Create new knowledge entry
                    conn.execute('''
                        INSERT INTO knowledge_base (category, prompt_template, success_rate,
                                                    usage_count, avg_elapsed, last_updated)
                        VALUES (?, ?, ?, 1, ?, ?)
                    ''', (cat, prompt[:200], 1.0 if is_success else 0.0, elapsed or 0, timestamp))

            # 5. Generate insights periodically
            _maybe_generate_insights(conn, timestamp)

            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Insight generation (self-learning)
# ---------------------------------------------------------------------------

def _maybe_generate_insights(conn, timestamp: float):
    """Generate insights based on accumulated data."""
    # Check total job count
    total = conn.execute('SELECT COUNT(*) FROM job_outcomes').fetchone()[0]

    # Generate insights every 10 new jobs
    if total % 10 != 0 or total < 10:
        return

    # Insight: Most popular categories
    cat_counts = conn.execute('''
        SELECT category, COUNT(*) as cnt
        FROM pattern_counts
        GROUP BY category
        ORDER BY cnt DESC
        LIMIT 5
    ''').fetchall()

    if cat_counts:
        popular = [dict(r) for r in cat_counts]
        conn.execute('''
            INSERT INTO insights (insight_type, content, confidence, timestamp)
            VALUES ('popular_categories', ?, 0.9, ?)
        ''', (json.dumps(popular), timestamp))

    # Insight: Best performing patterns
    best_patterns = conn.execute('''
        SELECT pattern_text, success_count, fail_count, category
        FROM pattern_counts
        WHERE success_count >= 2
        ORDER BY (success_count * 1.0 / (success_count + fail_count + 0.001)) DESC,
                 success_count DESC
        LIMIT 10
    ''').fetchall()

    if best_patterns:
        best = [{'text': r['pattern_text'], 'successes': r['success_count'],
                 'failures': r['fail_count'], 'category': r['category']}
                for r in best_patterns]
        conn.execute('''
            INSERT INTO insights (insight_type, content, confidence, timestamp)
            VALUES ('best_patterns', ?, 0.85, ?)
        ''', (json.dumps(best), timestamp))

    # Insight: Average job time trend
    avg_elapsed = conn.execute('''
        SELECT AVG(elapsed) FROM job_outcomes WHERE status = 'complete'
    ''').fetchone()[0]

    if avg_elapsed:
        conn.execute('''
            INSERT INTO insights (insight_type, content, confidence, timestamp)
            VALUES ('avg_elapsed', ?, 0.95, ?)
        ''', (json.dumps({'avg_elapsed': round(avg_elapsed, 1)}), timestamp))

    # Insight: Success rate
    success_rate = conn.execute('''
        SELECT
            SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0) * 1.0 / COUNT(*)
        FROM job_outcomes
    ''').fetchone()[0]

    if success_rate:
        conn.execute('''
            INSERT INTO insights (insight_type, content, confidence, timestamp)
            VALUES ('success_rate', ?, 0.95, ?)
        ''', (json.dumps({'success_rate': round(success_rate, 3)}), timestamp))


# ---------------------------------------------------------------------------
# Query functions (API-facing)
# ---------------------------------------------------------------------------

def get_suggestions_for_ip(ip: str, limit: int = 8) -> list:
    """
    Get personalized prompt suggestions for a specific IP.
    Based on their past successful edits and trending patterns.
    """
    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            suggestions = []

            # 1. User's own successful patterns (highest priority)
            user_patterns = conn.execute('''
                SELECT pattern_text, success_count, category
                FROM pattern_counts
                WHERE success_count > 0
                ORDER BY success_count DESC
                LIMIT ?
            ''', (limit,)).fetchall()

            # Filter to only patterns from this user's IP
            user_prompts = conn.execute('''
                SELECT prompt FROM job_outcomes
                WHERE ip = ? AND status = 'complete'
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (ip, limit)).fetchall()

            for r in user_prompts:
                prompt = r['prompt']
                if prompt and prompt not in [s['text'] for s in suggestions]:
                    categories = categorize_prompt(prompt)
                    suggestions.append({
                        'text': prompt,
                        'source': 'your_history',
                        'categories': categories,
                        'confidence': 0.9
                    })

            # 2. Trending patterns from all users (fill remaining slots)
            trending = conn.execute('''
                SELECT pattern_text, success_count, fail_count, category
                FROM pattern_counts
                WHERE success_count >= 1
                ORDER BY (success_count * 1.0 / (success_count + fail_count + 0.001)) DESC,
                         success_count DESC
                LIMIT ?
            ''', (limit,)).fetchall()

            for r in trending:
                text = r['pattern_text']
                if text and text not in [s['text'] for s in suggestions]:
                    suggestions.append({
                        'text': text,
                        'source': 'trending',
                        'success_rate': round(r['success_count'] / max(r['success_count'] + r['fail_count'], 1), 2),
                        'uses': r['success_count'] + r['fail_count'],
                        'category': r['category'],
                        'confidence': 0.7
                    })

            # 3. Category-based suggestions from knowledge base
            kb_entries = conn.execute('''
                SELECT prompt_template, category, success_rate
                FROM knowledge_base
                WHERE usage_count >= 1
                ORDER BY success_rate DESC, usage_count DESC
                LIMIT ?
            ''', (limit,)).fetchall()

            for r in kb_entries:
                text = r['prompt_template']
                if text and text not in [s['text'] for s in suggestions]:
                    suggestions.append({
                        'text': text,
                        'source': 'knowledge_base',
                        'category': r['category'],
                        'success_rate': round(r['success_rate'], 2),
                        'confidence': 0.6
                    })

            return suggestions[:limit]
        finally:
            conn.close()


def get_user_preferences(ip: str) -> dict:
    """Get learned preferences for a specific IP."""
    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            row = conn.execute(
                'SELECT * FROM user_preferences WHERE ip = ?', (ip,)
            ).fetchone()

            if not row:
                return {
                    'preferred_lightning': True,
                    'preferred_steps': 4,
                    'preferred_cfg': 1.0,
                    'preferred_guidance': 1.0,
                    'category_counts': {},
                    'total_jobs': 0,
                    'successful_jobs': 0,
                    'favorite_categories': [],
                }

            cat_counts = json.loads(row['category_counts'] or '{}')
            # Sort categories by usage count
            favorite_categories = sorted(cat_counts.keys(),
                                         key=lambda c: cat_counts[c], reverse=True)[:5]

            return {
                'preferred_lightning': bool(row['preferred_lightning']),
                'preferred_steps': int(row['preferred_steps']),
                'preferred_cfg': round(float(row['preferred_cfg']), 2),
                'preferred_guidance': round(float(row['preferred_guidance']), 2),
                'category_counts': cat_counts,
                'total_jobs': row['total_jobs'],
                'successful_jobs': row['successful_jobs'],
                'favorite_categories': favorite_categories,
            }
        finally:
            conn.close()


def get_trending_prompts(limit: int = 10) -> list:
    """Get the most popular/trending prompts across all users."""
    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            rows = conn.execute('''
                SELECT pattern_text, success_count, fail_count, category, last_seen
                FROM pattern_counts
                WHERE success_count >= 1
                ORDER BY last_seen DESC, success_count DESC
                LIMIT ?
            ''', (limit,)).fetchall()

            return [{
                'text': r['pattern_text'],
                'successes': r['success_count'],
                'failures': r['fail_count'],
                'success_rate': round(r['success_count'] / max(r['success_count'] + r['fail_count'], 1), 2),
                'category': r['category'],
            } for r in rows]
        finally:
            conn.close()


def get_memory_stats() -> dict:
    """Get overall memory system statistics."""
    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            total_jobs = conn.execute('SELECT COUNT(*) FROM job_outcomes').fetchone()[0]
            successful = conn.execute(
                "SELECT COUNT(*) FROM job_outcomes WHERE status = 'complete'"
            ).fetchone()[0]
            unique_patterns = conn.execute('SELECT COUNT(*) FROM pattern_counts').fetchone()[0]
            unique_users = conn.execute('SELECT COUNT(DISTINCT ip) FROM job_outcomes').fetchone()[0]
            kb_entries = conn.execute('SELECT COUNT(*) FROM knowledge_base').fetchone()[0]
            insights_count = conn.execute('SELECT COUNT(*) FROM insights').fetchone()[0]

            avg_elapsed = conn.execute('''
                SELECT AVG(elapsed) FROM job_outcomes WHERE status = 'complete'
            ''').fetchone()[0] or 0

            # Top categories
            top_categories = conn.execute('''
                SELECT category, success_count, fail_count
                FROM pattern_counts
                GROUP BY category
                ORDER BY (success_count + fail_count) DESC
                LIMIT 10
            ''').fetchall()

            return {
                'total_jobs_tracked': total_jobs,
                'successful_jobs': successful,
                'success_rate': round(successful / max(total_jobs, 1), 3),
                'unique_patterns_learned': unique_patterns,
                'unique_users': unique_users,
                'knowledge_entries': kb_entries,
                'insights_generated': insights_count,
                'avg_completion_time': round(avg_elapsed, 1),
                'top_categories': [{
                    'category': r['category'],
                    'successes': r['success_count'],
                    'total': r['success_count'] + r['fail_count'],
                } for r in top_categories],
            }
        finally:
            conn.close()


def get_recent_insights(limit: int = 5) -> list:
    """Get the most recent system-generated insights."""
    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            rows = conn.execute('''
                SELECT insight_type, content, confidence, timestamp
                FROM insights
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,)).fetchall()

            return [{
                'type': r['insight_type'],
                'content': json.loads(r['content']),
                'confidence': r['confidence'],
                'timestamp': r['timestamp'],
            } for r in rows]
        finally:
            conn.close()


def get_smart_default_settings(ip: str) -> dict:
    """
    Get smart default settings based on learned preferences.
    Falls back to system defaults if no data available.
    """
    prefs = get_user_preferences(ip)
    return {
        'use_lightning': prefs['preferred_lightning'],
        'num_inference_steps': prefs['preferred_steps'],
        'true_cfg_scale': prefs['preferred_cfg'],
        'guidance_scale': prefs['preferred_guidance'],
    }


def search_memory(query: str, limit: int = 10) -> list:
    """
    Search memory for similar prompts/patterns.
    Simple keyword-based search across all recorded patterns.
    """
    if not query:
        return []

    with MEMORY_LOCK:
        conn = _get_memory_db()
        try:
            lower_query = query.lower()
            keywords = lower_query.split()

            # Search in pattern_counts
            rows = conn.execute('''
                SELECT pattern_text, success_count, fail_count, category
                FROM pattern_counts
                WHERE success_count >= 1
                ORDER BY success_count DESC
                LIMIT 100
            ''').fetchall()

            results = []
            for r in rows:
                text = (r['pattern_text'] or '').lower()
                score = sum(1 for kw in keywords if kw and kw in text)
                if score > 0:
                    results.append({
                        'text': r['pattern_text'],
                        'successes': r['success_count'],
                        'failures': r['fail_count'],
                        'success_rate': round(r['success_count'] / max(r['success_count'] + r['fail_count'], 1), 2),
                        'category': r['category'],
                        'relevance': score,
                    })

            results.sort(key=lambda x: x['relevance'], reverse=True)
            return results[:limit]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Prompt enhancement (self-learning prompt improvement)
# ---------------------------------------------------------------------------

def enhance_prompt(prompt: str, ip: str = None) -> str:
    """
    Optionally enhance a prompt based on learned knowledge.
    Returns the original prompt (can be extended in the future).
    """
    categories = categorize_prompt(prompt)

    # If prompt is very short, suggest adding detail based on category
    if len(prompt.split()) <= 3 and categories:
        enhancements = {
            'style_watercolor': ' with detailed brush strokes and soft colors',
            'style_oil': ' with rich textures and detailed brushwork',
            'style_anime': ' with vibrant colors and clean line art',
            'style_cyberpunk': ' with neon lights and futuristic details',
            'style_vintage': ' with film grain and warm tones',
            'scene_sunset': ' with warm golden lighting and dramatic clouds',
            'scene_winter': ' with realistic snow and cold atmosphere',
            'light_cinematic': ' with cinematic lighting and color grading',
        }

        for cat in categories:
            if cat in enhancements:
                return prompt + enhancements[cat]

    return prompt


# ---------------------------------------------------------------------------
# Export for server integration
# ---------------------------------------------------------------------------

__all__ = [
    'record_job_outcome',
    'get_suggestions_for_ip',
    'get_user_preferences',
    'get_trending_prompts',
    'get_memory_stats',
    'get_recent_insights',
    'get_smart_default_settings',
    'search_memory',
    'enhance_prompt',
    'categorize_prompt',
    'PROMPT_CATEGORIES',
]
