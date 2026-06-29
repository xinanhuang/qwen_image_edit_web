# Persistent Intelligence & Memory Engine

## Overview

A self-learning system inspired by **pi-persistent-intelligence** and **Hermes agent** patterns. The memory engine tracks all image editing jobs, learns from successful patterns, and provides smart suggestions to improve the user experience over time.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Web Server (Flask)                   │
│                                                     │
│  User submits edit ──► Job completes                │
│       │                     │                        │
│       ▼                     ▼                        │
│  /api/edit            _finalize_job()               │
│                          │                           │
│                          ▼                           │
│              record_job_outcome()                    │
│                          │                           │
│                          ▼                           │
│              ┌──────────────────────┐               │
│              │  memory_engine.py    │               │
│              │                      │               │
│              │  memory.db (SQLite)  │               │
│              │  - job_outcomes      │               │
│              │  - pattern_counts    │               │
│              │  - user_preferences  │               │
│              │  - knowledge_base    │               │
│              │  - insights          │               │
│              └──────────────────────┘               │
│                          ▲                           │
│                          │                           │
│  Memory API Endpoints ───┘                           │
│  /api/memory/{suggestions,preferences,               │
│              trending,stats,insights,                │
│              settings,search,enhance}                │
└─────────────────────────────────────────────────────┘
```

## Features

### 1. Pattern Recognition
- Automatically categorizes prompts into 30+ editing categories
- Tracks success/failure rates per pattern
- Groups similar edits for trend analysis

### 2. Smart Suggestions
- Personalized prompt suggestions based on user history
- Trending prompts from all users
- Knowledge base recommendations

### 3. Self-Learning Preferences
- Learns preferred settings (lightning mode, steps, CFG, guidance)
- Uses Exponential Moving Average (EMA) for smooth preference updates
- Per-IP preference tracking

### 4. Knowledge Base
- Tracks "what works" for each category
- Success rate tracking per prompt template
- Average completion time learning

### 5. System Insights
- Auto-generates insights every 10 new jobs
- Tracks popular categories, best patterns, success rates
- Average completion time trends

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/suggestions?limit=8` | GET | Personalized prompt suggestions |
| `/api/memory/preferences` | GET | Learned user preferences |
| `/api/memory/trending?limit=10` | GET | Trending prompts across all users |
| `/api/memory/stats` | GET | Overall memory system statistics |
| `/api/memory/insights?limit=5` | GET | System-generated insights |
| `/api/memory/settings` | GET | Smart default settings for current IP |
| `/api/memory/search?q=query&limit=10` | GET | Search memory for similar prompts |
| `/api/memory/enhance` | POST | Enhance a prompt based on knowledge |

### Example: Get Suggestions
```bash
curl http://localhost:7860/api/memory/suggestions?limit=5
```

Response:
```json
{
  "ip": "100.94.158.36",
  "count": 5,
  "suggestions": [
    {
      "text": "Change the background to a sunset beach",
      "source": "your_history",
      "categories": ["edit_background", "scene_beach", "scene_sunset"],
      "confidence": 0.9
    },
    {
      "text": "Turn into a watercolor painting style",
      "source": "trending",
      "success_rate": 0.95,
      "uses": 12,
      "category": "style_watercolor",
      "confidence": 0.7
    }
  ]
}
```

### Example: Get Stats
```bash
curl http://localhost:7860/api/memory/stats
```

Response:
```json
{
  "total_jobs_tracked": 150,
  "successful_jobs": 142,
  "success_rate": 0.947,
  "unique_patterns_learned": 45,
  "unique_users": 12,
  "knowledge_entries": 18,
  "insights_generated": 15,
  "avg_completion_time": 22.3,
  "top_categories": [...]
}
```

## Prompt Categories

The system recognizes 30+ categories across these groups:

### Style Transfers
- `style_watercolor`, `style_oil`, `style_anime`, `style_cyberpunk`
- `style_vintage`, `style_sketch`, `style_pixel`, `style_3d`, `style_pop_art`

### Scene Changes
- `scene_sunset`, `scene_winter`, `scene_beach`, `scene_forest`
- `scene_city`, `scene_space`, `scene_fantasy`, `scene_storm`
- `scene_autumn`, `scene_spring`

### Subject Edits
- `edit_hair`, `edit_age_younger`, `edit_age_older`
- `edit_skin`, `edit_eyes`, `edit_smile`, `edit_clothes`, `edit_background`

### Lighting / Atmosphere
- `light_dramatic`, `light_soft`, `light_night`, `light_golden`
- `light_moody`, `light_cinematic`

### Quality / Enhancement
- `enhance_quality`, `enhance_color`, `enhance_bw`

## Database Schema

### `job_outcomes`
- Raw record of every job with its outcome
- Used for pattern analysis and user preference learning

### `pattern_counts`
- Aggregated counts per prompt pattern
- Tracks success/failure rates and average elapsed time

### `user_preferences`
- Per-IP learned preferences
- EMA-based learning from successful jobs

### `knowledge_base`
- "What works" entries per category
- Success rate and usage tracking

### `insights`
- System-generated insights
- Popular categories, best patterns, trends

## Frontend Integration

The Memory panel (🧠 tab) provides:
1. **System Stats** - Jobs tracked, success rate, patterns learned, unique users
2. **Smart Suggestions** - Clickable prompt suggestions from history + trending
3. **Trending Prompts** - Ranked by success rate and recency
4. **Learned Preferences** - User's preferred settings with "Apply" button
5. **System Insights** - Auto-generated insights about editing patterns

## Self-Learning Mechanism

1. **On Job Completion**: `record_job_outcome()` is called
2. **Pattern Update**: Prompt pattern counts are updated
3. **Preference Learning**: Successful jobs update user preferences (EMA, α=0.3)
4. **Knowledge Update**: Category knowledge base is updated
5. **Insight Generation**: Every 10 jobs, new insights are generated
6. **Smart Defaults**: On page load, learned settings are auto-applied

## Comparison to pi-persistent-intelligence

| Feature | pi-persistent-intelligence | This Implementation |
|---------|---------------------------|---------------------|
| Memory persistence | ✅ (file-based) | ✅ (SQLite) |
| Pattern learning | ✅ | ✅ |
| User preferences | ✅ | ✅ (per-IP) |
| Smart suggestions | ✅ | ✅ |
| Self-improving defaults | ✅ | ✅ |
| Knowledge base | ✅ | ✅ |
| System insights | ✅ | ✅ |
| Real-time API | ✅ | ✅ |
| Frontend integration | ✅ | ✅ (Memory tab) |

## Future Enhancements

- [ ] Embedding-based semantic search (requires sentence-transformers)
- [ ] Prompt quality scoring
- [ ] A/B testing for prompt variations
- [ ] Cross-user anonymized pattern sharing
- [ ] Time-based trend analysis
- [ ] Prompt template generation from successful patterns
