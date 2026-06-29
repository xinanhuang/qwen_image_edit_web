# Qwen-Image-Edit Prompt Keyword/提示词 Injection Guide

Based on:
- Reddit post: https://www.reddit.com/r/StableDiffusion/comments/1n1n81o/qwenimageedit_prompt_guide_the_complete_playbook/
- Paper: 2508.02324v1.pdf (Qwen-Image Technical Report)
- HuggingFace: https://huggingface.co/Qwen/Qwen-Image-Edit-2511

## 📝 Quick-Insert Prompt Templates

### Text Edits (Signs, Labels, Posters)
```
Replace text with '[TEXT]'. Keep original font, size, color, and perspective. Do not alter background.
```

### Object Edits
```
[ADD/REMOVE/REPLACE] [OBJECT]. Keep shadows, reflections, and texture consistent.
```

### Style Transfer
```
Re-render in [STYLE] style. Preserve character identity, clothing, and layout.
```

### Identity Control
```
Preserve face features: [HAIR/EYES/NOSE]. Keep lighting and background unchanged.
```

## 🔑 Keyword Injection Patterns

### Consistency Keywords
- `Keep everything else unchanged`
- `Preserve original style`
- `Maintain lighting and perspective`
- `No distortion, no warped text`

### Quality Keywords
- `high quality, sharp details, 4k`
- `photorealistic, professional`
- `clean edges, smooth blending`

### Negative Keywords
- `no duplicate faces`
- `no extra limbs`
- `no blurry text`
- `no unnatural colors`

## 💡 Best Practices (From Paper & Guide)

1. **Always add consistency phrases** - "Keep everything else unchanged"
2. **Lock identity** - "Preserve face/clothing features"
3. **Chain edits** - 2-3 smaller edits > 1 big edit
4. **Use negatives** - "no distortion, no warped text"
5. **Be specific** - Mention exact objects, positions, styles

## 🎯 Face Swap Specific

### For BFS Head V5
```
head_swap: start with Picture 1 as base, keeping lighting/environment. Remove head from Picture 1, replace with head from Picture 2. Preserve hair, eye color, nose structure of Picture 2. Copy eye direction, head rotation, micro expressions from Picture 1.
```

### Negative Prompt
```
bad quality, noise, blurry, worst quality, low resolution, blur, distortion, unnatural blending, cartoon, illustration, painting
```

## 📷 Camera & Lighting Controls

### Change Lighting
```
Relight the scene with a warm key light from the right and cool rim light from the back. Keep pose and background unchanged.
```

### Simulate Lens Choice
```
Render with a 35 mm lens, shallow depth of field, focus on subject's face. Preserve environment blur.
```

## 🚀 Final Thoughts

- **Add/Replace/Remove language works best** - "add-replace-remove" phrasing improves results
- **"Keep everything the same, don't change anything else" 100% works** - prevents drift
- **Natural lighting helps realism** - "natural lighting" prompt improves results
- **Chain edits** - 2-3 smaller edits > 1 big edit
