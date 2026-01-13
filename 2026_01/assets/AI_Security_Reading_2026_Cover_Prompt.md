# AI Security Reading Guide 2026 - Cover Image Generation

## 封面图片生成Prompt

### 推荐Prompt（复制到DALL-E 3 / Midjourney）

```
A sophisticated reading and learning visualization for AI security professionals.

Dark gradient background from deep navy (#1a1a2e) to charcoal (#16213e).

Central composition: Four floating books with glowing edges arranged in an ascending arc, each representing a key topic:
- Book 1: Neural network patterns (adversarial AI)
- Book 2: Binary code and sticker icons (ML attacks)
- Book 3: Shield with detection waves (EDR/endpoint)
- Book 4: World map with data visualization (threat intelligence)

Connecting elements: Glowing data streams and circuit patterns linking the books.

Accent colors: Electric blue (#0984e3), cyan (#00cec9), purple (#6c5ce7), subtle red warning accents (#e17055).

Style: Professional tech, authoritative, knowledge-focused.
Modern corporate aesthetic with subtle futuristic elements.

Include visual elements: Open book pages, floating code snippets, security shield icons, brain/AI neural patterns.

Reserved clean space at bottom for title overlay (dark, suitable for white text).

No text in image. No watermarks.
Resolution: 1792x1024 (DALL-E 3 native landscape).
```

### 替代Prompt（简洁版）

```
Professional AI security education visualization. Dark blue gradient background.

Four glowing books floating in ascending arrangement, connected by data streams.

Elements: Neural networks, binary code, security shields, global threat map patterns.

Colors: Navy blue, electric cyan, purple accents, subtle red warnings.

Modern corporate tech style. Clean bottom area for text.

No text. 1792x1024 resolution.
```

### 本地生成命令（mflux）

```bash
# 快速生成
./tools/generate_cover.sh ai-security 2026_01/assets "AI Security Reading Guide 2026"

# 高质量生成
python3 tools/cover_generator.py \
    --template whitepaper \
    --topic "AI Security Reading Guide 2026" \
    --output 2026_01/assets/AI_Security_Reading_2026_Cover.png \
    --mode local \
    --quality high \
    --crop
```

### 图片规格要求

| 平台 | 尺寸 | 文件名 |
|------|------|--------|
| 原图 | 1792×1024 | `AI_Security_Reading_2026_Cover_1792x1024.png` |
| LinkedIn | 1200×628 | `AI_Security_Reading_2026_Cover_LinkedIn.png` |
| Medium | 1200×600 | `AI_Security_Reading_2026_Cover_Medium.png` |
| Twitter/X | 1200×628 | `AI_Security_Reading_2026_Cover_Twitter.png` |
| GitHub | 1280×640 | `AI_Security_Reading_2026_Cover_GitHub.png` |

### 文字叠加建议

**标题**（英文）:
```
2026 AI Security Reading Guide
```

**副标题**:
```
4 Essential Books for Security Professionals
```

**中文标题**:
```
2026年AI安全必读书单
从对抗攻击到威胁情报的完整指南
```

---

*生成时间: 2026-01-06*
