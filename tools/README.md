# pubilie_doc 工具集

> **项目**: AI安全技术文章生成系统 - 自动化工具集
> **版本**: 2.1.0
> **更新日期**: 2026-01-10
> **总工具数**: 19个Python模块
> **AI策略**: CLI优先（Gemini/Codex/Claude → Ollama → API）

## 📋 工具概览

### Phase 1: 核心强化方案 (P0+P1)

| 工具 | 文件 | 功能 | 优先级 |
|------|------|------|--------|
| E1 深度AI检测增强 | `deep_ai_detector.py` | 多检测器融合、困惑度/突发性分析 | P0 |
| E2 攻击链映射器 | `attack_mapper.py` | MITRE ATT&CK映射、TTPs提取 | P0 |
| E3 CVE/漏洞监控器 | `cve_monitor.py` | NVD/GitHub Advisory实时监控 | P0 |
| E4 Multi-Agent管线 | `agents/` | CrewAI研究→写作→审核协作 | P1 |
| E5 规范链接管理 | `canonical_manager.py` | URL规范化、重复检测 | P1 |
| E6 Schema标记生成 | `schema_generator.py` | JSON-LD结构化数据 | P1 |
| E7 可读性优化器 | `readability_checker.py` | Flesch-Kincaid评分、中文适配 | P1 |

### Phase 2: 效率优化方案 (P2)

| 工具 | 文件 | 功能 | 优先级 |
|------|------|------|--------|
| E8 趋势监控器 | `trend_monitor.py` | 多源热点追踪、智能调度 | P2 |
| E9 本地转述模型 | `local_paraphraser.py` | Pegasus/T5/Ollama多后端转述 | P2 |
| E10 API限速管理 | `rate_limiter.py` | 令牌桶/滑动窗口、优先级队列 | P2 |
| E11 知识图谱 | `knowledge_graph.py` | Neo4j安全本体、GraphRAG增强 | P2 |
| E12 地理优化 | `geo_optimizer.py` | 时区适配、多语言SEO | P2 |

### Phase 3: 内容扩展方案 (P3)

| 工具 | 文件 | 功能 | 优先级 |
|------|------|------|--------|
| E13 视频生成器 | `video_generator.py` | 文章→视频、Edge-TTS、字幕 | P3 |
| E14 播客生成器 | `podcast_generator.py` | 多角色对话、RSS生成 | P3 |
| E15 社区自动化 | `community_automation.py` | 多平台发布、情感监控 | P3 |

### 基础工具

| 工具 | 文件 | 功能 |
|------|------|------|
| 封面生成器 | `cover_generator.py` | FLUX/DALL-E封面图生成 |
| 博客同步 | `sync_to_blog.py` | 同步到innora-website |
| SEO工具 | `unified_seo_tool.py` | 统一SEO优化 |
| 博客SEO修复 | `fix_blog_seo.py` | 博客SEO问题修复 |

---

## 🚀 快速开始

### 环境依赖

```bash
# 基础依赖
pip install openai anthropic aiohttp pydantic

# AI检测 (E1)
pip install transformers torch nltk

# 知识图谱 (E11)
pip install neo4j graphrag-sdk

# 视频/音频 (E13, E14)
pip install edge-tts ffmpeg-python moviepy

# 转述模型 (E9)
pip install sentencepiece
```

### 使用示例

```python
# E1: AI检测
from tools.deep_ai_detector import DeepAIDetector
detector = DeepAIDetector()
result = await detector.analyze("待检测文本...")
print(f"AI概率: {result.ai_probability:.2%}")

# E9: 文本转述
from tools.local_paraphraser import LocalParaphraser
paraphraser = LocalParaphraser()
result = await paraphraser.paraphrase("原始文本", style="formal")

# E11: 知识图谱
from tools.knowledge_graph import SecurityKnowledgeGraph
kg = SecurityKnowledgeGraph()
await kg.extract_from_article("文章内容...")

# E13: 视频生成
from tools.video_generator import VideoGenerator
generator = VideoGenerator()
await generator.generate_from_article(article, "标题", output_path)

# E14: 播客生成
from tools.podcast_generator import PodcastGenerator
podcast = PodcastGenerator()
episode = await podcast.generate_from_article(article, "标题")

# E15: 社区自动化
from tools.community_automation import CommunityManager
manager = CommunityManager()
await manager.publish_content(content, platforms=["twitter", "linkedin"])
```

---

## 📁 文件结构

```
tools/
├── README.md                  # 本文档
├── cover_generator.py         # 封面生成器
├── sync_to_blog.py            # 博客同步
│
├── # Phase 1: 核心强化
├── deep_ai_detector.py        # E1: AI检测增强
├── attack_mapper.py           # E2: MITRE ATT&CK映射
├── cve_monitor.py             # E3: CVE监控
├── canonical_manager.py       # E5: 规范链接
├── schema_generator.py        # E6: Schema标记
├── readability_checker.py     # E7: 可读性分析
│
├── # Phase 2: 效率优化
├── trend_monitor.py           # E8: 趋势监控
├── local_paraphraser.py       # E9: 本地转述
├── rate_limiter.py            # E10: API限速
├── knowledge_graph.py         # E11: 知识图谱
├── geo_optimizer.py           # E12: 地理优化
│
├── # Phase 3: 内容扩展
├── video_generator.py         # E13: 视频生成
├── podcast_generator.py       # E14: 播客生成
└── community_automation.py    # E15: 社区自动化
```

---

## 🔧 各工具详细说明

### E1: 深度AI检测增强 (deep_ai_detector.py)

多检测器融合方案，结合困惑度(Perplexity)和突发性(Burstiness)理论：

```python
class DeepAIDetector:
    """
    检测维度:
    - 困惑度分析: GPT-2模型计算文本困惑度
    - 突发性分析: 句子长度和词汇变化
    - N-gram分析: 重复模式检测
    - API检测: GPTZero, Originality.ai等
    """

    async def analyze(self, text: str) -> DetectionResult:
        # 返回综合检测结果
        pass
```

### E9: 本地转述模型 (local_paraphraser.py)

支持多种转述后端，降低AI检测风险：

```python
class LocalParaphraser:
    """
    后端选项:
    - PEGASUS: HuggingFace tuner007/pegasus_paraphrase
    - PARROT: T5-based prithivida/parrot_paraphraser_on_T5
    - OLLAMA: 本地LLM（支持中英文）
    - HYBRID: 自动选择（基于语言和长度）

    转述风格:
    - FORMAL: 正式学术风格
    - CASUAL: 口语化风格
    - TECHNICAL: 技术精确风格
    - CREATIVE: 创意表达风格
    """
```

### E11: 知识图谱 (knowledge_graph.py)

安全领域知识图谱，支持中英文术语映射：

```python
class SecurityKnowledgeGraph:
    """
    实体类型:
    - ThreatActor, Malware, Vulnerability
    - Technique, Tool, Campaign
    - Industry, Country, Infrastructure

    关系类型:
    - USES, TARGETS, EXPLOITS
    - MITIGATES, ATTRIBUTED_TO

    功能:
    - LLM实体提取
    - Neo4j持久化
    - GraphRAG查询增强
    """
```

### E13: 视频生成器 (video_generator.py)

文章到视频的自动化转换：

```python
class VideoGenerator:
    """
    生成流程:
    1. 文章 → 场景脚本
    2. 脚本 → Edge-TTS语音
    3. 语音 → SRT字幕
    4. 素材 → FFmpeg合成

    支持格式:
    - 竖屏(9:16): 抖音/快手/YouTube Shorts
    - 横屏(16:9): YouTube/B站
    - 方形(1:1): Instagram/LinkedIn
    """
```

### E14: 播客生成器 (podcast_generator.py)

多角色对话播客生成：

```python
class PodcastGenerator:
    """
    对话风格:
    - SOLO: 单人叙述
    - INTERVIEW: 采访问答
    - DISCUSSION: 多人讨论
    - STORYTELLING: 故事叙述

    功能:
    - 多声音角色分配
    - 背景音乐混合
    - RSS Feed生成
    """
```

### E15: 社区自动化 (community_automation.py)

多平台内容发布和互动管理：

```python
class CommunityManager:
    """
    支持平台:
    - Twitter/X, LinkedIn, Medium
    - Dev.to, 知乎, 微信公众号

    功能:
    - 内容适配（字数、格式、hashtag）
    - 最佳时间调度
    - 评论情感监控
    - LLM智能回复
    - A/B测试
    """
```

---

## 🔗 Claude Code集成

所有工具已配置为Claude Code Skills，在相关场景自动触发：

| Skill | 触发条件 |
|-------|----------|
| cover-generator | 创建新文章、请求封面 |
| content-extraction | 提取URL内容 |
| ai-security-writer | 撰写安全文章 |
| quality-verifying | 发布前质量检查 |

---

## 📊 实现进度

| 阶段 | 方案数 | 已完成 | 进度 |
|------|--------|--------|------|
| Phase 1 (P0+P1) | 7 | 7 | 100% |
| Phase 2 (P2) | 5 | 5 | 100% |
| Phase 3 (P3) | 3 | 3 | 100% |
| **总计** | **15** | **15** | **100%** |

---

## 🤖 AI CLI优先策略

工具集优先使用免费的AI CLI而非付费API：

### 优先级顺序
```
1️⃣ AI CLI工具（免费、高质量）
   - Gemini CLI: 技术调研、文本分析
   - Codex CLI: 代码生成、重构
   - Claude Code: 架构设计、复杂分析

2️⃣ 本地Ollama（离线可用）
   - qwen3:14b: 中文处理
   - gemma3:12b: 英文处理

3️⃣ 付费API（按需使用）
   - OpenAI API: 批量处理
   - ZeroGPT API: AI检测
```

### 任务与推荐CLI
| 任务 | 首选 | 备选 |
|------|------|------|
| 文本转述 | Gemini | Ollama |
| 知识提取 | Claude | Codex |
| AI检测分析 | Gemini | ZeroGPT API |
| 代码生成 | Codex | Claude |

### 快捷命令（~/.zshrc）
```bash
ai "prompt"           # 默认使用Gemini
ai-code "prompt"      # Codex (代码)
ai-arch "prompt"      # Claude (架构)
ai-research "topic"   # Gemini (调研)
ai-fast "prompt"      # Ollama (快速)
```

---

## ⚠️ 注意事项

1. **隐私安全**: 所有Python工具仅保留本地，不上传到公开仓库
2. **API密钥**: 请在`~/.env`配置所需API密钥
3. **资源需求**: 部分工具（E1, E9）需要GPU或大内存
4. **网络依赖**: E3, E8, E15需要网络访问
5. **AI调用**: 优先使用CLI工具，减少API成本

---

*更新日期: 2026-01-10*
*版本: 2.1.0*
*Ultrathink协议实现 + AI CLI优先策略*
