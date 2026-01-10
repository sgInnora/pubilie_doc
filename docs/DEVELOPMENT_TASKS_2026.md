# pubilie_doc 开发任务分解 2026

> **关联文档**: [ENHANCEMENT_PLAN_2026.md](./ENHANCEMENT_PLAN_2026.md)
> **生成时间**: 2026-01-10 10:00:00 +0800
> **版本**: v1.0

---

## 📋 任务总览

| 阶段 | 任务数 | 预计工期 | 优先级 |
|------|--------|----------|--------|
| Phase 1: 基础强化 | 12 | 2周 | 🔴 P0 |
| Phase 2: 能力增强 | 15 | 2周 | 🟡 P1 |
| Phase 3: 效率优化 | 10 | 2周 | 🟢 P2 |
| Phase 4: 长期演进 | 8 | 持续 | ⚪ P3 |

---

## 🔴 Phase 1: 基础强化 (Week 1-2)

### E1: GEO优化体系

#### T1.1 创建GEO优化Skill
- **文件**: `.claude/skills/geo-optimization/SKILL.md`
- **工期**: 0.5天
- **验收标准**: Skill可被正确触发，包含完整的优化指南

```yaml
任务详情:
  - 定义Skill元数据（name, description, triggers）
  - 编写GEO优化策略指南
  - 定义输出格式模板
  - 添加触发条件（文章发布前、SEO请求时）

依赖: 无
负责人: AI系统自动生成
```

#### T1.2 开发GEO优化工具
- **文件**: `tools/geo_optimizer.py`
- **工期**: 1天
- **验收标准**: 单元测试通过率≥80%

```python
# 核心功能
class GEOOptimizer:
    def extract_faq_pairs(self, content: str) -> list[dict]
    def generate_key_takeaways(self, content: str) -> str
    def identify_entities(self, content: str) -> list[Entity]
    def optimize_for_citation(self, content: str) -> str
    def generate_summary_block(self, content: str) -> str
```

#### T1.3 集成GEO到写作流程
- **修改**: `CLAUDE.md` 写作流程部分
- **工期**: 0.5天
- **验收标准**: 新文章自动执行GEO优化

---

### E2: Schema Markup自动生成

#### T2.1 开发Schema生成工具
- **文件**: `tools/schema_generator.py`
- **工期**: 1天
- **验收标准**: 生成的JSON-LD通过Google Rich Results Test

```python
# 核心功能
class SchemaGenerator:
    def generate_article_schema(self, metadata: dict) -> str
    def generate_faqpage_schema(self, faqs: list) -> str
    def generate_howto_schema(self, steps: list) -> str
    def generate_techarticle_schema(self, metadata: dict) -> str
    def inject_schema_to_html(self, html: str, schema: str) -> str
```

#### T2.2 创建Schema模板库
- **文件**: `templates/schema/`
- **工期**: 0.5天
- **内容**:
  - `article.json.j2`
  - `faqpage.json.j2`
  - `howto.json.j2`
  - `techarticle.json.j2`

#### T2.3 集成到发布流程
- **修改**: platform-publisher Agent
- **工期**: 0.5天
- **验收标准**: GitHub Pages版本自动包含Schema

---

### E3: Canonical URL管理系统

#### T3.1 开发Canonical管理工具
- **文件**: `tools/canonical_manager.py`
- **工期**: 0.5天

```python
# 核心功能
class CanonicalManager:
    PRIMARY_DOMAIN = "https://innora.ai/blog/"

    def generate_canonical_url(self, article_path: str) -> str
    def get_syndication_status(self, article: Article) -> dict
    def inject_canonical_frontmatter(self, content: str, url: str) -> str
```

#### T3.2 更新平台适配逻辑
- **修改**: `content-repurposing` Skill
- **工期**: 0.5天
- **内容**:
  - Medium版本添加canonicalUrl
  - Dev.to版本添加canonical_url frontmatter
  - LinkedIn版本添加Full Report链接

---

### E7: 可读性指标体系

#### T7.1 开发可读性检测工具
- **文件**: `tools/readability_checker.py`
- **工期**: 0.5天
- **依赖**: `pip install py-readability-metrics`

```python
# 核心功能
class ReadabilityChecker:
    def analyze(self, text: str) -> ReadabilityReport
    def get_grade_level_recommendation(self, score: float) -> str
    def suggest_improvements(self, text: str) -> list[str]
```

#### T7.2 集成到质量验证流程
- **修改**: `quality-verifying` Skill
- **工期**: 0.5天
- **验收标准**: 发布前核验清单包含可读性检查

---

## 🟡 Phase 2: 能力增强 (Week 3-4)

### E4: Multi-Agent内容管线

#### T4.1 安装CrewAI框架
- **命令**: `pip install crewai crewai-tools`
- **工期**: 0.25天

#### T4.2 定义Researcher Agent
- **文件**: `agents/researcher.py`
- **工期**: 1天

```python
researcher = Agent(
    role="Security Researcher",
    goal="发现高价值威胁情报主题并收集权威资料",
    backstory="资深网络安全研究员，专注APT和漏洞分析",
    tools=[WebSearchTool(), IntelOwlTool(), OpenCTITool()],
    allow_delegation=False,
    verbose=True
)
```

#### T4.3 定义Writer Agent
- **文件**: `agents/writer.py`
- **工期**: 1天

#### T4.4 定义Editor Agent
- **文件**: `agents/editor.py`
- **工期**: 1天

#### T4.5 定义Publisher Agent
- **文件**: `agents/publisher.py`
- **工期**: 1天

#### T4.6 创建Content Crew
- **文件**: `agents/content_crew.py`
- **工期**: 1天

```python
content_crew = Crew(
    agents=[researcher, writer, editor, publisher],
    tasks=[research_task, write_task, edit_task, publish_task],
    process=Process.sequential,
    verbose=True
)
```

#### T4.7 集成到n8n工作流
- **文件**: `workflows/n8n/multi-agent-content-pipeline.json`
- **工期**: 0.5天

---

### E5: 深度AI检测对抗层

#### T5.1 增强deep_ai_detector.py
- **修改**: `tools/deep_ai_detector.py`
- **工期**: 1天
- **新增**:
  - ZeroGPT API集成
  - Copyleaks API集成
  - 多检测器聚合评分

#### T5.2 开发对抗性重写器
- **文件**: `tools/adversarial_humanizer.py`
- **工期**: 1.5天

```python
class AdversarialHumanizer:
    def sentence_by_sentence_rewrite(self, text: str) -> str
    def semantic_preserving_paraphrase(self, sentence: str) -> str
    def inject_human_style(self, text: str) -> str
    def iterative_humanize(self, text: str, target_score: float) -> str
```

#### T5.3 部署本地检测模型（可选）
- **文件**: `models/ai_detector/`
- **工期**: 1天
- **模型**: RoBERTa-based AI detector
- **依赖**: transformers, torch

---

### E6: MITRE ATT&CK自动化

#### T6.1 集成ATT&CK STIX数据
- **文件**: `knowledge/attack_stix/`
- **工期**: 0.5天
- **来源**: https://github.com/mitre-attack/attack-stix-data

#### T6.2 开发TTP标注工具
- **文件**: `tools/attack_mapper.py`
- **工期**: 1.5天

```python
class ATTACKMapper:
    def identify_techniques(self, text: str) -> list[Technique]
    def map_to_attack_ids(self, techniques: list) -> list[str]
    def generate_navigator_layer(self, attack_ids: list) -> dict
    def export_stix_bundle(self, entities: list) -> dict
```

#### T6.3 增强threat-intel Skill
- **修改**: `.claude/skills/threat-intel/SKILL.md`
- **工期**: 0.5天
- **新增**:
  - 自动TTP标注指南
  - ATT&CK Navigator Layer生成
  - IOC提取与验证

#### T6.4 集成IntelOwl API
- **文件**: `tools/intelowl_client.py`
- **工期**: 1天

---

## 🟢 Phase 3: 效率优化 (Week 5-6)

### E8: n8n调度监控增强

#### T8.1 创建调度监控工作流
- **文件**: `workflows/n8n/content-scheduler-monitor.json`
- **工期**: 0.5天
- **功能**:
  - Cron定时触发
  - 执行状态日志
  - 失败告警（Telegram/Email）

#### T8.2 添加执行仪表板
- **文件**: `workflows/n8n/execution-dashboard.json`
- **工期**: 0.5天

---

### E10: 内容发现自动化

#### T10.1 集成Google Trends API
- **文件**: `tools/trend_monitor.py`
- **工期**: 0.5天

#### T10.2 集成CVE监控
- **文件**: `tools/cve_monitor.py`
- **工期**: 0.5天
- **数据源**: NVD API, CVE.org

#### T10.3 创建主题推荐工作流
- **文件**: `workflows/n8n/topic-recommender.json`
- **工期**: 1天

---

### E12: API限流与重试机制

#### T12.1 实现限流队列
- **文件**: `tools/rate_limiter.py`
- **工期**: 0.5天
- **依赖**: Redis (可选)

```python
class RateLimiter:
    def __init__(self, requests_per_minute: int):
        pass

    async def acquire(self) -> bool
    async def wait_for_slot(self) -> None
```

#### T12.2 实现指数退避重试
- **文件**: `tools/retry_handler.py`
- **工期**: 0.5天

---

## ⚪ Phase 4: 长期演进 (Month 2+)

### E9: 本地转述模型部署
- T9.1 下载T5/Pegasus模型
- T9.2 创建推理服务
- T9.3 集成到Ollama

### E11: 多语言知识图谱
- T11.1 设计术语Schema
- T11.2 部署Neo4j
- T11.3 创建术语管理工具

### E13-E15: 多媒体扩展
- T13: ZetaVideo深度集成
- T14: 音频/播客生成
- T15: 社区互动自动化

---

## 📊 任务依赖图

```
E1 (GEO) ─────┬──────> E2 (Schema)
              │
              └──────> E3 (Canonical)

E4 (Multi-Agent) ────> E5 (AI Detection)
                       │
                       └──> E6 (ATT&CK)

E7 (Readability) ────> 独立

E8 (n8n调度) ─────────> E10 (内容发现)
                       │
                       └──> E12 (限流)
```

---

## ✅ 验收标准清单

### Phase 1 验收
- [ ] GEO优化Skill可正常触发
- [ ] Schema生成工具通过Google Rich Results Test
- [ ] Canonical URL在Dev.to/Medium版本正确注入
- [ ] 可读性分数显示在质量报告中

### Phase 2 验收
- [ ] CrewAI Content Crew端到端执行成功
- [ ] 多层AI检测通过率≥90%
- [ ] ATT&CK技术自动标注准确率≥85%

### Phase 3 验收
- [ ] n8n调度任务执行成功率≥99%
- [ ] 内容发现每周推荐≥5个高价值主题
- [ ] API调用无限流错误

---

## 🔧 开发环境准备

### 依赖安装
```bash
# Python依赖
pip install crewai crewai-tools
pip install py-readability-metrics
pip install stix2
pip install pyintelowl

# n8n自托管（如需）
docker pull n8nio/n8n
```

### 环境变量
```bash
# .env
GPTZERO_API_KEY=xxx
ORIGINALITY_API_KEY=xxx
INTELOWL_API_KEY=xxx
INTELOWL_URL=http://localhost:8000
```

---

**文档生成**: Claude Opus 4.5
**最后更新**: 2026-01-10 10:00:00 +0800
