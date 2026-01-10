# GEO Optimization Skill

> **版本**: 1.0
> **创建时间**: 2026-01-10
> **触发条件**: 文章发布前、SEO优化请求、内容更新时

---

## 概述

GEO (Generative Engine Optimization) 是2026年取代传统SEO的新兴优化策略。本Skill用于优化内容以提升在AI搜索引擎（ChatGPT、Perplexity、Google AI Overviews、Claude）中的引用率和可见性。

## 核心原理

### 为什么GEO比SEO更重要（2026）

| 指标 | 数据 |
|------|------|
| ChatGPT周活用户 | 8亿+ (2025年10月) |
| Google AI Overviews覆盖率 | 15%搜索结果 |
| B2B买家使用AI决策 | 89% (Forrester) |
| AI搜索日查询量 | Perplexity 2.3亿/月 |

### GEO vs SEO 关键差异

| 维度 | 传统SEO | GEO |
|------|---------|-----|
| 目标 | 搜索结果排名 | AI回答中被引用 |
| 核心指标 | 关键词密度、反向链接 | 引用概率、上下文相关性 |
| 内容要求 | 关键词优化 | 权威性、原创数据、专家观点 |
| 成功标准 | 点击率(CTR) | 被AI引用次数 |

---

## 优化策略

### 策略1: 直接答案优化 (Zero-Click Content)

**原理**: AI搜索引擎优先引用能直接回答问题的内容。

**执行规则**:
```
✅ 强制要求:
- 文章开头30字内给出核心答案
- 使用"What is X? X is..."的直接回答格式
- 避免"在本文中，我们将探讨..."等迂回开头
```

**示例**:
```markdown
# ❌ 错误开头
本文将深入探讨零信任架构的核心概念及其在现代企业中的应用...

# ✅ 正确开头
零信任(Zero Trust)是一种安全框架，核心原则是"永不信任，始终验证"。
它要求对每个访问请求进行身份验证，无论来自内部还是外部网络。
```

### 策略2: 引用概率提升

**原理**: AI更倾向于引用包含原创数据、统计和专家观点的内容。

**执行规则**:
```
✅ 强制要求:
- 每篇文章至少包含3个独家数据/统计
- 引用权威来源(MITRE, NIST, Gartner等)
- 添加专家观点引用(带出处)
- 包含具体案例研究(非虚构)
```

**引用概率提升数据**:
| 内容类型 | 引用概率提升 |
|----------|--------------|
| 原创研究数据 | +40% |
| 行业统计 | +30% |
| 专家引用 | +25% |
| 案例研究 | +20% |

### 策略3: 实体标记与结构化

**原理**: 帮助AI理解内容中的实体(组织、技术、漏洞等)。

**执行规则**:
```
✅ 强制要求:
- 识别文章中的关键实体
- 使用Schema.org标记(配合E2 Schema工具)
- 明确实体关系(攻击者→使用→技术→针对→目标)
```

**实体类型**:
- **组织**: APT组织、安全厂商、目标企业
- **技术**: CVE编号、ATT&CK技术ID、工具名称
- **事件**: 安全事件、数据泄露、攻击活动

### 策略4: E-E-A-T强化

**原理**: Google和AI搜索引擎看重Experience(经验)、Expertise(专业)、Authoritativeness(权威)、Trustworthiness(可信)。

**执行规则**:
```
✅ 强制要求:
- 展示作者/团队资质
- 包含"我实际操作过"的真实体验描述
- 数据来源可追溯(引用链接)
- 更新日期清晰标注
```

---

## 输出格式

### GEO优化检查报告

```markdown
## 🎯 GEO优化报告

### 直接答案评分
- 开头是否直接回答: ✅/❌
- 核心答案字数: XX字
- 建议: [具体建议]

### 引用概率评分
- 原创数据数量: X个
- 权威引用数量: X个
- 专家观点数量: X个
- 案例研究数量: X个
- 总分: X/10

### 实体标记
- 识别实体: [实体列表]
- Schema类型建议: [类型]

### E-E-A-T评分
- 作者资质展示: ✅/❌
- 实际经验描述: ✅/❌
- 来源可追溯性: ✅/❌
- 更新日期标注: ✅/❌

### 优化建议
1. [建议1]
2. [建议2]
3. [建议3]
```

---

## 与其他Skill集成

| 集成Skill | 集成点 |
|-----------|--------|
| seo-optimization | 传统SEO+GEO双重优化 |
| quality-verifying | 发布前GEO检查 |
| humanized-writing | 人性化写作+GEO兼容 |

---

## 工具依赖

- `tools/geo_optimizer.py`: GEO分析和优化工具
- `tools/schema_generator.py`: 结构化数据生成(E2方案)

---

## 参考资源

- [GEO完整指南2026](https://www.superlines.io/articles/generative-engine-optimization-geo-guide)
- [How to Rank on ChatGPT, Perplexity](https://almcorp.com/blog/how-to-rank-on-chatgpt-perplexity-ai-search-engines-complete-guide-generative-engine-optimization/)
- [GEO Trends 2026](https://www.seo.com/blog/geo-trends/)

---

**创建者**: Claude Opus 4.5 (Ultrathink Protocol)
**最后更新**: 2026-01-10
