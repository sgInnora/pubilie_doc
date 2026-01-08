---
name: article-writing
description: Writes high-quality technical articles on AI security, threat intelligence, and cybersecurity topics. Use when creating new articles, drafting technical analysis, or producing bilingual documentation. Triggers on article creation, technical writing, or documentation requests.
allowed-tools: Read, Write, Edit, Grep, Glob, WebSearch, WebFetch
---

# Article Writing Skill

## Core Capabilities
Produces professional technical articles (≥12,000 words) with bilingual CN/EN versions.

## Prerequisites
Before writing, ensure:
1. ✅ Time verification passed (3-source check)
2. ✅ Research conducted (≥3 verified sources)
3. ✅ Topic not duplicated in existing articles

## Article Templates

### Threat Analysis Template
```markdown
# [威胁名称]深度技术分析与防御策略

> **注**：本文基于公开信息和行业趋势分析编写。具体数据请以官方最新信息为准。

## 执行摘要
[300-500字核心发现，包含关键数据点]

## 1. 威胁概述
### 1.1 发现历程
### 1.2 归属分析
### 1.3 影响范围

## 2. 技术深度分析
### 2.1 攻击链分析
### 2.2 工具与技术
### 2.3 基础设施

## 3. MITRE ATT&CK映射
| 战术 | 技术 | 编号 | 描述 |
|------|------|------|------|

## 4. 防御策略
### 4.1 检测方法
### 4.2 响应措施
### 4.3 预防建议

## 5. 威胁指标（IoCs）
### 5.1 文件哈希
### 5.2 网络指标
### 5.3 检测规则

## 6. 结论

## 参考文献
```

### Technology Deep-Dive Template
```markdown
# [技术名称]深度解析与实践指南

> **注**：本文基于公开信息和行业趋势分析编写。

## 执行摘要

## 1. 技术背景
### 1.1 发展历程
### 1.2 核心原理
### 1.3 应用场景

## 2. 技术架构
### 2.1 系统组件
### 2.2 工作流程
### 2.3 关键算法

## 3. 实现细节
### 3.1 代码示例
### 3.2 配置说明
### 3.3 最佳实践

## 4. 性能分析
### 4.1 基准测试
### 4.2 优化策略

## 5. 安全考虑

## 6. 结论与展望

## 参考文献
```

## Writing Standards

### Quality Requirements
- ✅ Minimum 12,000 words (CN version)
- ✅ Executive summary (300-500 words)
- ✅ Runnable code examples
- ✅ Verifiable data sources
- ✅ Disclaimer at top
- ✅ Complete references (≥5 sources)

### Prohibited
- ❌ Fabricated statistics
- ❌ Non-existent reports
- ❌ Unverified percentages
- ❌ Plagiarized content

## Terminology Reference
See: `WRITING_STYLE_GUIDE.md` §Terminology

| English | 中文 |
|---------|------|
| Prompt Injection | 提示注入 |
| Jailbreak | 越狱攻击 |
| Adversarial Attack | 对抗性攻击 |
| Red Teaming | 红队测试 |
| Zero-day | 零日漏洞 |
| APT | 高级持续性威胁 |

## File Naming Convention
```
{YYYY_MM}/
├── {主题}_CN.md
└── {Topic}_EN.md
```

## Writing Process
1. **Research** (30%): Use `research-conducting` skill
2. **Draft** (50%): Follow template, CN version first
3. **Verify** (20%): Use `quality-verifying` skill

## Usage Examples

### Example 1: APT Threat Report
```
User: "Write an article about the latest APT29 campaign"

Skill Actions:
1. Trigger `research-conducting` skill (minimum 5 sources)
2. Apply Threat Analysis Template from §Article Templates
3. Include MITRE ATT&CK mapping table
4. Generate CN/EN bilingual versions (≥12,000 words each)
5. Add IoCs section with detection rules
6. Output: Two files in {YYYY_MM}/ directory
```

### Example 2: Technology Deep-Dive
```
User: "Create an article analyzing FlipAttack's 81% success rate"

Skill Actions:
1. Research attack methodology and defenses bypassed
2. Apply Technology Deep-Dive Template
3. Include code examples for detection
4. Cite academic sources (arXiv, IEEE)
5. Add practical defense recommendations
6. Output: CN/EN versions with code blocks
```

### Example 3: Vulnerability Analysis
```
User: "Write a comprehensive analysis of CVE-2025-XXXX"

Skill Actions:
1. Verify CVE details from NVD and vendor advisories
2. Structure with attack chain analysis
3. Include proof-of-concept explanation (no live exploits)
4. Add detection rules (YARA/Sigma)
5. Provide patching and mitigation guidance
6. Output: Technical report with IoC appendix
```

## Post-Writing Checklist（2026-01-01新增）

> **文章完成后必须执行**

### 发布前必检项
```yaml
before_platform_distribution:
  1. link_verification:
     - "检查所有GitHub链接包含完整路径（/blob/main/...）"
     - "Full Report链接指向正确的GitHub版本"

  2. contact_consistency:
     - "联系邮箱统一为 security@innora.ai"
     - "作者信息统一为 Innora Security Research Team"

  3. cover_image:
     - "生成封面图片Prompt"
     - "保存到 assets/{Topic}_Cover.png"

  4. file_completeness:
     - "{Topic}_CN.md 存在"
     - "{Topic}_EN.md 存在"
```

### 验证命令
```bash
# 验证链接完整性
grep -n "github.com/sgInnora/pubilie_doc" article.md | grep -v "blob/main"
# 空输出 = 通过

# 验证邮箱统一
grep "@innora" article.md | grep -v "security@innora.ai"
# 空输出 = 通过
```

## Integration
- Requires `research-conducting` for sources
- Outputs to `quality-verifying` for review
- **New**: Runs post-writing checklist before `content-adapting`
- Feeds into `content-adapting` for distribution
