---
name: translating-docs
description: Translates technical documentation between languages while preserving technical accuracy, formatting, and context. Maintains terminology consistency using glossaries. Supports Chinese, English, Japanese, Korean, and European languages. Triggers when user asks to "translate", "localize", or "create multilingual docs".
---

# Technical Documentation Translation Skill

## Overview
Provides accurate, context-aware translation of technical documentation while preserving formatting, code examples, and domain-specific terminology.

## Supported Languages
- English ↔ Chinese (Simplified/Traditional)
- English ↔ Japanese
- English ↔ Korean
- English ↔ German/French/Spanish
- Cross-language pairs available

## Translation Process

```
Phase 1: Preparation
├── Analyze document structure
├── Extract technical terms
├── Build/load terminology glossary
└── Identify non-translatable elements

Phase 2: Translation
├── Translate content sections
├── Preserve code blocks
├── Maintain formatting
└── Apply consistent terminology

Phase 3: Review
├── Technical accuracy check
├── Terminology consistency
├── Format validation
└── Cultural adaptation

Phase 4: Finalization
├── Generate bilingual glossary
├── Create translation notes
└── Produce final document
```

## Terminology Management

### Glossary Template
```yaml
terminology:
  - source: "API endpoint"
    target_zh: "API 端点"
    target_ja: "APIエンドポイント"
    context: "REST API documentation"

  - source: "authentication"
    target_zh: "身份验证"
    target_ja: "認証"
    context: "Security context"

  - source: "deployment"
    target_zh: "部署"
    target_ja: "デプロイ"
    context: "DevOps context"

  - source: "container"
    target_zh: "容器"
    target_ja: "コンテナ"
    context: "Docker/Kubernetes"
```

### Common Technical Terms (EN → ZH)

| English | 中文 | Notes |
|---------|------|-------|
| Repository | 仓库 | Git context |
| Branch | 分支 | Git context |
| Commit | 提交 | Git context |
| Pull Request | 拉取请求 | GitHub workflow |
| Merge | 合并 | Git context |
| Deploy | 部署 | DevOps |
| Build | 构建 | CI/CD |
| Pipeline | 管道/流水线 | CI/CD |
| Container | 容器 | Docker |
| Cluster | 集群 | Kubernetes |
| Node | 节点 | Infrastructure |
| Service | 服务 | Microservices |
| Endpoint | 端点 | API |
| Middleware | 中间件 | Architecture |
| Framework | 框架 | Development |

## Translation Rules

### Preserve Elements
- Code blocks (``` ```)
- Inline code (` `)
- File paths and URLs
- Variable names
- Command examples
- Version numbers
- Technical abbreviations (API, SDK, CLI)

### Adapt Elements
- Date formats (MM/DD/YYYY → YYYY年MM月DD日)
- Currency symbols
- Measurement units
- Cultural references
- Humor and idioms

### Format Preservation
```markdown
<!-- Source (English) -->
# Quick Start Guide

Install the package:
```bash
npm install my-package
```

Configure the settings in `config.json`:
```json
{
  "apiKey": "your-api-key"
}
```

<!-- Target (Chinese) -->
# 快速入门指南

安装软件包：
```bash
npm install my-package
```

在 `config.json` 中配置设置：
```json
{
  "apiKey": "your-api-key"
}
```
```

## Output Format

### Translation Output Structure
```markdown
# {Translated Title}

> **Original**: {original_file_path}
> **Language**: {source} → {target}
> **Translated**: {date}

{Translated content with preserved formatting}

---

## Translation Notes

### Terminology Decisions
| Term | Translation | Rationale |
|------|-------------|-----------|
| {term1} | {translation1} | {reason} |

### Cultural Adaptations
- {adaptation_1}
- {adaptation_2}

### Untranslated Elements
- Code blocks: Preserved as-is
- Technical commands: Preserved as-is
- {other_elements}
```

### Bilingual Document Template
```markdown
# Title / 标题

## English Version

{English content}

---

## 中文版本

{Chinese content}

---

## Glossary / 术语表

| English | 中文 |
|---------|------|
| term1 | 术语1 |
| term2 | 术语2 |
```

## Quality Checklist
- [ ] All code blocks preserved unchanged
- [ ] Technical terms consistently translated
- [ ] Formatting matches source document
- [ ] Links and references updated
- [ ] Numbers and units correctly adapted
- [ ] No machine translation artifacts
- [ ] Natural, fluent target language
- [ ] Cultural references appropriately adapted

## Constraints
- Never translate code or command examples
- Maintain original document structure
- Use established terminology standards
- Preserve all Markdown/HTML formatting
- Include translator notes for ambiguous terms
- Flag cultural references that may not translate
