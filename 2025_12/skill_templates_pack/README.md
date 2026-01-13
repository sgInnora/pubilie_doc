# Claude Code Skills 模板包 / Claude Code Skills Template Pack

> **版本 / Version**: 1.0.0
> **日期 / Date**: 2025-12-31
> **作者 / Author**: Innora AI Security Team

## 🎯 概述 / Overview

本模板包包含12个精心设计的Claude Code Skills模板，覆盖软件开发全生命周期的关键场景。每个模板均遵循Anthropic官方最佳实践，可直接部署使用。

This template pack contains 12 carefully designed Claude Code Skills templates covering key scenarios throughout the software development lifecycle. Each template follows Anthropic's official best practices and can be deployed directly.

## 📦 模板列表 / Template List

| # | 模板名称 | Template Name | 用途 / Purpose |
|---|---------|---------------|----------------|
| 01 | reviewing-code | Code Review | 代码审查（安全、性能、可维护性）|
| 02 | generating-docs | Documentation | 生成项目文档（README、架构文档）|
| 03 | auditing-security | Security Audit | 安全审计（OWASP Top 10检查）|
| 04 | documenting-apis | API Documentation | API文档生成（OpenAPI 3.0）|
| 05 | generating-tests | Test Generation | 测试生成（单元/集成/E2E）|
| 06 | refactoring-code | Code Refactoring | 代码重构（设计模式、SOLID原则）|
| 07 | translating-docs | Translation | 技术文档翻译（中英日韩）|
| 08 | managing-projects | Project Management | 项目管理（任务分解、路线图）|
| 09 | analyzing-data | Data Analysis | 数据分析（统计、趋势、异常检测）|
| 10 | generating-reports | Report Generation | 报告生成（技术报告、事故报告）|
| 11 | integrating-cicd | CI/CD Integration | CI/CD配置（GitHub Actions、GitLab CI）|
| 12 | collaborating-teams | Team Collaboration | 团队协作（PR模板、代码审查指南）|

## 🚀 快速使用 / Quick Start

### 方法一：全局安装 / Global Installation
```bash
# 复制到用户级Skills目录
cp -r ./01-reviewing-code ~/.claude/skills/
cp -r ./02-generating-docs ~/.claude/skills/
# ... 依次复制其他模板
```

### 方法二：项目级安装 / Project-level Installation
```bash
# 复制到项目的.claude目录
cp -r ./01-reviewing-code ./.claude/skills/
```

### 方法三：选择性安装 / Selective Installation
根据需要只安装特定模板。

## 📁 目录结构 / Directory Structure

```
skill_templates_pack/
├── README.md                    # 本文件
├── 01-reviewing-code/
│   └── SKILL.md                # 代码审查技能
├── 02-generating-docs/
│   └── SKILL.md                # 文档生成技能
├── 03-auditing-security/
│   └── SKILL.md                # 安全审计技能
├── 04-documenting-apis/
│   └── SKILL.md                # API文档技能
├── 05-generating-tests/
│   └── SKILL.md                # 测试生成技能
├── 06-refactoring-code/
│   └── SKILL.md                # 代码重构技能
├── 07-translating-docs/
│   └── SKILL.md                # 文档翻译技能
├── 08-managing-projects/
│   └── SKILL.md                # 项目管理技能
├── 09-analyzing-data/
│   └── SKILL.md                # 数据分析技能
├── 10-generating-reports/
│   └── SKILL.md                # 报告生成技能
├── 11-integrating-cicd/
│   └── SKILL.md                # CI/CD集成技能
└── 12-collaborating-teams/
    └── SKILL.md                # 团队协作技能
```

## 🔧 自定义指南 / Customization Guide

### SKILL.md 结构 / Structure
```yaml
---
name: skill-name              # 技能名称（必填，≤64字符）
description: ...              # 描述和触发条件（必填，≤1024字符）
---

# Skill Title                 # Markdown正文（建议<500行）

## Overview
...

## Templates
...

## Constraints
...
```

### 最佳实践 / Best Practices
1. **描述使用第三人称**：如"Performs..."、"Generates..."
2. **包含触发关键词**：如"Triggers when user asks..."
3. **提供具体示例**：帮助Claude理解期望输出
4. **定义明确约束**：防止不当行为

## 📋 使用场景 / Use Cases

### 场景1：代码审查
```
用户: "Review this authentication module for security issues"
Claude: [自动激活 reviewing-code skill，执行安全审查]
```

### 场景2：文档生成
```
用户: "Generate API documentation for this Express router"
Claude: [自动激活 documenting-apis skill，生成OpenAPI规范]
```

### 场景3：CI/CD配置
```
用户: "Setup GitHub Actions for this Python project"
Claude: [自动激活 integrating-cicd skill，生成工作流配置]
```

## ⚠️ 注意事项 / Notes

1. **Token消耗**：Skills会增加上下文Token消耗，建议按需启用
2. **优先级**：项目级Skills优先于用户级，用户级优先于全局
3. **调试**：使用 `/skills` 命令查看已加载的Skills
4. **更新**：定期检查模板是否需要更新以匹配新版本Claude Code

## 📚 相关资源 / Related Resources

- [Claude Code Skills深度指南](../Claude_Code_Skills深度指南_从入门到精通_CN.md)
- [Claude Skills Complete Guide](../Claude_Skills_Complete_Guide_From_Beginner_to_Expert_EN.md)
- [Anthropic官方文档](https://docs.anthropic.com/en/docs/claude-code)

## 📄 许可证 / License

MIT License - 可自由使用、修改和分发

---

**制作 / Created by**: Innora AI Security Team
**联系 / Contact**: security@innora.ai
