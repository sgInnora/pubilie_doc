# 技术文章自动归档规则

*生效日期：2025年7月30日*

## 📁 目录结构规范

所有技术文章必须按照以下目录结构进行归档：

```
pubilie_doc/
├── README.md                          # 项目索引和导航
├── WRITING_STYLE_GUIDE.md            # 写作风格指南
├── ARTICLE_WRITING_ACCURACY_RULES.md # 准确性规则
├── ARTICLE_ACCURACY_CHECKLIST.md     # 检查清单
├── ARTICLE_ARCHIVING_RULES.md        # 本文档
├── CLAUDE.md                         # 项目配置
│
├── 2025_01/                          # 年份_月份目录
│   ├── 文章标题_CN.md               # 中文版本
│   ├── Article_Title_EN.md          # 英文版本
│   └── assets/                      # 文章资源文件
│       ├── images/                  # 图片
│       └── diagrams/                # 图表
│
├── 2025_02/
├── 2025_03/
└── ...
```

## 🗂️ 归档规则

### 1. 目录命名规范
- **格式**：`YYYY_MM`（例如：2025_07）
- **创建时机**：每月第一篇文章发布时自动创建
- **排序**：按时间顺序排列，最新月份在下

### 2. 文件命名规范

#### 中文文章
- **格式**：`[主题描述]_CN.md`
- **示例**：`AI驱动的攻击面管理产品深度分析与实践_CN.md`
- **要求**：
  - 使用中文标题
  - 避免特殊字符
  - 控制在50个字符以内

#### 英文文章
- **格式**：`[Topic_Description]_EN.md`
- **示例**：`AI_Driven_Attack_Surface_Management_Deep_Analysis_EN.md`
- **要求**：
  - 使用下划线分隔单词
  - 首字母大写
  - 避免缩写（除非广泛认可）

### 3. 资源文件归档
- 每篇文章的图片、图表等资源放在同月份的`assets`子目录
- 图片命名：`article_name_fig_01.png`
- 图表命名：`article_name_diagram_01.svg`

## 🔄 自动归档流程

### 新文章创建时
1. **检查当前日期**：使用`date`命令获取真实日期
2. **确定目标目录**：`YYYY_MM`格式
3. **创建目录**（如不存在）
4. **生成文件**：在正确目录下创建中英文版本
5. **更新索引**：自动更新README.md

### 归档命令示例
```bash
# 获取当前年月
YEAR_MONTH=$(date +"%Y_%m")

# 创建目录（如不存在）
mkdir -p pubilie_doc/$YEAR_MONTH/assets/{images,diagrams}

# 创建文章
touch pubilie_doc/$YEAR_MONTH/文章标题_CN.md
touch pubilie_doc/$YEAR_MONTH/Article_Title_EN.md
```

## 📋 README.md 自动更新

每次添加新文章时，README.md 应自动更新，格式如下：

```markdown
### 2025年7月文档
- **[文章标题](./2025_07/文章标题_CN.md)** | [English Version](./2025_07/Article_Title_EN.md)
  - 简短描述（1-2行）
```

## 🏷️ 元数据标准

每篇文章开头必须包含元数据：

```markdown
---
title: 文章标题
author: Innora安全研究团队
date: 2025-07-30
category: AI安全/威胁情报/渗透测试
tags: [标签1, 标签2, 标签3]
version: 1.0
---
```

## 🔍 搜索和导航

### 分类索引
在README.md中维护分类索引：

```markdown
## 按主题分类

### AI安全
- [文章1](路径)
- [文章2](路径)

### 威胁情报
- [文章3](路径)
- [文章4](路径)
```

### 时间索引
按时间倒序排列，最新文章在前

## ⚙️ 自动化脚本

### archive_article.sh
```bash
#!/bin/bash
# 自动归档新文章的脚本

# 获取参数
TITLE_CN=$1
TITLE_EN=$2

# 获取当前日期
CURRENT_DATE=$(date +"%Y-%m-%d")
YEAR_MONTH=$(date +"%Y_%m")

# 创建目录
mkdir -p "pubilie_doc/$YEAR_MONTH/assets/{images,diagrams}"

# 创建文章文件
cat > "pubilie_doc/$YEAR_MONTH/${TITLE_CN}_CN.md" << EOF
---
title: $TITLE_CN
author: Innora安全研究团队
date: $CURRENT_DATE
category: 待分类
tags: []
version: 1.0
---

> **注**：本文基于公开信息和行业趋势分析编写，旨在探讨[主题]。
> 具体产品功能和数据请以官方最新信息为准。

# $TITLE_CN

*作者：Innora安全研究团队 | 发布时间：$CURRENT_DATE*

## 执行摘要

[执行摘要内容]

**关键词：** 关键词1, 关键词2, 关键词3
EOF

# 更新README.md
# ... (自动更新逻辑)

echo "文章已归档到: pubilie_doc/$YEAR_MONTH/"
```

## 🚨 强制执行规则

1. **禁止手动创建在根目录**
   - 所有文章必须创建在对应的年月目录
   - 根目录只保留配置和指南文件

2. **禁止跨月修改**
   - 文章一旦归档，路径不可更改
   - 更新只能在原位置进行

3. **必须双语同步**
   - 中英文版本必须在同一目录
   - 不允许只有单语版本

## 📊 归档统计

### 月度统计
每月底自动生成统计：
- 本月发布文章数
- 各类别文章分布
- 阅读量统计（如有）

### 年度汇总
每年底生成年度报告：
- 全年文章总数
- 主题分布分析
- 热门文章排行

## 🔧 维护指南

### 定期检查
- 每周检查目录结构完整性
- 每月验证所有链接有效性
- 每季度审查归档规则适用性

### 清理规则
- 保留最近2年的详细归档
- 更早的文章可以按年归档
- 永久保留重要技术文档

## 💡 最佳实践

1. **提前规划**：月初规划当月文章主题
2. **批量处理**：使用脚本批量创建和归档
3. **版本控制**：所有变更通过git管理
4. **备份策略**：定期备份到云存储

---

*本规则将持续优化，确保文档管理的高效性和可维护性。*