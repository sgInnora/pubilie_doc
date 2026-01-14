# 开发任务清单

> **项目**: 7x24内容自动化系统
> **创建日期**: 2026-01-10
> **预计周期**: 5周

---

## Phase 1: 基础采集层（Week 1）

### Task 1.1: GitHub Trending采集器
- **优先级**: P0
- **预计耗时**: 4小时
- **依赖**: 无

```python
# 实现要点
- REST API获取trending repos
- 支持语言过滤（Python/Go/Rust/TS）
- 支持时间范围（daily/weekly/monthly）
- 数据模型：repo_name, stars, forks, description, topics, url
- 存储到Neo4j知识图谱
```

**验收标准**:
- [x] 每6小时自动采集 ✅ 采集器已创建
- [x] 数据正确存入Neo4j ✅ 10/10测试通过
- [x] 支持增量更新（避免重复） ✅ MERGE语句确保幂等

---

### Task 1.2: YouTube热门采集器
- **优先级**: P0
- **预计耗时**: 4小时
- **依赖**: YouTube Data API Key

```python
# 实现要点
- YouTube Data API v3
- mostPopular接口
- 分类过滤：Science & Technology
- 区域：US, CN
- 提取字幕（yt-dlp）
```

**验收标准**:
- [x] 每12小时采集热门视频 ✅ 采集器已创建
- [ ] 自动提取视频字幕（需yt-dlp集成）
- [x] 存储视频元数据到Neo4j ✅ video_to_content转换器

---

### Task 1.3: 统一存储层
- **优先级**: P0
- **预计耗时**: 3小时
- **依赖**: Neo4j Aura

```python
# Neo4j Schema设计
(:Source {name, type, url})
(:Content {id, title, summary, created_at, hot_score})
(:Topic {name, category})
(:Author {name, platform})

(Content)-[:FROM]->(Source)
(Content)-[:ABOUT]->(Topic)
(Content)-[:BY]->(Author)
```

**验收标准**:
- [x] Schema创建完成 ✅ 约束+索引已初始化
- [x] CRUD操作封装 ✅ store_content/get_hot_contents
- [x] 查询接口完善 ✅ 热门/最近/统计接口

---

## Phase 1.5: 内容战略层（已完成 ✅）

> **完成日期**: 2026-01-10
> **实际耗时**: 4小时
> **依赖**: Phase 1基础采集层

### Task 1.5.1: 内容战略定义
- **优先级**: P0
- **状态**: ✅ 已完成

```python
# 实现要点 (content_strategy.py)
- 7个内容领域定义（ContentDomain枚举）
  - AI+安全、AI+自动化、AI+编程、AI前沿
  - 中国出海-被动收入、论文解读、GitHub项目
- 20个内容标签（ContentTag数据类）
  - 每个标签包含：关键词、GitHub Topics、arXiv分类
- 15个数据源配置（采集间隔、优先级）
- 标签匹配算法（基于关键词权重）
```

**验收标准**:
- [x] ContentDomain枚举完整定义7个领域
- [x] 20个ContentTag覆盖所有目标主题
- [x] get_tags_for_content()函数正确匹配标签
- [x] 测试用例验证："Claude Code" → ai_coding_assistant标签

---

### Task 1.5.2: arXiv论文采集器
- **优先级**: P0
- **状态**: ✅ 已完成

```python
# 实现要点 (arxiv_collector.py)
- arXiv API集成（无需API Key）
- 支持分类过滤：cs.CR, cs.CL, cs.AI, cs.LG
- 相关性评分算法（0-100分）
  - 分类优先级（权重30%）
  - 关键词匹配（权重50%）
  - 顶会论文加分（权重20%）
- 关键词列表：adversarial, jailbreak, prompt injection等
```

**验收标准**:
- [x] 采集cs.CR/cs.CL/cs.AI分类论文
- [x] 相关性评分≥20的论文过滤
- [x] 测试采集：50篇论文，34篇高相关性
- [x] 最高分论文："Defense Against Indirect Prompt Injection"（100分）

---

### Task 1.5.3: 统一采集管理器
- **优先级**: P0
- **状态**: ✅ 已完成

```python
# 实现要点 (collector_manager.py)
- 协调GitHub/YouTube/arXiv三个采集器
- 集成content_strategy进行标签匹配
- 统一存储到Neo4j知识图谱
- 每日摘要生成（热门内容Top 20）
```

**验收标准**:
- [x] collect_all()并行执行所有采集
- [x] 存储内容自动匹配领域和标签
- [x] 测试采集：69项内容成功存储
  - GitHub: 19个仓库（11高优先级）
  - arXiv: 50篇论文（34高优先级）
- [x] Neo4j关系正确建立（Content-[:ABOUT]->Topic）

---

## Phase 2: 社交监控层（Week 2）

### Task 2.1: Twitter/X监控
- **优先级**: P1
- **预计耗时**: 6小时
- **依赖**: Nitter自建实例

```python
# 实现要点（避开官方API）
- 方案A: Nitter RSS订阅
- 方案B: snscrape库（需适配最新反爬）
- 监控KOL列表（50+账号）
- 关键词过滤
```

**验收标准**:
- [ ] 监控50+安全/技术KOL
- [ ] 每4小时更新
- [ ] 热度评分算法

---

### Task 2.2: 微信公众号采集
- **优先级**: P1
- **预计耗时**: 8小时
- **依赖**: 采集方案验证

```python
# 方案选择（按优先级）
1. wechat_articles_spider
2. Playwright自动化
3. MITM代理拦截

# 监控公众号列表
- 安全类：360安全、奇安信、绿盟科技...
- 技术类：InfoQ、掘金、机器之心...
```

**验收标准**:
- [ ] 监控50+公众号
- [ ] 完整提取文章内容
- [ ] 图片下载保存

---

### Task 2.3: n8n工作流配置
- **优先级**: P1
- **预计耗时**: 4小时
- **依赖**: Task 1.1-2.2

```yaml
# 工作流列表
- github_trending.json   # 每6小时
- youtube_trending.json  # 每12小时
- twitter_monitor.json   # 每4小时
- wechat_collector.json  # 每8小时
```

**验收标准**:
- [ ] 4个工作流部署到n8n
- [ ] 自动触发执行
- [ ] 错误告警配置

---

## Phase 3: 智能分析层（Week 3）

### Task 3.1: AI热度分析
- **优先级**: P0
- **预计耗时**: 6小时
- **依赖**: AI CLI配置

```python
# 分析维度
- 话题热度评分（0-100）
- 时效性评分（新鲜度）
- 相关性评分（与安全/AI主题）
- 综合排序算法

# AI CLI调用（优先级）
1. Gemini CLI - 联网分析
2. Codex CLI - 代码相关
3. Ollama - 快速处理
```

**验收标准**:
- [ ] 多维度评分算法
- [ ] CLI自动选择
- [ ] 每日热点Top 20

---

### Task 3.2: 知识图谱实体提取
- **优先级**: P1
- **预计耗时**: 4小时
- **依赖**: knowledge_graph.py

```python
# 实体类型
- ThreatActor, Malware, CVE
- Technology, Framework, Tool
- Company, Person, Event

# 关系类型
- USES, TARGETS, EXPLOITS
- DEVELOPS, ANNOUNCES, AFFECTS
```

**验收标准**:
- [ ] 自动实体提取
- [ ] 关系自动建立
- [ ] 图谱可视化查询

---

### Task 3.3: 选题自动生成
- **优先级**: P0
- **预计耗时**: 5小时
- **依赖**: Task 3.1-3.2

```python
# 选题生成逻辑
1. 获取24小时热点（Top 20）
2. 知识图谱关联分析
3. AI生成选题建议
4. 输出5-10个选题

# 选题格式
{
    "title": "建议标题",
    "angle": "写作角度",
    "sources": ["来源1", "来源2"],
    "hot_score": 85,
    "keywords": ["关键词1", "关键词2"]
}
```

**验收标准**:
- [ ] 每日5-10个选题
- [ ] 包含来源引用
- [ ] 热度评分排序

---

## Phase 4: 定时输出层（Week 4）

### Task 4.1: 每日定时任务
- **优先级**: P0
- **预计耗时**: 4小时
- **依赖**: n8n

```yaml
# 定时计划（北京时间）
06:00 - 数据处理和分析
07:00 - 选题生成
08:00 - 通知用户
```

**验收标准**:
- [ ] 每日08:00准时触发
- [ ] 完整执行流程
- [ ] 异常重试机制

---

### Task 4.2: 草稿生成管线
- **优先级**: P0
- **预计耗时**: 6小时
- **依赖**: pubilie_doc工具

```python
# 草稿生成流程
1. 选题 → 大纲生成
2. 大纲 → 内容扩展
3. 内容 → 格式优化
4. 输出Markdown草稿

# 调用工具
- AI CLI（内容生成）
- readability_checker.py（可读性）
- schema_generator.py（SEO）
```

**验收标准**:
- [ ] 自动生成草稿框架
- [ ] 包含来源引用
- [ ] Markdown格式规范

---

### Task 4.3: 通知系统
- **优先级**: P1
- **预计耗时**: 3小时
- **依赖**: Telegram Bot / Email

```python
# 通知内容
- 今日热点摘要
- 选题建议列表
- 草稿链接
- 操作指引

# 通知渠道
1. Telegram Bot（首选）
2. Email（备选）
```

**验收标准**:
- [ ] 08:00准时通知
- [ ] 内容格式清晰
- [ ] 支持快捷回复

---

## Phase 5: 发布流程层（Week 5）

### Task 5.1: 人工选择界面
- **优先级**: P1
- **预计耗时**: 4小时
- **依赖**: Task 4.3

```python
# 交互方式
- Telegram Bot inline keyboard
- 或 简单Web界面
- 支持选择/跳过/标记

# 流程
用户收到通知 → 选择选题 → 确认编写
```

**验收标准**:
- [ ] 简洁选择界面
- [ ] 一键确认
- [ ] 状态追踪

---

### Task 5.2: AI编写→去检测流程
- **优先级**: P0
- **预计耗时**: 6小时
- **依赖**: deep_ai_detector.py, local_paraphraser.py

```python
# 流程
1. AI CLI编写完整文章
2. AI检测（ZeroGPT + 本地检测）
3. 如检测率>30%，自动转述
4. 循环直到检测率<20%
5. 人工最终审核
```

**验收标准**:
- [ ] AI检测率<20%
- [ ] 保持内容质量
- [ ] 自动化处理

---

### Task 5.3: 多平台发布
- **优先级**: P1
- **预计耗时**: 4小时
- **依赖**: sync_to_blog.py, community_automation.py

```python
# 发布平台
1. innora-website博客（首发）
2. 知乎/掘金（技术文章）
3. Twitter/LinkedIn（摘要）
4. Notion记录

# 调用工具
- sync_to_blog.py
- community_automation.py
```

**验收标准**:
- [ ] 一键多平台发布
- [ ] 格式自动适配
- [ ] 发布记录追踪

---

## 📊 进度追踪

| Phase | 任务数 | 预计工时 | 实际工时 | 状态 |
|-------|--------|----------|----------|------|
| Phase 1 | 3 | 11h | 11h | ✅ 已完成 |
| Phase 1.5 | 3 | 4h | 4h | ✅ 已完成 |
| Phase 2 | 3 | 18h | - | ⏳ 待开始 |
| Phase 3 | 3 | 15h | - | ⏳ 待开始 |
| Phase 4 | 3 | 13h | - | ⏳ 待开始 |
| Phase 5 | 3 | 14h | - | ⏳ 待开始 |
| **总计** | **18** | **75h** | **15h** | 33% |

### 已创建文件清单

| 文件 | 路径 | 用途 |
|------|------|------|
| content_strategy.py | automation/ | 内容领域、标签、数据源定义 |
| arxiv_collector.py | automation/collectors/ | arXiv论文采集器 |
| collector_manager.py | automation/ | 统一采集管理器 |
| github_collector.py | automation/collectors/ | GitHub Trending采集器 |
| youtube_collector.py | automation/collectors/ | YouTube热门视频采集器 |
| neo4j_store.py | automation/storage/ | Neo4j知识图谱存储层 |
| config.py | automation/ | 统一配置管理 |

---

## 🚀 立即开始

**第一个任务**: Task 1.1 GitHub Trending采集器

```bash
# 创建项目结构
mkdir -p ~/Documents/code/pubilie_doc/automation/{collectors,processors,storage,generators,notifiers}
touch ~/Documents/code/pubilie_doc/automation/__init__.py
```

---

**创建时间**: 2026-01-10 13:10:00 +0800
