# Memos + Notion 知识库架构设计

> **项目**: pubilie_doc 知识库系统
> **版本**: 1.0.0
> **创建日期**: 2026-01-13
> **设计协议**: Ultrathink v2.7
> **预期效果**: 每日5000+文章自动入库、本地LLM处理、知识图谱融合

---

## 📋 系统概览

### 设计目标

| 目标 | 指标 | 实现方式 |
|------|------|----------|
| **大规模入库** | 5000+ 文章/天 | n8n 批量工作流 + 异步队列 |
| **快速检索** | <100ms 查询 | Memos 全文搜索 + PostgreSQL 索引 |
| **本地LLM处理** | 隐私优先 | Mac本机 Ollama + Qwen3-14B |
| **知识图谱融合** | 实体关联 | Neo4j + Memos Tag 系统 |
| **多端同步** | 随时访问 | Notion 云端 + Memos 本地 |

### 架构哲学

```
Memos = 快速捕获 + 本地存储 + 隐私优先
Notion = 结构化整理 + 团队协作 + 云端同步
知识图谱 = 实体关联 + 语义检索 + 深度分析
```

---

## 🏗️ 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Memos + Notion 知识库系统                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         数据采集层 (5000+/天)                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ GitHub  │ │ Twitter │ │ YouTube │ │  微信   │ │  arXiv  │       │   │
│  │  │Trending │ │  热点   │ │  热门   │ │ 公众号  │ │  论文   │       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │       │           │           │           │           │             │   │
│  │       └───────────┴───────────┴───────────┴───────────┘             │   │
│  │                               │                                      │   │
│  │                               ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                    n8n 工作流引擎                            │    │   │
│  │  │         (NAS: 192.168.80.2:5678)                            │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│              ┌────────────────┼────────────────┐                            │
│              ▼                ▼                ▼                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                           处理层                                       │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │ │
│  │  │   本地 LLM      │  │   内容分析器    │  │   Tag 生成器    │       │ │
│  │  │  (Mac Ollama)   │  │                 │  │                 │       │ │
│  │  │                 │  │ - 摘要生成      │  │ - 自动标签      │       │ │
│  │  │ - Qwen3-14B    │  │ - 关键词提取    │  │ - 分类映射      │       │ │
│  │  │ (高性能推理)   │  │ - 情感分析      │  │ - 优先级评估    │       │ │
│  │  │               │  │ - 相关性评分    │  │                 │       │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                               │                                             │
│              ┌────────────────┼────────────────┐                            │
│              ▼                ▼                ▼                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          存储层                                        │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │ │
│  │  │     Memos       │  │     Neo4j       │  │     Notion      │       │ │
│  │  │  (NAS Docker)   │  │  (NAS Docker)   │  │    (云端)       │       │ │
│  │  │                 │  │                 │  │                 │       │ │
│  │  │ - 快速捕获      │  │ - 实体关系      │  │ - 结构化整理    │       │ │
│  │  │ - 全文搜索      │  │ - 语义检索      │  │ - 团队协作      │       │ │
│  │  │ - Tag系统       │  │ - 大容量存储    │  │ - 可视化看板    │       │ │
│  │  │ - API驱动      │  │ - 图遍历查询    │  │                 │       │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │ │
│  │           │                   │                   │                   │ │
│  │           └───────────────────┼───────────────────┘                   │ │
│  │                               │                                       │ │
│  │                               ▼                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────┐     │ │
│  │  │                    PostgreSQL                                │     │ │
│  │  │                 (统一元数据存储)                              │     │ │
│  │  │  - 文章元数据、同步状态、处理记录、统计数据                   │     │ │
│  │  └─────────────────────────────────────────────────────────────┘     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                               │                                             │
│                               ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          应用层                                        │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │ │
│  │  │  每日选题推荐   │  │  智能检索助手   │  │  文章生成管线   │       │ │
│  │  │                 │  │                 │  │                 │       │ │
│  │  │ - 热点排序      │  │ - 自然语言查询  │  │ - 草稿生成      │       │ │
│  │  │ - 趋势分析      │  │ - 相关内容推荐  │  │ - 人性化优化    │       │ │
│  │  │ - 08:00 通知    │  │ - 知识图谱检索  │  │ - 多平台发布    │       │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 架构核心要点

| 组件 | 部署位置 | 核心职责 |
|------|----------|----------|
| **n8n** | NAS Docker (5678) | 工作流编排、任务调度 |
| **Memos** | NAS Docker (5230) | 快速捕获、本地存储、Tag系统 |
| **PostgreSQL** | NAS Docker (5432) | 统一元数据存储 |
| **Ollama** | **Mac本机 (11434)** | 本地LLM推理 (Qwen3-14B) |
| **Neo4j** | **NAS Docker (7474/7687)** | 知识图谱、大容量存储 |
| **Notion** | 云端 | 结构化整理、团队协作 |

> 💡 **设计决策**:
> - **Ollama → Mac**: 推理密集型，需要高算力 (Mac M3 50-80 tok/s vs NAS 5-10 tok/s)
> - **Neo4j → NAS**: 存储密集型，NAS大容量SSD适合图数据库持久化，无云端成本

---

## 📊 组件详细设计

### 1. Memos 部署配置 (NAS)

```yaml
# /volume1/docker/memos/docker-compose.yml
version: "3.9"

services:
  memos:
    image: neosmemo/memos:stable
    container_name: memos
    restart: unless-stopped
    ports:
      - "5230:5230"
    volumes:
      - ./data:/var/opt/memos
    environment:
      - MEMOS_MODE=prod
      - MEMOS_PORT=5230
      - MEMOS_DRIVER=postgres
      - MEMOS_DSN=postgresql://memos:memos_password@memos-db:5432/memos?sslmode=disable
    depends_on:
      - memos-db
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:5230/api/v1/ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  memos-db:
    image: postgres:16-alpine
    container_name: memos-db
    restart: unless-stopped
    volumes:
      - ./postgres:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: memos
      POSTGRES_PASSWORD: memos_password
      POSTGRES_DB: memos
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U memos -d memos"]
      interval: 30s
      timeout: 10s
      retries: 5

networks:
  default:
    name: knowledge-base
```

### 2. 本地 LLM 配置 (Mac Ollama)

> ⚠️ **重要变更**: Ollama 部署在 Mac 本机（非NAS），使用 Qwen3-14B 模型
> **原因**: NAS (AMD R1600, 32GB RAM) 性能不足以运行大规模LLM推理

#### 2.1 Mac 本机配置

```yaml
# Mac 本机 Ollama 配置
host: Mac (M3 Max / 128GB RAM)
ollama_url: http://192.168.80.1:11434  # Mac 内网 IP
model: qwen3:14b
performance:
  tokens_per_second: ~50-80 tok/s
  context_length: 32768
  memory_usage: ~12GB
```

#### 2.2 Mac 安装与配置

```bash
# 1. 安装 Ollama (如未安装)
brew install ollama

# 2. 拉取 Qwen3-14B 模型
ollama pull qwen3:14b

# 3. 配置 Ollama 监听所有网络接口（允许NAS访问）
# 编辑 ~/.zshrc 或 ~/.bashrc 添加：
export OLLAMA_HOST=0.0.0.0

# 4. 启动 Ollama 服务
ollama serve

# 5. 验证模型
ollama list
# 应显示: qwen3:14b

# 6. 测试推理
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3:14b", "prompt": "Hello", "stream": false}'
```

#### 2.3 网络配置（NAS 访问 Mac Ollama）

```bash
# 确认 Mac IP 地址
ifconfig en0 | grep "inet "
# 预期输出: inet 192.168.80.1 netmask...

# NAS 访问测试
ssh feng@192.168.80.2
curl http://192.168.80.1:11434/api/tags
# 应返回模型列表

# macOS 防火墙配置（如需要）
# System Preferences → Security & Privacy → Firewall → Allow ollama
```

#### 2.4 性能对比

| 部署方案 | 硬件 | Qwen3-14B 推理速度 | 适用场景 |
|----------|------|-------------------|----------|
| **Mac 本机** ✅ | M3 Max 128GB | **50-80 tok/s** | 推荐 |
| NAS Docker ❌ | AMD R1600 32GB | ~5-10 tok/s | 性能不足 |
| 云端 API | N/A | 取决于网络 | 成本高、隐私风险 |

### 3. Neo4j 知识图谱 (NAS Docker)

> 💡 Neo4j 是存储密集型服务，适合部署在 NAS 大容量存储上，无需云端成本

```yaml
# /volume1/docker/neo4j/docker-compose.yml
version: "3.9"

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"   # HTTP Browser
      - "7687:7687"   # Bolt protocol
    volumes:
      - ./data:/data
      - ./logs:/logs
      - ./plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/knowledge_graph_2026
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_pagecache_size=2G
      - NEO4J_dbms_memory_heap_initial__size=1G
      - NEO4J_dbms_memory_heap_max__size=2G
      # 允许外部访问
      - NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687
      - NEO4J_dbms_connector_http_listen__address=0.0.0.0:7474
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 5

networks:
  default:
    name: knowledge-base
```

#### 3.1 Neo4j 访问信息

| 项目 | 值 |
|------|-----|
| **Browser URL** | http://192.168.80.2:7474 |
| **Bolt URL** | bolt://192.168.80.2:7687 |
| **用户名** | neo4j |
| **密码** | knowledge_graph_2026 |

#### 3.2 Python 连接示例

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://192.168.80.2:7687",
    auth=("neo4j", "knowledge_graph_2026")
)

# 创建文章节点
def create_article(tx, article):
    tx.run("""
        MERGE (a:Article {id: $id})
        SET a.title = $title,
            a.summary = $summary,
            a.source = $source,
            a.hot_score = $hot_score,
            a.created_at = datetime()
    """, **article)

# 建立主题关联
def link_to_topic(tx, article_id, topic_name):
    tx.run("""
        MATCH (a:Article {id: $article_id})
        MERGE (t:Topic {name: $topic_name})
        MERGE (a)-[:ABOUT]->(t)
    """, article_id=article_id, topic_name=topic_name)
```

### 4. 数据模型设计

#### 3.1 PostgreSQL Schema (统一元数据)

```sql
-- 文章元数据表
CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,  -- github, twitter, youtube, wechat, arxiv
    source_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,  -- LLM生成的摘要
    url TEXT,
    author VARCHAR(255),

    -- 处理状态
    memos_id VARCHAR(50),  -- Memos中的ID
    notion_page_id VARCHAR(50),  -- Notion中的页面ID
    neo4j_node_id VARCHAR(50),  -- Neo4j中的节点ID

    -- 分析结果
    keywords JSONB,  -- 关键词数组
    tags VARCHAR(100)[],  -- 标签数组
    hot_score FLOAT,  -- 热度评分
    relevance_score FLOAT,  -- 相关性评分
    sentiment VARCHAR(20),  -- 情感分析

    -- 时间戳
    source_created_at TIMESTAMPTZ,
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    synced_to_memos_at TIMESTAMPTZ,
    synced_to_notion_at TIMESTAMPTZ,

    -- 唯一约束
    UNIQUE(source, source_id)
);

-- 索引
CREATE INDEX idx_articles_source ON articles(source);
CREATE INDEX idx_articles_tags ON articles USING GIN(tags);
CREATE INDEX idx_articles_hot_score ON articles(hot_score DESC);
CREATE INDEX idx_articles_collected_at ON articles(collected_at DESC);

-- 同步状态表
CREATE TABLE sync_status (
    id SERIAL PRIMARY KEY,
    article_id UUID REFERENCES articles(id),
    target VARCHAR(20) NOT NULL,  -- memos, notion, neo4j
    status VARCHAR(20) NOT NULL,  -- pending, success, failed
    error_message TEXT,
    retry_count INT DEFAULT 0,
    last_attempt_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 处理日志表
CREATE TABLE processing_logs (
    id SERIAL PRIMARY KEY,
    article_id UUID REFERENCES articles(id),
    processor VARCHAR(50) NOT NULL,  -- llm_summary, keyword_extract, tag_generate
    input_tokens INT,
    output_tokens INT,
    duration_ms INT,
    result JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### 3.2 Memos Tag 体系

```yaml
# Tag命名规范
tag_structure:
  # 来源标签
  source:
    - "#source/github"
    - "#source/twitter"
    - "#source/youtube"
    - "#source/wechat"
    - "#source/arxiv"

  # 主题分类
  topic:
    - "#topic/ai-security"
    - "#topic/llm"
    - "#topic/vulnerability"
    - "#topic/apt"
    - "#topic/threat-intel"
    - "#topic/devsecops"

  # 优先级
  priority:
    - "#priority/high"
    - "#priority/medium"
    - "#priority/low"

  # 处理状态
  status:
    - "#status/unread"
    - "#status/read"
    - "#status/processed"
    - "#status/published"

  # 时间标签
  time:
    - "#time/2026-01"
    - "#time/week-02"
```

#### 3.3 Neo4j 知识图谱模型

```cypher
// 节点类型
(:Article {
  id: string,
  title: string,
  summary: string,
  url: string,
  source: string,
  hot_score: float,
  created_at: datetime
})

(:Topic {
  name: string,
  description: string
})

(:Author {
  name: string,
  platform: string,
  profile_url: string
})

(:Technology {
  name: string,
  category: string  // language, framework, tool
})

(:Vulnerability {
  cve_id: string,
  severity: string,
  cvss: float
})

(:APTGroup {
  name: string,
  aliases: list<string>,
  country: string
})

// 关系类型
(:Article)-[:ABOUT]->(:Topic)
(:Article)-[:WRITTEN_BY]->(:Author)
(:Article)-[:MENTIONS]->(:Technology)
(:Article)-[:DISCUSSES]->(:Vulnerability)
(:Article)-[:TRACKS]->(:APTGroup)
(:Article)-[:RELATED_TO]->(:Article)
```

---

## 🔄 n8n 工作流设计

### 4.1 主工作流: 文章采集入库

```yaml
workflow_name: "Knowledge Base - Article Ingestion"
schedule: "*/30 * * * *"  # 每30分钟

nodes:
  - trigger:
      type: n8n-nodes-base.scheduleTrigger
      interval: 30 minutes

  - collect_github:
      type: n8n-nodes-base.httpRequest
      url: "http://pubilie-api:8000/api/v1/collectors/github/trending"

  - collect_twitter:
      type: n8n-nodes-base.httpRequest
      url: "http://pubilie-api:8000/api/v1/collectors/twitter/monitor"

  - merge_results:
      type: n8n-nodes-base.merge
      mode: append

  - filter_duplicates:
      type: n8n-nodes-base.code
      code: |
        // 根据source_id去重
        const seen = new Set();
        return $input.all().filter(item => {
          const key = `${item.json.source}:${item.json.source_id}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });

  - process_with_llm:
      type: n8n-nodes-base.httpRequest
      url: "http://192.168.80.1:11434/api/generate"  # Mac Ollama
      method: POST
      body: |
        {
          "model": "qwen3:14b",
          "prompt": "请为以下内容生成100字摘要和5个关键词...",
          "stream": false
        }

  - save_to_memos:
      type: n8n-nodes-memos.createMemo
      url: "http://memos:5230"
      content: "{{ $json.title }}\n\n{{ $json.summary }}\n\n{{ $json.tags.join(' ') }}"

  - save_to_postgres:
      type: n8n-nodes-base.postgres
      operation: insert
      table: articles

  - update_neo4j:
      type: n8n-nodes-base.httpRequest
      url: "http://neo4j:7474/db/data/cypher"
```

### 4.2 每日选题生成工作流

```yaml
workflow_name: "Knowledge Base - Daily Topic Generation"
schedule: "0 7 * * *"  # 每天07:00

nodes:
  - trigger:
      type: n8n-nodes-base.scheduleTrigger
      time: "07:00"

  - fetch_yesterday_articles:
      type: n8n-nodes-base.postgres
      query: |
        SELECT * FROM articles
        WHERE collected_at > NOW() - INTERVAL '24 hours'
        ORDER BY hot_score DESC
        LIMIT 100

  - analyze_trends:
      type: n8n-nodes-base.httpRequest
      url: "http://192.168.80.1:11434/api/generate"  # Mac Ollama
      body: |
        {
          "model": "qwen3:14b",
          "prompt": "分析以下100条热点，生成10个选题建议..."
        }

  - generate_drafts:
      type: n8n-nodes-base.code
      code: |
        // 生成选题草稿框架
        const topics = $input.first().json.topics;
        return topics.map(topic => ({
          json: {
            title: topic.title,
            outline: topic.outline,
            references: topic.sources,
            estimated_words: 2000
          }
        }));

  - save_to_notion:
      type: n8n-nodes-base.notion
      operation: createPage
      database_id: "{{ $env.NOTION_TOPICS_DB }}"

  - notify_telegram:
      type: n8n-nodes-base.telegram
      chatId: "{{ $env.TELEGRAM_CHAT_ID }}"
      text: |
        📰 每日选题推荐 ({{ $today }})

        {{ $json.topics.map((t, i) => `${i+1}. ${t.title}`).join('\n') }}

        详情查看 Notion 👉 [链接]
```

### 4.3 Memos 到 Notion 同步工作流

```yaml
workflow_name: "Knowledge Base - Memos to Notion Sync"
schedule: "0 */2 * * *"  # 每2小时

nodes:
  - trigger:
      type: n8n-nodes-base.scheduleTrigger
      interval: 2 hours

  - fetch_unsynced_memos:
      type: n8n-nodes-memos.searchMemos
      url: "http://memos:5230"
      filter: "#status/processed -#synced/notion"

  - transform_for_notion:
      type: n8n-nodes-base.code
      code: |
        return $input.all().map(memo => ({
          json: {
            title: memo.json.content.split('\n')[0].substring(0, 100),
            content: memo.json.content,
            tags: extractTags(memo.json.content),
            source_url: memo.json.resource?.url,
            created_at: memo.json.createdAt
          }
        }));

  - create_notion_page:
      type: n8n-nodes-base.notion
      operation: createPage
      database_id: "{{ $env.NOTION_KNOWLEDGE_DB }}"
      properties:
        Title: "{{ $json.title }}"
        Tags: "{{ $json.tags }}"
        Source: "{{ $json.source_url }}"

  - update_memos_tag:
      type: n8n-nodes-memos.updateMemo
      addTags: ["#synced/notion"]
```

---

## 🧠 本地 LLM 处理管线

### 5.1 处理流程

```python
# automation/processors/llm_processor.py
from typing import Optional
import httpx
from pydantic import BaseModel

class LLMProcessor:
    """本地LLM处理器 - 基于Mac Ollama (Qwen3-14B)"""

    def __init__(self, base_url: str = "http://192.168.80.1:11434"):
        self.base_url = base_url
        self.default_model = "qwen3:14b"

    async def generate_summary(
        self,
        content: str,
        max_length: int = 100
    ) -> str:
        """生成文章摘要"""
        prompt = f"""请为以下内容生成一个{max_length}字以内的中文摘要，
要求：简洁、准确、突出核心观点。

内容：
{content[:3000]}

摘要："""

        response = await self._generate(prompt)
        return response.strip()

    async def extract_keywords(
        self,
        content: str,
        count: int = 5
    ) -> list[str]:
        """提取关键词"""
        prompt = f"""请从以下内容中提取{count}个最重要的关键词。
仅返回关键词，用逗号分隔。

内容：
{content[:3000]}

关键词："""

        response = await self._generate(prompt)
        keywords = [k.strip() for k in response.split(',')]
        return keywords[:count]

    async def classify_topic(
        self,
        content: str
    ) -> list[str]:
        """分类主题标签"""
        topics = [
            "ai-security", "llm", "vulnerability", "apt",
            "threat-intel", "devsecops", "cloud-security",
            "web3", "mobile-security", "red-team"
        ]

        prompt = f"""根据以下内容，从给定的主题列表中选择最匹配的1-3个主题。

可选主题：{', '.join(topics)}

内容：
{content[:2000]}

选择的主题（用逗号分隔）："""

        response = await self._generate(prompt)
        selected = [t.strip() for t in response.split(',')]
        return [t for t in selected if t in topics]

    async def assess_relevance(
        self,
        content: str,
        target_topics: list[str]
    ) -> float:
        """评估内容相关性（0-1）"""
        prompt = f"""评估以下内容与目标主题的相关程度。
返回0-1之间的分数，1表示高度相关。

目标主题：{', '.join(target_topics)}

内容：
{content[:2000]}

相关性分数（仅返回数字）："""

        response = await self._generate(prompt)
        try:
            score = float(response.strip())
            return min(max(score, 0), 1)
        except ValueError:
            return 0.5

    async def _generate(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> str:
        """调用Ollama API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model or self.default_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
```

### 5.2 批量处理任务

```python
# automation/tasks/batch_processor.py
import asyncio
from automation.processors.llm_processor import LLMProcessor
from automation.storage.memos_client import MemosClient
from automation.storage.pg_store import PostgresStore

class BatchProcessor:
    """批量文章处理任务"""

    def __init__(self):
        self.llm = LLMProcessor()
        self.memos = MemosClient("http://memos:5230")
        self.pg = PostgresStore()

    async def process_batch(self, batch_size: int = 50):
        """处理一批待处理文章"""
        # 获取待处理文章
        articles = await self.pg.get_pending_articles(limit=batch_size)

        # 并发处理
        tasks = [self.process_article(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
        success = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - success

        return {"processed": success, "failed": failed}

    async def process_article(self, article: dict):
        """处理单篇文章"""
        content = article["content"] or article["title"]

        # 1. 生成摘要
        summary = await self.llm.generate_summary(content)

        # 2. 提取关键词
        keywords = await self.llm.extract_keywords(content)

        # 3. 分类主题
        topics = await self.llm.classify_topic(content)

        # 4. 评估相关性
        relevance = await self.llm.assess_relevance(
            content,
            ["ai-security", "llm", "vulnerability"]
        )

        # 5. 构建Memos内容
        tags = [
            f"#source/{article['source']}",
            f"#priority/{'high' if relevance > 0.7 else 'medium' if relevance > 0.4 else 'low'}",
            "#status/unread"
        ] + [f"#topic/{t}" for t in topics]

        memo_content = f"""## {article['title']}

{summary}

**关键词**: {', '.join(keywords)}
**来源**: {article['url']}
**采集时间**: {article['collected_at']}

{' '.join(tags)}"""

        # 6. 保存到Memos
        memo_id = await self.memos.create_memo(memo_content)

        # 7. 更新PostgreSQL
        await self.pg.update_article(
            article["id"],
            {
                "summary": summary,
                "keywords": keywords,
                "tags": topics,
                "relevance_score": relevance,
                "memos_id": memo_id,
                "processed_at": "NOW()"
            }
        )

        return {"id": article["id"], "memo_id": memo_id}
```

---

## 📡 API 接口设计

### 6.1 Memos REST API 封装

```python
# automation/storage/memos_client.py
import httpx
from typing import Optional, List

class MemosClient:
    """Memos API客户端"""

    def __init__(self, base_url: str, access_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}" if access_token else ""
        }

    async def create_memo(
        self,
        content: str,
        visibility: str = "PRIVATE"
    ) -> str:
        """创建Memo"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/memos",
                headers=self.headers,
                json={
                    "content": content,
                    "visibility": visibility
                }
            )
            response.raise_for_status()
            return response.json()["name"].split("/")[-1]

    async def search_memos(
        self,
        filter_expr: str,
        limit: int = 50
    ) -> List[dict]:
        """搜索Memos"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/memos",
                headers=self.headers,
                params={
                    "filter": filter_expr,
                    "pageSize": limit
                }
            )
            response.raise_for_status()
            return response.json().get("memos", [])

    async def update_memo(
        self,
        memo_id: str,
        content: Optional[str] = None,
        add_tags: Optional[List[str]] = None
    ) -> dict:
        """更新Memo"""
        memo = await self.get_memo(memo_id)

        if add_tags:
            current_content = memo["content"]
            new_tags = " ".join(add_tags)
            content = f"{current_content}\n{new_tags}"

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.base_url}/api/v1/memos/{memo_id}",
                headers=self.headers,
                json={"content": content}
            )
            response.raise_for_status()
            return response.json()

    async def get_memo(self, memo_id: str) -> dict:
        """获取单个Memo"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/memos/{memo_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
```

### 6.2 统一查询接口

```python
# automation/api/routes/knowledge.py
from fastapi import APIRouter, Query
from typing import Optional, List

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

@router.get("/search")
async def search_knowledge(
    query: str,
    sources: Optional[List[str]] = Query(default=None),
    topics: Optional[List[str]] = Query(default=None),
    min_relevance: float = 0.5,
    limit: int = 20
):
    """
    统一知识检索接口

    支持：
    - 全文搜索（Memos）
    - 语义检索（Neo4j）
    - 结构化过滤（PostgreSQL）
    """
    results = []

    # 1. PostgreSQL 结构化查询
    pg_results = await pg_store.search(
        query=query,
        sources=sources,
        topics=topics,
        min_relevance=min_relevance,
        limit=limit
    )
    results.extend(pg_results)

    # 2. Memos 全文搜索
    memos_results = await memos_client.search_memos(
        filter_expr=f'content ~ "{query}"',
        limit=limit
    )

    # 3. Neo4j 图检索（相关文章推荐）
    related = await neo4j_store.find_related(
        keywords=query.split(),
        limit=10
    )

    return {
        "results": results,
        "memos_count": len(memos_results),
        "related_articles": related
    }

@router.get("/stats")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    return {
        "total_articles": await pg_store.count_articles(),
        "today_collected": await pg_store.count_today(),
        "by_source": await pg_store.count_by_source(),
        "by_topic": await pg_store.count_by_topic(),
        "pending_sync": await pg_store.count_pending_sync()
    }
```

---

## 📈 监控与告警

### 7.1 健康检查

```yaml
# n8n工作流: Knowledge Base Health Check
schedule: "*/5 * * * *"  # 每5分钟

checks:
  - name: Memos API
    url: http://memos:5230/api/v1/ping
    expected_status: 200

  - name: Mac Ollama Service
    url: http://192.168.80.1:11434/api/tags
    expected_status: 200
    description: Mac本机Ollama (Qwen3-14B)

  - name: PostgreSQL
    command: pg_isready -h postgres -p 5432

  - name: Neo4j (NAS)
    url: http://192.168.80.2:7474/
    expected_status: 200
    description: NAS本地知识图谱

alerts:
  - type: telegram
    on: failure
    message: "⚠️ 知识库服务异常: {{ $json.failed_checks }}"
```

### 7.2 性能指标

```python
# automation/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 采集指标
articles_collected = Counter(
    'knowledge_articles_collected_total',
    'Total articles collected',
    ['source']
)

# 处理指标
processing_duration = Histogram(
    'knowledge_processing_duration_seconds',
    'Article processing duration',
    ['processor']
)

# 同步指标
sync_status = Gauge(
    'knowledge_sync_pending_count',
    'Pending sync count',
    ['target']
)

# LLM指标
llm_tokens_used = Counter(
    'knowledge_llm_tokens_total',
    'Total LLM tokens used',
    ['model', 'operation']
)
```

---

## 🚀 部署步骤

### Phase 1: 基础设施 (Day 1)

```bash
# ===== NAS 部署 Memos =====
ssh feng@192.168.80.2
cd /volume1/docker
mkdir -p memos && cd memos
# 上传docker-compose.yml
docker-compose up -d

# 验证Memos
curl http://localhost:5230/api/v1/ping

# ===== NAS 部署 Neo4j =====
cd /volume1/docker
mkdir -p neo4j && cd neo4j
# 上传docker-compose.yml
docker-compose up -d

# 验证Neo4j (等待30秒启动)
sleep 30
curl http://localhost:7474/
# 访问 Browser: http://192.168.80.2:7474

# ===== Mac 配置 Ollama (本机执行) =====
# 在 Mac 本机终端执行

# 1. 安装 Ollama (如未安装)
brew install ollama

# 2. 配置监听所有接口（允许NAS访问）
echo 'export OLLAMA_HOST=0.0.0.0' >> ~/.zshrc
source ~/.zshrc

# 3. 拉取 Qwen3-14B 模型
ollama pull qwen3:14b

# 4. 启动服务（建议使用 launchd 或 tmux 保持运行）
ollama serve

# 5. 验证服务
curl http://localhost:11434/api/tags

# ===== 从 NAS 测试 Mac Ollama 连通性 =====
ssh feng@192.168.80.2
curl http://192.168.80.1:11434/api/tags
# 应返回 qwen3:14b 模型信息
```

### Phase 2: 数据库迁移 (Day 2)

```bash
# 1. 创建PostgreSQL表
docker exec -i n8n-postgres psql -U n8n -d n8n < schema.sql

# 2. 配置n8n凭据
# - Memos API Token
# - Notion Integration Token
# - Ollama endpoint

# 3. 导入n8n工作流
curl -X POST http://192.168.80.2:5678/api/v1/workflows \
  -H "Authorization: Bearer $N8N_API_KEY" \
  -H "Content-Type: application/json" \
  -d @workflows/article_ingestion.json
```

### Phase 3: 集成测试 (Day 3)

```bash
# 1. 测试采集管线
curl http://localhost:8000/api/v1/collectors/github/trending

# 2. 测试LLM处理 (从NAS访问Mac Ollama)
curl -X POST http://192.168.80.1:11434/api/generate \
  -d '{"model":"qwen3:14b","prompt":"Hello","stream":false}'

# 3. 测试Memos创建
curl -X POST http://localhost:5230/api/v1/memos \
  -H "Authorization: Bearer $MEMOS_TOKEN" \
  -d '{"content":"测试Memo #test"}'

# 4. 端到端测试
python -m pytest tests/integration/test_knowledge_base.py
```

---

## 📊 预期效果

| 指标 | 目标 | 验证方式 |
|------|------|----------|
| 每日入库量 | 5000+ 文章 | PostgreSQL count |
| 处理延迟 | <5分钟 | 采集→入库时间差 |
| 查询响应 | <100ms | API 监控 |
| LLM处理成本 | $0 (本地) | 无云端调用 |
| 存储增长 | ~1GB/月 | 磁盘监控 |
| 同步成功率 | >99% | 同步状态统计 |

---

## 📎 相关资源

### 参考文档
- [Memos API 文档](https://www.usememos.com/docs/api/overview)
- [Ollama API 参考](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [n8n Memos 社区节点](https://www.npmjs.com/package/n8n-nodes-memos)

### 配置文件位置
- Memos (NAS): `/volume1/docker/memos/docker-compose.yml`
- Neo4j (NAS): `/volume1/docker/neo4j/docker-compose.yml`
- Ollama (Mac): `~/.ollama/` (模型存储)
- Ollama 环境: `~/.zshrc` (OLLAMA_HOST=0.0.0.0)
- n8n工作流: `/volume1/docker/n8n/workflows/`

### 相关项目
- pubilie_doc: `/Users/anwu/Documents/code/pubilie_doc/`
- Notion同步工具: `~/Documents/code/tools/notion-archive/`

---

**创建时间**: 2026-01-13 21:00:00 +0800
**更新时间**: 2026-01-13 21:30:00 +0800
**设计者**: Claude Opus 4.5 (Ultrathink Protocol v2.7)
**状态**: 设计完成，待实施
**架构变更**:
- Ollama: NAS Docker → Mac 本机 (Qwen3-14B，性能优化)
- Neo4j: 云端 Aura → NAS Docker (大容量存储，零成本)
