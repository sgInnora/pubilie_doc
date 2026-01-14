# Reddit 集成技术指南

> **项目**: pubilie_doc Reddit 发布与监控
> **版本**: 1.0.0
> **创建日期**: 2026-01-13
> **状态**: 调研完成，待实施

---

## 📋 系统概览

### 集成目标

| 功能 | 描述 | 优先级 |
|------|------|--------|
| **内容发布** | 自动发布 AI/安全文章到相关 subreddit | P0 |
| **热点监控** | 监控目标 subreddit 的热门帖子 | P0 |
| **关键词追踪** | 跟踪特定关键词的新帖子 | P1 |
| **互动管理** | 监控评论和回复通知 | P2 |

---

## 🔐 1. Reddit API 认证

### 1.1 创建 Reddit App

1. 访问 [Reddit App Preferences](https://old.reddit.com/prefs/apps/)
2. 点击 "create another app..."
3. 填写信息：
   - **name**: `pubilie-bot`
   - **type**: 选择 `script` (个人使用)
   - **description**: `Content automation for AI/Security articles`
   - **redirect uri**: `http://localhost:8080`
4. 点击 "create app"

### 1.2 获取凭据

创建后获取：
- **client_id**: 应用名称下方的 14+ 字符字符串
- **client_secret**: `secret` 旁边的 27+ 字符字符串

```
┌─────────────────────────────────────────┐
│ pubilie-bot                             │
│ personal use script                     │
│ ──────────────────────────────────────  │
│ client_id: Ab3CdEfGhIjKlM              │  ← 这个
│ secret: AbCdEfGhIjKlMnOpQrStUvWxYz123  │  ← 这个
│ redirect uri: http://localhost:8080     │
└─────────────────────────────────────────┘
```

### 1.3 认证方式对比

| 方式 | 适用场景 | 复杂度 |
|------|----------|--------|
| **Password Flow** | 个人脚本，单账户 | ⭐ 简单 |
| **Code Flow** | 多用户应用，需用户授权 | ⭐⭐⭐ 复杂 |
| **Refresh Token** | 长期运行的自动化任务 | ⭐⭐ 中等 |

**推荐**: 使用 Password Flow 进行自动化发布

---

## 📊 2. API 限制

### 2.1 Rate Limits

| 认证类型 | 限制 | 时间窗口 |
|----------|------|----------|
| **OAuth 认证** | 100 QPM | 10分钟平均 |
| **未认证** | 10 QPM | - |

> ⚠️ **重要**: 自 2023-07-01 起，未认证请求会被阻断

### 2.2 响应头监控

```python
# 从响应头获取限制信息
X-Ratelimit-Used: 45        # 当前周期已使用
X-Ratelimit-Remaining: 55   # 剩余可用
X-Ratelimit-Reset: 120      # 重置倒计时（秒）
```

### 2.3 发布限制

| 限制类型 | 要求 |
|----------|------|
| 账号年龄 | 通常需要 >7 天 |
| Karma | 部分 subreddit 要求 >10 |
| 发帖间隔 | 同一 subreddit 约 10 分钟 |
| 全局发帖 | 约 1 帖/分钟 |

---

## 🎯 3. 目标 Subreddit 列表

### 3.1 AI/机器学习 (发布 + 监控)

| Subreddit | 成员数 | 内容类型 | 自发布适合度 |
|-----------|--------|----------|--------------|
| r/artificial | 1M+ | 新闻/讨论 | ⭐⭐⭐ 高 |
| r/MachineLearning | 3M+ | 技术/论文 | ⭐⭐ 中 |
| r/ArtificialIntelligence | 1.4M+ | 新闻/产品 | ⭐⭐⭐ 高 |
| r/LocalLLaMA | 500K+ | 开源LLM | ⭐⭐⭐ 高 |
| r/ChatGPT | 9M+ | ChatGPT相关 | ⭐⭐ 中 |
| r/OpenAI | 2M+ | OpenAI产品 | ⭐⭐ 中 |
| r/learnmachinelearning | 400K+ | 教程/入门 | ⭐⭐ 中 |
| r/Singularity | 1.8M+ | AI未来 | ⭐⭐ 中 |
| r/AGI | 62K+ | AGI讨论 | ⭐⭐ 中 |

### 3.2 网络安全 (发布 + 监控)

| Subreddit | 成员数 | 内容类型 | 自发布适合度 |
|-----------|--------|----------|--------------|
| r/cybersecurity | 4M+ | 综合安全 | ⭐⭐⭐ 高 |
| r/netsec | 600K+ | 技术深度 | ⭐⭐ 中(严格) |
| r/hacking | 3M+ | 黑客技术 | ⭐⭐ 中 |
| r/InfoSecNews | 100K+ | 安全新闻 | ⭐⭐⭐ 高 |
| r/ethicalhacking | 200K+ | 渗透测试 | ⭐⭐ 中 |
| r/cybersecurityai | 新 | AI+安全 | ⭐⭐⭐ 高 |
| r/learncybersecurity | 100K+ | 安全入门 | ⭐⭐ 中 |

### 3.3 推荐发布策略

```yaml
# 发布优先级配置
publishing_strategy:
  tier_1:  # 高优先级，每周1-2篇
    - r/artificial
    - r/cybersecurity
    - r/LocalLLaMA
  tier_2:  # 中优先级，每周1篇
    - r/ArtificialIntelligence
    - r/InfoSecNews
    - r/cybersecurityai
  tier_3:  # 低优先级，精选内容
    - r/MachineLearning
    - r/netsec
```

---

## 🐍 4. Python 实现 (PRAW)

### 4.1 安装

```bash
pip install praw asyncpraw
```

### 4.2 基础配置

```python
# config/reddit_config.py
import praw
from praw.models import Subreddit
import os

class RedditClient:
    """Reddit API 客户端封装"""

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent="pubilie-bot/1.0 by u/your_username"
        )

    def verify_auth(self) -> bool:
        """验证认证状态"""
        try:
            return self.reddit.user.me() is not None
        except Exception as e:
            print(f"Auth failed: {e}")
            return False
```

### 4.3 发布功能

```python
# publishers/reddit_publisher.py
from typing import Optional, List
import time

class RedditPublisher:
    """Reddit 内容发布器"""

    def __init__(self, client: RedditClient):
        self.reddit = client.reddit
        self.last_post_time = {}

    def submit_text_post(
        self,
        subreddit: str,
        title: str,
        content: str,
        flair_id: Optional[str] = None
    ) -> dict:
        """
        发布文字帖子

        Args:
            subreddit: 目标 subreddit 名称
            title: 帖子标题
            content: Markdown 格式内容
            flair_id: 可选的 flair ID

        Returns:
            包含帖子信息的字典
        """
        # 检查发布间隔
        self._check_rate_limit(subreddit)

        sub = self.reddit.subreddit(subreddit)

        try:
            submission = sub.submit(
                title=title,
                selftext=content,
                flair_id=flair_id
            )

            self.last_post_time[subreddit] = time.time()

            return {
                "success": True,
                "id": submission.id,
                "url": f"https://reddit.com{submission.permalink}",
                "subreddit": subreddit
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "subreddit": subreddit
            }

    def submit_link_post(
        self,
        subreddit: str,
        title: str,
        url: str,
        flair_id: Optional[str] = None
    ) -> dict:
        """发布链接帖子"""
        self._check_rate_limit(subreddit)

        sub = self.reddit.subreddit(subreddit)

        try:
            submission = sub.submit(
                title=title,
                url=url,
                flair_id=flair_id
            )

            self.last_post_time[subreddit] = time.time()

            return {
                "success": True,
                "id": submission.id,
                "url": f"https://reddit.com{submission.permalink}",
                "subreddit": subreddit
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _check_rate_limit(self, subreddit: str, min_interval: int = 600):
        """检查发布间隔（默认10分钟）"""
        if subreddit in self.last_post_time:
            elapsed = time.time() - self.last_post_time[subreddit]
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                print(f"Rate limit: waiting {wait_time:.0f}s for r/{subreddit}")
                time.sleep(wait_time)
```

### 4.4 监控功能

```python
# collectors/reddit_collector.py
from typing import List, Generator
from datetime import datetime, timedelta

class RedditCollector:
    """Reddit 内容采集器"""

    def __init__(self, client: RedditClient):
        self.reddit = client.reddit

    def get_hot_posts(
        self,
        subreddit: str,
        limit: int = 25
    ) -> List[dict]:
        """获取热门帖子"""
        sub = self.reddit.subreddit(subreddit)
        posts = []

        for post in sub.hot(limit=limit):
            posts.append(self._parse_post(post))

        return posts

    def get_new_posts(
        self,
        subreddit: str,
        limit: int = 25
    ) -> List[dict]:
        """获取最新帖子"""
        sub = self.reddit.subreddit(subreddit)
        posts = []

        for post in sub.new(limit=limit):
            posts.append(self._parse_post(post))

        return posts

    def search_posts(
        self,
        query: str,
        subreddit: str = "all",
        sort: str = "relevance",
        time_filter: str = "week",
        limit: int = 25
    ) -> List[dict]:
        """
        搜索帖子

        Args:
            query: 搜索关键词
            subreddit: 目标 subreddit，"all" 为全站搜索
            sort: relevance, hot, top, new, comments
            time_filter: hour, day, week, month, year, all
            limit: 返回数量
        """
        sub = self.reddit.subreddit(subreddit)
        posts = []

        for post in sub.search(
            query,
            sort=sort,
            time_filter=time_filter,
            limit=limit
        ):
            posts.append(self._parse_post(post))

        return posts

    def stream_new_posts(
        self,
        subreddits: List[str]
    ) -> Generator[dict, None, None]:
        """
        实时流监控新帖子

        Args:
            subreddits: subreddit 列表，用 + 连接

        Yields:
            新帖子字典
        """
        sub_str = "+".join(subreddits)
        sub = self.reddit.subreddit(sub_str)

        for post in sub.stream.submissions(skip_existing=True):
            yield self._parse_post(post)

    def _parse_post(self, post) -> dict:
        """解析帖子为标准字典格式"""
        return {
            "id": post.id,
            "title": post.title,
            "author": str(post.author) if post.author else "[deleted]",
            "subreddit": post.subreddit.display_name,
            "url": f"https://reddit.com{post.permalink}",
            "external_url": post.url if not post.is_self else None,
            "content": post.selftext if post.is_self else None,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat(),
            "is_self": post.is_self,
            "flair": post.link_flair_text
        }
```

### 4.5 异步版本 (asyncpraw)

```python
# collectors/async_reddit_collector.py
import asyncpraw
import asyncio
from typing import List

class AsyncRedditCollector:
    """异步 Reddit 采集器"""

    def __init__(self):
        self.reddit = asyncpraw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent="pubilie-bot/1.0 by u/your_username"
        )

    async def get_hot_from_multiple(
        self,
        subreddits: List[str],
        limit: int = 10
    ) -> List[dict]:
        """并发获取多个 subreddit 的热门帖子"""
        tasks = [
            self._get_hot(sub, limit)
            for sub in subreddits
        ]
        results = await asyncio.gather(*tasks)

        # 合并并按分数排序
        all_posts = []
        for posts in results:
            all_posts.extend(posts)

        return sorted(all_posts, key=lambda x: x["score"], reverse=True)

    async def _get_hot(self, subreddit: str, limit: int) -> List[dict]:
        """获取单个 subreddit 热门帖子"""
        sub = await self.reddit.subreddit(subreddit)
        posts = []

        async for post in sub.hot(limit=limit):
            posts.append({
                "id": post.id,
                "title": post.title,
                "subreddit": subreddit,
                "score": post.score,
                "url": f"https://reddit.com{post.permalink}"
            })

        return posts

    async def close(self):
        """关闭连接"""
        await self.reddit.close()
```

---

## 🔄 5. n8n 集成

### 5.1 n8n Reddit 节点

n8n 提供官方 Reddit 节点，支持：
- **Post**: 获取帖子、搜索
- **Profile**: 获取用户信息
- **Subreddit**: 获取 subreddit 信息

### 5.2 OAuth2 凭据配置

在 n8n 中创建 Reddit OAuth2 凭据：

```yaml
# n8n 凭据配置
name: Reddit OAuth2
type: redditOAuth2Api
data:
  clientId: "YOUR_CLIENT_ID"
  clientSecret: "YOUR_CLIENT_SECRET"
  accessToken: ""  # 首次授权后自动填充
  refreshToken: ""
```

### 5.3 发布工作流示例

```json
{
  "name": "Reddit Auto Publisher",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [{"field": "hours", "hoursInterval": 6}]
        }
      },
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "operation": "search",
        "query": "SELECT * FROM articles WHERE published_to_reddit = false AND hot_score > 0.7 LIMIT 1"
      },
      "name": "Get Pending Article",
      "type": "n8n-nodes-base.postgres",
      "position": [450, 300]
    },
    {
      "parameters": {
        "resource": "post",
        "operation": "submit",
        "subreddit": "={{ $json.target_subreddit }}",
        "kind": "self",
        "title": "={{ $json.title }}",
        "text": "={{ $json.reddit_content }}"
      },
      "name": "Submit to Reddit",
      "type": "n8n-nodes-base.reddit",
      "position": [650, 300]
    },
    {
      "parameters": {
        "operation": "update",
        "query": "UPDATE articles SET published_to_reddit = true, reddit_post_id = '{{ $json.id }}' WHERE id = '{{ $('Get Pending Article').first().json.id }}'"
      },
      "name": "Update Status",
      "type": "n8n-nodes-base.postgres",
      "position": [850, 300]
    }
  ],
  "connections": {
    "Schedule Trigger": {"main": [[{"node": "Get Pending Article"}]]},
    "Get Pending Article": {"main": [[{"node": "Submit to Reddit"}]]},
    "Submit to Reddit": {"main": [[{"node": "Update Status"}]]}
  }
}
```

### 5.4 监控工作流示例

```json
{
  "name": "Reddit Hot Posts Monitor",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [{"field": "hours", "hoursInterval": 2}]
        }
      },
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "resource": "post",
        "operation": "getAll",
        "subreddit": "artificial+cybersecurity+LocalLLaMA",
        "returnAll": false,
        "limit": 50,
        "filters": {
          "sort": "hot"
        }
      },
      "name": "Get Hot Posts",
      "type": "n8n-nodes-base.reddit",
      "position": [450, 300]
    },
    {
      "parameters": {
        "jsCode": "// 过滤高分帖子\nconst posts = $input.all();\nreturn posts.filter(item => item.json.score > 100).map(item => ({\n  json: {\n    title: item.json.title,\n    subreddit: item.json.subreddit,\n    score: item.json.score,\n    url: item.json.url,\n    source: 'reddit'\n  }\n}));"
      },
      "name": "Filter High Score",
      "type": "n8n-nodes-base.code",
      "position": [650, 300]
    },
    {
      "parameters": {
        "operation": "insert",
        "table": "articles",
        "columns": "title, source, url, hot_score, collected_at"
      },
      "name": "Save to DB",
      "type": "n8n-nodes-base.postgres",
      "position": [850, 300]
    }
  ]
}
```

---

## ⚠️ 6. 最佳实践与风险规避

### 6.1 避免 Shadowban

| 风险行为 | 安全做法 |
|----------|----------|
| 快速连续发帖 | 每 subreddit 间隔 >10 分钟 |
| 纯自我推广 | 遵循 10:1 规则（10条互动:1条推广） |
| 相同内容多发 | 针对不同社区定制内容 |
| 标题党/误导 | 使用准确描述性标题 |
| 忽略社区规则 | 阅读并遵守每个 subreddit 规则 |

### 6.2 内容质量要求

```yaml
# 发布前检查清单
pre_publish_checklist:
  - title_length: 60-300 字符
  - content_length: >500 字符（self post）
  - has_value: 提供新信息或独特观点
  - not_duplicate: 检查是否已发布过
  - fits_subreddit: 匹配目标社区主题
  - proper_flair: 选择正确的 flair
```

### 6.3 推荐发布频率

| 账号状态 | 建议频率 |
|----------|----------|
| 新账号 (<30天) | 1 帖/天 |
| 成长期 (30-90天) | 2-3 帖/天 |
| 成熟账号 (>90天) | 5-10 帖/天 |

### 6.4 Karma 积累策略

1. **评论优先**: 在目标 subreddit 有价值地评论
2. **回答问题**: 在 r/learnmachinelearning 等帮助新手
3. **分享资源**: 分享有用的工具和教程
4. **参与讨论**: 技术讨论中提供专业见解

---

## 📁 7. 环境配置

### 7.1 环境变量

```bash
# .env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
REDDIT_USER_AGENT=pubilie-bot/1.0 by u/your_username
```

### 7.2 praw.ini 配置（推荐）

```ini
# ~/.config/praw.ini 或项目根目录/praw.ini
[pubilie]
client_id=your_client_id
client_secret=your_client_secret
username=your_username
password=your_password
user_agent=pubilie-bot/1.0 by u/your_username
```

使用：
```python
reddit = praw.Reddit("pubilie")
```

---

## 📊 8. 数据库 Schema 扩展

```sql
-- 添加 Reddit 相关字段到 articles 表
ALTER TABLE articles ADD COLUMN IF NOT EXISTS reddit_post_id VARCHAR(20);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS reddit_subreddit VARCHAR(50);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS published_to_reddit BOOLEAN DEFAULT false;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS reddit_score INT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS reddit_published_at TIMESTAMPTZ;

-- Reddit 监控记录表
CREATE TABLE IF NOT EXISTS reddit_monitored_posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reddit_id VARCHAR(20) UNIQUE NOT NULL,
    subreddit VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(50),
    url TEXT,
    score INT,
    num_comments INT,
    created_utc TIMESTAMPTZ,
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT false
);

CREATE INDEX idx_reddit_posts_subreddit ON reddit_monitored_posts(subreddit);
CREATE INDEX idx_reddit_posts_score ON reddit_monitored_posts(score DESC);
```

---

## 📎 参考资源

### 官方文档
- [PRAW Documentation](https://praw.readthedocs.io/en/stable/)
- [Reddit API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki)
- [n8n Reddit Integration](https://n8n.io/integrations/reddit/)

### 教程
- [JC Chouinard - Reddit API Guide](https://www.jcchouinard.com/reddit-api/)
- [GeeksforGeeks - PRAW Tutorial](https://www.geeksforgeeks.org/python/python-praw-python-reddit-api-wrapper/)

### Subreddit 列表
- [Best AI Subreddits 2025](https://usefulai.com/subreddits)
- [Top 50 Cybersecurity Subreddits](https://www.sentinelone.com/blog/top-50-subreddits-for-cybersecurity-and-infosec/)
- [Awesome Cybersecurity Subreddits](https://github.com/d0midigi/awesome-cybersecurity-subreddits)

---

**创建时间**: 2026-01-13 22:00:00 +0800
**设计者**: Claude Opus 4.5
**状态**: 调研完成，待实施
