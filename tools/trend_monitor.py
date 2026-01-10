#!/usr/bin/env python3
"""
趋势监控工具 (E10)
==================

监控多个数据源的热门话题和趋势，为内容创作提供选题建议。

数据源:
- Google Trends (通过pytrends)
- Hacker News Top Stories
- Reddit 安全相关子版
- Twitter/X Trending (通过API)
- GitHub Trending Repositories

证据来源:
- Google Trends API: https://pypi.org/project/pytrends/
- Hacker News API: https://github.com/HackerNews/API
- Reddit API: https://www.reddit.com/dev/api

作者: Claude Opus 4.5 (Ultrathink Protocol v2.7)
创建时间: 2026-01-10
"""

import os
import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import time
import re

# 尝试导入可选依赖
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendSource(Enum):
    """趋势数据源"""
    GOOGLE_TRENDS = "google_trends"
    HACKER_NEWS = "hacker_news"
    REDDIT = "reddit"
    GITHUB = "github"
    TWITTER = "twitter"


@dataclass
class TrendItem:
    """趋势项目"""
    title: str
    source: TrendSource
    url: Optional[str] = None
    score: float = 0.0  # 热度分数 (0-100)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """生成唯一ID"""
        content = f"{self.source.value}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class TrendReport:
    """趋势报告"""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    sources_checked: List[str] = field(default_factory=list)
    total_items: int = 0
    items: List[TrendItem] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "generated_at": self.generated_at,
            "sources_checked": self.sources_checked,
            "total_items": self.total_items,
            "items": [asdict(item) for item in self.items],
            "recommendations": self.recommendations
        }


class TrendMonitor:
    """
    趋势监控器

    聚合多个数据源的热门话题，生成内容选题建议。

    使用示例:
    ```python
    monitor = TrendMonitor()
    report = monitor.scan_all()
    print(f"发现 {report.total_items} 个趋势话题")

    # 获取安全相关推荐
    security_topics = monitor.get_recommendations(category="security")
    ```
    """

    # 安全相关关键词
    SECURITY_KEYWORDS = [
        "security", "vulnerability", "cve", "exploit", "malware", "ransomware",
        "breach", "hack", "cyber", "threat", "apt", "zero-day", "zeroday",
        "phishing", "attack", "backdoor", "trojan", "botnet", "ddos",
        "encryption", "authentication", "authorization", "firewall", "siem",
        "安全", "漏洞", "攻击", "恶意软件", "勒索软件", "钓鱼", "入侵"
    ]

    # AI相关关键词
    AI_KEYWORDS = [
        "ai", "artificial intelligence", "machine learning", "llm", "gpt",
        "chatgpt", "claude", "gemini", "openai", "anthropic", "deepmind",
        "neural network", "deep learning", "transformer", "agent", "rag",
        "人工智能", "机器学习", "大模型", "智能体"
    ]

    def __init__(
        self,
        google_trends_geo: str = "US",
        reddit_subreddits: List[str] = None,
        cache_ttl: int = 3600
    ):
        self.google_trends_geo = google_trends_geo
        self.reddit_subreddits = reddit_subreddits or [
            "netsec", "cybersecurity", "hacking", "malware", "privacy",
            "artificial", "MachineLearning", "LocalLLaMA"
        ]
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}

        # 初始化HTTP会话
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "TrendMonitor/1.0 (pubilie-doc)"
            })
        else:
            self.session = None
            logger.warning("requests库未安装，部分功能不可用")

    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache_time:
            return False
        elapsed = (datetime.now() - self._cache_time[key]).total_seconds()
        return elapsed < self.cache_ttl

    def _categorize_topic(self, title: str) -> tuple:
        """分类话题并提取标签"""
        title_lower = title.lower()
        tags = []
        category = "general"

        # 检查安全关键词
        for kw in self.SECURITY_KEYWORDS:
            if kw.lower() in title_lower:
                category = "security"
                tags.append(kw)

        # 检查AI关键词
        for kw in self.AI_KEYWORDS:
            if kw.lower() in title_lower:
                if category == "general":
                    category = "ai"
                tags.append(kw)

        # 限制标签数量
        tags = list(set(tags))[:5]

        return category, tags

    def fetch_google_trends(self, keywords: List[str] = None) -> List[TrendItem]:
        """获取Google Trends数据"""
        if not HAS_PYTRENDS:
            logger.warning("pytrends未安装，跳过Google Trends")
            return []

        cache_key = f"google_trends_{self.google_trends_geo}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            pytrend = TrendReq(hl='en-US', tz=360)

            # 获取实时热搜
            trending = pytrend.trending_searches(pn=self.google_trends_geo.lower())

            items = []
            for idx, row in trending.iterrows():
                title = row[0]
                category, tags = self._categorize_topic(title)

                item = TrendItem(
                    title=title,
                    source=TrendSource.GOOGLE_TRENDS,
                    score=100 - idx,  # 排名越靠前分数越高
                    category=category,
                    tags=tags,
                    url=f"https://trends.google.com/trends/explore?q={title.replace(' ', '%20')}"
                )
                items.append(item)

            self._cache[cache_key] = items
            self._cache_time[cache_key] = datetime.now()

            logger.info(f"Google Trends: 获取 {len(items)} 个热搜话题")
            return items

        except Exception as e:
            logger.error(f"Google Trends获取失败: {e}")
            return []

    def fetch_hacker_news(self, limit: int = 30) -> List[TrendItem]:
        """获取Hacker News热门故事"""
        if not self.session:
            return []

        cache_key = "hacker_news"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            # 获取Top Stories IDs
            top_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            resp = self.session.get(top_url, timeout=10)
            story_ids = resp.json()[:limit]

            items = []
            for idx, story_id in enumerate(story_ids):
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_resp = self.session.get(story_url, timeout=5)
                story = story_resp.json()

                if not story or story.get("type") != "story":
                    continue

                title = story.get("title", "")
                category, tags = self._categorize_topic(title)

                # 计算分数：基于得分和评论数
                hn_score = story.get("score", 0)
                comments = story.get("descendants", 0)
                normalized_score = min(100, (hn_score / 500 + comments / 200) * 50)

                item = TrendItem(
                    title=title,
                    source=TrendSource.HACKER_NEWS,
                    url=story.get("url") or f"https://news.ycombinator.com/item?id={story_id}",
                    score=normalized_score,
                    category=category,
                    tags=tags,
                    metadata={
                        "hn_id": story_id,
                        "hn_score": hn_score,
                        "comments": comments,
                        "author": story.get("by")
                    }
                )
                items.append(item)

            self._cache[cache_key] = items
            self._cache_time[cache_key] = datetime.now()

            logger.info(f"Hacker News: 获取 {len(items)} 个热门故事")
            return items

        except Exception as e:
            logger.error(f"Hacker News获取失败: {e}")
            return []

    def fetch_reddit(self, limit: int = 20) -> List[TrendItem]:
        """获取Reddit热门帖子"""
        if not self.session:
            return []

        cache_key = "reddit"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        items = []

        for subreddit in self.reddit_subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
                resp = self.session.get(url, timeout=10)

                if resp.status_code != 200:
                    logger.warning(f"Reddit r/{subreddit} 请求失败: {resp.status_code}")
                    continue

                data = resp.json()
                posts = data.get("data", {}).get("children", [])

                for post_data in posts:
                    post = post_data.get("data", {})
                    title = post.get("title", "")
                    category, tags = self._categorize_topic(title)

                    # 基于upvotes和评论计算分数
                    upvotes = post.get("ups", 0)
                    comments = post.get("num_comments", 0)
                    normalized_score = min(100, (upvotes / 1000 + comments / 100) * 30)

                    item = TrendItem(
                        title=title,
                        source=TrendSource.REDDIT,
                        url=f"https://reddit.com{post.get('permalink', '')}",
                        score=normalized_score,
                        category=category,
                        tags=tags + [subreddit],
                        metadata={
                            "subreddit": subreddit,
                            "upvotes": upvotes,
                            "comments": comments,
                            "author": post.get("author")
                        }
                    )
                    items.append(item)

                # 避免Reddit限流
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Reddit r/{subreddit} 获取失败: {e}")

        self._cache[cache_key] = items
        self._cache_time[cache_key] = datetime.now()

        logger.info(f"Reddit: 获取 {len(items)} 个热门帖子")
        return items

    def fetch_github_trending(self, language: str = None, since: str = "daily") -> List[TrendItem]:
        """获取GitHub Trending仓库"""
        if not self.session:
            return []

        cache_key = f"github_{language or 'all'}_{since}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        try:
            # 使用非官方API (github-trending-api)
            url = "https://api.gitterapp.com/repositories"
            params = {"since": since}
            if language:
                params["language"] = language

            resp = self.session.get(url, params=params, timeout=15)

            if resp.status_code != 200:
                # 备用: 直接解析GitHub页面会比较复杂，这里使用模拟数据
                logger.warning("GitHub Trending API不可用，跳过")
                return []

            repos = resp.json()
            items = []

            for idx, repo in enumerate(repos[:30]):
                title = f"{repo.get('author', '')}/{repo.get('name', '')}"
                description = repo.get("description", "")
                full_title = f"{title}: {description}" if description else title

                category, tags = self._categorize_topic(full_title)
                tags.append(repo.get("language", "").lower()) if repo.get("language") else None

                # 基于stars计算分数
                stars = repo.get("stars", 0)
                forks = repo.get("forks", 0)
                normalized_score = min(100, (stars / 10000 + forks / 1000) * 50)

                item = TrendItem(
                    title=full_title,
                    source=TrendSource.GITHUB,
                    url=repo.get("url") or f"https://github.com/{title}",
                    score=normalized_score,
                    category=category,
                    tags=[t for t in tags if t],
                    metadata={
                        "stars": stars,
                        "forks": forks,
                        "language": repo.get("language"),
                        "stars_today": repo.get("currentPeriodStars", 0)
                    }
                )
                items.append(item)

            self._cache[cache_key] = items
            self._cache_time[cache_key] = datetime.now()

            logger.info(f"GitHub Trending: 获取 {len(items)} 个热门仓库")
            return items

        except Exception as e:
            logger.error(f"GitHub Trending获取失败: {e}")
            return []

    def scan_all(self, include_sources: List[TrendSource] = None) -> TrendReport:
        """
        扫描所有数据源

        Args:
            include_sources: 要包含的数据源列表，None表示全部

        Returns:
            TrendReport: 趋势报告
        """
        sources = include_sources or list(TrendSource)
        all_items = []
        checked_sources = []

        for source in sources:
            try:
                if source == TrendSource.GOOGLE_TRENDS:
                    items = self.fetch_google_trends()
                elif source == TrendSource.HACKER_NEWS:
                    items = self.fetch_hacker_news()
                elif source == TrendSource.REDDIT:
                    items = self.fetch_reddit()
                elif source == TrendSource.GITHUB:
                    items = self.fetch_github_trending()
                else:
                    items = []

                all_items.extend(items)
                checked_sources.append(source.value)

            except Exception as e:
                logger.error(f"扫描 {source.value} 失败: {e}")

        # 去重（基于标题相似度）
        unique_items = self._deduplicate_items(all_items)

        # 按分数排序
        unique_items.sort(key=lambda x: x.score, reverse=True)

        # 生成推荐
        recommendations = self._generate_recommendations(unique_items)

        return TrendReport(
            sources_checked=checked_sources,
            total_items=len(unique_items),
            items=unique_items,
            recommendations=recommendations
        )

    def _deduplicate_items(self, items: List[TrendItem]) -> List[TrendItem]:
        """基于标题相似度去重"""
        seen_titles = set()
        unique = []

        for item in items:
            # 简化标题用于比较
            simplified = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '', item.title.lower())

            if simplified not in seen_titles:
                seen_titles.add(simplified)
                unique.append(item)

        return unique

    def _generate_recommendations(self, items: List[TrendItem], top_n: int = 10) -> List[Dict[str, Any]]:
        """生成内容创作推荐"""
        recommendations = []

        # 按类别分组
        by_category = {}
        for item in items:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)

        # 优先推荐安全和AI相关话题
        priority_categories = ["security", "ai", "general"]

        for category in priority_categories:
            if category in by_category:
                category_items = by_category[category][:5]

                for item in category_items:
                    if len(recommendations) >= top_n:
                        break

                    recommendations.append({
                        "topic": item.title,
                        "category": item.category,
                        "source": item.source.value,
                        "score": item.score,
                        "url": item.url,
                        "tags": item.tags,
                        "suggested_angles": self._suggest_angles(item),
                        "priority": "high" if item.category == "security" else "medium"
                    })

        return recommendations

    def _suggest_angles(self, item: TrendItem) -> List[str]:
        """为话题建议创作角度"""
        angles = []

        if item.category == "security":
            angles = [
                "威胁分析与防御策略",
                "技术细节深度解读",
                "企业安全团队应对指南",
                "攻击者视角与防御者视角对比"
            ]
        elif item.category == "ai":
            angles = [
                "技术原理解析",
                "安全影响评估",
                "实战应用案例",
                "与现有技术对比分析"
            ]
        else:
            angles = [
                "行业影响分析",
                "技术趋势解读",
                "入门指南",
                "最佳实践总结"
            ]

        return angles[:2]

    def get_recommendations(
        self,
        category: str = None,
        min_score: float = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取内容推荐

        Args:
            category: 筛选类别 (security/ai/general)
            min_score: 最低分数
            limit: 返回数量

        Returns:
            推荐列表
        """
        report = self.scan_all()

        filtered = report.recommendations

        if category:
            filtered = [r for r in filtered if r["category"] == category]

        if min_score > 0:
            filtered = [r for r in filtered if r["score"] >= min_score]

        return filtered[:limit]


# ============================================================================
# 主函数
# ============================================================================

def main():
    """演示趋势监控"""
    print("=" * 60)
    print("趋势监控工具 (E10) - 演示")
    print("=" * 60)

    monitor = TrendMonitor()

    # 扫描所有数据源
    print("\n正在扫描数据源...")
    report = monitor.scan_all(include_sources=[
        TrendSource.HACKER_NEWS,
        TrendSource.REDDIT
    ])

    print(f"\n检查的数据源: {report.sources_checked}")
    print(f"发现话题总数: {report.total_items}")

    # 按类别统计
    categories = {}
    for item in report.items:
        cat = item.category
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n类别分布:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat}: {count}")

    # 显示推荐
    print(f"\n内容创作推荐 (Top 5):")
    for idx, rec in enumerate(report.recommendations[:5], 1):
        print(f"\n  {idx}. [{rec['category'].upper()}] {rec['topic'][:60]}...")
        print(f"     来源: {rec['source']}, 分数: {rec['score']:.1f}")
        print(f"     建议角度: {', '.join(rec['suggested_angles'])}")

    # 保存报告
    output_path = "/tmp/trend_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {output_path}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
