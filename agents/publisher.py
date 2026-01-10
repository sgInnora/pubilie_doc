#!/usr/bin/env python3
"""
Publisher Agent

负责多平台发布、格式适配和SEO优化。
"""

from typing import Optional, List, Any, Dict
from pathlib import Path
from datetime import datetime
import re
from .config import AGENT_ROLES, AgentConfig


def create_publisher_agent(
    config: Optional[AgentConfig] = None,
    tools: Optional[List[Any]] = None,
    **kwargs
):
    """
    创建Publisher Agent

    Args:
        config: Agent配置
        tools: 可用工具列表
        **kwargs: 额外参数覆盖默认配置

    Returns:
        CrewAI Agent实例
    """
    try:
        from crewai import Agent
    except ImportError:
        from .researcher import MockAgent
        return MockAgent('publisher', AGENT_ROLES['publisher'])

    config = config or AgentConfig()
    role_config = AGENT_ROLES['publisher']

    agent_params = {
        'role': role_config['role'],
        'goal': role_config['goal'],
        'backstory': role_config['backstory'],
        'verbose': config.verbose,
        'allow_delegation': kwargs.get('allow_delegation', False),
        'max_iter': config.max_iter,
        'max_retry_limit': config.max_retry_limit,
    }

    if config.llm_model:
        agent_params['llm'] = config.llm_model

    if tools:
        agent_params['tools'] = tools

    agent_params.update(kwargs)

    return Agent(**agent_params)


# 平台配置
PLATFORM_CONFIGS = {
    'github': {
        'name': 'GitHub',
        'format': 'markdown',
        'max_length': None,
        'features': ['code_blocks', 'tables', 'images', 'anchors'],
        'naming_convention': '{title}_EN.md / {title}_CN.md',
        'frontmatter': False,
        'canonical_method': 'readme_link',
    },
    'medium': {
        'name': 'Medium',
        'format': 'markdown',
        'max_length': 15000,  # 约15分钟阅读
        'features': ['images', 'embeds', 'quotes', 'code_blocks'],
        'naming_convention': 'N/A (web editor)',
        'frontmatter': False,
        'canonical_method': 'api',
        'api_field': 'canonicalUrl',
    },
    'devto': {
        'name': 'Dev.to',
        'format': 'markdown',
        'max_length': None,
        'features': ['code_blocks', 'liquid_tags', 'embeds'],
        'naming_convention': 'article-title.md',
        'frontmatter': True,
        'frontmatter_fields': ['title', 'published', 'tags', 'canonical_url', 'cover_image'],
        'canonical_method': 'frontmatter',
    },
    'linkedin': {
        'name': 'LinkedIn',
        'format': 'rich_text',
        'max_length': 3000,  # 文章模式
        'features': ['images', 'mentions', 'hashtags'],
        'naming_convention': 'N/A (web editor)',
        'frontmatter': False,
        'canonical_method': 'footer_link',
    },
    'twitter': {
        'name': 'Twitter/X',
        'format': 'threads',
        'max_length': 280,  # 单条
        'thread_max': 25,
        'features': ['images', 'links', 'hashtags', 'mentions'],
        'naming_convention': 'N/A (API/web)',
        'frontmatter': False,
        'canonical_method': 'link_card',
    },
}


class PlatformAdapter:
    """平台内容适配器"""

    def __init__(self, primary_domain: str = "https://innora.ai/blog"):
        self.primary_domain = primary_domain
        self.platforms = PLATFORM_CONFIGS

    def adapt_for_platform(
        self,
        content: str,
        platform: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        为指定平台适配内容

        Args:
            content: 原始Markdown内容
            platform: 目标平台
            metadata: 文章元数据

        Returns:
            适配后的内容
        """
        if platform not in self.platforms:
            raise ValueError(f"Unknown platform: {platform}. "
                             f"Available: {list(self.platforms.keys())}")

        config = self.platforms[platform]
        metadata = metadata or {}

        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else metadata.get('title', 'Untitled')

        # 生成Canonical URL
        canonical_url = self._generate_canonical_url(title, metadata)

        # 根据平台处理
        if platform == 'github':
            return self._adapt_github(content, title, canonical_url)
        elif platform == 'medium':
            return self._adapt_medium(content, title, canonical_url)
        elif platform == 'devto':
            return self._adapt_devto(content, title, canonical_url, metadata)
        elif platform == 'linkedin':
            return self._adapt_linkedin(content, title, canonical_url)
        elif platform == 'twitter':
            return self._adapt_twitter(content, title, canonical_url)

        return content

    def _generate_canonical_url(self, title: str, metadata: Dict) -> str:
        """生成Canonical URL"""
        if 'canonical_url' in metadata:
            return metadata['canonical_url']

        # 从标题生成slug
        slug = title.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = slug.strip('-')[:50]

        date = metadata.get('date', datetime.now().strftime('%Y/%m'))
        return f"{self.primary_domain}/{date}/{slug}"

    def _adapt_github(self, content: str, title: str, canonical_url: str) -> str:
        """适配GitHub格式"""
        # GitHub Markdown保持原样，添加顶部链接
        header = f"> 📄 Full article: [{title}]({canonical_url})\n\n"
        return header + content

    def _adapt_medium(self, content: str, title: str, canonical_url: str) -> str:
        """适配Medium格式"""
        # Medium需要通过API设置canonicalUrl，内容保持Markdown
        # 添加底部信息
        footer = f"\n\n---\n\n*Originally published at [{self.primary_domain}]({canonical_url})*"
        return content + footer

    def _adapt_devto(
        self,
        content: str,
        title: str,
        canonical_url: str,
        metadata: Dict
    ) -> str:
        """适配Dev.to格式"""
        # 生成frontmatter
        tags = metadata.get('tags', ['security', 'cybersecurity'])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]
        tags = tags[:4]  # Dev.to最多4个标签

        frontmatter = f"""---
title: "{title}"
published: true
tags: {', '.join(tags)}
canonical_url: {canonical_url}
---

"""
        # 移除原有frontmatter如果有
        content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)

        return frontmatter + content

    def _adapt_linkedin(self, content: str, title: str, canonical_url: str) -> str:
        """适配LinkedIn格式"""
        config = self.platforms['linkedin']
        max_length = config['max_length']

        # 移除Markdown格式
        text = self._strip_markdown(content)

        # 截断到限制长度
        if len(text) > max_length:
            text = text[:max_length - 100] + "...\n\n"
            text += f"📖 Read the full article: {canonical_url}"
        else:
            text += f"\n\n---\n\n*Originally published at: {canonical_url}*"

        # 添加hashtags
        text += "\n\n#Cybersecurity #InfoSec #ThreatIntelligence"

        return text

    def _adapt_twitter(self, content: str, title: str, canonical_url: str) -> str:
        """适配Twitter线程格式"""
        # 提取关键点生成线程
        sections = re.split(r'\n##\s+', content)
        threads = []

        # 首条推文
        first_tweet = f"🔒 {title}\n\nA thread 🧵\n\n{canonical_url}"
        threads.append(first_tweet)

        # 从各section提取要点
        for section in sections[1:6]:  # 最多5个section
            lines = section.split('\n')
            section_title = lines[0].strip()
            # 提取第一个要点
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    point = self._strip_markdown(line)[:250]
                    tweet = f"📌 {section_title}\n\n{point}"
                    threads.append(tweet)
                    break

        # 结尾推文
        threads.append(f"📖 Full analysis here: {canonical_url}\n\nLike and retweet if you found this useful! 🙏")

        return "\n\n---\n\n".join(threads)

    def _strip_markdown(self, text: str) -> str:
        """移除Markdown格式"""
        # 移除标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # 移除链接，保留文字
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 移除粗体/斜体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # 移除代码块
        text = re.sub(r'```[\s\S]*?```', '[code block]', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # 移除图片
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)

        return text.strip()


class PublishingChecklist:
    """发布检查清单"""

    CHECKLIST = {
        'pre_publish': [
            'AI检测率 < 15%',
            '可读性评分 > 60',
            '拼写和语法检查通过',
            '所有链接可访问',
            '图片有alt文本',
            'Schema Markup已添加',
        ],
        'seo': [
            '标题包含关键词',
            '描述在155字符内',
            '有H2/H3层级结构',
            '内部链接已添加',
            'Canonical URL已设置',
        ],
        'platform_specific': {
            'github': ['README链接更新', '目录结构正确'],
            'medium': ['封面图片已上传', '标签已添加（最多5个）'],
            'devto': ['frontmatter完整', '标签已添加（最多4个）'],
            'linkedin': ['摘要版本已准备', 'hashtags已添加'],
            'twitter': ['线程已分割', '关键点已提炼'],
        },
    }

    @classmethod
    def get_checklist(cls, platform: Optional[str] = None) -> Dict:
        """获取发布检查清单"""
        checklist = {
            'pre_publish': cls.CHECKLIST['pre_publish'],
            'seo': cls.CHECKLIST['seo'],
        }

        if platform and platform in cls.CHECKLIST['platform_specific']:
            checklist['platform_specific'] = cls.CHECKLIST['platform_specific'][platform]

        return checklist


# 发布时间建议
OPTIMAL_POSTING_TIMES = {
    'github': {
        'best_days': ['Tuesday', 'Wednesday', 'Thursday'],
        'best_hours': ['10:00', '14:00'],
        'timezone': 'UTC',
    },
    'medium': {
        'best_days': ['Tuesday', 'Wednesday'],
        'best_hours': ['08:00', '11:00'],
        'timezone': 'EST',
    },
    'devto': {
        'best_days': ['Monday', 'Tuesday', 'Wednesday'],
        'best_hours': ['07:00', '12:00'],
        'timezone': 'UTC',
    },
    'linkedin': {
        'best_days': ['Tuesday', 'Wednesday', 'Thursday'],
        'best_hours': ['08:00', '10:00', '12:00'],
        'timezone': 'EST',
    },
    'twitter': {
        'best_days': ['Wednesday', 'Thursday'],
        'best_hours': ['09:00', '12:00', '17:00'],
        'timezone': 'EST',
    },
}


def get_optimal_posting_time(platform: str) -> Dict:
    """获取平台最佳发布时间"""
    if platform not in OPTIMAL_POSTING_TIMES:
        return {'message': f'No data for {platform}'}
    return OPTIMAL_POSTING_TIMES[platform]
