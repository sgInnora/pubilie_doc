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
    'wechat': {
        'name': '微信公众号',
        'format': 'rich_text_html',
        'max_length': 20000,  # 字符限制
        'features': ['images', 'quotes', 'code_blocks', 'tables', 'cards'],
        'naming_convention': 'wechat_{slug}.md',
        'frontmatter': True,
        'frontmatter_fields': ['title', 'author', 'cover_image', 'summary', 'original'],
        'canonical_method': 'original_link',
        'image_specs': {
            'cover': {'width': 900, 'height': 383, 'ratio': '2.35:1'},
            'thumb': {'width': 200, 'height': 200, 'ratio': '1:1'},
            'content': {'max_width': 900},
        },
        'style_guide': {
            'font_size': 16,
            'line_height': 1.75,
            'paragraph_spacing': 15,
            'heading_sizes': {'h1': 22, 'h2': 20, 'h3': 18},
            'accent_color': '#1890ff',
        },
    },
    'zhihu': {
        'name': '知乎',
        'format': 'markdown',
        'max_length': 50000,
        'features': ['images', 'formulas', 'tables', 'code_blocks', 'citations'],
        'naming_convention': 'zhihu_{slug}.md',
        'frontmatter': False,
        'canonical_method': 'footer_link',
    },
    'xiaohongshu': {
        'name': '小红书',
        'format': 'rich_text',
        'max_length': 1000,  # 笔记正文
        'features': ['images', 'emojis', 'hashtags', 'mentions'],
        'naming_convention': 'xhs_{slug}.md',
        'frontmatter': False,
        'canonical_method': 'none',
        'image_specs': {
            'cover': {'ratio': '3:4', 'min_width': 1080},
            'content': {'max_count': 18},
        },
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
        elif platform == 'wechat':
            return self._adapt_wechat(content, title, canonical_url, metadata)
        elif platform == 'zhihu':
            return self._adapt_zhihu(content, title, canonical_url, metadata)
        elif platform == 'xiaohongshu':
            return self._adapt_xiaohongshu(content, title, canonical_url, metadata)

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

    def _adapt_wechat(
        self,
        content: str,
        title: str,
        canonical_url: str,
        metadata: Dict
    ) -> str:
        """
        适配微信公众号格式

        微信公众号特点：
        - 支持富文本HTML格式
        - 封面图尺寸：900×383（2.35:1）
        - 正文图片最大宽度900px
        - 行间距1.75，段间距15px
        - 主色#1890ff
        """
        config = self.platforms['wechat']

        # 生成frontmatter
        author = metadata.get('author', 'AI研究员')
        summary = metadata.get('summary', '')
        if not summary:
            # 自动提取摘要（第一段）
            first_para = re.search(r'^(?!#)(.+?)(?:\n\n|\n#)', content, re.DOTALL)
            summary = first_para.group(1).strip()[:120] if first_para else title

        frontmatter = f"""---
title: "{title}"
author: "{author}"
cover_image: "需要上传900×383封面图"
summary: "{summary}"
original: true
canonical_url: "{canonical_url}"
platform: wechat
---

"""
        # 移除原有frontmatter
        content = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)

        # 转换Markdown为微信友好格式
        adapted = self._markdown_to_wechat(content)

        # 添加底部信息
        footer = f"""

---

**原文链接**：{canonical_url}

**关于作者**
{author}，关注AI与科技创业。欢迎交流。

"""
        return frontmatter + adapted + footer

    def _markdown_to_wechat(self, content: str) -> str:
        """
        将Markdown转换为微信公众号友好格式

        处理规则：
        - 保留层级标题结构
        - 转换代码块为引用格式
        - 表格保持Markdown格式（微信支持）
        - 添加视觉分隔符
        """
        # 转换粗体强调（微信支持）
        result = content

        # 转换引用块（添加左边框样式提示）
        result = re.sub(
            r'^>\s*(.+)$',
            r'> 💬 \1',
            result,
            flags=re.MULTILINE
        )

        # 转换代码块为引用格式（微信代码显示有限）
        def code_to_quote(match):
            lang = match.group(1) or 'code'
            code = match.group(2).strip()
            # 保持代码但添加标记
            return f"\n📋 **代码 ({lang})**\n```\n{code}\n```\n"

        result = re.sub(
            r'```(\w+)?\n([\s\S]*?)```',
            code_to_quote,
            result
        )

        # 添加段落分隔（微信需要明显分隔）
        result = re.sub(r'\n\n', '\n\n　\n\n', result)

        # 转换分隔线
        result = re.sub(r'^---$', '\n━━━━━━━━━━━━━━━\n', result, flags=re.MULTILINE)

        return result

    def _adapt_zhihu(
        self,
        content: str,
        title: str,
        canonical_url: str,
        metadata: Dict
    ) -> str:
        """
        适配知乎格式

        知乎特点：
        - 支持完整Markdown
        - 支持LaTeX公式
        - 支持引用和参考文献
        - 专栏文章无字数限制
        """
        # 知乎基本保持Markdown格式
        # 添加底部信息
        footer = f"""

---

**本文首发于**：{canonical_url}

欢迎关注我的知乎专栏，获取更多AI与科技创业的深度分析。

"""
        # 添加参考文献格式化
        if '来源' in content or 'Source' in content:
            footer += "\n**参考资料**已在文中标注。\n"

        return content + footer

    def _adapt_xiaohongshu(
        self,
        content: str,
        title: str,
        canonical_url: str,
        metadata: Dict
    ) -> str:
        """
        适配小红书格式

        小红书特点：
        - 笔记正文限制1000字
        - 需要精炼核心观点
        - 大量使用emoji
        - hashtag格式 #话题#
        - 图片为主，文字为辅
        """
        config = self.platforms['xiaohongshu']
        max_length = config['max_length']

        # 提取核心观点
        points = []

        # 提取所有加粗文本作为核心观点
        bold_matches = re.findall(r'\*\*([^*]+)\*\*', content)
        points.extend(bold_matches[:5])

        # 提取列表项
        list_items = re.findall(r'^[-*]\s+(.+)$', content, re.MULTILINE)
        points.extend(list_items[:5])

        # 生成小红书风格内容
        emojis = ['🔥', '💡', '✨', '📌', '🎯', '💪', '🚀', '⭐']

        note = f"【{title}】\n\n"

        for i, point in enumerate(points[:6]):
            emoji = emojis[i % len(emojis)]
            clean_point = self._strip_markdown(point)[:100]
            note += f"{emoji} {clean_point}\n\n"

        # 添加hashtags
        tags = metadata.get('tags', ['AI创业', '科技趋势', '超级个体'])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',')]

        note += "\n"
        for tag in tags[:10]:
            note += f"#{tag}# "

        # 截断到限制长度
        if len(note) > max_length:
            note = note[:max_length - 50] + "...\n\n更多内容见主页～"

        # 添加封面图说明
        frontmatter = f"""---
platform: xiaohongshu
title: "{title}"
cover_ratio: "3:4"
cover_min_width: 1080
image_count: 建议9张图
---

"""
        return frontmatter + note


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
            'wechat': [
                '封面图已上传（900×383）',
                '正文图片宽度≤900px',
                '原创声明已勾选',
                '赞赏功能已开启',
                '阅读原文链接已设置',
                '无敏感词（已检测）',
                'AI内容标识已添加',
            ],
            'zhihu': [
                '专栏已选择',
                '话题标签已添加',
                '参考文献格式正确',
                '原创声明已勾选',
            ],
            'xiaohongshu': [
                '封面图比例3:4',
                '图片数量≤18张',
                '正文≤1000字',
                'hashtag格式正确（#话题#）',
                'AI内容已标识',
            ],
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
    'wechat': {
        'best_days': ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'best_hours': ['08:00', '12:00', '20:00', '22:00'],
        'timezone': 'Asia/Shanghai',
        'notes': '早8点通勤、午休、晚间阅读高峰',
    },
    'zhihu': {
        'best_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],
        'best_hours': ['10:00', '14:00', '21:00'],
        'timezone': 'Asia/Shanghai',
        'notes': '工作日知识消费活跃',
    },
    'xiaohongshu': {
        'best_days': ['Friday', 'Saturday', 'Sunday'],
        'best_hours': ['12:00', '18:00', '21:00', '22:00'],
        'timezone': 'Asia/Shanghai',
        'notes': '周末女性用户活跃度高',
    },
}


def get_optimal_posting_time(platform: str) -> Dict:
    """获取平台最佳发布时间"""
    if platform not in OPTIMAL_POSTING_TIMES:
        return {'message': f'No data for {platform}'}
    return OPTIMAL_POSTING_TIMES[platform]
