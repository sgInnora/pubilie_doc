#!/usr/bin/env python3
"""
Canonical URL 管理工具

管理多平台发布的Canonical URL，确保SEO权重归集到主站。
支持Medium、Dev.to、LinkedIn等平台的Canonical设置。

版本: 1.0
创建时间: 2026-01-10
"""

import re
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


@dataclass
class CanonicalConfig:
    """Canonical URL配置"""
    primary_domain: str = "https://innora.ai/blog"
    syndication_delay_hours: int = 48  # 分发延迟（小时）
    platforms: dict = None

    def __post_init__(self):
        if self.platforms is None:
            self.platforms = {
                'medium': {
                    'supports_canonical': True,
                    'method': 'api',  # API设置canonicalUrl
                    'format': 'canonical_url'
                },
                'devto': {
                    'supports_canonical': True,
                    'method': 'frontmatter',  # frontmatter中设置
                    'format': 'canonical_url: {url}'
                },
                'linkedin': {
                    'supports_canonical': False,
                    'method': 'footer_link',  # 文末添加链接
                    'format': '*Originally published at: [{title}]({url})*'
                },
                'twitter': {
                    'supports_canonical': False,
                    'method': 'link_card',  # 链接卡片
                    'format': 'Read full article: {url}'
                },
                'github': {
                    'supports_canonical': False,
                    'method': 'readme_link',
                    'format': '> Full article: [{title}]({url})'
                }
            }


class CanonicalManager:
    """Canonical URL管理器"""

    def __init__(self, config: Optional[CanonicalConfig] = None):
        self.config = config or CanonicalConfig()

    def generate_canonical_url(self, article_path: str, slug: Optional[str] = None) -> str:
        """
        生成文章的Canonical URL

        Args:
            article_path: 文章文件路径 (如 2026_01/Article_EN.md)
            slug: 自定义slug (可选)

        Returns:
            完整的Canonical URL
        """
        path = Path(article_path)

        # 从路径提取日期
        date_match = re.search(r'(\d{4})_(\d{2})', str(path))
        if date_match:
            year, month = date_match.groups()
        else:
            now = datetime.now()
            year, month = now.strftime('%Y'), now.strftime('%m')

        # 生成slug
        if not slug:
            # 从文件名生成
            filename = path.stem
            # 移除语言后缀
            slug = re.sub(r'_(CN|EN|GitHub|LinkedIn|Medium|Twitter)$', '', filename, flags=re.IGNORECASE)
            # 转换为URL友好格式
            slug = slug.lower()
            slug = re.sub(r'[^a-z0-9]+', '-', slug)
            slug = slug.strip('-')

        return f"{self.config.primary_domain}/{year}/{month}/{slug}"

    def get_syndication_status(self, article_path: str, publish_date: Optional[str] = None) -> dict:
        """
        检查文章的分发状态

        Args:
            article_path: 文章路径
            publish_date: 主站发布日期 (YYYY-MM-DD)

        Returns:
            {
                'can_syndicate': bool,
                'hours_remaining': int,
                'recommended_platforms': list
            }
        """
        if publish_date:
            pub_dt = datetime.strptime(publish_date, '%Y-%m-%d')
        else:
            # 假设今天发布
            pub_dt = datetime.now()

        delay = timedelta(hours=self.config.syndication_delay_hours)
        syndication_time = pub_dt + delay
        now = datetime.now()

        can_syndicate = now >= syndication_time
        hours_remaining = max(0, int((syndication_time - now).total_seconds() / 3600))

        return {
            'can_syndicate': can_syndicate,
            'hours_remaining': hours_remaining,
            'syndication_time': syndication_time.strftime('%Y-%m-%d %H:%M'),
            'recommended_platforms': list(self.config.platforms.keys())
        }

    def inject_canonical_frontmatter(self, content: str, canonical_url: str) -> str:
        """
        向Markdown内容注入Canonical URL (frontmatter)

        用于Dev.to等支持frontmatter的平台
        """
        # 检查是否已有frontmatter
        if content.startswith('---'):
            # 在现有frontmatter中添加
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                # 检查是否已有canonical_url
                if 'canonical_url:' not in frontmatter:
                    frontmatter += f"\ncanonical_url: {canonical_url}\n"
                return f"---{frontmatter}---{parts[2]}"

        # 创建新的frontmatter
        new_frontmatter = f"""---
canonical_url: {canonical_url}
---

"""
        return new_frontmatter + content

    def inject_footer_link(self, content: str, canonical_url: str, title: str = "Full Report") -> str:
        """
        向内容末尾添加原文链接

        用于LinkedIn等不支持Canonical的平台
        """
        footer = f"\n\n---\n\n*Originally published at: [{title}]({canonical_url})*"

        # 检查是否已有类似链接
        if 'Originally published' in content or canonical_url in content:
            return content

        return content + footer

    def generate_platform_content(self, content: str, platform: str,
                                   canonical_url: str, title: str = "") -> str:
        """
        为指定平台生成带Canonical的内容

        Args:
            content: 原始内容
            platform: 平台名称 (medium, devto, linkedin, etc.)
            canonical_url: Canonical URL
            title: 文章标题

        Returns:
            处理后的内容
        """
        platform_config = self.config.platforms.get(platform.lower())
        if not platform_config:
            return content

        method = platform_config['method']

        if method == 'frontmatter':
            return self.inject_canonical_frontmatter(content, canonical_url)
        elif method == 'footer_link':
            return self.inject_footer_link(content, canonical_url, title or "Full Report")
        elif method == 'link_card':
            # Twitter格式
            footer = f"\n\n🔗 Read full article: {canonical_url}"
            return content + footer
        elif method == 'readme_link':
            # GitHub格式
            header = f"> 📄 Full article: [{title or 'Read More'}]({canonical_url})\n\n"
            return header + content

        return content

    def batch_process(self, articles_dir: str) -> list:
        """
        批量处理目录下的文章

        Args:
            articles_dir: 文章目录

        Returns:
            处理结果列表
        """
        results = []
        articles_path = Path(articles_dir)

        for md_file in articles_path.glob('**/*.md'):
            # 跳过非文章文件
            if any(skip in md_file.name for skip in ['README', 'CLAUDE', 'TODO', 'GUIDE']):
                continue

            canonical_url = self.generate_canonical_url(str(md_file))
            results.append({
                'file': str(md_file),
                'canonical_url': canonical_url
            })

        return results

    def generate_sitemap_entry(self, article_path: str, canonical_url: str,
                               lastmod: Optional[str] = None) -> str:
        """生成sitemap XML条目"""
        if not lastmod:
            lastmod = datetime.now().strftime('%Y-%m-%d')

        return f"""  <url>
    <loc>{canonical_url}</loc>
    <lastmod>{lastmod}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>"""

    def format_report(self, articles: list) -> str:
        """格式化Canonical URL报告"""
        output = "## 📌 Canonical URL 管理报告\n\n"
        output += f"**主站域名**: {self.config.primary_domain}\n"
        output += f"**分发延迟**: {self.config.syndication_delay_hours}小时\n\n"

        output += "### 文章Canonical URL列表\n\n"
        output += "| 文件 | Canonical URL |\n"
        output += "|------|---------------|\n"

        for article in articles:
            filename = Path(article['file']).name
            output += f"| {filename} | {article['canonical_url']} |\n"

        output += "\n### 平台支持情况\n\n"
        output += "| 平台 | 支持Canonical | 设置方法 |\n"
        output += "|------|--------------|----------|\n"

        for platform, config in self.config.platforms.items():
            supports = '✅' if config['supports_canonical'] else '❌'
            output += f"| {platform.capitalize()} | {supports} | {config['method']} |\n"

        return output


def main():
    parser = argparse.ArgumentParser(description='Canonical URL管理工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # generate命令
    gen_parser = subparsers.add_parser('generate', help='生成Canonical URL')
    gen_parser.add_argument('file', help='文章文件路径')
    gen_parser.add_argument('--slug', help='自定义slug')

    # inject命令
    inject_parser = subparsers.add_parser('inject', help='注入Canonical URL到内容')
    inject_parser.add_argument('file', help='文章文件路径')
    inject_parser.add_argument('--platform', '-p', required=True,
                               choices=['medium', 'devto', 'linkedin', 'twitter', 'github'],
                               help='目标平台')
    inject_parser.add_argument('--output', '-o', help='输出文件路径')

    # status命令
    status_parser = subparsers.add_parser('status', help='检查分发状态')
    status_parser.add_argument('file', help='文章文件路径')
    status_parser.add_argument('--publish-date', help='发布日期 (YYYY-MM-DD)')

    # batch命令
    batch_parser = subparsers.add_parser('batch', help='批量处理目录')
    batch_parser.add_argument('directory', help='文章目录')
    batch_parser.add_argument('--output', '-o', help='输出报告路径')

    args = parser.parse_args()

    manager = CanonicalManager()

    if args.command == 'generate':
        url = manager.generate_canonical_url(args.file, args.slug)
        print(f"Canonical URL: {url}")

    elif args.command == 'inject':
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"错误: 文件不存在 - {args.file}")
            return 1

        content = file_path.read_text(encoding='utf-8')
        canonical_url = manager.generate_canonical_url(args.file)

        # 提取标题
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Full Report"

        result = manager.generate_platform_content(
            content, args.platform, canonical_url, title
        )

        if args.output:
            Path(args.output).write_text(result, encoding='utf-8')
            print(f"已保存到: {args.output}")
        else:
            print(result)

    elif args.command == 'status':
        status = manager.get_syndication_status(args.file, args.publish_date)
        if status['can_syndicate']:
            print("✅ 可以开始分发到第三方平台")
        else:
            print(f"⏳ 距离可分发还需 {status['hours_remaining']} 小时")
        print(f"建议分发平台: {', '.join(status['recommended_platforms'])}")

    elif args.command == 'batch':
        articles = manager.batch_process(args.directory)
        report = manager.format_report(articles)

        if args.output:
            Path(args.output).write_text(report, encoding='utf-8')
            print(f"报告已保存到: {args.output}")
        else:
            print(report)

    else:
        parser.print_help()

    return 0


if __name__ == '__main__':
    exit(main())
