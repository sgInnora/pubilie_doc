#!/usr/bin/env python3
"""
Schema Markup 自动生成工具

自动为技术文章生成结构化数据(JSON-LD)，支持:
- Article/BlogPosting
- TechArticle
- FAQPage
- HowTo

版本: 1.0
创建时间: 2026-01-10
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path


@dataclass
class ArticleMetadata:
    """文章元数据"""
    title: str = ""
    description: str = ""
    author: str = "Innora Security Research Team"
    author_url: str = "https://innora.ai/team"
    date_published: str = ""
    date_modified: str = ""
    keywords: list = field(default_factory=list)
    canonical_url: str = ""
    image_url: str = ""
    language: str = "en"
    word_count: int = 0


class SchemaGenerator:
    """Schema Markup 生成器"""

    PUBLISHER_INFO = {
        "@type": "Organization",
        "name": "Innora",
        "url": "https://innora.ai",
        "logo": {
            "@type": "ImageObject",
            "url": "https://innora.ai/logo.png",
            "width": 600,
            "height": 60
        }
    }

    AUTHOR_INFO = {
        "@type": "Organization",
        "name": "Innora Security Research Team",
        "url": "https://innora.ai/team"
    }

    def __init__(self, base_url: str = "https://innora.ai/blog"):
        self.base_url = base_url

    def parse_frontmatter(self, content: str) -> dict:
        """解析Markdown frontmatter"""
        frontmatter = {}

        # 匹配YAML frontmatter
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if match:
            yaml_content = match.group(1)
            # 简单解析（不依赖PyYAML）
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    frontmatter[key] = value

        return frontmatter

    def extract_metadata(self, content: str, file_path: Optional[str] = None) -> ArticleMetadata:
        """从Markdown内容提取元数据"""
        metadata = ArticleMetadata()

        # 解析frontmatter
        fm = self.parse_frontmatter(content)
        metadata.title = fm.get('title', '')
        metadata.description = fm.get('description', fm.get('summary', ''))
        metadata.author = fm.get('author', 'Innora Security Research Team')
        metadata.date_published = fm.get('date', fm.get('created', ''))
        metadata.date_modified = fm.get('modified', fm.get('updated', metadata.date_published))
        metadata.canonical_url = fm.get('canonical_url', '')
        metadata.image_url = fm.get('image', fm.get('cover', ''))
        metadata.language = fm.get('lang', fm.get('language', 'en'))

        # 解析keywords
        keywords_str = fm.get('keywords', fm.get('tags', ''))
        if keywords_str:
            metadata.keywords = [k.strip() for k in keywords_str.split(',')]

        # 如果没有标题，从H1提取
        if not metadata.title:
            h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if h1_match:
                metadata.title = h1_match.group(1).strip()

        # 如果没有描述，取第一段
        if not metadata.description:
            # 跳过frontmatter和标题
            body = re.sub(r'^---.*?---\s*', '', content, flags=re.DOTALL)
            body = re.sub(r'^#.*?\n', '', body)
            paragraphs = [p.strip() for p in body.split('\n\n') if p.strip() and not p.startswith('#')]
            if paragraphs:
                metadata.description = paragraphs[0][:160]

        # 计算字数
        clean_content = re.sub(r'[#*`\[\](){}]', '', content)
        metadata.word_count = len(clean_content.split())

        # 从文件路径推断日期
        if file_path and not metadata.date_published:
            path_match = re.search(r'(\d{4})_(\d{2})', file_path)
            if path_match:
                year, month = path_match.groups()
                metadata.date_published = f"{year}-{month}-01"

        # 默认今天
        if not metadata.date_published:
            metadata.date_published = datetime.now().strftime('%Y-%m-%d')
        if not metadata.date_modified:
            metadata.date_modified = metadata.date_published

        return metadata

    def generate_article_schema(self, metadata: ArticleMetadata) -> dict:
        """生成Article Schema"""
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": metadata.title[:110],  # Google限制110字符
            "description": metadata.description[:160],
            "author": self.AUTHOR_INFO.copy(),
            "publisher": self.PUBLISHER_INFO.copy(),
            "datePublished": metadata.date_published,
            "dateModified": metadata.date_modified,
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": metadata.canonical_url or self.base_url
            },
            "inLanguage": metadata.language,
            "wordCount": metadata.word_count
        }

        if metadata.keywords:
            schema["keywords"] = metadata.keywords

        if metadata.image_url:
            schema["image"] = {
                "@type": "ImageObject",
                "url": metadata.image_url,
                "width": 1200,
                "height": 630
            }

        if metadata.author and metadata.author != "Innora Security Research Team":
            schema["author"] = {
                "@type": "Person",
                "name": metadata.author
            }

        return schema

    def generate_techarticle_schema(self, metadata: ArticleMetadata) -> dict:
        """生成TechArticle Schema（用于技术深度文章）"""
        schema = self.generate_article_schema(metadata)
        schema["@type"] = "TechArticle"
        schema["proficiencyLevel"] = "Expert"

        return schema

    def generate_blogposting_schema(self, metadata: ArticleMetadata) -> dict:
        """生成BlogPosting Schema"""
        schema = self.generate_article_schema(metadata)
        schema["@type"] = "BlogPosting"

        return schema

    def generate_faqpage_schema(self, faqs: list) -> dict:
        """
        生成FAQPage Schema

        Args:
            faqs: [{'question': str, 'answer': str}, ...]

        Returns:
            FAQPage Schema dict
        """
        schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": []
        }

        for faq in faqs:
            qa = {
                "@type": "Question",
                "name": faq['question'],
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq['answer']
                }
            }
            schema["mainEntity"].append(qa)

        return schema

    def generate_howto_schema(self, title: str, steps: list, description: str = "") -> dict:
        """
        生成HowTo Schema

        Args:
            title: 教程标题
            steps: [{'name': str, 'text': str}, ...]
            description: 教程描述

        Returns:
            HowTo Schema dict
        """
        schema = {
            "@context": "https://schema.org",
            "@type": "HowTo",
            "name": title,
            "step": []
        }

        if description:
            schema["description"] = description

        for i, step in enumerate(steps, 1):
            step_schema = {
                "@type": "HowToStep",
                "position": i,
                "name": step.get('name', f"Step {i}"),
                "text": step.get('text', '')
            }
            schema["step"].append(step_schema)

        return schema

    def extract_faqs(self, content: str) -> list:
        """从内容中提取FAQ"""
        faqs = []

        # 模式1: 问题标题 (## 什么是...?)
        question_headers = re.findall(
            r'^##\s*(.+\?)\s*\n((?:(?!^#).+\n?)+)',
            content, re.MULTILINE
        )
        for q, a in question_headers:
            answer = a.strip()
            # 清理Markdown格式
            answer = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', answer)
            answer = re.sub(r'[*_`#]', '', answer)
            faqs.append({
                'question': q.strip(),
                'answer': answer[:500]
            })

        # 模式2: Q&A格式
        qa_matches = re.findall(
            r'(?:\*\*)?(?:Q|问|Question)(?:\*\*)?[:：]\s*(.+?)\n+(?:\*\*)?(?:A|答|Answer)(?:\*\*)?[:：]\s*(.+?)(?=\n+(?:\*\*)?(?:Q|问|Question)|\n+#|\Z)',
            content, re.DOTALL | re.IGNORECASE
        )
        for q, a in qa_matches:
            answer = a.strip()
            answer = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', answer)
            answer = re.sub(r'[*_`#]', '', answer)
            faqs.append({
                'question': q.strip(),
                'answer': answer[:500]
            })

        return faqs

    def extract_howto_steps(self, content: str) -> list:
        """从内容中提取HowTo步骤"""
        steps = []

        # 模式1: 数字列表 (1. 2. 3.)
        numbered_steps = re.findall(
            r'^\d+\.\s*\*\*(.+?)\*\*[:：]?\s*(.+?)(?=^\d+\.|\n\n|\Z)',
            content, re.MULTILINE | re.DOTALL
        )
        for name, text in numbered_steps:
            steps.append({
                'name': name.strip(),
                'text': text.strip()[:500]
            })

        # 模式2: Step标题
        step_headers = re.findall(
            r'^###?\s*(?:Step|步骤)\s*\d+[:：]?\s*(.+?)\n((?:(?!^#).+\n?)+)',
            content, re.MULTILINE
        )
        if step_headers and not steps:
            for name, text in step_headers:
                steps.append({
                    'name': name.strip(),
                    'text': text.strip()[:500]
                })

        return steps

    def to_json_ld(self, schema: dict) -> str:
        """转换为JSON-LD HTML代码块"""
        json_str = json.dumps(schema, ensure_ascii=False, indent=2)
        return f'<script type="application/ld+json">\n{json_str}\n</script>'

    def generate_all_schemas(self, content: str, file_path: Optional[str] = None) -> dict:
        """
        生成所有适用的Schema

        Returns:
            {'article': schema, 'faq': schema, 'howto': schema}
        """
        result = {}

        # 提取元数据
        metadata = self.extract_metadata(content, file_path)

        # 判断文章类型
        content_lower = content.lower()
        is_tech = any(kw in content_lower for kw in ['cve-', 'apt', 'mitre', 'attack', 't1', '漏洞', '威胁'])
        is_howto = any(kw in content_lower for kw in ['how to', '如何', 'step by step', '步骤', '教程'])

        # 生成主Schema
        if is_tech:
            result['article'] = self.generate_techarticle_schema(metadata)
        else:
            result['article'] = self.generate_article_schema(metadata)

        # 检测FAQ
        faqs = self.extract_faqs(content)
        if faqs:
            result['faq'] = self.generate_faqpage_schema(faqs)

        # 检测HowTo
        if is_howto:
            steps = self.extract_howto_steps(content)
            if steps:
                result['howto'] = self.generate_howto_schema(
                    metadata.title,
                    steps,
                    metadata.description
                )

        return result


def main():
    parser = argparse.ArgumentParser(description='Schema Markup生成工具')
    parser.add_argument('file', help='要处理的Markdown文件路径')
    parser.add_argument('--type', '-t', choices=['article', 'tech', 'blog', 'faq', 'howto', 'all'],
                        default='all', help='Schema类型')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--html', action='store_true', help='输出HTML格式(script标签)')

    args = parser.parse_args()

    # 读取文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误: 文件不存在 - {args.file}")
        return 1

    content = file_path.read_text(encoding='utf-8')

    # 生成Schema
    generator = SchemaGenerator()

    if args.type == 'all':
        schemas = generator.generate_all_schemas(content, str(file_path))
        if args.html:
            output = '\n\n'.join(generator.to_json_ld(s) for s in schemas.values())
        else:
            output = json.dumps(schemas, ensure_ascii=False, indent=2)
    else:
        metadata = generator.extract_metadata(content, str(file_path))
        if args.type == 'article':
            schema = generator.generate_article_schema(metadata)
        elif args.type == 'tech':
            schema = generator.generate_techarticle_schema(metadata)
        elif args.type == 'blog':
            schema = generator.generate_blogposting_schema(metadata)
        elif args.type == 'faq':
            faqs = generator.extract_faqs(content)
            schema = generator.generate_faqpage_schema(faqs)
        elif args.type == 'howto':
            steps = generator.extract_howto_steps(content)
            schema = generator.generate_howto_schema(metadata.title, steps, metadata.description)

        if args.html:
            output = generator.to_json_ld(schema)
        else:
            output = json.dumps(schema, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"Schema已保存到: {args.output}")
    else:
        print(output)

    return 0


if __name__ == '__main__':
    exit(main())
