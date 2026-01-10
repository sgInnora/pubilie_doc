#!/usr/bin/env python3
"""
Custom Tools for Content Agents

为Multi-Agent系统提供的自定义工具集。
"""

import re
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class WebSearchTool:
    """网络搜索工具"""

    name = "web_search"
    description = "搜索互联网获取最新信息和研究资料"

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def _run(self, query: str) -> str:
        """执行搜索"""
        # 集成DuckDuckGo或其他搜索API
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                return json.dumps(results, ensure_ascii=False, indent=2)
        except ImportError:
            return f"[模拟搜索结果] 查询: {query}\n请安装 duckduckgo-search: pip install duckduckgo-search"


class ThreatIntelTool:
    """威胁情报工具"""

    name = "threat_intel"
    description = "查询威胁情报数据源获取APT、漏洞和攻击信息"

    def __init__(self):
        self.sources = ['MITRE ATT&CK', 'NVD', 'VirusTotal', 'AlienVault OTX']

    def _run(self, query: str, source: str = 'all') -> str:
        """查询威胁情报"""
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources_queried': self.sources if source == 'all' else [source],
            'findings': []
        }

        # 模拟查询结果（实际实现应连接真实API）
        if 'CVE' in query.upper():
            results['findings'].append({
                'type': 'vulnerability',
                'id': query.upper(),
                'severity': 'HIGH',
                'description': f'Vulnerability details for {query}'
            })
        elif 'APT' in query.upper():
            results['findings'].append({
                'type': 'threat_actor',
                'name': query,
                'origin': 'Unknown',
                'targets': ['Government', 'Finance', 'Technology']
            })

        return json.dumps(results, ensure_ascii=False, indent=2)


class ReadabilityTool:
    """可读性分析工具"""

    name = "readability_check"
    description = "分析文本的可读性指标"

    def _run(self, text: str) -> str:
        """分析可读性"""
        # 使用项目中已有的readability_checker
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
            from readability_checker import ReadabilityChecker

            checker = ReadabilityChecker()
            report = checker.analyze(text)
            return checker.format_report(report)
        except ImportError:
            # 简化版实现
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            return json.dumps({
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_words_per_sentence': len(words) / max(len(sentences), 1)
            }, ensure_ascii=False)


class AIDetectionTool:
    """AI检测工具"""

    name = "ai_detection"
    description = "检测文本的AI生成概率"

    # AI典型用词黑名单
    AI_PATTERNS = [
        r'\bfurthermore\b', r'\bmoreover\b', r'\bnevertheless\b',
        r'\bconsequently\b', r'\bsubsequently\b', r'\bthus\b',
        r'\bhence\b', r'\bthereby\b', r'\bwhereby\b',
        r'\b此外\b', r'\b综上所述\b', r'\b值得注意的是\b',
        r'\b总而言之\b', r'\b需要强调的是\b',
    ]

    def _run(self, text: str) -> str:
        """检测AI痕迹"""
        text_lower = text.lower()
        detections = []

        for pattern in self.AI_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detections.extend(matches)

        # 计算基于规则的AI概率估计
        word_count = len(text.split())
        ai_word_density = len(detections) / max(word_count, 1) * 100

        result = {
            'ai_patterns_found': len(detections),
            'patterns': list(set(detections))[:10],
            'ai_word_density': f"{ai_word_density:.2f}%",
            'estimated_ai_probability': min(ai_word_density * 5, 100),
            'recommendation': '需要人性化优化' if ai_word_density > 0.5 else '通过'
        }

        return json.dumps(result, ensure_ascii=False, indent=2)


class HumanizationTool:
    """文本人性化工具"""

    name = "humanize_text"
    description = "将文本进行人性化处理，消除AI痕迹"

    REPLACEMENTS = {
        # 英文替换
        'furthermore': ['another thing', 'also', 'plus'],
        'moreover': ['what\'s more', 'and', 'on top of that'],
        'nevertheless': ['but still', 'anyway', 'that said'],
        'consequently': ['so', 'as a result', 'because of this'],
        'subsequently': ['then', 'after that', 'later on'],
        # 中文替换
        '此外': ['另外', '还有个事儿', '说到这个'],
        '综上所述': ['总的来说', '说到底', '我的看法是'],
        '值得注意的是': ['有意思的是', '不过话说回来', '我发现'],
        '总而言之': ['反正', '说白了', '简单来说'],
        '需要强调的是': ['特别想说', '我觉得很重要的是', '敲黑板'],
    }

    HUMAN_PHRASES = [
        '说真的', '坦白讲', '你懂的', '这个...怎么说呢',
        '我个人觉得', '以我的经验来看', '记得有一次',
        '前几天', '话说回来', '不开玩笑',
    ]

    def _run(self, text: str) -> str:
        """人性化处理"""
        import random

        result = text

        # 替换AI典型用词
        for ai_word, alternatives in self.REPLACEMENTS.items():
            if ai_word.lower() in result.lower():
                replacement = random.choice(alternatives)
                result = re.sub(
                    re.escape(ai_word),
                    replacement,
                    result,
                    flags=re.IGNORECASE
                )

        return result


class SchemaMarkupTool:
    """Schema Markup生成工具"""

    name = "generate_schema"
    description = "为文章生成Schema.org结构化数据"

    def _run(self, title: str, description: str, author: str = "Security Researcher") -> str:
        """生成Schema Markup"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
            from schema_generator import SchemaGenerator, ArticleMetadata

            generator = SchemaGenerator()
            metadata = ArticleMetadata(
                title=title,
                description=description,
                author=author,
                date_published=datetime.now().strftime('%Y-%m-%d'),
            )
            schema = generator.generate_article_schema(metadata)
            return generator.to_json_ld(schema)
        except ImportError:
            # 简化版
            schema = {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": title[:110],
                "description": description[:160],
                "author": {"@type": "Person", "name": author},
                "datePublished": datetime.now().strftime('%Y-%m-%d'),
            }
            return f'<script type="application/ld+json">\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n</script>'


class CanonicalURLTool:
    """Canonical URL管理工具"""

    name = "manage_canonical"
    description = "生成和管理Canonical URL"

    def __init__(self, primary_domain: str = "https://innora.ai/blog"):
        self.primary_domain = primary_domain

    def _run(self, article_path: str, platform: str = 'all') -> str:
        """生成Canonical URL配置"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
            from canonical_manager import CanonicalManager

            manager = CanonicalManager()
            canonical_url = manager.generate_canonical_url(article_path)
            status = manager.get_syndication_status(article_path)

            return json.dumps({
                'canonical_url': canonical_url,
                'syndication_status': status,
                'platform_formats': {
                    'devto': f'canonical_url: {canonical_url}',
                    'medium': f'canonicalUrl: {canonical_url}',
                    'linkedin': f'Originally published at: {canonical_url}',
                }
            }, ensure_ascii=False, indent=2)
        except ImportError:
            # 简化版
            slug = Path(article_path).stem.lower().replace('_', '-')
            return json.dumps({
                'canonical_url': f'{self.primary_domain}/{slug}',
                'platform': platform
            }, ensure_ascii=False)


class ATTACKMappingTool:
    """MITRE ATT&CK映射工具"""

    name = "attack_mapping"
    description = "将威胁情报映射到MITRE ATT&CK框架"

    def _run(self, text: str) -> str:
        """执行ATT&CK映射"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
            from attack_mapper import ATTACKMapper

            mapper = ATTACKMapper()
            mapping = mapper.map_text(text)
            return mapper.format_report(mapping)
        except ImportError:
            # 简化版
            techniques = []
            if 'phishing' in text.lower():
                techniques.append({'id': 'T1566', 'name': 'Phishing'})
            if 'ransomware' in text.lower():
                techniques.append({'id': 'T1486', 'name': 'Data Encrypted for Impact'})
            return json.dumps({'techniques': techniques}, ensure_ascii=False)


# 工具注册表
TOOL_REGISTRY = {
    'web_search': WebSearchTool,
    'threat_intel': ThreatIntelTool,
    'readability': ReadabilityTool,
    'ai_detection': AIDetectionTool,
    'humanize': HumanizationTool,
    'schema_markup': SchemaMarkupTool,
    'canonical_url': CanonicalURLTool,
    'attack_mapping': ATTACKMappingTool,
}


def get_tool(name: str, **kwargs):
    """获取工具实例"""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name](**kwargs)


def get_all_tools(**kwargs) -> List:
    """获取所有工具实例"""
    return [cls(**kwargs) for cls in TOOL_REGISTRY.values()]
