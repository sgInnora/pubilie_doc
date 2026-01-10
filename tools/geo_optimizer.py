#!/usr/bin/env python3
"""
GEO (Generative Engine Optimization) 优化工具

用于优化内容以提升在AI搜索引擎中的引用率和可见性。
支持ChatGPT、Perplexity、Google AI Overviews等平台。

版本: 1.0
创建时间: 2026-01-10
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class GEOReport:
    """GEO优化报告数据类"""
    # 直接答案评分
    has_direct_answer: bool = False
    opening_word_count: int = 0
    direct_answer_score: float = 0.0

    # 引用概率评分
    original_data_count: int = 0
    authority_citations: int = 0
    expert_quotes: int = 0
    case_studies: int = 0
    citation_score: float = 0.0

    # 实体标记
    entities: list = field(default_factory=list)
    suggested_schema: str = "Article"

    # E-E-A-T评分
    has_author_credentials: bool = False
    has_experience_description: bool = False
    has_traceable_sources: bool = False
    has_update_date: bool = False
    eeat_score: float = 0.0

    # 总分
    total_score: float = 0.0
    recommendations: list = field(default_factory=list)


class GEOOptimizer:
    """GEO优化器主类"""

    # 权威来源关键词
    AUTHORITY_SOURCES = [
        'MITRE', 'NIST', 'Gartner', 'Forrester', 'IDC',
        'IEEE', 'ACM', 'OWASP', 'CISA', 'FBI', 'NSA',
        'Microsoft', 'Google', 'Amazon', 'CrowdStrike',
        'Mandiant', 'Recorded Future', 'Palo Alto'
    ]

    # 数据指标关键词
    DATA_INDICATORS = [
        r'\d+%', r'\d+\s*(million|billion|万|亿)',
        r'increased by', r'decreased by', r'growth of',
        r'统计', r'数据显示', r'研究表明', r'调查发现'
    ]

    # 专家引用模式
    EXPERT_PATTERNS = [
        r'according to\s+\w+',
        r'said\s+\w+',
        r'\w+\s+表示',
        r'\w+\s+指出',
        r'研究员\s+\w+',
        r'专家\s+\w+'
    ]

    # 案例研究关键词
    CASE_STUDY_KEYWORDS = [
        'case study', 'real-world', 'example',
        '案例', '实例', '实际应用', '真实场景'
    ]

    # 安全实体模式
    SECURITY_ENTITIES = {
        'cve': r'CVE-\d{4}-\d+',
        'apt': r'APT\d+|APT-\d+',
        'attack_id': r'T\d{4}(?:\.\d{3})?',
        'malware': r'[A-Z][a-z]+(?:Bot|Trojan|Ransomware|Worm|Backdoor)',
    }

    # 迂回开头黑名单
    INDIRECT_OPENINGS = [
        '在本文中', '本文将', '让我们探讨', '让我们深入',
        'In this article', 'This article will', 'Let us explore',
        '随着.*的发展', '近年来', '众所周知'
    ]

    def __init__(self):
        pass

    def analyze(self, content: str) -> GEOReport:
        """
        分析内容的GEO优化程度

        Args:
            content: Markdown格式的文章内容

        Returns:
            GEOReport: GEO优化报告
        """
        report = GEOReport()

        # 1. 分析直接答案
        report = self._analyze_direct_answer(content, report)

        # 2. 分析引用概率
        report = self._analyze_citation_probability(content, report)

        # 3. 提取实体
        report = self._extract_entities(content, report)

        # 4. 分析E-E-A-T
        report = self._analyze_eeat(content, report)

        # 5. 计算总分
        report.total_score = self._calculate_total_score(report)

        # 6. 生成建议
        report.recommendations = self._generate_recommendations(report)

        return report

    def _analyze_direct_answer(self, content: str, report: GEOReport) -> GEOReport:
        """分析直接答案评分"""
        lines = content.split('\n')

        # 跳过frontmatter和标题，找到正文开头
        in_frontmatter = False
        first_paragraph = ""

        for line in lines:
            if line.strip() == '---':
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                continue
            if line.startswith('#'):
                continue
            if line.strip():
                first_paragraph = line.strip()
                break

        # 计算开头字数
        report.opening_word_count = len(first_paragraph)

        # 检查是否是迂回开头
        is_indirect = any(
            re.search(pattern, first_paragraph, re.IGNORECASE)
            for pattern in self.INDIRECT_OPENINGS
        )

        # 检查是否是直接定义/回答格式
        direct_patterns = [
            r'^[\w\u4e00-\u9fa5]+\s*(是|is|are|refers to|means)',
            r'^(What|How|Why|When|Where).*\?.*:',
            r'^\*\*[\w\u4e00-\u9fa5]+\*\*\s*[:：]',
        ]
        is_direct = any(
            re.search(pattern, first_paragraph, re.IGNORECASE)
            for pattern in direct_patterns
        )

        report.has_direct_answer = is_direct and not is_indirect

        # 评分: 直接回答10分，开头简洁(<50字)+5分
        score = 0
        if report.has_direct_answer:
            score += 10
        if report.opening_word_count < 50:
            score += 3
        elif report.opening_word_count < 100:
            score += 1

        report.direct_answer_score = min(score, 10)

        return report

    def _analyze_citation_probability(self, content: str, report: GEOReport) -> GEOReport:
        """分析引用概率评分"""
        # 统计原创数据
        for pattern in self.DATA_INDICATORS:
            report.original_data_count += len(re.findall(pattern, content, re.IGNORECASE))

        # 统计权威引用
        for source in self.AUTHORITY_SOURCES:
            if source.lower() in content.lower():
                report.authority_citations += 1

        # 统计专家引用
        for pattern in self.EXPERT_PATTERNS:
            report.expert_quotes += len(re.findall(pattern, content, re.IGNORECASE))

        # 统计案例研究
        for keyword in self.CASE_STUDY_KEYWORDS:
            if keyword.lower() in content.lower():
                report.case_studies += 1

        # 计算引用概率分数 (0-10)
        score = 0
        score += min(report.original_data_count * 1.5, 4)  # 最多4分
        score += min(report.authority_citations * 1, 3)     # 最多3分
        score += min(report.expert_quotes * 0.5, 2)         # 最多2分
        score += min(report.case_studies * 1, 1)            # 最多1分

        report.citation_score = min(score, 10)

        return report

    def _extract_entities(self, content: str, report: GEOReport) -> GEOReport:
        """提取安全相关实体"""
        entities = []

        for entity_type, pattern in self.SECURITY_ENTITIES.items():
            matches = re.findall(pattern, content)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match
                })

        report.entities = entities

        # 根据实体类型建议Schema
        if any(e['type'] == 'cve' for e in entities):
            report.suggested_schema = "TechArticle"
        elif any(e['type'] == 'apt' for e in entities):
            report.suggested_schema = "Article"  # 威胁情报文章
        else:
            report.suggested_schema = "BlogPosting"

        return report

    def _analyze_eeat(self, content: str, report: GEOReport) -> GEOReport:
        """分析E-E-A-T评分"""
        content_lower = content.lower()

        # 检查作者资质
        author_patterns = [
            r'author:', r'作者:', r'by\s+\w+',
            r'security\s+researcher', r'安全研究员',
            r'research\s+team', r'研究团队'
        ]
        report.has_author_credentials = any(
            re.search(p, content, re.IGNORECASE) for p in author_patterns
        )

        # 检查实际经验描述
        experience_patterns = [
            r'我(们)?实际', r'我(们)?测试', r'我(们)?发现',
            r'in our experience', r'we tested', r'we found',
            r'实战', r'实操', r'hands-on'
        ]
        report.has_experience_description = any(
            re.search(p, content, re.IGNORECASE) for p in experience_patterns
        )

        # 检查来源可追溯性
        source_patterns = [
            r'https?://', r'来源:', r'source:', r'reference:',
            r'\[.*\]\(http', r'引用自', r'according to'
        ]
        report.has_traceable_sources = any(
            re.search(p, content, re.IGNORECASE) for p in source_patterns
        )

        # 检查更新日期
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}', r'\d{4}年\d{1,2}月',
            r'last\s+updated', r'最后更新', r'更新时间'
        ]
        report.has_update_date = any(
            re.search(p, content, re.IGNORECASE) for p in date_patterns
        )

        # 计算E-E-A-T分数
        score = 0
        if report.has_author_credentials:
            score += 2.5
        if report.has_experience_description:
            score += 2.5
        if report.has_traceable_sources:
            score += 2.5
        if report.has_update_date:
            score += 2.5

        report.eeat_score = score

        return report

    def _calculate_total_score(self, report: GEOReport) -> float:
        """计算总分"""
        # 权重: 直接答案25%, 引用概率40%, E-E-A-T35%
        total = (
            report.direct_answer_score * 0.25 +
            report.citation_score * 0.40 +
            report.eeat_score * 0.35
        )
        return round(total, 2)

    def _generate_recommendations(self, report: GEOReport) -> list:
        """生成优化建议"""
        recommendations = []

        # 直接答案建议
        if not report.has_direct_answer:
            recommendations.append(
                "开头应直接回答核心问题，避免迂回表达。"
                "使用'X是...'或'X refers to...'格式开始。"
            )

        # 引用概率建议
        if report.original_data_count < 3:
            recommendations.append(
                f"当前仅有{report.original_data_count}个数据点，"
                "建议添加至少3个原创统计/数据以提升引用概率。"
            )

        if report.authority_citations < 2:
            recommendations.append(
                "建议引用更多权威来源(MITRE, NIST, Gartner等)以增强可信度。"
            )

        # E-E-A-T建议
        if not report.has_author_credentials:
            recommendations.append(
                "建议添加作者/团队资质说明以提升权威性。"
            )

        if not report.has_experience_description:
            recommendations.append(
                "建议添加实际操作经验描述，"
                "如'我们实际测试发现...'以增强Experience维度。"
            )

        if not report.has_traceable_sources:
            recommendations.append(
                "建议添加可追溯的引用链接，提升内容可信度。"
            )

        return recommendations

    def generate_key_takeaways(self, content: str, count: int = 3) -> list:
        """
        生成文章关键要点（用于GEO优化）

        Args:
            content: 文章内容
            count: 要点数量

        Returns:
            关键要点列表
        """
        # 提取标题作为要点基础
        headers = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)

        takeaways = []
        for header in headers[:count]:
            # 清理标题
            clean_header = re.sub(r'[\d\.\)]+\s*', '', header).strip()
            if clean_header and len(clean_header) > 5:
                takeaways.append(clean_header)

        return takeaways[:count]

    def generate_faq_pairs(self, content: str) -> list:
        """
        从内容中提取FAQ对（用于FAQPage Schema）

        Args:
            content: 文章内容

        Returns:
            FAQ对列表 [{'question': str, 'answer': str}]
        """
        faqs = []

        # 模式1: 标题是问题格式
        question_headers = re.findall(
            r'^##\s*(.+\?)\s*\n((?:(?!^#).+\n?)+)',
            content, re.MULTILINE
        )
        for q, a in question_headers:
            faqs.append({
                'question': q.strip(),
                'answer': a.strip()[:500]  # 限制答案长度
            })

        # 模式2: Q: A: 格式
        qa_pattern = re.findall(
            r'(?:Q|问)[:：]\s*(.+?)\n(?:A|答)[:：]\s*(.+?)(?=\n(?:Q|问)[:：]|\n#|\Z)',
            content, re.DOTALL
        )
        for q, a in qa_pattern:
            faqs.append({
                'question': q.strip(),
                'answer': a.strip()[:500]
            })

        return faqs

    def format_report(self, report: GEOReport) -> str:
        """格式化GEO报告为Markdown"""
        output = f"""## 🎯 GEO优化报告

### 直接答案评分: {report.direct_answer_score}/10
- 开头是否直接回答: {'✅' if report.has_direct_answer else '❌'}
- 开头字数: {report.opening_word_count}字

### 引用概率评分: {report.citation_score}/10
- 原创数据数量: {report.original_data_count}个
- 权威引用数量: {report.authority_citations}个
- 专家观点数量: {report.expert_quotes}个
- 案例研究数量: {report.case_studies}个

### 实体识别
- 识别实体数: {len(report.entities)}个
- 实体列表: {', '.join([e['value'] for e in report.entities[:10]])}
- 建议Schema: {report.suggested_schema}

### E-E-A-T评分: {report.eeat_score}/10
- 作者资质展示: {'✅' if report.has_author_credentials else '❌'}
- 实际经验描述: {'✅' if report.has_experience_description else '❌'}
- 来源可追溯性: {'✅' if report.has_traceable_sources else '❌'}
- 更新日期标注: {'✅' if report.has_update_date else '❌'}

### 📊 总分: {report.total_score}/10

### 💡 优化建议
"""
        for i, rec in enumerate(report.recommendations, 1):
            output += f"{i}. {rec}\n"

        if not report.recommendations:
            output += "暂无建议，内容GEO优化良好！\n"

        return output


def main():
    parser = argparse.ArgumentParser(description='GEO优化工具')
    parser.add_argument('file', help='要分析的Markdown文件路径')
    parser.add_argument('--output', '-o', help='输出报告路径(可选)')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')

    args = parser.parse_args()

    # 读取文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误: 文件不存在 - {args.file}")
        return 1

    content = file_path.read_text(encoding='utf-8')

    # 分析
    optimizer = GEOOptimizer()
    report = optimizer.analyze(content)

    # 输出
    if args.json:
        output = json.dumps({
            'direct_answer_score': report.direct_answer_score,
            'citation_score': report.citation_score,
            'eeat_score': report.eeat_score,
            'total_score': report.total_score,
            'entities': report.entities,
            'recommendations': report.recommendations
        }, ensure_ascii=False, indent=2)
    else:
        output = optimizer.format_report(report)

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"报告已保存到: {args.output}")
    else:
        print(output)

    return 0


if __name__ == '__main__':
    exit(main())
