#!/usr/bin/env python3
"""
可读性指标检测工具

分析文本的可读性指标，包括:
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Coleman-Liau Index
- Automated Readability Index (ARI)

版本: 1.0
创建时间: 2026-01-10
"""

import re
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ReadabilityReport:
    """可读性分析报告"""
    # 基础统计
    total_words: int = 0
    total_sentences: int = 0
    total_syllables: int = 0
    total_characters: int = 0
    complex_words: int = 0  # 3+音节词

    # 可读性指标
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    coleman_liau: float = 0.0
    automated_readability_index: float = 0.0

    # 综合评分
    average_grade_level: float = 0.0
    difficulty_level: str = ""  # Easy, Moderate, Difficult, Very Difficult

    # 改进建议
    suggestions: list = field(default_factory=list)


class ReadabilityChecker:
    """可读性检测器"""

    # 难度级别阈值
    DIFFICULTY_THRESHOLDS = {
        'Easy': (0, 6),           # 小学水平
        'Moderate': (6, 10),      # 初中水平
        'Difficult': (10, 14),    # 高中水平
        'Very Difficult': (14, 20),  # 大学水平
        'Academic': (20, 100)     # 研究生水平
    }

    # 目标受众推荐
    AUDIENCE_RECOMMENDATIONS = {
        'general': (6, 8),        # 普通大众
        'tech_blog': (8, 12),     # 技术博客
        'academic': (12, 16),     # 学术文章
        'security_report': (10, 14)  # 安全报告
    }

    def __init__(self, target_grade: float = 10.0):
        """
        初始化检测器

        Args:
            target_grade: 目标阅读等级（默认10，高中水平）
        """
        self.target_grade = target_grade

    def _count_syllables(self, word: str) -> int:
        """
        估算单词音节数（英文）

        基于规则的简化算法
        """
        word = word.lower().strip()
        if not word:
            return 0

        # 常见例外
        exceptions = {
            'the': 1, 'a': 1, 'an': 1, 'and': 1, 'or': 1,
            'is': 1, 'are': 1, 'was': 1, 'were': 1,
            'have': 1, 'has': 1, 'had': 1,
            'security': 4, 'vulnerability': 6, 'authentication': 5,
            'authorization': 5, 'infrastructure': 4, 'implementation': 5
        }
        if word in exceptions:
            return exceptions[word]

        # 基本规则
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False

        for i, char in enumerate(word):
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # 调整规则
        # 结尾的e通常不发音
        if word.endswith('e') and count > 1:
            count -= 1
        # 结尾的le通常是一个音节
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        # 结尾的ed通常不增加音节（除非前面是t或d）
        if word.endswith('ed') and len(word) > 2 and word[-3] not in 'td':
            count = max(1, count)

        return max(1, count)

    def _count_chinese_characters(self, text: str) -> int:
        """统计中文字符数"""
        return len(re.findall(r'[\u4e00-\u9fff]', text))

    def _is_complex_word(self, word: str) -> bool:
        """判断是否为复杂词（3+音节）"""
        return self._count_syllables(word) >= 3

    def _tokenize_sentences(self, text: str) -> list:
        """分句"""
        # 处理常见句末标点
        sentences = re.split(r'[.!?。！？]+', text)
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_words(self, text: str) -> list:
        """分词（仅英文）"""
        # 移除中文字符后分词
        text_en = re.sub(r'[\u4e00-\u9fff]', ' ', text)
        words = re.findall(r'[a-zA-Z]+', text_en)
        return [w.lower() for w in words if len(w) > 0]

    def analyze(self, text: str) -> ReadabilityReport:
        """
        分析文本可读性

        Args:
            text: 要分析的文本

        Returns:
            ReadabilityReport: 可读性报告
        """
        report = ReadabilityReport()

        # 清理Markdown格式
        clean_text = self._clean_markdown(text)

        # 基础统计
        sentences = self._tokenize_sentences(clean_text)
        words = self._tokenize_words(clean_text)
        chinese_chars = self._count_chinese_characters(clean_text)

        report.total_sentences = len(sentences)
        report.total_words = len(words) + chinese_chars  # 中文字符计入
        report.total_characters = len(re.sub(r'\s', '', clean_text))

        # 如果词数太少，返回默认报告
        if len(words) < 10:
            report.suggestions.append("文本过短，无法进行准确的可读性分析")
            report.difficulty_level = "Unknown"
            return report

        # 计算英文统计（用于可读性公式）
        report.total_syllables = sum(self._count_syllables(w) for w in words)
        report.complex_words = sum(1 for w in words if self._is_complex_word(w))

        # 计算各项指标（使用英文词汇统计）
        word_count = len(words)
        sentence_count = max(1, report.total_sentences)
        syllable_count = report.total_syllables
        char_count = sum(len(w) for w in words)
        complex_count = report.complex_words

        # Flesch Reading Ease: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        # 分数越高越易读（0-100）
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / max(1, word_count)
        report.flesch_reading_ease = max(0, min(100,
            206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        ))

        # Flesch-Kincaid Grade Level
        report.flesch_kincaid_grade = (
            0.39 * words_per_sentence +
            11.8 * syllables_per_word -
            15.59
        )

        # Gunning Fog Index
        complex_ratio = complex_count / max(1, word_count)
        report.gunning_fog = 0.4 * (words_per_sentence + 100 * complex_ratio)

        # SMOG Index (需要至少30句)
        if sentence_count >= 3:
            report.smog_index = 1.0430 * (complex_count * (30 / sentence_count)) ** 0.5 + 3.1291
        else:
            report.smog_index = report.flesch_kincaid_grade  # 回退

        # Coleman-Liau Index
        L = (char_count / word_count) * 100  # 每100词的字母数
        S = (sentence_count / word_count) * 100  # 每100词的句子数
        report.coleman_liau = 0.0588 * L - 0.296 * S - 15.8

        # Automated Readability Index (ARI)
        report.automated_readability_index = (
            4.71 * (char_count / word_count) +
            0.5 * words_per_sentence -
            21.43
        )

        # 计算平均等级
        grades = [
            report.flesch_kincaid_grade,
            report.gunning_fog,
            report.smog_index,
            report.coleman_liau,
            report.automated_readability_index
        ]
        report.average_grade_level = sum(grades) / len(grades)

        # 确定难度级别
        avg = report.average_grade_level
        for level, (low, high) in self.DIFFICULTY_THRESHOLDS.items():
            if low <= avg < high:
                report.difficulty_level = level
                break
        else:
            report.difficulty_level = "Academic"

        # 生成改进建议
        report.suggestions = self._generate_suggestions(report)

        return report

    def _clean_markdown(self, text: str) -> str:
        """清理Markdown格式"""
        # 移除代码块
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        # 移除链接，保留文本
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 移除图片
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 移除标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # 移除列表标记
        text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # 移除粗体/斜体
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # 移除表格分隔符
        text = re.sub(r'\|[-:]+\|', '', text)
        text = re.sub(r'\|', ' ', text)

        return text

    def _generate_suggestions(self, report: ReadabilityReport) -> list:
        """生成改进建议"""
        suggestions = []

        # 基于目标等级的建议
        diff = report.average_grade_level - self.target_grade

        if diff > 3:
            suggestions.append(f"文本阅读难度过高（{report.average_grade_level:.1f}级），建议降至{self.target_grade:.0f}级以下")
        elif diff > 1:
            suggestions.append(f"文本阅读难度略高（{report.average_grade_level:.1f}级），可考虑简化")
        elif diff < -3:
            suggestions.append(f"文本阅读难度较低（{report.average_grade_level:.1f}级），可适当增加专业深度")

        # 基于指标的具体建议
        if report.total_sentences > 0:
            words_per_sentence = report.total_words / report.total_sentences
            if words_per_sentence > 25:
                suggestions.append(f"句子过长（平均{words_per_sentence:.1f}词/句），建议拆分长句，目标15-20词/句")
            elif words_per_sentence < 10:
                suggestions.append(f"句子过短（平均{words_per_sentence:.1f}词/句），可适当组合相关内容")

        if report.total_words > 0:
            complex_ratio = report.complex_words / report.total_words
            if complex_ratio > 0.2:
                suggestions.append(f"复杂词汇过多（{complex_ratio*100:.1f}%），建议用简单词替换部分专业术语")

        if report.flesch_reading_ease < 30:
            suggestions.append("Flesch阅读难度评分过低，文本非常难读，建议大幅简化")
        elif report.flesch_reading_ease < 50:
            suggestions.append("Flesch阅读难度评分较低，适合专业读者，普通读者可能难以理解")

        # GEO优化建议
        if report.difficulty_level in ['Very Difficult', 'Academic']:
            suggestions.append("对于AI搜索引擎引用，建议在文章开头提供简化的摘要或关键要点")

        return suggestions

    def get_grade_level_recommendation(self, audience: str = 'general') -> tuple:
        """
        获取目标受众的推荐阅读等级

        Args:
            audience: 目标受众类型

        Returns:
            (min_grade, max_grade): 推荐等级范围
        """
        return self.AUDIENCE_RECOMMENDATIONS.get(audience, (8, 12))

    def format_report(self, report: ReadabilityReport) -> str:
        """格式化输出报告"""
        output = "## 📊 可读性分析报告\n\n"

        # 基础统计
        output += "### 基础统计\n"
        output += f"- 总词数: {report.total_words}\n"
        output += f"- 总句数: {report.total_sentences}\n"
        output += f"- 复杂词数: {report.complex_words}"
        if report.total_words > 0:
            output += f" ({report.complex_words/report.total_words*100:.1f}%)"
        output += "\n"
        output += f"- 总音节数: {report.total_syllables}\n\n"

        # 可读性指标
        output += "### 可读性指标\n"
        output += f"| 指标 | 分数 | 说明 |\n"
        output += f"|------|------|------|\n"
        output += f"| Flesch Reading Ease | {report.flesch_reading_ease:.1f} | 0-100，越高越易读 |\n"
        output += f"| Flesch-Kincaid Grade | {report.flesch_kincaid_grade:.1f} | 美国学校年级 |\n"
        output += f"| Gunning Fog | {report.gunning_fog:.1f} | 需要的教育年限 |\n"
        output += f"| SMOG Index | {report.smog_index:.1f} | 理解所需年级 |\n"
        output += f"| Coleman-Liau | {report.coleman_liau:.1f} | 年级水平 |\n"
        output += f"| ARI | {report.automated_readability_index:.1f} | 年级水平 |\n\n"

        # 综合评估
        output += "### 综合评估\n"
        output += f"- **平均阅读等级**: {report.average_grade_level:.1f}\n"
        output += f"- **难度级别**: {report.difficulty_level}\n\n"

        # 难度对照表
        output += "### 难度对照\n"
        output += "| 级别 | 年级范围 | 适合受众 |\n"
        output += "|------|----------|----------|\n"
        output += "| Easy | 0-6 | 小学生、普通大众 |\n"
        output += "| Moderate | 6-10 | 初中生、博客读者 |\n"
        output += "| Difficult | 10-14 | 高中生、技术人员 |\n"
        output += "| Very Difficult | 14-20 | 大学生、专业人士 |\n"
        output += "| Academic | 20+ | 研究生、学术领域 |\n\n"

        # 改进建议
        if report.suggestions:
            output += "### 改进建议\n"
            for i, suggestion in enumerate(report.suggestions, 1):
                output += f"{i}. {suggestion}\n"

        return output


def main():
    parser = argparse.ArgumentParser(description='可读性分析工具')
    parser.add_argument('file', help='要分析的Markdown文件路径')
    parser.add_argument('--target', '-t', type=float, default=10.0,
                        help='目标阅读等级（默认10）')
    parser.add_argument('--audience', '-a',
                        choices=['general', 'tech_blog', 'academic', 'security_report'],
                        default='tech_blog', help='目标受众')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')

    args = parser.parse_args()

    # 读取文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误: 文件不存在 - {args.file}")
        return 1

    content = file_path.read_text(encoding='utf-8')

    # 分析
    checker = ReadabilityChecker(target_grade=args.target)
    report = checker.analyze(content)

    # 输出
    if args.json:
        import json
        from dataclasses import asdict
        output = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    else:
        output = checker.format_report(report)

        # 添加受众推荐
        min_grade, max_grade = checker.get_grade_level_recommendation(args.audience)
        output += f"\n### 受众推荐\n"
        output += f"- 目标受众: {args.audience}\n"
        output += f"- 推荐等级: {min_grade}-{max_grade}\n"

        if report.average_grade_level < min_grade:
            output += f"- 判定: ⚠️ 难度偏低，可增加专业深度\n"
        elif report.average_grade_level > max_grade:
            output += f"- 判定: ⚠️ 难度偏高，建议简化\n"
        else:
            output += f"- 判定: ✅ 难度适中，符合目标受众\n"

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f"报告已保存到: {args.output}")
    else:
        print(output)

    return 0


if __name__ == '__main__':
    exit(main())
