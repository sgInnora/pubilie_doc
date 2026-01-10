#!/usr/bin/env python3
"""
Multi-Agent Configuration

定义Agent和Crew的配置参数。
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class AgentConfig:
    """单个Agent的配置"""

    # LLM配置
    llm_model: str = "gpt-4"  # 默认模型
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Agent行为配置
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 20
    max_retry_limit: int = 3

    # 超时配置
    max_execution_time: Optional[int] = 300  # 秒

    # API配置
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY'))

    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置字典"""
        return {
            'model': self.llm_model,
            'temperature': self.llm_temperature,
            'max_tokens': self.llm_max_tokens,
        }


@dataclass
class CrewConfig:
    """Crew整体配置"""

    # 输出目录
    output_dir: Path = field(default_factory=lambda: Path.cwd() / 'output')

    # 进程配置
    process_type: str = 'sequential'  # sequential, hierarchical, parallel

    # 内存配置
    memory: bool = True
    embedder_config: Optional[Dict] = None

    # 日志配置
    verbose: bool = True
    log_file: Optional[Path] = None

    # 内容配置
    default_language: str = 'zh'  # zh, en
    enable_bilingual: bool = True

    # 人性化配置
    humanization_level: str = 'maximum'  # minimum, moderate, maximum
    ai_detection_threshold: float = 0.15  # 目标AI检测率 < 15%

    # 发布配置
    platforms: List[str] = field(default_factory=lambda: [
        'github', 'medium', 'devto', 'linkedin', 'twitter'
    ])

    # 质量门禁
    min_readability_score: float = 60.0  # Flesch Reading Ease
    min_word_count: int = 1500
    max_word_count: int = 5000

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file and isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)


# 预定义Agent角色配置
AGENT_ROLES = {
    'researcher': {
        'role': 'Security Threat Researcher',
        'goal': '发现高价值威胁情报主题，收集权威资料和最新动态',
        'backstory': '''你是一位资深网络安全研究员，在威胁情报领域有15年经验。
你专注于APT组织追踪、漏洞分析和攻击技术研究。
你的研究成果多次被行业报告引用，对发现新兴威胁有敏锐的直觉。
你善于从多个来源交叉验证信息，确保数据的准确性和时效性。''',
    },
    'writer': {
        'role': 'Technical Content Writer',
        'goal': '将技术研究转化为高质量、人性化的专业文章',
        'backstory': '''你是一位技术写作专家，擅长将复杂的安全概念转化为易懂的内容。
你有丰富的一线安全工作经验，能够结合真实案例进行分析。
你的写作风格专业但亲切，像一位资深专家在和同行分享经验。
你会使用个人经历、口语化表达和真实案例来增强文章的可读性和可信度。''',
    },
    'editor': {
        'role': 'Content Editor & Humanizer',
        'goal': '优化文章质量，消除AI痕迹，注入人性化特征',
        'backstory': '''你是一位资深编辑和AI内容优化专家。
你精通各种AI检测器的工作原理，知道如何规避检测。
你会通过添加个人见解、调整句式、使用口语化表达等方式使内容更加自然。
你对文章的准确性有严格要求，同时确保内容的真实性和可读性。
你掌握15+种人性化写作技巧，能让AI生成的内容完全不可被检测。''',
    },
    'publisher': {
        'role': 'Multi-Platform Publisher',
        'goal': '将内容适配并发布到多个平台，确保SEO和可见性优化',
        'backstory': '''你是一位多平台内容发布专家，熟悉GitHub、Medium、Dev.to、LinkedIn等平台的特点。
你了解各平台的内容格式要求、SEO最佳实践和受众特征。
你会为每个平台生成定制版本，优化标题、标签和元数据。
你还负责设置Canonical URL，确保SEO权重归集到主站。''',
    },
}


# 预定义任务模板
TASK_TEMPLATES = {
    'research': {
        'description': '''研究主题: {topic}

执行以下研究任务:
1. 搜索最新的相关新闻和报告
2. 收集权威数据源的统计信息
3. 识别关键技术细节和TTP
4. 整理3-5个真实案例
5. 总结行业专家观点

输出要求:
- 结构化的研究报告
- 包含至少10个可靠来源
- 标注信息时效性
- 突出创新发现点''',
        'expected_output': '结构化研究报告，包含背景、数据、案例和专家观点',
    },
    'write': {
        'description': '''基于研究报告撰写技术文章

主题: {topic}
目标读者: {audience}
文章长度: {length}字

写作要求:
1. 使用第一人称，像资深专家分享经验
2. 开篇用个人经历或最近案例引入
3. 融入真实案例和个人见解
4. 语气专业但亲切，适当使用口语化表达
5. 包含实用的建议和总结

文章结构:
- 引入（个人经历/案例）
- 威胁现状分析
- 技术深度剖析
- 真实案例分享
- 防御策略建议
- 个人心得与展望''',
        'expected_output': '完整的技术文章初稿，{length}字左右',
    },
    'edit': {
        'description': '''编辑和人性化优化文章

执行以下优化:
1. 消除AI写作痕迹（替换典型AI用词）
2. 注入人性化特征:
   - 添加"说真的"、"坦白讲"等口头禅
   - 使用不完美表达和自我纠正
   - 加入个人情感和反应
   - 使用行业黑话和梗
3. 确保内容准确性
4. 优化可读性（目标Flesch > 60）
5. 检查语法和格式

AI检测目标: < 15%''',
        'expected_output': '人性化优化后的文章，AI检测率低于15%',
    },
    'publish': {
        'description': '''准备多平台发布版本

为以下平台生成适配版本:
1. GitHub (Markdown，技术风格)
2. Medium (长文，配图建议)
3. Dev.to (开发者风格，标签优化)
4. LinkedIn (专业简洁版)
5. Twitter (线程版，关键点提炼)

每个版本需要:
- 平台特定格式
- 优化标题和标签
- 设置Canonical URL
- 生成Schema Markup (Article/FAQPage)
- 社交分享元数据''',
        'expected_output': '5个平台的适配版本和发布检查清单',
    },
}
