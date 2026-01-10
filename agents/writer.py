#!/usr/bin/env python3
"""
Writer Agent

负责将研究转化为高质量技术文章。
"""

from typing import Optional, List, Any
from .config import AGENT_ROLES, AgentConfig


def create_writer_agent(
    config: Optional[AgentConfig] = None,
    tools: Optional[List[Any]] = None,
    **kwargs
):
    """
    创建Writer Agent

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
        return MockAgent('writer', AGENT_ROLES['writer'])

    config = config or AgentConfig()
    role_config = AGENT_ROLES['writer']

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


# 写作风格模板
WRITING_STYLES = {
    'expert_sharing': {
        'name': '专家分享型',
        'description': '像资深专家和同行交流，专业但亲切',
        'characteristics': [
            '使用第一人称',
            '融入个人经历和案例',
            '适当使用行业术语和黑话',
            '口语化表达',
            '承认不确定性',
        ],
        'opening_templates': [
            '说真的，最近这波{topic}让我想起了去年处理的一个案子...',
            '前几天和老王聊天，他提到一个关于{topic}的有趣观点...',
            '记得是上个月吧，我们团队遇到了一个{topic}的棘手问题...',
        ],
        'transition_phrases': [
            '不过话说回来',
            '还有个事儿特别值得注意',
            '这个...怎么说呢',
            '坦白讲',
            '以我的经验来看',
        ],
        'closing_templates': [
            '就先聊这么多吧，欢迎大家一起讨论。',
            '以上就是我的一些个人见解，仅供参考。',
            '安全这个事儿，还是需要大家一起努力的。',
        ],
    },
    'tutorial': {
        'name': '教程指南型',
        'description': '手把手教学，步骤清晰',
        'characteristics': [
            '使用第二人称',
            '步骤编号清晰',
            '包含代码示例',
            '提供注意事项',
            '给出常见问题解答',
        ],
    },
    'analysis': {
        'name': '深度分析型',
        'description': '技术深度剖析，数据支撑',
        'characteristics': [
            '使用数据和统计',
            '多角度分析',
            '引用权威来源',
            '提供对比分析',
            '预测趋势',
        ],
    },
    'news': {
        'name': '新闻资讯型',
        'description': '快速传递信息，客观报道',
        'characteristics': [
            '使用倒金字塔结构',
            '核心信息前置',
            '简洁明了',
            '引用多方观点',
            '时效性强',
        ],
    },
}


# 文章结构模板
ARTICLE_STRUCTURES = {
    'threat_analysis': {
        'name': '威胁分析结构',
        'sections': [
            {'name': '引入', 'description': '个人经历或最近案例引入'},
            {'name': '威胁概述', 'description': '威胁背景和现状'},
            {'name': '技术分析', 'description': '攻击技术深度剖析'},
            {'name': '案例分析', 'description': '真实案例分享（脱敏）'},
            {'name': '防御策略', 'description': '实用防御建议'},
            {'name': '总结展望', 'description': '个人心得和趋势预测'},
        ],
    },
    'vulnerability_report': {
        'name': '漏洞报告结构',
        'sections': [
            {'name': '概述', 'description': '漏洞基本信息'},
            {'name': '技术原理', 'description': '漏洞成因分析'},
            {'name': '影响评估', 'description': '影响范围和严重性'},
            {'name': 'PoC分析', 'description': '利用方式（谨慎）'},
            {'name': '修复方案', 'description': '官方修复和临时方案'},
            {'name': '检测建议', 'description': '检测规则和IOC'},
        ],
    },
    'tool_review': {
        'name': '工具评测结构',
        'sections': [
            {'name': '背景介绍', 'description': '工具定位和使用场景'},
            {'name': '安装配置', 'description': '环境准备和安装步骤'},
            {'name': '功能演示', 'description': '核心功能使用'},
            {'name': '实战测试', 'description': '真实场景测试'},
            {'name': '优劣分析', 'description': '优点缺点对比'},
            {'name': '使用建议', 'description': '最佳实践推荐'},
        ],
    },
}


def get_writing_style(style_name: str) -> dict:
    """获取写作风格配置"""
    if style_name not in WRITING_STYLES:
        raise ValueError(f"Unknown style: {style_name}. "
                         f"Available: {list(WRITING_STYLES.keys())}")
    return WRITING_STYLES[style_name]


def get_article_structure(structure_name: str) -> dict:
    """获取文章结构模板"""
    if structure_name not in ARTICLE_STRUCTURES:
        raise ValueError(f"Unknown structure: {structure_name}. "
                         f"Available: {list(ARTICLE_STRUCTURES.keys())}")
    return ARTICLE_STRUCTURES[structure_name]
