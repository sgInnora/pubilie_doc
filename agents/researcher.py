#!/usr/bin/env python3
"""
Researcher Agent

负责研究主题、收集资料和整理情报。
"""

from typing import Optional, List, Any
from .config import AGENT_ROLES, AgentConfig


def create_researcher_agent(
    config: Optional[AgentConfig] = None,
    tools: Optional[List[Any]] = None,
    **kwargs
):
    """
    创建Researcher Agent

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
        # 返回模拟Agent用于测试
        return MockAgent('researcher', AGENT_ROLES['researcher'])

    config = config or AgentConfig()
    role_config = AGENT_ROLES['researcher']

    # 合并配置
    agent_params = {
        'role': role_config['role'],
        'goal': role_config['goal'],
        'backstory': role_config['backstory'],
        'verbose': config.verbose,
        'allow_delegation': kwargs.get('allow_delegation', False),
        'max_iter': config.max_iter,
        'max_retry_limit': config.max_retry_limit,
    }

    # 添加LLM配置
    if config.llm_model:
        agent_params['llm'] = config.llm_model

    # 添加工具
    if tools:
        agent_params['tools'] = tools

    # 覆盖额外参数
    agent_params.update(kwargs)

    return Agent(**agent_params)


class MockAgent:
    """模拟Agent用于无CrewAI环境测试"""

    def __init__(self, name: str, role_config: dict):
        self.name = name
        self.role = role_config['role']
        self.goal = role_config['goal']
        self.backstory = role_config['backstory']

    def execute(self, task: str) -> str:
        """模拟执行任务"""
        return f"[{self.role}] 已完成任务: {task[:100]}..."


# 预定义的研究任务模板
RESEARCH_TASKS = {
    'threat_landscape': {
        'name': '威胁态势研究',
        'description': '''研究当前网络安全威胁态势:
1. 收集近期重大安全事件
2. 分析攻击趋势和新技术
3. 整理APT组织活动情报
4. 统计漏洞利用数据
5. 总结行业预测和建议''',
    },
    'vulnerability_analysis': {
        'name': '漏洞深度分析',
        'description': '''对指定漏洞进行深度研究:
1. 漏洞技术原理
2. 影响范围评估
3. 在野利用情况
4. 修复方案建议
5. 检测规则推荐''',
    },
    'apt_tracking': {
        'name': 'APT追踪研究',
        'description': '''追踪APT组织活动:
1. 组织背景和归属
2. 历史攻击战役
3. 常用TTP分析
4. IOC指标收集
5. 防御建议''',
    },
    'emerging_tech': {
        'name': '新兴技术研究',
        'description': '''研究新兴安全技术:
1. 技术原理介绍
2. 应用场景分析
3. 优劣势对比
4. 市场现状
5. 发展趋势预测''',
    },
}


def get_research_task(task_type: str) -> dict:
    """获取预定义研究任务"""
    if task_type not in RESEARCH_TASKS:
        raise ValueError(f"Unknown task type: {task_type}. "
                         f"Available: {list(RESEARCH_TASKS.keys())}")
    return RESEARCH_TASKS[task_type]
