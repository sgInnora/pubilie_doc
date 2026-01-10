#!/usr/bin/env python3
"""
pubilie_doc Multi-Agent Content Pipeline

基于CrewAI框架的多智能体内容生成管线。
实现 Researcher → Writer → Editor → Publisher 协作流程。

版本: 1.0
创建时间: 2026-01-10
"""

from .config import AgentConfig, CrewConfig
from .content_crew import ContentCrew, ContentRequest, ContentOutput
from .researcher import create_researcher_agent
from .writer import create_writer_agent
from .editor import create_editor_agent
from .publisher import create_publisher_agent

__all__ = [
    'AgentConfig',
    'CrewConfig',
    'ContentCrew',
    'ContentRequest',
    'ContentOutput',
    'create_researcher_agent',
    'create_writer_agent',
    'create_editor_agent',
    'create_publisher_agent',
]

__version__ = '1.0.0'
