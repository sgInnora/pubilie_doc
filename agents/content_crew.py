#!/usr/bin/env python3
"""
Content Crew - Multi-Agent Content Pipeline

协调 Researcher → Writer → Editor → Publisher 的完整内容生产流程。
基于CrewAI框架实现多智能体协作。

版本: 1.0
创建时间: 2026-01-10
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .config import AgentConfig, CrewConfig, TASK_TEMPLATES
from .researcher import create_researcher_agent
from .writer import create_writer_agent
from .editor import create_editor_agent
from .publisher import create_publisher_agent
from .tools import (
    WebSearchTool, ThreatIntelTool, ReadabilityTool,
    AIDetectionTool, HumanizationTool, SchemaMarkupTool,
    CanonicalURLTool, ATTACKMappingTool
)


@dataclass
class ContentRequest:
    """内容生成请求"""
    topic: str
    audience: str = "企业安全团队和技术决策者"
    length: int = 3000  # 字数
    language: str = "zh"  # zh, en, both
    style: str = "expert_sharing"  # 写作风格
    platforms: List[str] = field(default_factory=lambda: ['github', 'medium', 'linkedin'])
    keywords: List[str] = field(default_factory=list)
    humanization_level: str = "maximum"  # minimum, moderate, maximum

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'topic': self.topic,
            'audience': self.audience,
            'length': self.length,
            'language': self.language,
            'style': self.style,
            'platforms': self.platforms,
            'keywords': self.keywords,
            'humanization_level': self.humanization_level,
        }


@dataclass
class ContentOutput:
    """内容生成输出"""
    research_report: str = ""
    draft_article: str = ""
    edited_article: str = ""
    platform_versions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)


class ContentCrew:
    """
    内容生产Crew

    协调多个Agent完成从研究到发布的完整流程。
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        crew_config: Optional[CrewConfig] = None
    ):
        """
        初始化Content Crew

        Args:
            agent_config: Agent配置
            crew_config: Crew配置
        """
        self.agent_config = agent_config or AgentConfig()
        self.crew_config = crew_config or CrewConfig()

        # 初始化工具
        self.tools = self._initialize_tools()

        # 初始化Agents
        self.agents = self._initialize_agents()

        # CrewAI Crew对象（延迟初始化）
        self._crew = None

    def _initialize_tools(self) -> Dict[str, Any]:
        """初始化工具集"""
        return {
            'web_search': WebSearchTool(),
            'threat_intel': ThreatIntelTool(),
            'readability': ReadabilityTool(),
            'ai_detection': AIDetectionTool(),
            'humanize': HumanizationTool(),
            'schema_markup': SchemaMarkupTool(),
            'canonical_url': CanonicalURLTool(),
            'attack_mapping': ATTACKMappingTool(),
        }

    def _initialize_agents(self) -> Dict[str, Any]:
        """初始化Agent集合"""
        return {
            'researcher': create_researcher_agent(
                config=self.agent_config,
                tools=[
                    self.tools['web_search'],
                    self.tools['threat_intel'],
                    self.tools['attack_mapping'],
                ]
            ),
            'writer': create_writer_agent(
                config=self.agent_config,
                tools=[self.tools['readability']]
            ),
            'editor': create_editor_agent(
                config=self.agent_config,
                tools=[
                    self.tools['ai_detection'],
                    self.tools['humanize'],
                    self.tools['readability'],
                ]
            ),
            'publisher': create_publisher_agent(
                config=self.agent_config,
                tools=[
                    self.tools['schema_markup'],
                    self.tools['canonical_url'],
                ]
            ),
        }

    def _create_tasks(self, request: ContentRequest) -> List[Any]:
        """
        创建任务列表

        Args:
            request: 内容请求

        Returns:
            CrewAI Task列表
        """
        try:
            from crewai import Task
        except ImportError:
            # 返回模拟任务
            return self._create_mock_tasks(request)

        tasks = []

        # Task 1: 研究
        research_task = Task(
            description=TASK_TEMPLATES['research']['description'].format(
                topic=request.topic
            ),
            expected_output=TASK_TEMPLATES['research']['expected_output'],
            agent=self.agents['researcher'],
        )
        tasks.append(research_task)

        # Task 2: 写作
        write_task = Task(
            description=TASK_TEMPLATES['write']['description'].format(
                topic=request.topic,
                audience=request.audience,
                length=request.length,
            ),
            expected_output=TASK_TEMPLATES['write']['expected_output'].format(
                length=request.length
            ),
            agent=self.agents['writer'],
            context=[research_task],  # 依赖研究任务
        )
        tasks.append(write_task)

        # Task 3: 编辑与人性化
        edit_task = Task(
            description=TASK_TEMPLATES['edit']['description'],
            expected_output=TASK_TEMPLATES['edit']['expected_output'],
            agent=self.agents['editor'],
            context=[write_task],
        )
        tasks.append(edit_task)

        # Task 4: 发布准备
        publish_task = Task(
            description=TASK_TEMPLATES['publish']['description'],
            expected_output=TASK_TEMPLATES['publish']['expected_output'],
            agent=self.agents['publisher'],
            context=[edit_task],
        )
        tasks.append(publish_task)

        return tasks

    def _create_mock_tasks(self, request: ContentRequest) -> List[Dict]:
        """创建模拟任务用于测试"""
        return [
            {'name': 'research', 'topic': request.topic},
            {'name': 'write', 'topic': request.topic, 'length': request.length},
            {'name': 'edit', 'level': request.humanization_level},
            {'name': 'publish', 'platforms': request.platforms},
        ]

    def _get_or_create_crew(self, tasks: List[Any]):
        """获取或创建CrewAI Crew"""
        try:
            from crewai import Crew, Process

            process_map = {
                'sequential': Process.sequential,
                'hierarchical': Process.hierarchical,
            }

            return Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=process_map.get(self.crew_config.process_type, Process.sequential),
                verbose=self.crew_config.verbose,
                memory=self.crew_config.memory,
            )
        except ImportError:
            return None

    def kickoff(self, request: ContentRequest) -> ContentOutput:
        """
        启动内容生产流程

        Args:
            request: 内容请求

        Returns:
            内容输出
        """
        output = ContentOutput()
        output.metadata['request'] = request.to_dict()
        output.metadata['start_time'] = datetime.now().isoformat()

        print(f"🚀 启动Content Crew")
        print(f"   主题: {request.topic}")
        print(f"   目标平台: {', '.join(request.platforms)}")
        print("=" * 60)

        # 创建任务
        tasks = self._create_tasks(request)

        # 尝试使用CrewAI
        crew = self._get_or_create_crew(tasks)

        if crew:
            # 使用CrewAI执行
            print("\n📋 使用CrewAI执行任务...")
            result = crew.kickoff()
            output.edited_article = str(result)
        else:
            # 回退到模拟执行
            print("\n📋 使用模拟模式执行（未安装CrewAI）...")
            output = self._mock_execution(request, output)

        output.metadata['end_time'] = datetime.now().isoformat()

        # 生成质量报告
        output.quality_report = self._generate_quality_report(output)

        return output

    def _mock_execution(
        self,
        request: ContentRequest,
        output: ContentOutput
    ) -> ContentOutput:
        """
        模拟执行流程（用于无CrewAI环境）

        Args:
            request: 内容请求
            output: 输出对象

        Returns:
            填充后的输出
        """
        # Step 1: 模拟研究
        print("\n📚 Step 1: Researcher Agent - 研究主题...")
        output.research_report = f"""
# 研究报告: {request.topic}

## 背景
[基于网络搜索的背景信息]

## 关键发现
1. 发现1
2. 发现2
3. 发现3

## 数据支撑
- 来源1: [统计数据]
- 来源2: [专家观点]

## 案例
- 案例1: [脱敏案例描述]
"""

        # Step 2: 模拟写作
        print("📝 Step 2: Writer Agent - 撰写文章...")
        output.draft_article = f"""
# {request.topic}

> 作者: 资深安全研究员
> 日期: {datetime.now().strftime('%Y年%m月')}

说真的，最近这个话题特别火。前几天和同行交流时，大家都在讨论...

## 背景

{output.research_report}

## 深度分析

以我的经验来看，这个问题需要从几个角度来分析...

## 建议

1. 建议一
2. 建议二
3. 建议三

## 结语

就先聊这么多吧，欢迎大家一起讨论。
"""

        # Step 3: 模拟编辑
        print("✏️ Step 3: Editor Agent - 编辑优化...")
        from .editor import TextHumanizer
        humanizer = TextHumanizer(level=request.humanization_level)
        output.edited_article = humanizer.humanize(output.draft_article, request.language)

        # Step 4: 模拟发布准备
        print("📤 Step 4: Publisher Agent - 准备发布...")
        from .publisher import PlatformAdapter
        adapter = PlatformAdapter()

        for platform in request.platforms:
            try:
                adapted = adapter.adapt_for_platform(
                    output.edited_article,
                    platform,
                    {'title': request.topic, 'tags': request.keywords}
                )
                output.platform_versions[platform] = adapted
                print(f"   ✅ {platform} 版本已生成")
            except Exception as e:
                print(f"   ⚠️ {platform} 生成失败: {e}")

        return output

    def _generate_quality_report(self, output: ContentOutput) -> Dict:
        """生成质量报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
        }

        # AI检测
        ai_detector = self.tools['ai_detection']
        if output.edited_article:
            ai_result = json.loads(ai_detector._run(output.edited_article))
            report['checks']['ai_detection'] = ai_result

        # 可读性
        readability = self.tools['readability']
        if output.edited_article:
            readability_result = readability._run(output.edited_article)
            report['checks']['readability'] = readability_result

        # 平台版本数量
        report['platforms_generated'] = len(output.platform_versions)

        return report

    def save_output(self, output: ContentOutput, output_dir: Optional[Path] = None):
        """
        保存输出到文件

        Args:
            output: 内容输出
            output_dir: 输出目录
        """
        output_dir = output_dir or self.crew_config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存主文章
        if output.edited_article:
            article_path = output_dir / f"article_{timestamp}_CN.md"
            article_path.write_text(output.edited_article, encoding='utf-8')
            print(f"📄 主文章: {article_path}")

        # 保存平台版本
        for platform, content in output.platform_versions.items():
            platform_path = output_dir / f"article_{timestamp}_{platform}.md"
            platform_path.write_text(content, encoding='utf-8')
            print(f"📄 {platform}: {platform_path}")

        # 保存质量报告
        report_path = output_dir / f"quality_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(output.quality_report, f, ensure_ascii=False, indent=2)
        print(f"📊 质量报告: {report_path}")

        # 保存元数据
        metadata_path = output_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(output.metadata, f, ensure_ascii=False, indent=2)
        print(f"📋 元数据: {metadata_path}")


def main():
    """CLI入口"""
    import argparse

    parser = argparse.ArgumentParser(description='Content Crew - Multi-Agent Content Pipeline')
    parser.add_argument('topic', help='文章主题')
    parser.add_argument('--audience', '-a', default='企业安全团队', help='目标读者')
    parser.add_argument('--length', '-l', type=int, default=3000, help='目标字数')
    parser.add_argument('--language', choices=['zh', 'en', 'both'], default='zh', help='语言')
    parser.add_argument('--platforms', '-p', nargs='+',
                        default=['github', 'medium', 'linkedin'],
                        help='目标平台')
    parser.add_argument('--output', '-o', help='输出目录')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')

    args = parser.parse_args()

    # 创建请求
    request = ContentRequest(
        topic=args.topic,
        audience=args.audience,
        length=args.length,
        language=args.language,
        platforms=args.platforms,
    )

    # 配置
    agent_config = AgentConfig(verbose=args.verbose)
    crew_config = CrewConfig(verbose=args.verbose)
    if args.output:
        crew_config.output_dir = Path(args.output)

    # 创建Crew并执行
    crew = ContentCrew(agent_config=agent_config, crew_config=crew_config)
    output = crew.kickoff(request)

    # 保存输出
    crew.save_output(output)

    print("\n" + "=" * 60)
    print("🎉 内容生成完成!")
    print(f"📊 AI检测结果: {output.quality_report.get('checks', {}).get('ai_detection', {})}")

    return 0


if __name__ == '__main__':
    exit(main())
