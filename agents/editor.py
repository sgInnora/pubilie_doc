#!/usr/bin/env python3
"""
Editor Agent

负责编辑优化、人性化处理和质量把控。
"""

from typing import Optional, List, Any
import re
from .config import AGENT_ROLES, AgentConfig


def create_editor_agent(
    config: Optional[AgentConfig] = None,
    tools: Optional[List[Any]] = None,
    **kwargs
):
    """
    创建Editor Agent

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
        return MockAgent('editor', AGENT_ROLES['editor'])

    config = config or AgentConfig()
    role_config = AGENT_ROLES['editor']

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


# 人性化策略库
HUMANIZATION_STRATEGIES = {
    # 策略1: AI用词替换
    'word_replacement': {
        'name': 'AI用词替换',
        'description': '替换典型的AI写作用词',
        'rules': {
            # 英文替换
            'furthermore': ['another thing', 'also', 'plus', 'and'],
            'moreover': ["what's more", 'and', 'on top of that', 'besides'],
            'nevertheless': ['but still', 'anyway', 'that said', 'even so'],
            'consequently': ['so', 'as a result', 'therefore', 'thus'],
            'subsequently': ['then', 'after that', 'later on', 'next'],
            'however': ['but', 'though', 'that said', 'still'],
            'therefore': ['so', 'thus', "that's why", 'hence'],
            'additionally': ['also', 'plus', 'and', 'on top of that'],
            'in conclusion': ['to wrap up', 'all in all', 'bottom line'],
            'it is important to note': ['worth mentioning', 'keep in mind', 'note that'],
            # 中文替换
            '此外': ['另外', '还有个事儿', '说到这个', '顺便说一下'],
            '综上所述': ['总的来说', '说到底', '我的看法是', '简单总结一下'],
            '值得注意的是': ['有意思的是', '不过话说回来', '我发现', '特别想说的是'],
            '总而言之': ['反正', '说白了', '简单来说', '一句话'],
            '需要强调的是': ['特别想说', '我觉得很重要的是', '敲黑板', '划重点'],
            '在此背景下': ['在这种情况下', '说到这儿', '这么一来'],
            '由此可见': ['可以看出', '这说明', '所以'],
            '不可否认': ['确实', '没错', '这点我承认'],
        },
    },

    # 策略2: 口语化表达注入
    'colloquial_injection': {
        'name': '口语化表达注入',
        'description': '添加口头禅和口语化表达',
        'phrases': {
            'zh': [
                '说真的', '坦白讲', '你懂的', '这个...怎么说呢',
                '我个人觉得', '以我的经验来看', '实话实说',
                '话说回来', '不开玩笑', '认真的', '真的',
                '就我观察', '不夸张地说', '老实说',
            ],
            'en': [
                'honestly', 'to be frank', 'you know', 'I mean',
                'in my experience', 'truth be told', 'look',
                "here's the thing", 'no kidding', 'seriously',
                'from what I can tell', 'not gonna lie',
            ],
        },
    },

    # 策略3: 不完美表达
    'imperfection': {
        'name': '不完美表达',
        'description': '添加人类自然的不完美表达',
        'patterns': {
            'self_correction': [
                '等等，让我想想...', '不对，应该是...',
                '哦对了，差点忘了说...', '嗯...这个具体数字我记不太清了，大概是...',
            ],
            'uncertainty': [
                '我不太确定，但是...', '可能我说的不够准确，',
                '如果我没记错的话...', '大概是这样吧...',
            ],
            'thinking_pause': [
                '...', '——', '嗯', '这个嘛',
            ],
        },
    },

    # 策略4: 个人经历注入
    'personal_experience': {
        'name': '个人经历注入',
        'description': '添加个人经历和案例',
        'templates': [
            '记得有一次...', '前几天我遇到过类似的情况...',
            '上个月处理的一个案子...', '去年我们团队就碰到过...',
            '我有个朋友在xxx公司，他说...', '昨天和老王聊天时他提到...',
        ],
    },

    # 策略5: 情感表达
    'emotional_expression': {
        'name': '情感表达',
        'description': '添加情感反应和个人态度',
        'expressions': {
            'surprise': ['说实话我也惊了', '看到这个我愣了好几秒', '这波操作真的6'],
            'concern': ['这个问题真的让人头疼', '看着就让人担心', '这种情况太常见了'],
            'humor': ['黑客的创意真是...让人哭笑不得', '这UI设计水平越来越高了（苦笑）'],
            'frustration': ['有时候确实挺无力的', '这种事儿见多了也习惯了'],
        },
    },

    # 策略6: 行业黑话
    'industry_jargon': {
        'name': '行业黑话',
        'description': '使用行业内部用语和梗',
        'terms': {
            'security': [
                '这个0day被爆出来的时候，整个圈子都炸了',
                '又是周五晚上搞事情', '经典的钓鱼套路',
                '蓝队日常', '红蓝对抗', 'APT演员',
            ],
        },
    },
}


class TextHumanizer:
    """文本人性化处理器"""

    def __init__(self, level: str = 'maximum'):
        """
        初始化人性化处理器

        Args:
            level: 人性化级别 (minimum, moderate, maximum)
        """
        self.level = level
        self.strategies = HUMANIZATION_STRATEGIES

    def humanize(self, text: str, language: str = 'zh') -> str:
        """
        对文本进行人性化处理

        Args:
            text: 原始文本
            language: 语言 (zh, en)

        Returns:
            人性化后的文本
        """
        import random

        result = text

        # 策略1: 替换AI用词
        for ai_word, alternatives in self.strategies['word_replacement']['rules'].items():
            if ai_word.lower() in result.lower():
                replacement = random.choice(alternatives)
                result = re.sub(
                    re.escape(ai_word),
                    replacement,
                    result,
                    flags=re.IGNORECASE
                )

        if self.level in ['moderate', 'maximum']:
            # 策略2: 在适当位置注入口语化表达
            phrases = self.strategies['colloquial_injection']['phrases'].get(language, [])
            if phrases:
                # 在段落开头随机添加
                paragraphs = result.split('\n\n')
                for i in range(1, len(paragraphs), 3):  # 每隔几段添加
                    if random.random() > 0.5 and paragraphs[i].strip():
                        phrase = random.choice(phrases)
                        paragraphs[i] = f"{phrase}，{paragraphs[i]}"
                result = '\n\n'.join(paragraphs)

        if self.level == 'maximum':
            # 策略3: 添加不完美表达
            imperfections = self.strategies['imperfection']['patterns']
            # 随机添加自我纠正
            if random.random() > 0.7:
                correction = random.choice(imperfections['self_correction'])
                # 在文章中间某处插入
                mid_point = len(result) // 2
                insert_point = result.find('。', mid_point)
                if insert_point > 0:
                    result = result[:insert_point + 1] + correction + result[insert_point + 1:]

        return result

    def detect_ai_patterns(self, text: str) -> dict:
        """
        检测文本中的AI模式

        Returns:
            检测结果字典
        """
        patterns_found = []
        text_lower = text.lower()

        for ai_word in self.strategies['word_replacement']['rules'].keys():
            if ai_word.lower() in text_lower:
                count = text_lower.count(ai_word.lower())
                patterns_found.append({'pattern': ai_word, 'count': count})

        word_count = len(text.split())
        total_ai_words = sum(p['count'] for p in patterns_found)
        ai_density = total_ai_words / max(word_count, 1) * 100

        return {
            'patterns_found': patterns_found,
            'total_ai_patterns': len(patterns_found),
            'ai_word_density': f"{ai_density:.2f}%",
            'estimated_ai_probability': min(ai_density * 5, 100),
            'needs_humanization': ai_density > 0.5,
        }


# 编辑检查清单
EDITING_CHECKLIST = {
    'accuracy': {
        'name': '准确性检查',
        'items': [
            '统计数据有可靠来源',
            '技术细节正确无误',
            '案例描述合理可信',
            '不包含虚构的组织/产品',
            '时间线逻辑一致',
        ],
    },
    'readability': {
        'name': '可读性检查',
        'items': [
            '句子长度适中（平均20-25词）',
            '段落不超过5句',
            '使用清晰的标题层级',
            '专业术语有解释',
            'Flesch Reading Ease > 60',
        ],
    },
    'humanization': {
        'name': '人性化检查',
        'items': [
            '无典型AI用词',
            '有个人见解和经历',
            '语气自然亲切',
            '有适当的情感表达',
            '使用口语化过渡',
        ],
    },
    'structure': {
        'name': '结构检查',
        'items': [
            '开头吸引人',
            '逻辑流畅',
            '论点有支撑',
            '结尾有力',
            '包含可执行建议',
        ],
    },
}


def get_editing_checklist() -> dict:
    """获取完整编辑检查清单"""
    return EDITING_CHECKLIST
