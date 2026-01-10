#!/usr/bin/env python3
"""
deep_ai_detector.py - 深度学习AI检测模块 v2.0

用途:
    补充check_ai_traces.sh的规则检测，使用深度学习API进行第三层验证
    支持多检测器聚合评分，提升检测准确性

背景:
    Pangram Labs研究表明Perplexity/Burstiness规则检测假阳性率1-50%
    深度学习模型可提供更准确的检测结果
    多检测器聚合可进一步降低误判率

支持的检测服务:
    - GPTZero API (推荐，学术用途免费额度)
    - Originality.ai API (商业付费，精度最高)
    - ZeroGPT API (免费额度，快速检测)
    - Copyleaks API (企业级，支持多语言)
    - Winston AI (备选)
    - 自定义LLM (Qwen/Claude/GPT)

新增功能 (v2.0):
    - 多检测器聚合评分
    - ZeroGPT/Copyleaks API支持
    - 加权平均置信度计算
    - 检测器一致性分析

用法:
    # 单文件检测
    python deep_ai_detector.py --file article.md

    # 多检测器聚合模式
    python deep_ai_detector.py --file article.md --aggregate

    # JSON输出 (供n8n调用)
    python deep_ai_detector.py --file article.md --json

    # 批量检测
    python deep_ai_detector.py --dir 2026_01/ --pattern "*.md"

作者: Innora Security Research Team
版本: 2.0 | 日期: 2026-01-10
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 可选依赖
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".env")
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# ════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════

@dataclass
class DetectorConfig:
    """检测器配置"""
    # 服务选择: gptzero, originality, winston, custom_llm
    service: str = "gptzero"

    # GPTZero配置
    gptzero_api_key: str = ""
    gptzero_endpoint: str = "https://api.gptzero.me/v2/predict/text"

    # Originality.ai配置
    originality_api_key: str = ""
    originality_endpoint: str = "https://api.originality.ai/api/v1/scan/ai"

    # Winston AI配置
    winston_api_key: str = ""
    winston_endpoint: str = "https://api.gowinston.ai/v2/detect"

    # ZeroGPT配置 (v2.0新增)
    zerogpt_api_key: str = ""
    zerogpt_endpoint: str = "https://api.zerogpt.com/api/detect/detectText"

    # Copyleaks配置 (v2.0新增)
    copyleaks_api_key: str = ""
    copyleaks_email: str = ""
    copyleaks_endpoint: str = "https://api.copyleaks.com/v2/writer-detector"

    # 自定义LLM配置 (使用vLLM或OpenAI兼容API)
    custom_llm_endpoint: str = ""
    custom_llm_api_key: str = ""
    custom_llm_model: str = "Qwen/Qwen2.5-72B-Instruct"

    # 检测阈值
    ai_threshold: float = 0.7  # >0.7 判定为AI生成
    human_threshold: float = 0.3  # <0.3 判定为人类写作

    # 缓存
    cache_dir: str = ".ai_detection_cache"
    cache_ttl_hours: int = 24

    @classmethod
    def from_env(cls) -> "DetectorConfig":
        """从环境变量加载配置"""
        return cls(
            service=os.getenv("AI_DETECTOR_SERVICE", "gptzero"),
            gptzero_api_key=os.getenv("GPTZERO_API_KEY", ""),
            originality_api_key=os.getenv("ORIGINALITY_API_KEY", ""),
            winston_api_key=os.getenv("WINSTON_API_KEY", ""),
            zerogpt_api_key=os.getenv("ZEROGPT_API_KEY", ""),
            copyleaks_api_key=os.getenv("COPYLEAKS_API_KEY", ""),
            copyleaks_email=os.getenv("COPYLEAKS_EMAIL", ""),
            custom_llm_endpoint=os.getenv("VLLM_HOST", os.getenv("CUSTOM_LLM_ENDPOINT", "")),
            custom_llm_api_key=os.getenv("VLLM_API_KEY", os.getenv("CUSTOM_LLM_API_KEY", "")),
            custom_llm_model=os.getenv("CUSTOM_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
            ai_threshold=float(os.getenv("AI_DETECTOR_THRESHOLD", "0.7")),
        )


# ════════════════════════════════════════════════════════════════
# 检测结果
# ════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """检测结果"""
    file_path: str
    service: str
    timestamp: str

    # 核心分数 (0-1, 1=完全AI生成)
    ai_probability: float
    human_probability: float
    mixed_probability: float

    # 判定结果
    classification: str  # "ai", "human", "mixed", "uncertain"
    confidence: float

    # 详细分析 (服务相关)
    sentence_scores: Optional[List[float]] = None
    paragraph_scores: Optional[List[float]] = None
    highlighted_ai_sentences: Optional[List[str]] = None

    # 元数据
    text_length: int = 0
    language: str = "auto"
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class AggregateResult:
    """多检测器聚合结果 (v2.0新增)"""
    file_path: str
    timestamp: str

    # 聚合分数
    weighted_ai_probability: float
    weighted_human_probability: float
    consensus_classification: str  # 多数投票结果
    consensus_confidence: float
    agreement_ratio: float  # 检测器一致性 (0-1)

    # 各检测器结果
    individual_results: List[Dict] = field(default_factory=list)
    successful_detectors: int = 0
    failed_detectors: int = 0

    # 推荐
    recommendation: str = ""
    needs_review: bool = False  # 结果分歧时需要人工审核

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════
# 检测服务适配器
# ════════════════════════════════════════════════════════════════

class BaseDetector:
    """检测器基类"""

    def __init__(self, config: DetectorConfig):
        self.config = config

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        raise NotImplementedError

    def _create_error_result(self, file_path: str, error: str) -> DetectionResult:
        return DetectionResult(
            file_path=file_path,
            service=self.__class__.__name__,
            timestamp=datetime.now().isoformat(),
            ai_probability=0.5,
            human_probability=0.5,
            mixed_probability=0.0,
            classification="uncertain",
            confidence=0.0,
            error=error
        )


class GPTZeroDetector(BaseDetector):
    """GPTZero API检测器"""

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        if not self.config.gptzero_api_key:
            return self._create_error_result(file_path, "GPTZERO_API_KEY not configured")

        if not HAS_REQUESTS:
            return self._create_error_result(file_path, "requests library not installed")

        try:
            response = requests.post(
                self.config.gptzero_endpoint,
                headers={
                    "X-Api-Key": self.config.gptzero_api_key,
                    "Content-Type": "application/json"
                },
                json={"document": text[:50000]},  # GPTZero限制50k字符
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # 解析GPTZero响应
            doc = data.get("documents", [{}])[0]
            ai_prob = doc.get("completely_generated_prob", 0.5)
            mixed_prob = doc.get("average_generated_prob", 0.5)

            # 分类逻辑
            if ai_prob > self.config.ai_threshold:
                classification = "ai"
                confidence = ai_prob
            elif ai_prob < self.config.human_threshold:
                classification = "human"
                confidence = 1 - ai_prob
            else:
                classification = "mixed"
                confidence = 1 - abs(ai_prob - 0.5) * 2

            return DetectionResult(
                file_path=file_path,
                service="GPTZero",
                timestamp=datetime.now().isoformat(),
                ai_probability=ai_prob,
                human_probability=1 - ai_prob,
                mixed_probability=mixed_prob,
                classification=classification,
                confidence=confidence,
                sentence_scores=doc.get("sentences", []),
                text_length=len(text)
            )

        except Exception as e:
            return self._create_error_result(file_path, str(e))


class OriginalityDetector(BaseDetector):
    """Originality.ai API检测器"""

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        if not self.config.originality_api_key:
            return self._create_error_result(file_path, "ORIGINALITY_API_KEY not configured")

        if not HAS_REQUESTS:
            return self._create_error_result(file_path, "requests library not installed")

        try:
            response = requests.post(
                self.config.originality_endpoint,
                headers={
                    "X-OAI-API-KEY": self.config.originality_api_key,
                    "Content-Type": "application/json"
                },
                json={"content": text[:25000]},  # Originality限制25k字符
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            ai_score = data.get("score", {}).get("ai", 0.5)
            original_score = data.get("score", {}).get("original", 0.5)

            if ai_score > self.config.ai_threshold:
                classification = "ai"
            elif ai_score < self.config.human_threshold:
                classification = "human"
            else:
                classification = "mixed"

            return DetectionResult(
                file_path=file_path,
                service="Originality.ai",
                timestamp=datetime.now().isoformat(),
                ai_probability=ai_score,
                human_probability=original_score,
                mixed_probability=0.0,
                classification=classification,
                confidence=max(ai_score, original_score),
                text_length=len(text)
            )

        except Exception as e:
            return self._create_error_result(file_path, str(e))


class ZeroGPTDetector(BaseDetector):
    """ZeroGPT API检测器 (v2.0新增)"""

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        if not self.config.zerogpt_api_key:
            return self._create_error_result(file_path, "ZEROGPT_API_KEY not configured")

        if not HAS_REQUESTS:
            return self._create_error_result(file_path, "requests library not installed")

        try:
            response = requests.post(
                self.config.zerogpt_endpoint,
                headers={
                    "ApiKey": self.config.zerogpt_api_key,
                    "Content-Type": "application/json"
                },
                json={"input_text": text[:50000]},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # 解析ZeroGPT响应
            # ZeroGPT返回fake_percentage (0-100)
            fake_pct = data.get("data", {}).get("fakePercentage", 50)
            ai_prob = fake_pct / 100.0

            if ai_prob > self.config.ai_threshold:
                classification = "ai"
            elif ai_prob < self.config.human_threshold:
                classification = "human"
            else:
                classification = "mixed"

            return DetectionResult(
                file_path=file_path,
                service="ZeroGPT",
                timestamp=datetime.now().isoformat(),
                ai_probability=ai_prob,
                human_probability=1 - ai_prob,
                mixed_probability=0.0,
                classification=classification,
                confidence=max(ai_prob, 1 - ai_prob),
                highlighted_ai_sentences=data.get("data", {}).get("h", []),
                text_length=len(text)
            )

        except Exception as e:
            return self._create_error_result(file_path, str(e))


class CopyleaksDetector(BaseDetector):
    """Copyleaks AI检测器 (v2.0新增，企业级)"""

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._access_token = None

    def _get_access_token(self) -> Optional[str]:
        """获取Copyleaks访问令牌"""
        if self._access_token:
            return self._access_token

        if not self.config.copyleaks_email or not self.config.copyleaks_api_key:
            return None

        try:
            response = requests.post(
                "https://id.copyleaks.com/v3/account/login/api",
                json={
                    "email": self.config.copyleaks_email,
                    "key": self.config.copyleaks_api_key
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            self._access_token = data.get("access_token")
            return self._access_token
        except Exception:
            return None

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        if not self.config.copyleaks_api_key:
            return self._create_error_result(file_path, "COPYLEAKS_API_KEY not configured")

        if not HAS_REQUESTS:
            return self._create_error_result(file_path, "requests library not installed")

        token = self._get_access_token()
        if not token:
            return self._create_error_result(file_path, "Failed to authenticate with Copyleaks")

        try:
            response = requests.post(
                self.config.copyleaks_endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json={"text": text[:25000]},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # 解析Copyleaks响应
            summary = data.get("summary", {})
            ai_prob = summary.get("ai", 0.5)
            human_prob = summary.get("human", 0.5)

            if ai_prob > self.config.ai_threshold:
                classification = "ai"
            elif ai_prob < self.config.human_threshold:
                classification = "human"
            else:
                classification = "mixed"

            return DetectionResult(
                file_path=file_path,
                service="Copyleaks",
                timestamp=datetime.now().isoformat(),
                ai_probability=ai_prob,
                human_probability=human_prob,
                mixed_probability=1 - ai_prob - human_prob if ai_prob + human_prob < 1 else 0,
                classification=classification,
                confidence=max(ai_prob, human_prob),
                text_length=len(text)
            )

        except Exception as e:
            return self._create_error_result(file_path, str(e))


class CustomLLMDetector(BaseDetector):
    """自定义LLM检测器 (Qwen/Claude/GPT兼容)"""

    DETECTION_PROMPT = """Analyze the following text and determine if it was written by AI or a human.

Analyze these indicators:
1. Sentence structure variation (humans have more varied patterns)
2. Word choice (AI tends to use formal, academic vocabulary)
3. Logical flow (AI is often too structured and predictable)
4. Idioms and colloquialisms (humans use more informal expressions)
5. Personal opinions and emotional expressions (AI is often neutral)

Text to analyze:
---
{text}
---

Respond in JSON format ONLY:
{{
  "ai_probability": 0.0-1.0,
  "human_probability": 0.0-1.0,
  "classification": "ai" | "human" | "mixed",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "ai_indicators": ["indicator1", "indicator2"],
  "human_indicators": ["indicator1", "indicator2"]
}}"""

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        if not self.config.custom_llm_endpoint:
            return self._create_error_result(file_path, "CUSTOM_LLM_ENDPOINT not configured")

        if not HAS_REQUESTS:
            return self._create_error_result(file_path, "requests library not installed")

        try:
            # 截取前5000字符用于分析
            sample_text = text[:5000]
            prompt = self.DETECTION_PROMPT.format(text=sample_text)

            response = requests.post(
                f"{self.config.custom_llm_endpoint}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.custom_llm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.custom_llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # 解析LLM响应
            content = data["choices"][0]["message"]["content"]

            # 提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return self._create_error_result(file_path, "Failed to parse LLM response as JSON")

            result = json.loads(json_match.group())

            return DetectionResult(
                file_path=file_path,
                service=f"CustomLLM ({self.config.custom_llm_model})",
                timestamp=datetime.now().isoformat(),
                ai_probability=result.get("ai_probability", 0.5),
                human_probability=result.get("human_probability", 0.5),
                mixed_probability=0.0,
                classification=result.get("classification", "uncertain"),
                confidence=result.get("confidence", 0.5),
                highlighted_ai_sentences=result.get("ai_indicators", []),
                text_length=len(text)
            )

        except Exception as e:
            return self._create_error_result(file_path, str(e))


class FallbackDetector(BaseDetector):
    """本地规则检测器 (无需API，作为降级方案)"""

    # AI典型话术 (来自check_ai_traces.sh)
    AI_PHRASES_CN = [
        "让我们深入", "让我们来看", "让我们探讨",
        "值得注意的是", "值得一提",
        "综上所述", "总的来说", "总而言之",
        "不可否认", "毫无疑问", "显而易见",
        "在当今.*时代", "在.*的背景下",
        "我们可以得出", "可以得出结论",
        "这意味着", "这表明", "这说明"
    ]

    AI_PHRASES_EN = [
        "let's dive into", "let's explore", "let's take a look",
        "it's worth noting", "it is worth mentioning", "notably",
        "in conclusion", "to summarize", "to sum up", "all in all",
        "undeniably", "undoubtedly", "without a doubt",
        "in today's world", "in the current landscape",
        "we can conclude", "it can be concluded",
        "it's important to note", "it is essential to"
    ]

    def detect(self, text: str, file_path: str = "") -> DetectionResult:
        import re

        # 检测语言
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        lang = "cn" if cn_chars > en_words else "en"

        phrases = self.AI_PHRASES_CN if lang == "cn" else self.AI_PHRASES_EN

        # 统计AI话术
        ai_phrase_count = 0
        matched_phrases = []
        for phrase in phrases:
            matches = re.findall(phrase, text, re.IGNORECASE)
            if matches:
                ai_phrase_count += len(matches)
                matched_phrases.extend(matches)

        # 计算Burstiness (句子长度变化)
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) > 1:
            lengths = [len(s) for s in sentences]
            avg = sum(lengths) / len(lengths)
            variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
            stddev = variance ** 0.5
            burstiness = stddev / avg if avg > 0 else 0
        else:
            burstiness = 0.5

        # 综合评分
        # AI话术惩罚: 每个话术增加0.05 AI概率
        # Burstiness奖励: 高变化减少AI概率
        base_ai_prob = 0.5
        ai_prob = base_ai_prob + (ai_phrase_count * 0.05) - (burstiness * 0.2)
        ai_prob = max(0.0, min(1.0, ai_prob))

        if ai_prob > self.config.ai_threshold:
            classification = "ai"
        elif ai_prob < self.config.human_threshold:
            classification = "human"
        else:
            classification = "mixed"

        return DetectionResult(
            file_path=file_path,
            service="FallbackDetector (rule-based)",
            timestamp=datetime.now().isoformat(),
            ai_probability=ai_prob,
            human_probability=1 - ai_prob,
            mixed_probability=0.0,
            classification=classification,
            confidence=0.6,  # 规则检测置信度较低
            highlighted_ai_sentences=matched_phrases[:10],
            text_length=len(text),
            language=lang
        )


# ════════════════════════════════════════════════════════════════
# 主检测器
# ════════════════════════════════════════════════════════════════

class DeepAIDetector:
    """深度AI检测器主类"""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig.from_env()
        self._init_detector()

    # 检测器权重 (基于准确性和可靠性)
    DETECTOR_WEIGHTS = {
        "GPTZero": 0.25,
        "Originality.ai": 0.30,  # 最高准确性
        "ZeroGPT": 0.15,
        "Copyleaks": 0.20,
        "CustomLLM": 0.10,
        "FallbackDetector (rule-based)": 0.05
    }

    def _init_detector(self):
        """初始化检测器"""
        self.detectors = {
            "gptzero": GPTZeroDetector,
            "originality": OriginalityDetector,
            "zerogpt": ZeroGPTDetector,
            "copyleaks": CopyleaksDetector,
            "custom_llm": CustomLLMDetector,
            "fallback": FallbackDetector
        }

        detector_class = self.detectors.get(self.config.service, FallbackDetector)
        self.detector = detector_class(self.config)
        self.fallback = FallbackDetector(self.config)

    def detect_file(self, file_path: str) -> DetectionResult:
        """检测单个文件"""
        path = Path(file_path)

        if not path.exists():
            return DetectionResult(
                file_path=file_path,
                service="DeepAIDetector",
                timestamp=datetime.now().isoformat(),
                ai_probability=0.5,
                human_probability=0.5,
                mixed_probability=0.0,
                classification="uncertain",
                confidence=0.0,
                error=f"File not found: {file_path}"
            )

        # 读取文件内容
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return DetectionResult(
                file_path=file_path,
                service="DeepAIDetector",
                timestamp=datetime.now().isoformat(),
                ai_probability=0.5,
                human_probability=0.5,
                mixed_probability=0.0,
                classification="uncertain",
                confidence=0.0,
                error=f"Failed to read file: {e}"
            )

        # 移除Markdown元数据 (frontmatter)
        import re
        content = re.sub(r'^---[\s\S]*?---\n', '', content)

        # 尝试主检测器，失败时使用降级方案
        result = self.detector.detect(content, file_path)

        if result.error and self.config.service != "fallback":
            print(f"⚠️ Primary detector failed: {result.error}")
            print("   Falling back to rule-based detection...")
            result = self.fallback.detect(content, file_path)

        return result

    def detect_directory(self, dir_path: str, pattern: str = "*.md") -> List[DetectionResult]:
        """检测目录下所有文件"""
        path = Path(dir_path)
        results = []

        for file in sorted(path.glob(pattern)):
            print(f"📄 Analyzing: {file.name}...")
            result = self.detect_file(str(file))
            results.append(result)

        return results

    def aggregate_detect(self, file_path: str, services: Optional[List[str]] = None) -> AggregateResult:
        """
        多检测器聚合检测 (v2.0新增)

        Args:
            file_path: 文件路径
            services: 要使用的检测服务列表，默认使用所有已配置的

        Returns:
            AggregateResult: 聚合检测结果
        """
        path = Path(file_path)
        timestamp = datetime.now().isoformat()

        # 读取文件
        if not path.exists():
            return AggregateResult(
                file_path=file_path,
                timestamp=timestamp,
                weighted_ai_probability=0.5,
                weighted_human_probability=0.5,
                consensus_classification="uncertain",
                consensus_confidence=0.0,
                agreement_ratio=0.0,
                recommendation="文件不存在",
                needs_review=True
            )

        try:
            content = path.read_text(encoding="utf-8")
            import re
            content = re.sub(r'^---[\s\S]*?---\n', '', content)
        except Exception as e:
            return AggregateResult(
                file_path=file_path,
                timestamp=timestamp,
                weighted_ai_probability=0.5,
                weighted_human_probability=0.5,
                consensus_classification="uncertain",
                consensus_confidence=0.0,
                agreement_ratio=0.0,
                recommendation=f"文件读取失败: {e}",
                needs_review=True
            )

        # 确定要使用的检测器
        if services is None:
            services = ["gptzero", "zerogpt", "originality", "copyleaks", "fallback"]

        # 运行所有检测器
        results = []
        for service_name in services:
            detector_class = self.detectors.get(service_name)
            if not detector_class:
                continue

            detector = detector_class(self.config)
            print(f"  🔍 Running {service_name}...")
            result = detector.detect(content, file_path)
            results.append(result)

        # 分离成功和失败的结果
        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]

        if not successful:
            return AggregateResult(
                file_path=file_path,
                timestamp=timestamp,
                weighted_ai_probability=0.5,
                weighted_human_probability=0.5,
                consensus_classification="uncertain",
                consensus_confidence=0.0,
                agreement_ratio=0.0,
                individual_results=[r.to_dict() for r in results],
                successful_detectors=0,
                failed_detectors=len(failed),
                recommendation="所有检测器都失败，请检查API配置",
                needs_review=True
            )

        # 计算加权平均
        total_weight = 0.0
        weighted_ai_sum = 0.0
        classifications = []

        for result in successful:
            weight = self.DETECTOR_WEIGHTS.get(result.service, 0.1)
            weighted_ai_sum += result.ai_probability * weight
            total_weight += weight
            classifications.append(result.classification)

        weighted_ai_prob = weighted_ai_sum / total_weight if total_weight > 0 else 0.5
        weighted_human_prob = 1 - weighted_ai_prob

        # 多数投票确定分类
        from collections import Counter
        classification_counts = Counter(classifications)
        consensus_classification = classification_counts.most_common(1)[0][0]
        consensus_count = classification_counts[consensus_classification]

        # 计算一致性
        agreement_ratio = consensus_count / len(successful) if successful else 0.0

        # 计算置信度 (基于一致性和加权概率)
        if consensus_classification == "ai":
            base_confidence = weighted_ai_prob
        elif consensus_classification == "human":
            base_confidence = weighted_human_prob
        else:
            base_confidence = 1 - abs(weighted_ai_prob - 0.5) * 2

        consensus_confidence = base_confidence * agreement_ratio

        # 生成推荐
        if agreement_ratio >= 0.8:
            if consensus_classification == "ai":
                recommendation = "高度一致判定为AI生成，建议进行人性化改写"
            elif consensus_classification == "human":
                recommendation = "高度一致判定为人类写作，可以发布"
            else:
                recommendation = "判定为混合内容，建议审核后发布"
            needs_review = False
        elif agreement_ratio >= 0.6:
            recommendation = "检测器意见有分歧，建议人工审核"
            needs_review = True
        else:
            recommendation = "检测器严重分歧，强烈建议人工审核"
            needs_review = True

        return AggregateResult(
            file_path=file_path,
            timestamp=timestamp,
            weighted_ai_probability=weighted_ai_prob,
            weighted_human_probability=weighted_human_prob,
            consensus_classification=consensus_classification,
            consensus_confidence=consensus_confidence,
            agreement_ratio=agreement_ratio,
            individual_results=[r.to_dict() for r in results],
            successful_detectors=len(successful),
            failed_detectors=len(failed),
            recommendation=recommendation,
            needs_review=needs_review
        )


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def print_result(result: DetectionResult, json_output: bool = False):
    """打印检测结果"""
    if json_output:
        print(result.to_json())
        return

    # 颜色定义
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

    print(f"\n{BLUE}╔════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BLUE}║           深度学习AI检测报告                               ║{NC}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{NC}")
    print(f"\n{CYAN}文件: {result.file_path}{NC}")
    print(f"{CYAN}服务: {result.service}{NC}")
    print(f"{CYAN}时间: {result.timestamp}{NC}")

    print(f"\n{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{YELLOW}检测结果{NC}")
    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")

    # 分类结果
    if result.classification == "ai":
        color = RED
        icon = "❌"
        label = "AI生成"
    elif result.classification == "human":
        color = GREEN
        icon = "✅"
        label = "人类写作"
    else:
        color = YELLOW
        icon = "⚠️"
        label = "混合/不确定"

    print(f"\n   分类结果: {color}{icon} {label}{NC}")
    print(f"   置信度:   {result.confidence:.1%}")
    print(f"\n   AI概率:   {result.ai_probability:.1%}")
    print(f"   人类概率: {result.human_probability:.1%}")

    if result.highlighted_ai_sentences:
        print(f"\n   {RED}检测到的AI指标:{NC}")
        for indicator in result.highlighted_ai_sentences[:5]:
            print(f"   - {indicator}")

    if result.error:
        print(f"\n   {RED}错误: {result.error}{NC}")

    print()


def print_aggregate_result(result: AggregateResult, json_output: bool = False):
    """打印聚合检测结果 (v2.0新增)"""
    if json_output:
        print(result.to_json())
        return

    # 颜色定义
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

    print(f"\n{BLUE}╔════════════════════════════════════════════════════════════╗{NC}")
    print(f"{BLUE}║           多检测器聚合分析报告 (v2.0)                       ║{NC}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{NC}")
    print(f"\n{CYAN}文件: {result.file_path}{NC}")
    print(f"{CYAN}时间: {result.timestamp}{NC}")

    print(f"\n{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{YELLOW}聚合结果{NC}")
    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")

    # 分类结果
    if result.consensus_classification == "ai":
        color = RED
        icon = "❌"
        label = "AI生成"
    elif result.consensus_classification == "human":
        color = GREEN
        icon = "✅"
        label = "人类写作"
    else:
        color = YELLOW
        icon = "⚠️"
        label = "混合/不确定"

    print(f"\n   共识分类: {color}{icon} {label}{NC}")
    print(f"   共识置信度: {result.consensus_confidence:.1%}")
    print(f"   检测器一致性: {result.agreement_ratio:.1%}")
    print(f"\n   加权AI概率: {result.weighted_ai_probability:.1%}")
    print(f"   加权人类概率: {result.weighted_human_probability:.1%}")

    print(f"\n   成功检测器: {result.successful_detectors}")
    print(f"   失败检测器: {result.failed_detectors}")

    # 各检测器详情
    if result.individual_results:
        print(f"\n{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
        print(f"{YELLOW}各检测器结果{NC}")
        print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")

        for r in result.individual_results:
            service = r.get("service", "Unknown")
            ai_prob = r.get("ai_probability", 0.5)
            classification = r.get("classification", "unknown")
            error = r.get("error")

            if error:
                print(f"   {RED}✗ {service}: 失败 - {error[:50]}{NC}")
            else:
                if classification == "ai":
                    c = RED
                elif classification == "human":
                    c = GREEN
                else:
                    c = YELLOW
                print(f"   {c}• {service}: {classification} ({ai_prob:.1%} AI){NC}")

    # 推荐
    print(f"\n{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{YELLOW}建议{NC}")
    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"\n   {result.recommendation}")
    if result.needs_review:
        print(f"   {RED}⚠️ 需要人工审核{NC}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="深度学习AI检测工具 v2.0 - 支持多检测器聚合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单检测器模式
  python deep_ai_detector.py --file article.md
  python deep_ai_detector.py --file article.md --service gptzero

  # 多检测器聚合模式 (v2.0新增)
  python deep_ai_detector.py --file article.md --aggregate
  python deep_ai_detector.py --file article.md --aggregate --json

  # 批量检测
  python deep_ai_detector.py --dir 2026_01/ --pattern "*.md"

环境变量配置:
  GPTZERO_API_KEY        GPTZero API密钥
  ZEROGPT_API_KEY        ZeroGPT API密钥 (v2.0新增)
  ORIGINALITY_API_KEY    Originality.ai API密钥
  COPYLEAKS_API_KEY      Copyleaks API密钥 (v2.0新增)
  COPYLEAKS_EMAIL        Copyleaks 邮箱 (v2.0新增)
  AI_DETECTOR_SERVICE    默认检测服务 (gptzero/originality/zerogpt/copyleaks/custom_llm/fallback)
  AI_DETECTOR_THRESHOLD  AI判定阈值 (默认0.7)
"""
    )

    parser.add_argument("--file", "-f", help="要检测的文件路径")
    parser.add_argument("--dir", "-d", help="要检测的目录路径")
    parser.add_argument("--pattern", "-p", default="*.md", help="文件匹配模式 (默认: *.md)")
    parser.add_argument("--json", "-j", action="store_true", help="输出JSON格式")
    parser.add_argument("--aggregate", "-a", action="store_true",
                       help="使用多检测器聚合模式 (v2.0新增)")
    parser.add_argument("--service", "-s",
                       choices=["gptzero", "originality", "zerogpt", "copyleaks", "custom_llm", "fallback"],
                       help="指定检测服务")

    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.print_help()
        sys.exit(1)

    # 初始化检测器
    config = DetectorConfig.from_env()
    if args.service:
        config.service = args.service

    detector = DeepAIDetector(config)

    if args.file:
        if args.aggregate:
            # 多检测器聚合模式
            print(f"🔄 使用多检测器聚合模式分析: {args.file}")
            result = detector.aggregate_detect(args.file)
            print_aggregate_result(result, args.json)
        else:
            # 单检测器模式
            result = detector.detect_file(args.file)
            print_result(result, args.json)

    elif args.dir:
        results = detector.detect_directory(args.dir, args.pattern)

        if args.json:
            print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        else:
            for result in results:
                print_result(result, False)

            # 汇总
            ai_count = sum(1 for r in results if r.classification == "ai")
            human_count = sum(1 for r in results if r.classification == "human")
            mixed_count = sum(1 for r in results if r.classification in ("mixed", "uncertain"))

            print(f"\n{'='*60}")
            print(f"汇总: AI={ai_count}, 人类={human_count}, 混合/不确定={mixed_count}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
