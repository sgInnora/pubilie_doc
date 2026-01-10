#!/usr/bin/env python3
"""
API限流与重试机制 (E12)
========================

基于Ultrathink协议v2.7的API弹性化工具，提供：
- 指数退避重试 (Exponential Backoff with Jitter)
- 熔断器模式 (Circuit Breaker Pattern)
- 速率限制 (Rate Limiting with Token Bucket)
- Redis可选缓存支持

证据来源:
- OpenAI Cookbook: https://cookbook.openai.com/examples/how_to_handle_rate_limits
- Tenacity官方文档: https://tenacity.readthedocs.io/
- circuitbreaker PyPI: https://pypi.org/project/circuitbreaker/
- Medium最佳实践: https://medium.com/neural-engineer/implementing-effective-api-rate-limiting-in-python

作者: Claude Opus 4.5 (Ultrathink Protocol v2.7)
创建时间: 2026-01-10
"""

import time
import random
import logging
import functools
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, TypeVar, Dict, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import json
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 5
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 60.0  # 最大延迟（秒）
    exponential_base: float = 2.0  # 指数基数
    jitter: bool = True  # 是否添加抖动
    retry_on_exceptions: Tuple[type, ...] = (Exception,)
    retry_on_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5  # 失败阈值
    recovery_timeout: float = 30.0  # 恢复超时（秒）
    half_open_max_calls: int = 3  # 半开状态最大调用数
    success_threshold: int = 2  # 半开状态成功阈值


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    requests_per_minute: int = 60
    requests_per_second: Optional[int] = None
    burst_size: int = 10  # 突发容量


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"  # 正常状态
    OPEN = "open"  # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态（试探性恢复）


# ============================================================================
# 指数退避重试器
# ============================================================================

class ExponentialBackoffRetry:
    """
    指数退避重试器

    实现了带抖动的指数退避算法，用于处理API速率限制和临时故障。

    算法: delay = min(max_delay, base_delay * (exponential_base ** attempt))
    抖动: delay = delay * random.uniform(0.5, 1.5)

    使用示例:
    ```python
    retry = ExponentialBackoffRetry(max_attempts=5, base_delay=1.0)

    @retry
    def call_api():
        response = requests.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()
    ```
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "total_delay": 0.0
        }

    def calculate_delay(self, attempt: int) -> float:
        """计算第N次重试的延迟时间"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # 添加±50%的随机抖动，防止雪崩
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        # 检查异常类型
        if isinstance(exception, self.config.retry_on_exceptions):
            return True

        # 检查HTTP状态码（如果是requests异常）
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            if status_code in self.config.retry_on_status_codes:
                return True

        return False

    def get_retry_after(self, exception: Exception) -> Optional[float]:
        """从Retry-After头获取等待时间"""
        if hasattr(exception, 'response') and exception.response is not None:
            retry_after = exception.response.headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    # 可能是日期格式，解析它
                    pass
        return None

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """装饰器模式"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(self.config.max_attempts):
                self._stats["total_attempts"] += 1

                try:
                    result = func(*args, **kwargs)
                    self._stats["successful_attempts"] += 1
                    return result

                except Exception as e:
                    last_exception = e
                    self._stats["failed_attempts"] += 1

                    if not self.should_retry(e):
                        logger.warning(f"不可重试的异常: {type(e).__name__}: {e}")
                        raise

                    if attempt < self.config.max_attempts - 1:
                        # 优先使用Retry-After头
                        delay = self.get_retry_after(e) or self.calculate_delay(attempt)
                        self._stats["total_delay"] += delay

                        logger.warning(
                            f"重试 {attempt + 1}/{self.config.max_attempts}, "
                            f"延迟 {delay:.2f}s, 错误: {type(e).__name__}: {e}"
                        )
                        time.sleep(delay)

            logger.error(f"达到最大重试次数 ({self.config.max_attempts})")
            raise last_exception

        return wrapper

    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return self._stats.copy()


# ============================================================================
# 熔断器
# ============================================================================

class CircuitBreaker:
    """
    熔断器模式实现

    防止对不可用服务的持续调用，允许系统快速失败并自动恢复。

    状态转换:
    - CLOSED → OPEN: 连续失败次数达到阈值
    - OPEN → HALF_OPEN: 恢复超时后
    - HALF_OPEN → CLOSED: 成功调用达到阈值
    - HALF_OPEN → OPEN: 调用失败

    使用示例:
    ```python
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

    @breaker
    def call_external_service():
        # 可能失败的外部调用
        pass
    ```
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None, name: str = "default"):
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._state_change_callbacks: List[Callable] = []

    @property
    def state(self) -> CircuitState:
        """获取当前状态，自动检查是否应该转换到HALF_OPEN"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置（从OPEN转到HALF_OPEN）"""
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState):
        """状态转换"""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        logger.info(f"熔断器 [{self.name}]: {old_state.value} → {new_state.value}")

        # 触发回调
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"状态变更回调失败: {e}")

    def _record_success(self):
        """记录成功调用"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # 重置失败计数
                self._failure_count = 0

    def _record_failure(self):
        """记录失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def on_state_change(self, callback: Callable[[CircuitState, CircuitState], None]):
        """注册状态变更回调"""
        self._state_change_callbacks.append(callback)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """装饰器模式"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"熔断器 [{self.name}] 已打开，拒绝调用"
                )

            if current_state == CircuitState.HALF_OPEN:
                with self._lock:
                    if self._half_open_calls >= self.config.half_open_max_calls:
                        raise CircuitBreakerOpenError(
                            f"熔断器 [{self.name}] 半开状态调用数达到上限"
                        )
                    self._half_open_calls += 1

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise

        return wrapper

    def get_status(self) -> dict:
        """获取熔断器状态"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "half_open_max_calls": self.config.half_open_max_calls,
                    "success_threshold": self.config.success_threshold
                }
            }


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass


# ============================================================================
# 令牌桶速率限制器
# ============================================================================

class TokenBucketRateLimiter:
    """
    令牌桶速率限制器

    使用令牌桶算法实现平滑的速率限制，支持突发流量。

    算法原理:
    - 桶以固定速率填充令牌
    - 每次请求消耗一个令牌
    - 桶满时停止填充
    - 桶空时阻塞请求

    使用示例:
    ```python
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst_size=10)

    @limiter
    def call_api():
        # API调用
        pass

    # 或者手动调用
    limiter.acquire()  # 阻塞直到获得令牌
    call_api()
    ```
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()

        # 计算每秒令牌生成速率
        if self.config.requests_per_second:
            self._tokens_per_second = self.config.requests_per_second
        else:
            self._tokens_per_second = self.config.requests_per_minute / 60.0

        self._max_tokens = self.config.burst_size
        self._tokens = float(self._max_tokens)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

        # 统计
        self._stats = {
            "requests_total": 0,
            "requests_throttled": 0,
            "total_wait_time": 0.0
        }

    def _refill(self):
        """填充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._tokens_per_second
        )
        self._last_update = now

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取一个令牌

        Args:
            timeout: 最大等待时间（秒），None表示无限等待

        Returns:
            是否成功获取令牌
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= 1:
                    self._tokens -= 1
                    self._stats["requests_total"] += 1
                    return True

                # 计算需要等待的时间
                tokens_needed = 1 - self._tokens
                wait_time = tokens_needed / self._tokens_per_second

            # 检查超时
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    self._stats["requests_throttled"] += 1
                    return False

            self._stats["total_wait_time"] += wait_time
            time.sleep(wait_time)

    def try_acquire(self) -> bool:
        """尝试获取令牌（非阻塞）"""
        with self._lock:
            self._refill()

            if self._tokens >= 1:
                self._tokens -= 1
                self._stats["requests_total"] += 1
                return True

            self._stats["requests_throttled"] += 1
            return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """装饰器模式"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            self.acquire()
            return func(*args, **kwargs)

        return wrapper

    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return self._stats.copy()

    @property
    def available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens


# ============================================================================
# 综合API客户端包装器
# ============================================================================

class ResilientAPIClient:
    """
    弹性API客户端

    组合使用速率限制、重试和熔断器，提供完整的API调用保护。

    使用示例:
    ```python
    client = ResilientAPIClient(
        rate_limit=RateLimitConfig(requests_per_minute=60),
        retry=RetryConfig(max_attempts=3),
        circuit_breaker=CircuitBreakerConfig(failure_threshold=5)
    )

    @client.protect
    def call_api():
        response = requests.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()

    # 或者使用上下文管理器
    with client:
        result = call_api()
    ```
    """

    def __init__(
        self,
        rate_limit: Optional[RateLimitConfig] = None,
        retry: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
        name: str = "default"
    ):
        self.name = name
        self._rate_limiter = TokenBucketRateLimiter(rate_limit) if rate_limit else None
        self._retry = ExponentialBackoffRetry(retry) if retry else None
        self._circuit_breaker = CircuitBreaker(circuit_breaker, name) if circuit_breaker else None

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        保护函数装饰器

        应用顺序: 速率限制 → 熔断器 → 重试
        """
        wrapped = func

        # 重试（最内层）
        if self._retry:
            wrapped = self._retry(wrapped)

        # 熔断器
        if self._circuit_breaker:
            wrapped = self._circuit_breaker(wrapped)

        # 速率限制（最外层）
        if self._rate_limiter:
            wrapped = self._rate_limiter(wrapped)

        return wrapped

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_status(self) -> dict:
        """获取客户端状态"""
        status = {"name": self.name}

        if self._rate_limiter:
            status["rate_limiter"] = {
                "available_tokens": self._rate_limiter.available_tokens,
                "stats": self._rate_limiter.stats
            }

        if self._retry:
            status["retry"] = {
                "stats": self._retry.stats
            }

        if self._circuit_breaker:
            status["circuit_breaker"] = self._circuit_breaker.get_status()

        return status


# ============================================================================
# 预配置客户端
# ============================================================================

# OpenAI API客户端配置
OPENAI_CONFIG = ResilientAPIClient(
    rate_limit=RateLimitConfig(requests_per_minute=60, burst_size=20),
    retry=RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=60.0,
        retry_on_status_codes=(429, 500, 502, 503, 504)
    ),
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=60.0
    ),
    name="openai"
)

# 通用Web API客户端配置
WEB_API_CONFIG = ResilientAPIClient(
    rate_limit=RateLimitConfig(requests_per_minute=120, burst_size=30),
    retry=RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=30.0
    ),
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30.0
    ),
    name="web_api"
)

# 威胁情报API客户端配置
THREAT_INTEL_CONFIG = ResilientAPIClient(
    rate_limit=RateLimitConfig(requests_per_minute=30, burst_size=5),
    retry=RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=120.0
    ),
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=120.0
    ),
    name="threat_intel"
)


# ============================================================================
# 便捷装饰器
# ============================================================================

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
):
    """
    便捷重试装饰器

    使用示例:
    ```python
    @with_retry(max_attempts=5, base_delay=2.0)
    def call_api():
        # API调用
        pass
    ```
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter
    )
    return ExponentialBackoffRetry(config)


def with_rate_limit(
    requests_per_minute: int = 60,
    burst_size: int = 10
):
    """
    便捷速率限制装饰器

    使用示例:
    ```python
    @with_rate_limit(requests_per_minute=30)
    def call_api():
        # API调用
        pass
    ```
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        burst_size=burst_size
    )
    return TokenBucketRateLimiter(config)


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    name: str = "default"
):
    """
    便捷熔断器装饰器

    使用示例:
    ```python
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    def call_external_service():
        # 外部服务调用
        pass
    ```
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    return CircuitBreaker(config, name)


# ============================================================================
# 主函数（演示）
# ============================================================================

def main():
    """演示API限流与重试机制"""
    print("=" * 60)
    print("API限流与重试机制 (E12) - 演示")
    print("=" * 60)

    # 1. 测试指数退避重试
    print("\n1. 指数退避重试测试")
    print("-" * 40)

    retry = ExponentialBackoffRetry(RetryConfig(max_attempts=3, base_delay=0.1))

    call_count = 0

    @retry
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception(f"模拟失败 {call_count}/3")
        return "成功!"

    try:
        result = flaky_function()
        print(f"结果: {result}")
        print(f"统计: {retry.stats}")
    except Exception as e:
        print(f"最终失败: {e}")

    # 2. 测试熔断器
    print("\n2. 熔断器测试")
    print("-" * 40)

    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0),
        name="test_breaker"
    )

    @breaker
    def unstable_service():
        raise Exception("服务不可用")

    for i in range(5):
        try:
            unstable_service()
        except CircuitBreakerOpenError as e:
            print(f"调用 {i+1}: 熔断器已打开")
        except Exception as e:
            print(f"调用 {i+1}: 失败 - {e}")

    print(f"熔断器状态: {breaker.get_status()}")

    # 3. 测试速率限制
    print("\n3. 速率限制测试")
    print("-" * 40)

    limiter = TokenBucketRateLimiter(
        RateLimitConfig(requests_per_minute=600, burst_size=5)  # 10 req/sec
    )

    start = time.time()
    for i in range(10):
        limiter.acquire()
        print(f"请求 {i+1} @ {time.time() - start:.2f}s")

    print(f"速率限制器统计: {limiter.stats}")

    # 4. 综合客户端
    print("\n4. 综合API客户端测试")
    print("-" * 40)

    client = ResilientAPIClient(
        rate_limit=RateLimitConfig(requests_per_minute=120),
        retry=RetryConfig(max_attempts=2, base_delay=0.1),
        circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
        name="demo_client"
    )

    @client.protect
    def demo_api_call():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    for i in range(3):
        result = demo_api_call()
        print(f"调用 {i+1}: {result}")

    print(f"客户端状态: {json.dumps(client.get_status(), indent=2, default=str)}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
