"""
统一的 LLM API 调用接口
支持 OpenAI 格式 API，可配置 base_url 使用代理或其他 API 服务
内置滑动窗口速率限制器 (500 RPM / 20M TPM)
"""
import os
import json
import time
import threading
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache
from collections import deque


# ============== 默认配置 ==============
_DEFAULT_API_KEY = os.environ.get("API_KEY", "")
_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
_DEFAULT_MODEL = "MiniMax-M2.7-highspeed"

# API 速率限制
_RPM_LIMIT = 200         # 每分钟最大请求数
_TPM_LIMIT = 10_000_000   # 每分钟最大 token 数


# ============================================================
# 滑动窗口速率限制器
# ============================================================
class RateLimiter:
    """线程安全的滑动窗口速率限制器"""

    def __init__(self, rpm_limit: int = _RPM_LIMIT, tpm_limit: int = _TPM_LIMIT, window_sec: float = 60.0):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_sec = window_sec
        self._lock = threading.Lock()
        self._events: deque = deque()  # (timestamp, estimated_tokens)

    def _trim(self, now: float):
        cutoff = now - self.window_sec
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _estimate_tokens(self, messages: list, response_len: int = 0) -> int:
        chars = sum(len(m.get("content", "")) for m in messages) + response_len
        return max(1, chars // 2)

    def wait_if_needed(self, messages: list, estimated_response_tokens: int = 256):
        """在必要时等待以遵守速率限制。返回 (等待秒数, 预估消耗token数)"""
        now = time.time()
        estimated = self._estimate_tokens(messages, estimated_response_tokens)
        wait = 0.0

        with self._lock:
            self._trim(now)
            req_count = len(self._events)
            tok_count = sum(e[1] for e in self._events)

            if req_count >= self.rpm_limit or (tok_count + estimated) > self.tpm_limit:
                if self._events:
                    oldest_ts = self._events[0][0]
                    wait = oldest_ts + self.window_sec - now

            if wait <= 0:
                self._events.append((now, estimated))

        if wait > 0:
            time.sleep(wait)
            with self._lock:
                now2 = time.time()
                self._trim(now2)
                self._events.append((now2, estimated))
            return wait, estimated

        return 0.0, estimated


# 全局速率限制器
_global_rate_limiter = RateLimiter(rpm_limit=_RPM_LIMIT, tpm_limit=_TPM_LIMIT)


# ============================================================
# API 客户端
# ============================================================
class APIClient:
    """统一 API 调用客户端（内置速率限制 + 自动重试）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        disable_thinking: bool = True,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or _DEFAULT_API_KEY
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL
        self.default_model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.disable_thinking = disable_thinking
        self.rate_limiter = rate_limiter or _global_rate_limiter
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        disable_thinking: Optional[bool] = None,
        **kwargs
    ) -> str:
        model = model or self.default_model

        # deepseek / MiniMax thinking 禁用
        use_disable = self.disable_thinking if disable_thinking is None else disable_thinking
        if use_disable:
            model_lower = (model or "").lower()
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            if "deepseek" in model_lower:
                extra_body.setdefault("thinking", {"type": "disabled"})
            elif "minimax" in model_lower:
                # MiniMax 通过 enable_thinking=false 禁用
                extra_body.setdefault("enable_thinking", False)
            if extra_body:
                kwargs["extra_body"] = extra_body

        # 速率限制
        waited, _ = self.rate_limiter.wait_if_needed(messages, max_tokens or 256)
        if waited > 0.3:
            print(f"  [RateLimit] waited {waited:.1f}s")

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, top_p=top_p,
                    frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                    stop=stop, **kwargs
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  API failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} retries: {e}")

    def chat_with_json(self, messages, model=None, **kwargs):
        text = self.chat(messages, model=model, **kwargs)
        text = text.strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        start = text.find("{")
        end = text.rfind("}") + 1
        try:
            return json.loads(text[start:end]) if start >= 0 and end > start else json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}\nResponse: {text}")

    def batch_chat(self, messages_list, model=None, delay=0.0, **kwargs):
        results = []
        for i, msgs in enumerate(messages_list):
            try:
                results.append(self.chat(msgs, model=model, **kwargs))
            except Exception as e:
                print(f"  Batch {i} failed: {e}")
                results.append("")
            if i < len(messages_list) - 1 and delay > 0:
                time.sleep(delay)
        return results


# ============================================================
# 全局快捷函数
# ============================================================
_default_client: Optional[APIClient] = None


def get_client(api_key=None, base_url=None, model=_DEFAULT_MODEL, **kwargs):
    global _default_client
    if _default_client is None:
        _default_client = APIClient(api_key=api_key, base_url=base_url, model=model, **kwargs)
    return _default_client


def chat(messages, model=None, **kwargs):
    return get_client(model=model).chat(messages, **kwargs)


def chat_with_json(messages, model=None, **kwargs):
    return get_client(model=model).chat_with_json(messages, **kwargs)


if __name__ == "__main__":
    print("APIClient ready | Rate limits: 500 RPM / 20M TPM")
