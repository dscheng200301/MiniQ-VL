"""
统一的 LLM API 调用接口
支持 OpenAI 格式 API，可配置 base_url 使用代理或其他 API 服务
"""
import os
import json
import time
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache


class APIClient:
    """统一 API 调用客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-ai/DeepSeek-V4-Pro",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key: API 密钥，默认从环境变量 OPENAI_API_KEY 读取
            base_url: API base URL，默认从环境变量 OPENAI_BASE_URL 读取
            model: 默认模型名称（默认使用 DeepSeek-V4-Pro）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-f9a2e5f5f04e49689fbc3c036f3f61a0")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        self.default_model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            print("Warning: No API key provided, set OPENAI_API_KEY environment variable")
        
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
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
        **kwargs
    ) -> str:
        """
        发送对话请求

        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称，默认使用 self.default_model
            temperature: 采样温度
            max_tokens: 最大 token 数
            top_p: nucleus sampling
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            stop: 停止词列表

        Returns:
            生成的文本内容
        """
        model = model or self.default_model
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API call failed after {self.max_retries} retries: {e}")

    def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        发送对话请求并解析 JSON 响应

        Args:
            messages: 对话消息列表
            model: 模型名称

        Returns:
            解析后的 JSON 对象
        """
        text = self.chat(messages, model=model, **kwargs)
        try:
            # 尝试提取 JSON（处理可能的前缀/后缀）
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {text}")

    def batch_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: Optional[str] = None,
        delay: float = 0.5,
        **kwargs
    ) -> List[str]:
        """
        批量发送对话请求

        Args:
            messages_list: 多个对话消息列表
            model: 模型名称
            delay: 请求间隔（秒），避免限流

        Returns:
            生成的文本列表
        """
        results = []
        for i, messages in enumerate(messages_list):
            try:
                result = self.chat(messages, model=model, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Batch request {i} failed: {e}")
                results.append("")
            
            if i < len(messages_list) - 1 and delay > 0:
                time.sleep(delay)
        
        return results


# 全局默认客户端实例
_default_client: Optional[APIClient] = None


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs
) -> APIClient:
    """
    获取全局 API 客户端（单例模式）

    Args:
        api_key: API 密钥
        base_url: API base URL
        model: 默认模型
        **kwargs: 其他参数

    Returns:
        APIClient 实例
    """
    global _default_client
    
    if _default_client is None:
        _default_client = APIClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
    return _default_client


def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    **kwargs
) -> str:
    """
    快捷函数：使用全局客户端发送对话请求
    """
    return get_client(model=model or "gpt-4o").chat(messages, **kwargs)


def chat_with_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    **kwargs
) -> Any:
    """
    快捷函数：使用全局客户端发送对话请求并解析 JSON
    """
    return get_client(model=model or "gpt-4o").chat_with_json(messages, **kwargs)


# 示例用法
if __name__ == "__main__":
    # 使用 DeepSeek-V4-Pro（默认配置）
    # 设置环境变量：
    #   export OPENAI_API_KEY=your_api_key
    # 
    # 创建客户端并调用：
    # client = APIClient()  # 默认使用 DeepSeek-V4-Pro
    # response = client.chat([{"role": "user", "content": "Hello!"}])
    
    # 自定义配置：
    # client = APIClient(
    #     model="deepseek-ai/DeepSeek-V4-Pro",
    #     base_url="https://api.deepseek.com/v1"
    # )
    # response = client.chat([{"role": "user", "content": "请描述这张图片"}])
    
    print("APIClient API wrapper ready")
    print("Usage:")
    print("  from utils.api_client import chat, get_client, APIClient")
