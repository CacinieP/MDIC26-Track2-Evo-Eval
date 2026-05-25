"""
Universal LLM Client - Supports Anthropic-compatible and OpenAI-compatible APIs.

Auto-detects provider type from config:
- Anthropic-compatible: GLM (智谱), StepFun (阶跃星辰), direct Anthropic
- OpenAI-compatible: OpenRouter, DeepSeek, any /v1/chat/completions endpoint
- Fallback: no LLM, heuristic only

Usage:
    client = LLMClient({
        "provider": "anthropic",
        "base_url": "https://open.bigmodel.cn/api/anthropic",
        "api_key": "...",
        "model": "glm-5.1",
    })
    response = await client.generate("Hello")
"""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger


class LLMClient:
    """
    Universal LLM client that supports Anthropic-compatible and OpenAI-compatible APIs.

    Anthropic-compatible providers:
        - GLM (智谱): https://open.bigmodel.cn/api/anthropic
        - StepFun (阶跃星辰): via proxy or direct
        - Anthropic official: https://api.anthropic.com

    OpenAI-compatible providers:
        - OpenRouter: https://openrouter.ai/api/v1
        - DeepSeek: https://api.deepseek.com/v1
        - Any /v1/chat/completions endpoint
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.provider = config.get("provider", "anthropic").lower()
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "glm-5.1")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.1)

        # Auto-detect from environment if not configured
        if not self.api_key:
            self.api_key = (
                os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            )
        if not self.base_url:
            self.base_url = (
                os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or ""
            )

        # Auto-detect provider type from base_url
        if self.provider == "anthropic" or "anthropic" in self.base_url or "bigmodel" in self.base_url:
            self.provider = "anthropic"
        elif self.provider in ("openai", "openai_compatible") or "/v1" in self.base_url or "openrouter" in self.base_url:
            self.provider = "openai"

        self._client = None
        logger.info(f"LLMClient: provider={self.provider}, model={self.model}, base_url={self.base_url[:50]}...")

    def _ensure_client(self):
        """Lazy-init the underlying SDK client."""
        if self._client is not None:
            return True

        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.base_url or None,
                )
                logger.info(f"LLMClient: Anthropic-compatible client initialized ({self.base_url[:40]})")
                return True
            except ImportError:
                logger.warning("anthropic package not installed")
                return False

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
                logger.info(f"LLMClient: OpenAI-compatible client initialized")
                return True
            except ImportError:
                logger.warning("openai package not installed")
                return False

        return False

    async def generate(self, prompt: str, system: str = "") -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User message / prompt text
            system: Optional system prompt

        Returns:
            Generated text string
        """
        if not self._ensure_client():
            logger.warning("LLMClient: no client available, returning empty")
            return ""

        try:
            if self.provider == "anthropic":
                return await self._call_anthropic(prompt, system)
            else:
                return await self._call_openai(prompt, system)
        except Exception as e:
            logger.error(f"LLMClient.generate failed: {e}")
            return ""

    async def _call_anthropic(self, prompt: str, system: str = "") -> str:
        """Call Anthropic-compatible API (GLM, StepFun, etc.)."""
        import asyncio

        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        # Run sync client in executor to avoid blocking
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(**kwargs)
        )

        # Extract text from response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def _call_openai(self, prompt: str, system: str = "") -> str:
        """Call OpenAI-compatible API."""
        import asyncio

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": self.temperature or 0.1,
        }

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(**kwargs)
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or ""
        return ""

    @property
    def is_available(self) -> bool:
        """Check if the client can be used."""
        return bool(self.api_key) and self._ensure_client()


def create_llm_client(config: dict | None = None) -> LLMClient | None:
    """
    Create LLM client from project config or environment variables.

    Priority:
    1. ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL (Claude Code settings: GLM/StepFun)
    2. OPENAI_API_KEY + OPENAI_BASE_URL
    3. config.llm.planner section
    4. Returns None (heuristic fallback)
    """
    from src.utils.config import load_config

    config = config or load_config()

    # --- Priority 1: Anthropic-compatible env vars (GLM, StepFun, etc.) ---
    anthropic_key = (
        os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    if anthropic_key and anthropic_url:
        model = (
            os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
            or "glm-5.1"
        )
        client = LLMClient({
            "provider": "anthropic",
            "base_url": anthropic_url,
            "api_key": anthropic_key,
            "model": model,
        })
        if client.is_available:
            logger.info(f"LLM from ANTHROPIC env: {model} @ {anthropic_url[:50]}")
            return client

    # --- Priority 2: OpenAI-compatible env vars ---
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_url = os.environ.get("OPENAI_BASE_URL", "")
    if openai_key and openai_url:
        client = LLMClient({
            "provider": "openai",
            "base_url": openai_url,
            "api_key": openai_key,
            "model": "gpt-4o",
        })
        if client.is_available:
            logger.info(f"LLM from OPENAI env: gpt-4o @ {openai_url[:50]}")
            return client

    # --- Priority 3: Config file llm.planner section ---
    llm_cfg = config.get("llm", {})
    planner_cfg = llm_cfg.get("planner", {})
    if planner_cfg and planner_cfg.get("provider", "none") != "none":
        api_key = planner_cfg.get("api_key", "")
        # Skip if the key wasn't expanded (still has ${...})
        if api_key and not api_key.startswith("${"):
            client = LLMClient(planner_cfg)
            if client.is_available:
                logger.info(f"LLM from config: {planner_cfg.get('model', '?')}")
                return client

    logger.info("No LLM client available, will use heuristic fallback")
    return None
