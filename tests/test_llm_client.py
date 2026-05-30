"""
Tests for src/utils/llm_client.py — LLMClient and create_llm_client.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable (same pattern as conftest.py)
# ---------------------------------------------------------------------------
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Helper: clean all LLM-related env vars before each test
_LLM_ENV_VARS = [
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
]


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    """Remove all LLM env vars before every test to prevent leakage."""
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


# ===========================================================================
# Test 1: Initialisation with no API keys
# ===========================================================================
class TestLLMClientInit:
    """LLMClient should degrade gracefully when no credentials are given."""

    def test_defaults_with_empty_config(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({})
        assert client.provider == "anthropic"
        assert client.api_key == ""
        assert client.base_url == ""
        assert client.model == "glm-5.1"
        assert client.max_tokens == 4096
        assert client.temperature == 0.1

    def test_defaults_with_none_config(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient(None)
        assert client.api_key == ""

    def test_is_available_false_without_key(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({"api_key": "", "base_url": ""})
        # No API key → is_available must be False
        assert client.is_available is False

    def test_init_reads_env_vars_as_fallback(self, monkeypatch):
        """When config has no key, constructor should fall back to env vars."""
        from src.utils.llm_client import LLMClient

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-from-env")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

        client = LLMClient({})  # no key in config
        assert client.api_key == "test-key-from-env"
        assert client.base_url == "https://api.anthropic.com"


# ===========================================================================
# Test 2: Provider auto-detection from base_url
# ===========================================================================
class TestProviderDetection:
    """Provider type should be inferred from base_url patterns."""

    @pytest.mark.parametrize(
        "base_url, expected_provider",
        [
            ("https://api.anthropic.com", "anthropic"),
            ("https://open.bigmodel.cn/api/anthropic", "anthropic"),
            ("https://open.bigmodel.cn/api/anthropic/v1", "anthropic"),
        ],
    )
    def test_anthropic_urls_detected(self, base_url, expected_provider):
        """URLs containing 'anthropic' or 'bigmodel' → anthropic provider."""
        from src.utils.llm_client import LLMClient

        client = LLMClient({"base_url": base_url, "api_key": "k"})
        assert client.provider == expected_provider

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://openrouter.ai/api/v1",
            "https://api.deepseek.com/v1",
            "https://custom.llm.host/v1/chat/completions",
        ],
    )
    def test_openai_urls_detected_with_explicit_provider(self, base_url):
        """OpenAI-compatible URLs need explicit provider='openai' since default is 'anthropic'."""
        from src.utils.llm_client import LLMClient

        client = LLMClient({"provider": "openai", "base_url": base_url, "api_key": "k"})
        assert client.provider == "openai"

    def test_explicit_openai_provider(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "openai",
            "base_url": "https://example.com/v1",
            "api_key": "k",
        })
        assert client.provider == "openai"

    def test_explicit_openai_compatible_provider(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "openai_compatible",
            "api_key": "k",
        })
        assert client.provider == "openai"


# ===========================================================================
# Test 3: create_llm_client factory with environment variables
# ===========================================================================
class TestCreateLLMClient:
    """create_llm_client should resolve credentials from env / config."""

    def test_anthropic_env_vars_take_priority(self, monkeypatch):
        from src.utils.llm_client import create_llm_client

        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

        with patch("src.utils.config.load_config", return_value={}):
            client = create_llm_client()
        assert client is not None
        assert client.provider == "anthropic"
        assert client.api_key == "sk-ant-test"

    def test_openai_env_vars_used_as_second_priority(self, monkeypatch):
        from src.utils.llm_client import create_llm_client

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        with patch("src.utils.config.load_config", return_value={}):
            client = create_llm_client()
        assert client is not None
        assert client.provider == "openai"
        assert client.api_key == "sk-openai-test"

    def test_returns_none_when_no_credentials(self, monkeypatch):
        from src.utils.llm_client import create_llm_client

        with patch("src.utils.config.load_config", return_value={}):
            result = create_llm_client()
        assert result is None

    def test_config_file_planner_section_used(self, monkeypatch):
        from src.utils.llm_client import create_llm_client

        mock_config = {
            "llm": {
                "planner": {
                    "provider": "openai",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "cfg-key-deepseek",
                    "model": "deepseek-chat",
                }
            }
        }

        with patch("src.utils.config.load_config", return_value=mock_config):
            client = create_llm_client()
        assert client is not None
        assert client.provider == "openai"
        assert client.model == "deepseek-chat"


# ===========================================================================
# Test 4: generate method with mocked client
# ===========================================================================
class TestGenerate:
    """generate() should route to the correct SDK and return text."""

    @pytest.mark.asyncio
    async def test_anthropic_generate_returns_text(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "anthropic",
            "api_key": "fake",
            "base_url": "https://api.anthropic.com",
            "model": "test-model",
        })

        # Build a fake response with a text block
        fake_block = MagicMock()
        fake_block.text = "Hello from Anthropic!"
        fake_response = MagicMock()
        fake_response.content = [fake_block]

        mock_sdk_client = MagicMock()
        mock_sdk_client.messages.create.return_value = fake_response

        # Inject mock before calling generate
        client._client = mock_sdk_client

        result = await client.generate("Hi there")
        assert result == "Hello from Anthropic!"
        mock_sdk_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_generate_returns_text(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "openai",
            "api_key": "fake",
            "base_url": "https://api.deepseek.com/v1",
            "model": "test-model",
        })

        # Build a fake OpenAI response
        fake_message = MagicMock()
        fake_message.content = "Hello from OpenAI!"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]

        mock_sdk_client = MagicMock()
        mock_sdk_client.chat.completions.create.return_value = fake_response

        client._client = mock_sdk_client

        result = await client.generate("Hi there", system="You are helpful")
        assert result == "Hello from OpenAI!"
        mock_sdk_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_returns_empty_when_no_client(self):
        from src.utils.llm_client import LLMClient

        # No API key and no env vars (autouse fixture cleans them)
        client = LLMClient({"api_key": "", "base_url": ""})
        result = await client.generate("anything")
        assert result == ""

    @pytest.mark.asyncio
    async def test_anthropic_system_prompt_forwarded(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "anthropic",
            "api_key": "fake",
            "base_url": "https://api.anthropic.com",
        })

        fake_block = MagicMock()
        fake_block.text = "ok"
        fake_response = MagicMock()
        fake_response.content = [fake_block]

        mock_sdk_client = MagicMock()
        mock_sdk_client.messages.create.return_value = fake_response
        client._client = mock_sdk_client

        await client.generate("prompt", system="sys")
        call_kwargs = mock_sdk_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "sys"


# ===========================================================================
# Test 5: Error handling when API call fails
# ===========================================================================
class TestErrorHandling:
    """generate() must catch exceptions and return empty string."""

    @pytest.mark.asyncio
    async def test_anthropic_api_exception_returns_empty(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "anthropic",
            "api_key": "fake",
            "base_url": "https://api.anthropic.com",
        })

        mock_sdk_client = MagicMock()
        mock_sdk_client.messages.create.side_effect = RuntimeError("API timeout")
        client._client = mock_sdk_client

        result = await client.generate("Hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_openai_api_exception_returns_empty(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "openai",
            "api_key": "fake",
            "base_url": "https://api.deepseek.com/v1",
        })

        mock_sdk_client = MagicMock()
        mock_sdk_client.chat.completions.create.side_effect = ConnectionError("network error")
        client._client = mock_sdk_client

        result = await client.generate("Hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_auth_error_returns_empty(self):
        from src.utils.llm_client import LLMClient

        client = LLMClient({
            "provider": "anthropic",
            "api_key": "bad-key",
            "base_url": "https://api.anthropic.com",
        })

        mock_sdk_client = MagicMock()
        mock_sdk_client.messages.create.side_effect = PermissionError("401 Unauthorized")
        client._client = mock_sdk_client

        result = await client.generate("test")
        assert result == ""
