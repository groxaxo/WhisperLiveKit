import sys

from whisperlivekit.config import WhisperLiveKitConfig
from whisperlivekit.local_agreement.backends import (
    DEFAULT_OPENAI_API_BASE_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_API_MODEL,
    get_openai_api_defaults,
)
from whisperlivekit.parse_args import parse_args


def test_config_defaults_use_parakeet_backend():
    config = WhisperLiveKitConfig()

    assert config.backend_policy == "localagreement"
    assert config.backend == "openai-api"


def test_parse_args_defaults_use_parakeet_backend(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wlk"])

    config = parse_args()

    assert config.backend_policy == "localagreement"
    assert config.backend == "openai-api"


def test_openai_api_defaults_target_local_parakeet(monkeypatch):
    monkeypatch.delenv("WHISPERLIVEKIT_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("WHISPERLIVEKIT_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("WHISPERLIVEKIT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    defaults = get_openai_api_defaults()

    assert defaults == {
        "base_url": DEFAULT_OPENAI_API_BASE_URL,
        "api_key": DEFAULT_OPENAI_API_KEY,
        "model": DEFAULT_OPENAI_API_MODEL,
    }


def test_openai_api_defaults_fall_back_to_standard_openai_env(monkeypatch):
    monkeypatch.delenv("WHISPERLIVEKIT_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("WHISPERLIVEKIT_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("WHISPERLIVEKIT_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")

    defaults = get_openai_api_defaults()

    assert defaults == {
        "base_url": "http://localhost:9999/v1",
        "api_key": "fallback-key",
        "model": DEFAULT_OPENAI_API_MODEL,
    }


def test_openai_api_defaults_allow_env_overrides(monkeypatch):
    monkeypatch.setenv("WHISPERLIVEKIT_OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("WHISPERLIVEKIT_OPENAI_MODEL", "custom-model")
    monkeypatch.setenv("WHISPERLIVEKIT_OPENAI_API_KEY", "custom-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")

    defaults = get_openai_api_defaults()

    assert defaults == {
        "base_url": "http://localhost:1234/v1",
        "api_key": "custom-key",
        "model": "custom-model",
    }
