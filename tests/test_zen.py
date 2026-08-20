import json
from unittest.mock import MagicMock, patch

import pytest

from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.llm.router import ModelRouter
from kaggle_agent.llm.zen_client import ZenClient, ZenError
from kaggle_agent.paths import repo_root


def test_router_models_from_settings():
    s = load_settings(repo_root())
    c = load_competition("rsna_knee", repo_root())
    r = ModelRouter.build(s, c)
    assert r.model("plan")
    assert r.model("vision")


def test_from_env_none_without_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert ZenClient.from_env() is None


def test_deepseek_provider_uses_flash_only(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-zen")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-x")
    from kaggle_agent.config import load_settings
    from kaggle_agent.llm.fallback import FallbackClient, build_llm_client
    from kaggle_agent.paths import repo_root

    s = load_settings(repo_root())
    assert s.zen_model("plan") == "deepseek-v4-flash"
    assert s.zen_model("code") == "deepseek-v4-flash"
    assert "pro" not in s.zen_model("distill")
    client = build_llm_client(s)
    assert isinstance(client, ZenClient)
    assert client.base_url == "https://api.deepseek.com"
    assert not isinstance(client, FallbackClient)


def test_zen_free_rotates_after_deepseek_limit():
    from kaggle_agent.llm.fallback import FallbackClient, ProviderSpec

    seen: list[str] = []

    def chat_a(model, messages, **k):  # noqa: ANN001
        seen.append(model)
        raise ZenError("Zen HTTP 429: rate limit")

    def chat_b(model, messages, **k):  # noqa: ANN001
        seen.append(model)
        return "ok-lightning"

    a = ZenClient(api_key="k", base_url="https://opencode.ai/zen/v1")
    b = ZenClient(api_key="k", base_url="https://opencode.ai/zen/v1")
    a.chat = chat_a  # type: ignore[method-assign]
    b.chat = chat_b  # type: ignore[method-assign]
    client = FallbackClient(
        [
            ProviderSpec("zen:deepseek-v4-flash-free", a, {"*": "deepseek-v4-flash-free"}),
            ProviderSpec(
                "zen:nemotron-3.5-lightning-free",
                b,
                {"*": "nemotron-3.5-lightning-free"},
            ),
        ]
    )
    assert client.chat_text("deepseek-v4-flash-free", "s", "u") == "ok-lightning"
    assert seen == ["deepseek-v4-flash-free", "nemotron-3.5-lightning-free"]


def test_from_env_uses_deepseek_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    client = ZenClient.from_env()
    assert client is not None
    assert client.api_key == "sk-ds"
    assert client.base_url == "https://api.deepseek.com"


def test_chat_parses_response(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    client = ZenClient(api_key="test-key", base_url="https://example.test/v1")

    payload = {"choices": [{"message": {"content": "hypothesis: try 2d cnn\napproach: baseline\nsteps: a;b"}}]}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    with patch("urllib.request.urlopen", return_value=Resp()):
        out = client.chat_text("gpt-5.5", "sys", "user")
    assert "hypothesis" in out


def test_chat_reads_openai_tool_calls(monkeypatch):
    client = ZenClient(api_key="test-key", base_url="https://api.deepseek.com")
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "harvest_cards",
                                "arguments": '{"reset": true}',
                            }
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    with patch("urllib.request.urlopen", return_value=Resp()):
        out = client.chat(
            "deepseek-v4-flash",
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "harvest_cards"}}],
        )
    assert out == ""
    assert client.last_tool_calls == [("harvest_cards", {"reset": True})]
    assert client.last_usage["tokens_out"] == 4


def test_fallback_skips_to_hetznez_on_429(monkeypatch):
    from kaggle_agent.llm.fallback import FallbackClient, ProviderSpec

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    first = ZenClient(api_key="a", base_url="https://primary.test/v1")
    second = ZenClient(api_key="b", base_url="https://hetznez.test/v1")

    def boom(*a, **k):
        raise ZenError("Zen HTTP 429: rate limit")

    def ok(*a, **k):
        return "from-hetznez"

    first.chat = boom  # type: ignore[method-assign]
    second.chat = ok  # type: ignore[method-assign]
    client = FallbackClient(
        [
            ProviderSpec("nvidia", first, {"*": "m1"}),
            ProviderSpec("hetznez", second, {"*": "m2"}),
        ]
    )
    assert client.chat_text("m1", "s", "u") == "from-hetznez"


def test_chat_http_error(monkeypatch):
    import urllib.error

    client = ZenClient(api_key="k", base_url="https://example.test/v1")

    def boom(*a, **k):
        raise urllib.error.HTTPError("https://x", 401, "no", hdrs=None, fp=MagicMock(read=lambda: b"denied"))

    with patch("urllib.request.urlopen", side_effect=boom):
        with pytest.raises(ZenError):
            client.chat_text("m", "s", "u")
