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
    assert ZenClient.from_env() is None


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


def test_chat_http_error(monkeypatch):
    import urllib.error

    client = ZenClient(api_key="k", base_url="https://example.test/v1")

    def boom(*a, **k):
        raise urllib.error.HTTPError("https://x", 401, "no", hdrs=None, fp=MagicMock(read=lambda: b"denied"))

    with patch("urllib.request.urlopen", side_effect=boom):
        with pytest.raises(ZenError):
            client.chat_text("m", "s", "u")
