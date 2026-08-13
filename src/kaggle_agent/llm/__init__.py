"""OpenCode Zen LLM access."""

from kaggle_agent.llm.router import ModelRouter
from kaggle_agent.llm.zen_client import ZenClient, ZenError

__all__ = ["ModelRouter", "ZenClient", "ZenError"]
