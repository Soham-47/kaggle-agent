"""Pick Zen model id by role from competition config + settings."""

from __future__ import annotations

from dataclasses import dataclass

from kaggle_agent.config import CompetitionConfig, Settings
from kaggle_agent.llm.zen_client import ZenClient


@dataclass
class ModelRouter:
    settings: Settings
    competition: CompetitionConfig
    client: ZenClient | None

    @classmethod
    def build(cls, settings: Settings, competition: CompetitionConfig) -> ModelRouter:
        zen = settings.raw.get("zen", {})
        base = str(zen.get("base_url", "https://opencode.ai/zen/v1"))
        return cls(
            settings=settings,
            competition=competition,
            client=ZenClient.from_env(base_url=base),
        )

    def model(self, role: str) -> str:
        """role: plan | code | distill | vision"""
        return self.competition.model_for(role, self.settings)

    def available(self) -> bool:
        return self.client is not None

    def _client(self) -> ZenClient:
        if self.client is None:
            raise RuntimeError("OPENCODE_API_KEY not set")
        return self.client

    def plan(self, system: str, user: str) -> str:
        return self._client().chat_text(self.model("plan"), system, user, max_tokens=800)

    def code(self, system: str, user: str) -> str:
        """Short implement brief from method cards. Deterministic CODE still applies it."""
        return self._client().chat_text(self.model("code"), system, user, max_tokens=700)

    def distill(self, system: str, user: str) -> str:
        return self._client().chat_text(self.model("distill"), system, user, max_tokens=1200)

    def vision(self, system: str, user: str, image_urls: list[str]) -> str:
        return self._client().chat_vision(
            self.model("vision"), system, user, image_urls, max_tokens=800
        )
