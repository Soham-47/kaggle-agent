"""Rollback helper kept separate from promotion policy."""

from kaggle_agent.supervisor.generation import RuntimeGeneration
from kaggle_agent.supervisor.promote import GenerationPromotion


def rollback(promotion: GenerationPromotion, previous: RuntimeGeneration) -> None:
    promotion.rollback(previous)
