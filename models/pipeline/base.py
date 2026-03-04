from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .types import InferenceRequest, InferenceResult


class BasePipeline(ABC):
    """Unified interface that all model adapters must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (used for logs/result archiving)."""

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Run inference for a single sample."""

    def batch_predict(self, requests: Iterable[InferenceRequest]) -> list[InferenceResult]:
        """Default batch inference (can be overridden by subclasses)."""
        return [self.predict(req) for req in requests]
