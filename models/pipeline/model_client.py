from __future__ import annotations

from typing import Protocol

from .types import InferenceRequest, InferenceResult


class ModelClient(Protocol):
    """Unified model client interface used by the pipeline."""

    @property
    def model_name(self) -> str:
        ...

    def predict(self, request: InferenceRequest) -> InferenceResult:
        ...
