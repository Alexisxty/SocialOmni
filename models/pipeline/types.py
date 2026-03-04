from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class InferenceRequest:
    """Unified inference request schema."""

    video_path: str
    question: str
    options: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    """Unified inference result schema."""

    answer: str
    raw_response: Optional[str] = None
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
