from __future__ import annotations

from typing import Dict, Type

from .base import BasePipeline


_REGISTRY: Dict[str, Type[BasePipeline]] = {}


def register(name: str):
    """Register a model adapter."""
    def _wrap(cls: Type[BasePipeline]) -> Type[BasePipeline]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get(name: str) -> Type[BasePipeline]:
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' not registered")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())
