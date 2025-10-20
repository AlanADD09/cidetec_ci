from typing import Dict
from .base import ValidationStrategy
from .strategies import HoldoutStrategy, RepeatedHoldoutStrategy, KFoldStrategy, LOOStrategy, RepeatedKFoldStrategy

def make_validation_strategy(name: str, params: Dict) -> ValidationStrategy:
    name = (name or "").lower()
    if name == "holdout":
        return HoldoutStrategy(params)
    if name == "repeated_holdout":
        return RepeatedHoldoutStrategy(params)
    if name == "kfold":
        return KFoldStrategy(params)
    if name == "repeated_kfold":
        return RepeatedKFoldStrategy(params)
    if name == "loo":
        return LOOStrategy(params)
    raise ValueError(f"Estrategia de validación no soportada: {name}")
