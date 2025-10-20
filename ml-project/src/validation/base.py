from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

@dataclass
class CVResult:
    mode: str
    fold_metrics: List[Dict[str, Any]]          # por fold/split (si aplica)
    y_true_all: np.ndarray                       # OOF/global
    y_pred_all: np.ndarray                       # OOF/global

class ValidationStrategy(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params or {}

    @abstractmethod
    def run(self, X, y, build_pipeline_fn) -> CVResult:
        """Ejecuta la validación y devuelve predicciones OOF + métricas por fold (si aplica)."""
        ...
