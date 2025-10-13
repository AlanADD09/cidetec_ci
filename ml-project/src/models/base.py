from abc import ABC, abstractmethod
from typing import Dict, Any
from sklearn.base import BaseEstimator

class ClassifierStrategy(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def build(self) -> BaseEstimator:
        """Return a configured sklearn estimator."""
        raise NotImplementedError
