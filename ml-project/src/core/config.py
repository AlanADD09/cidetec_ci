from dataclasses import dataclass
from typing import List, Optional
import yaml

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class DataConfig:
    dataset_path: str
    target: str
    numerical_features: List[str]
    categorical_features: List[str]

@dataclass
class ValidationConfig:
    strategy: str  # "holdout" | "kfold"
    holdout: dict
    kfold: dict

@dataclass
class ModelConfig:
    name: str
    params: dict

@dataclass
class EvalConfig:
    save_confusion_matrix_png: bool = True
    save_classification_report_csv: bool = True
