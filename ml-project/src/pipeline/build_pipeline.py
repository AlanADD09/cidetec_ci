from sklearn.pipeline import Pipeline
from typing import List, Dict
from ..data.preprocess import build_preprocessor
from ..models.knn import KNNStrategy
from ..models.nearest_centroid import NearestCentroidStrategy

STRATEGIES = {
    "knn": KNNStrategy,
    "nearest_centroid": NearestCentroidStrategy,
}

def build_pipeline(model_name: str, model_params: Dict, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    scale_numeric = (model_name == "knn")
    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=scale_numeric)
    if model_name not in STRATEGIES:
        raise ValueError(f"Modelo no soportado: {model_name}")
    model = STRATEGIES[model_name](model_params).build()
    return Pipeline(steps=[("preprocess", pre), ("model", model)])
