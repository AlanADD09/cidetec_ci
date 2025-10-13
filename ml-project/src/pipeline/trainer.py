import joblib
from typing import Dict, Tuple
from .build_pipeline import build_pipeline
from ..evaluation.metrics import compute_metrics

def fit_and_evaluate(
    X_train, y_train, X_test, y_test,
    model_name: str, model_params: Dict,
    num_cols, cat_cols, model_out_path: str
) -> Dict:
    pipe = build_pipeline(model_name, model_params, num_cols, cat_cols)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    joblib.dump(pipe, model_out_path)
    return metrics
