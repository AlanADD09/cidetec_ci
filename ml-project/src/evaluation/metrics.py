from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
from typing import Dict, Any

def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "accuracy": float(acc),
        "f1_weighted": float(f1w),
    }
