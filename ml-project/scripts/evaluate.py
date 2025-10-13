import argparse
import os
import json
import joblib
import pandas as pd
from src.core.config import load_yaml, DataConfig, EvalConfig
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import save_confusion_matrix, save_json

def main(args):
    cfg_master = load_yaml(args.config)
    data_cfg = DataConfig(**load_yaml(cfg_master.get("data_config", "configs/data.yaml")))
    eval_cfg = EvalConfig(**load_yaml(cfg_master.get("eval_config", "configs/eval.yaml")))

    pipe = joblib.load(args.model_path)
    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet")[data_cfg.target]

    y_pred = pipe.predict(X)
    metrics = compute_metrics(y, y_pred)
    os.makedirs("experiments/figures", exist_ok=True)

    save_json(metrics, os.path.join("experiments/runs", "metrics_eval_fullset.json"))

    if eval_cfg.save_confusion_matrix_png:
        import numpy as np
        labels = sorted(list(map(str, set(y))))
        cm = np.array(metrics["confusion_matrix"])
        save_confusion_matrix(cm, labels, os.path.join("experiments/figures", "confusion_matrix.png"))

    if eval_cfg.save_classification_report_csv:
        import csv
        rep = metrics["classification_report"]
        with open(os.path.join("experiments/runs", "classification_report.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label","precision","recall","f1-score","support"]
            writer.writerow(header)
            for k, v in rep.items():
                if isinstance(v, dict) and all(m in v for m in ["precision","recall","f1-score","support"]):
                    writer.writerow([k, v["precision"], v["recall"], v["f1-score"], v["support"]])

    print("Evaluación completa. Artefactos en experiments/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-path", default="experiments/runs/model_knn.joblib")
    main(parser.parse_args())
