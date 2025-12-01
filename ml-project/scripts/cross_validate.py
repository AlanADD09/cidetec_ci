import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.core.config import load_yaml, DataConfig, ValidationConfig, ModelConfig
from src.core.utils import ensure_dir, set_seed
from src.pipeline.build_pipeline import build_pipeline
from src.validation.factory import make_validation_strategy
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import save_confusion_matrix

def main(args):
    cfg_master = load_yaml(args.config)
    # data_cfg  = DataConfig(**load_yaml(cfg_master.get("data_config", "configs/data.yaml")))
    data_cfg  = DataConfig(**load_yaml(cfg_master.get("data_config", "configs/data_water.yaml")))
    val_cfg   = ValidationConfig(**load_yaml(cfg_master.get("validation_config", "configs/validation.yaml")))
    model_cfg = ModelConfig(**load_yaml(args.model_config or cfg_master.get("model_config", "configs/model_knn.yaml")))

    set_seed(42)

    # Datos
    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet")[data_cfg.target].to_numpy()

    # Builder del pipeline para cada run (evita estado compartido)
    def build_pipe():
        return build_pipeline(
            model_name=model_cfg.name,
            model_params=model_cfg.params,
            num_cols=data_cfg.numerical_features,
            cat_cols=data_cfg.categorical_features,
        )

    # Ejecutar estrategia de validación
    strategy = make_validation_strategy(val_cfg.strategy, getattr(val_cfg, val_cfg.strategy, {}) or {})
    cvres = strategy.run(X, y, build_pipe)

    # Métricas globales (OOF)
    global_metrics = compute_metrics(cvres.y_true_all, cvres.y_pred_all)

    # Artefactos
    ensure_dir("experiments/runs")
    out_dir = "experiments/runs"

    # OOF CSV
    oof_df = pd.DataFrame({
        "index": np.arange(len(cvres.y_true_all)),
        "y_true": cvres.y_true_all,
        "y_pred": cvres.y_pred_all
    })
    oof_path = os.path.join(out_dir, f"oof_{model_cfg.name}_{cvres.mode}.csv")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8")

    # Matriz de confusión única (global)
    labels_ord = np.unique(y)
    cm = confusion_matrix(cvres.y_true_all, cvres.y_pred_all, labels=labels_ord)
    cm_png = os.path.join(out_dir, f"confusion_matrix_{model_cfg.name}_{cvres.mode}.png")
    save_confusion_matrix(cm, labels=[str(l) for l in labels_ord], out_png=cm_png)

    # JSON consolidado
    output = {
        "strategy": val_cfg.strategy,
        "mode": cvres.mode,
        "fold_metrics": cvres.fold_metrics,    # vacío en LOO/Holdout
        "global_metrics": global_metrics,      # accuracy, f1w, report, cm
        "oof_path": oof_path,
        "confusion_matrix_png": cm_png,
        "confusion_matrix": cm.tolist()
    }
    out_json = os.path.join(out_dir, f"cv_metrics_{model_cfg.name}_{cvres.mode}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] {cvres.mode} listo. Acc={global_metrics['accuracy']:.4f} | F1w={global_metrics['f1_weighted']:.4f}")
    print(f"Artefactos: {out_json} | {oof_path} | {cm_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default=None)
    main(parser.parse_args())
