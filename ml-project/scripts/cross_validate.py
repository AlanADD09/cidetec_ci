import argparse, os, json
import numpy as np
import pandas as pd

from src.core.config import load_yaml, DataConfig, ValidationConfig, ModelConfig
from src.core.utils import ensure_dir, set_seed
from src.pipeline.build_pipeline import build_pipeline
from src.data.splitters import make_kfold, make_loo
from src.evaluation.metrics import compute_metrics

def main(args):
    cfg_master = load_yaml(args.config)
    data_cfg = DataConfig(**load_yaml(cfg_master.get("data_config", "configs/data.yaml")))
    val_cfg  = ValidationConfig(**load_yaml(cfg_master.get("validation_config", "configs/validation.yaml")))
    model_cfg = ModelConfig(**load_yaml(args.model_config or cfg_master.get("model_config", "configs/model_knn.yaml")))

    set_seed(42)

    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet")[data_cfg.target].to_numpy()

    # Selección del splitter
    if val_cfg.strategy == "kfold":
        cv = make_kfold(
            n_splits=val_cfg.kfold["n_splits"],
            stratified=val_cfg.kfold.get("stratified", True),
            shuffle=val_cfg.kfold.get("shuffle", True),
            random_state=val_cfg.kfold.get("random_state", 42),
        )
    elif val_cfg.strategy == "loo":
        cv = make_loo()
    else:
        raise ValueError(f"Estrategia no soportada en cross_validate.py: {val_cfg.strategy}")

    pipe = build_pipeline(model_cfg.name, model_cfg.params,
                          data_cfg.numerical_features, data_cfg.categorical_features)

    # Acumuladores
    y_true_all, y_pred_all = [], []
    fold_metrics = []

    # Iterar folds: fit en train, predict en test (1 muestra en LOO)
    for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y[tr], y[te]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # guardar para métricas globales
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        # métrica por fold (opcional)
        m = compute_metrics(y_test, y_pred)
        fold_metrics.append({"fold": fold_idx, "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"]})

    # Métricas globales (sobre todas las predicciones CV)
    from src.evaluation.metrics import compute_metrics as compute
    global_metrics = compute(np.array(y_true_all), np.array(y_pred_all))

    # Persistencia
    ensure_dir("experiments/runs")
    out_dir = "experiments/runs"
    with open(os.path.join(out_dir, f"cv_metrics_{model_cfg.name}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "strategy": val_cfg.strategy,
            "fold_metrics": fold_metrics,
            "global_metrics": global_metrics
        }, f, ensure_ascii=False, indent=2)

    print(f"CV ({val_cfg.strategy}) completa. Métricas en {out_dir}/cv_metrics_{model_cfg.name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default=None)
    main(parser.parse_args())
