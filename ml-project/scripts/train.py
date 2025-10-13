import argparse
import os
import pandas as pd
from src.core.config import load_yaml, DataConfig, ValidationConfig, ModelConfig
from src.core.utils import ensure_dir, set_seed
from src.data.splitters import holdout
from src.pipeline.trainer import fit_and_evaluate

def main(args):
    cfg_master = load_yaml(args.config)

    data_cfg = DataConfig(**load_yaml(cfg_master.get("data_config", "configs/data.yaml")))
    val_cfg = ValidationConfig(**load_yaml(cfg_master.get("validation_config", "configs/validation.yaml")))
    model_cfg = ModelConfig(**load_yaml(args.model_config or cfg_master.get("model_config", "configs/model_knn.yaml")))

    set_seed(42)

    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet")[data_cfg.target]

    if val_cfg.strategy == "holdout":
        X_train, X_test, y_train, y_test = holdout(
            X, y,
            test_size=val_cfg.holdout["test_size"],
            random_state=val_cfg.holdout["random_state"],
            stratify=val_cfg.holdout.get("stratify", True)
        )
    else:
        raise NotImplementedError("Usa holdout o implementa KFold en cross_validate.py")

    ensure_dir("experiments/runs")
    model_out = os.path.join("experiments/runs", f"model_{model_cfg.name}.joblib")

    metrics = fit_and_evaluate(
        X_train, y_train, X_test, y_test,
        model_cfg.name, model_cfg.params,
        data_cfg.numerical_features, data_cfg.categorical_features,
        model_out
    )

    import json
    with open(os.path.join("experiments/runs", f"metrics_{model_cfg.name}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Entrenamiento completo. Métricas guardadas en experiments/runs/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-config", default=None, help="Ruta a un YAML de modelo específico")
    main(parser.parse_args())
