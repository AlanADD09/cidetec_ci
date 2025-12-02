import argparse, os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.core.config import load_yaml, DataConfig, ModelConfig
from src.core.utils import ensure_dir, set_seed
from src.pipeline.build_pipeline import build_pipeline
from src.validation.factory import make_validation_strategy
from src.evaluation.metrics import compute_metrics
from src.evaluation.reporter import save_confusion_matrix

def main(args):
    bench_cfg = load_yaml(args.benchmark_config)

    data_config_path = args.data_config or bench_cfg.get("data_config")
    model_config_path = args.model_config or bench_cfg.get("model_config")

    if not data_config_path:
        raise ValueError("Debe especificar un data_config ya sea en el YAML o vía --data-config")
    if not model_config_path:
        raise ValueError("Debe especificar un model_config ya sea en el YAML o vía --model-config")

    data_cfg  = DataConfig(**load_yaml(data_config_path))
    model_cfg = ModelConfig(**load_yaml(model_config_path))
    runs_cfg  = bench_cfg["runs"]

    set_seed(42)

    # Datos
    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet")[data_cfg.target].to_numpy()
    labels_ord = np.unique(y)

    # Builder de pipeline (nuevo por cada fit)
    def build_pipe():
        return build_pipeline(
            model_name=model_cfg.name,
            model_params=model_cfg.params,
            num_cols=data_cfg.numerical_features,
            cat_cols=data_cfg.categorical_features,
        )

    # Carpeta de salida por timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join("experiments", "runs", f"bench_{ts}")
    ensure_dir(root_out)

    summary_rows = []

    for run in runs_cfg:
        strategy_name = run["strategy"]
        params = run.get("params", {})
        print(f"==> Ejecutando {strategy_name} con params={params}")

        strategy = make_validation_strategy(strategy_name, params)
        cvres = strategy.run(X, y, build_pipe)

        # Métricas globales y artefactos OOF
        global_metrics = compute_metrics(cvres.y_true_all, cvres.y_pred_all)

        # Carpeta por estrategia
        out_dir = os.path.join(root_out, strategy_name)
        ensure_dir(out_dir)

        # OOF por estrategia
        oof_df = pd.DataFrame({
            "index": np.arange(len(cvres.y_true_all)),
            "y_true": cvres.y_true_all,
            "y_pred": cvres.y_pred_all
        })
        oof_path = os.path.join(out_dir, f"oof_{model_cfg.name}_{cvres.mode}.csv")
        oof_df.to_csv(oof_path, index=False, encoding="utf-8")

        # Confusion matrix única (global)
        cm = confusion_matrix(cvres.y_true_all, cvres.y_pred_all, labels=labels_ord)
        cm_png = os.path.join(out_dir, f"confusion_matrix_{model_cfg.name}_{cvres.mode}.png")
        save_confusion_matrix(cm, labels=[str(l) for l in labels_ord], out_png=cm_png)

        # Guardar fold_metrics (si existen) y JSON consolidado
        out_json = os.path.join(out_dir, f"metrics_{model_cfg.name}_{cvres.mode}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": strategy_name,
                "params": params,
                "mode": cvres.mode,
                "fold_metrics": cvres.fold_metrics,
                "global_metrics": global_metrics,
                "oof_path": oof_path,
                "confusion_matrix_png": cm_png,
                "confusion_matrix": cm.tolist()
            }, f, ensure_ascii=False, indent=2)

        # Resumen tabular
        summary_rows.append({
            "strategy": strategy_name,
            "mode": cvres.mode,
            "accuracy": global_metrics["accuracy"],
            "f1_weighted": global_metrics["f1_weighted"],
            "oof_path": oof_path,
            "cm_png": cm_png,
            "metrics_json": out_json
        })

        print(f"[OK] {strategy_name}: Acc={global_metrics['accuracy']:.4f} | F1w={global_metrics['f1_weighted']:.4f}")

    # Exporta resumen CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(root_out, "summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"\nResumen guardado en: {summary_csv}\nCarpeta: {root_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default="configs/benchmark.yaml", help="Ruta al YAML maestro del benchmark")
    parser.add_argument("--data-config", default=None, help="Sobrescribe el data_config definido en el YAML")
    parser.add_argument("--model-config", default=None, help="Sobrescribe el model_config definido en el YAML")
    main(parser.parse_args())
