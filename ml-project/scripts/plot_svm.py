import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.config import load_yaml, DataConfig, ModelConfig
from src.core.utils import ensure_dir
from src.pipeline.build_pipeline import build_pipeline


def parse_feature_pair(value: str) -> List[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    if len(cols) != 2:
        raise argparse.ArgumentTypeError("Debes especificar exactamente dos columnas separadas por coma.")
    return cols


def plot_decision_regions(pipe, X, y, features: List[str], out_path: str):
    f0, f1 = features
    x_min, x_max = X[f0].min() - 0.5, X[f0].max() + 0.5
    y_min, y_max = X[f1].min() - 0.5, X[f1].max() + 0.5

    grid_step = max((x_max - x_min), (y_max - y_min)) / 200
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=features)
    Z = pipe.predict(grid)
    classes = np.unique(y)
    class_to_int = {cls: idx for idx, cls in enumerate(classes)}
    Z_int = np.vectorize(class_to_int.get)(Z).reshape(xx.shape)

    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("coolwarm", len(classes))

    contour_levels = np.arange(len(classes) + 1) - 0.5
    ax.contourf(xx, yy, Z_int, levels=contour_levels, alpha=0.25, cmap=cmap)

    for cls in classes:
        mask = y == cls
        ax.scatter(
            X.loc[mask, f0],
            X.loc[mask, f1],
            label=str(cls),
            edgecolors="k",
            s=40,
        )

    svc = pipe.named_steps["model"]
    preprocess = pipe.named_steps["preprocess"]
    try:
        support_orig = preprocess.inverse_transform(svc.support_vectors_)
        ax.scatter(
            support_orig[:, 0],
            support_orig[:, 1],
            s=120,
            facecolors="none",
            edgecolors="k",
            linewidths=1.5,
            label="Support vectors",
        )
    except Exception:
        pass

    if getattr(svc, "kernel", None) == "linear" and len(getattr(svc, "classes_", [])) == 2:
        w = svc.coef_[0]
        b = svc.intercept_[0]
        xs = np.linspace(x_min, x_max, 200)
        if w[1] != 0:
            ys = -(w[0] / w[1]) * xs - b / w[1]
            ax.plot(xs, ys, "k--", label="Hyperplane")

    ax.set_xlabel(f0)
    ax.set_ylabel(f1)
    ax.set_title("SVM decision regions")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualiza la frontera de decisión de un SVM en 2D")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model_svm.yaml")
    parser.add_argument("--features", type=parse_feature_pair, default="petal_length,petal_width")
    parser.add_argument("--output", default="experiments/figures/svm_decision.png")
    args = parser.parse_args()

    data_cfg = DataConfig(**load_yaml(args.data_config))
    model_cfg = ModelConfig(**load_yaml(args.model_config))
    if model_cfg.name != "svm":
        raise ValueError("Este script está diseñado únicamente para el modelo SVM.")

    X = pd.read_parquet("data/processed/X.parquet")
    y = pd.read_parquet("data/processed/y.parquet")[data_cfg.target]

    for col in args.features:
        if col not in X.columns:
            raise ValueError(f"La columna '{col}' no existe en data/processed/X.parquet")

    X_pair = X[list(args.features)].copy()

    pipe = build_pipeline(
        model_name=model_cfg.name,
        model_params=model_cfg.params,
        num_cols=list(args.features),
        cat_cols=[],
    )
    pipe.fit(X_pair, y)

    plot_decision_regions(pipe, X_pair, y.to_numpy(), list(args.features), args.output)
    print(f"Gráfico guardado en {args.output}")


if __name__ == "__main__":
    main()
