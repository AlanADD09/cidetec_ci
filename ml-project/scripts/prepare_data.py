import argparse
import os
import pandas as pd
from src.core.config import load_yaml, DataConfig
from src.core.utils import ensure_dir
from src.data.dataset_io import write_parquet

def main(args):
    data_cfg = load_yaml(args.data_config)
    data_cfg = DataConfig(**data_cfg)

    df = pd.read_csv(data_cfg.dataset_path)
    X = df[data_cfg.numerical_features + data_cfg.categorical_features]
    y = df[data_cfg.target]

    ensure_dir("data/processed")
    write_parquet(X, "data/processed/X.parquet")
    write_parquet(pd.DataFrame({data_cfg.target: y}), "data/processed/y.parquet")
    print("Datos preparados en data/processed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    main(parser.parse_args())
