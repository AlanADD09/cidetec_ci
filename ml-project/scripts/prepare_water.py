import pandas as pd
import numpy as np
from src.core.utils import ensure_dir

def main():
    df = pd.read_csv("data/raw/water_potability.csv")

    # Asegura dtype correcto del target (0/1)
    if df["Potability"].dtype != np.int64 and df["Potability"].dtype != np.int32:
        df["Potability"] = df["Potability"].astype(int)

    # Separa X / y (no imputamos aquí; se imputará en el pipeline)
    y = df[["Potability"]].copy()
    X = df.drop(columns=["Potability"]).copy()

    ensure_dir("data/processed")
    X.to_parquet("data/processed/X.parquet", index=False)
    y.to_parquet("data/processed/y.parquet", index=False)
    print("Dataset Water Potability preparado en data/processed/")

if __name__ == "__main__":
    main()