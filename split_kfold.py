#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

def main():
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.rename("target")
    df = pd.concat([X, y], axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df["fold"] = -1

    for fold, (_, test_idx) in enumerate(skf.split(X, y)):
        df.loc[test_idx, "fold"] = fold

    out_path = Path("kfold_processed.data")
    df.to_csv(out_path, index=False)

    print("File saved in:", out_path.resolve())
    print(df["fold"].value_counts().sort_index())

if __name__ == "__main__":
    main()