#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.rename("target")
    df = pd.concat([X, y], axis=1)
    
    """
    X_train = in subset for training
    X_test  = in subset for test
    y_train = tags for X_train
    y_test  = tasg for X_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42, #seed
        stratify=y #same portion for train and test classes, avoids unbalanced sets
        #shuffle is on true
    )

    df["split"] = "train"
    df.loc[X_test.index, "split"] = "test"

    out_path = Path("holdout_processed.data")
    df.to_csv(out_path, index=False)

    print("File saved in:", out_path.resolve())
    print("Sizes -> train:", len(X_train), "| test:", len(X_test))

if __name__ == "__main__":
    main()
