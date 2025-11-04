from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessor(num_cols: List[str], cat_cols: List[str], *, scale_numeric: bool) -> ColumnTransformer:
    transformers = []
    if num_cols:
        if scale_numeric:
            num = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
        else:
            num = SimpleImputer(strategy="median")
        transformers.append(("num", num, num_cols))

    if cat_cols:
        transformers.append(("cat", SimpleImputer(strategy="most_frequent"), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)