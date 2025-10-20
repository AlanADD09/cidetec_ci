from typing import Dict, Any, List
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedShuffleSplit,
    StratifiedKFold, KFold, LeaveOneOut, RepeatedStratifiedKFold, RepeatedKFold
)
from .base import ValidationStrategy, CVResult

# ---------- HOLDOUT ----------
class HoldoutStrategy(ValidationStrategy):
    def run(self, X, y, build_pipeline_fn) -> CVResult:
        test_size   = self.params.get("test_size", 0.2)
        random_state= self.params.get("random_state", 42)
        stratify    = self.params.get("stratify", True)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        pipe = build_pipeline_fn()
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        return CVResult(
            mode="holdout",
            fold_metrics=[],                 # no aplica
            y_true_all=y_te,
            y_pred_all=y_pred
        )

# ---------- REPEATED HOLDOUT ----------
class RepeatedHoldoutStrategy(ValidationStrategy):
    def run(self, X, y, build_pipeline_fn) -> CVResult:
        test_size   = self.params.get("test_size", 0.2)
        n_splits    = self.params.get("n_splits", 30)
        random_state= self.params.get("random_state", 42)
        stratified  = self.params.get("stratified", True)

        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state
        ) if stratified else ShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state
        )

        y_true_all, y_pred_all = [], []
        fold_metrics: List[Dict[str, Any]] = []

        from sklearn.metrics import accuracy_score, f1_score
        for fold, (tr, te) in enumerate(splitter.split(X, y)):
            pipe = build_pipeline_fn()
            pipe.fit(X.iloc[tr], y[tr])
            yp = pipe.predict(X.iloc[te])

            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(yp.tolist())

            fold_metrics.append({
                "split": fold,
                "accuracy": float(accuracy_score(y[te], yp)),
                "f1_weighted": float(f1_score(y[te], yp, average="weighted", zero_division=0))
            })

        return CVResult(
            mode="repeated_holdout",
            fold_metrics=fold_metrics,
            y_true_all=np.array(y_true_all),
            y_pred_all=np.array(y_pred_all)
        )

# ---------- K-FOLD ----------
class KFoldStrategy(ValidationStrategy):
    def run(self, X, y, build_pipeline_fn) -> CVResult:
        n_splits = self.params.get("n_splits", 5)
        stratified = self.params.get("stratified", True)
        shuffle = self.params.get("shuffle", True)
        random_state = self.params.get("random_state", 42)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) \
             if stratified else KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        from sklearn.metrics import accuracy_score, f1_score
        y_true_all, y_pred_all = [], []
        fold_metrics: List[Dict[str, Any]] = []

        for fold, (tr, te) in enumerate(cv.split(X, y)):
            pipe = build_pipeline_fn()
            pipe.fit(X.iloc[tr], y[tr])
            yp = pipe.predict(X.iloc[te])

            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(yp.tolist())

            fold_metrics.append({
                "fold": fold,
                "accuracy": float(accuracy_score(y[te], yp)),
                "f1_weighted": float(f1_score(y[te], yp, average="weighted", zero_division=0))
            })

        return CVResult(
            mode="kfold",
            fold_metrics=fold_metrics,
            y_true_all=np.array(y_true_all),
            y_pred_all=np.array(y_pred_all)
        )

# ---------- REPEATED K-FOLD ----------
class RepeatedKFoldStrategy(ValidationStrategy):
    def run(self, X, y, build_pipeline_fn) -> CVResult:
        n_splits  = self.params.get("n_splits", 5)
        n_repeats = self.params.get("n_repeats", 10)
        random_state = self.params.get("random_state", 42)
        stratified = self.params.get("stratified", True)

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        ) if stratified else RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

        from sklearn.metrics import accuracy_score, f1_score
        y_true_all, y_pred_all = [], []
        fold_metrics: List[Dict[str, Any]] = []

        for fold, (tr, te) in enumerate(cv.split(X, y)):
            pipe = build_pipeline_fn()
            pipe.fit(X.iloc[tr], y[tr])
            yp = pipe.predict(X.iloc[te])

            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(yp.tolist())

            # opcional: métricas por split
            acc = float(accuracy_score(y[te], yp))
            f1w = float(f1_score(y[te], yp, average="weighted", zero_division=0))
            fold_metrics.append({"split": fold, "accuracy": acc, "f1_weighted": f1w})

        return CVResult(
            mode="repeated_kfold",
            fold_metrics=fold_metrics,
            y_true_all=np.array(y_true_all),
            y_pred_all=np.array(y_pred_all)
        )

# ---------- LEAVE-ONE-OUT ----------
class LOOStrategy(ValidationStrategy):
    def run(self, X, y, build_pipeline_fn) -> CVResult:
        cv = LeaveOneOut()
        y_true_all, y_pred_all = [], []

        for _, (tr, te) in enumerate(cv.split(X, y)):
            pipe = build_pipeline_fn()
            pipe.fit(X.iloc[tr], y[tr])
            yp = pipe.predict(X.iloc[te])
            y_true_all.extend(y[te].tolist())
            y_pred_all.extend(yp.tolist())

        return CVResult(
            mode="loo",
            fold_metrics=[],  # por-fold no tiene sentido con 1 muestra
            y_true_all=np.array(y_true_all),
            y_pred_all=np.array(y_pred_all)
        )
