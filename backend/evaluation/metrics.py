from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


def _float_dict(d: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _float_dict(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, float):
            out[k] = v
        else:
            out[k] = v
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    # SMAPE: stable for mixed-scale targets (raw MAPE explodes when |y_true| is tiny).
    num = 2.0 * np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = float(np.mean(num / den) * 100.0)
    return {
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "r2_score": r2,
        "explained_variance": evs,
        "smape_pct": smape,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return _float_dict(rep)


def print_metric_block(title: str, metrics: dict[str, Any], indent: str = "  ") -> None:
    print(title)
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{indent}{k}:")
            for sk, sv in v.items():
                if isinstance(sv, float):
                    print(f"{indent}  {sk}: {sv:.6f}" if abs(sv) < 1e4 else f"{indent}  {sk}: {sv:.4g}")
                else:
                    print(f"{indent}  {sk}: {sv}")
        elif isinstance(v, float):
            print(f"{indent}{k}: {v:.6f}" if abs(v) < 1e4 else f"{indent}{k}: {v:.4g}")
        else:
            print(f"{indent}{k}: {v}")
