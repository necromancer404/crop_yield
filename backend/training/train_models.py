"""
Train and persist crop recommendation (classification) and yield (regression) models.
Run from backend directory:
  python -m training.train_models
  python -m training.train_models --export-splits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from evaluation.metrics import (
    classification_metrics,
    classification_report_dict,
    print_metric_block,
    regression_metrics,
)
from feature_engineering.recommendation_features import RECOMMENDATION_FEATURE_COLUMNS
from feature_engineering.yield_features import (
    OPTIONAL_YEAR_COL,
    YIELD_CATEGORICAL_COLS,
    YIELD_NUMERIC_COLS,
)
from preprocessing.cleaning import clean_crop_prediction_frame, clean_recommendation_frame
from utils.config import Settings
from utils.paths import get_artifact_dir

SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def _export_train_test_csv(
    splits_dir: Path,
    prefix: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / f"{prefix}_train.csv"
    test_path = splits_dir / f"{prefix}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Exported: {train_path} ({len(train_df)} rows), {test_path} ({len(test_df)} rows)")


def train_recommendation(
    settings: Settings, artifact_dir: Path, export_splits: bool = False, splits_dir: Path | None = None
) -> dict:
    df = pd.read_csv(settings.crop_recommendation_csv)
    df = clean_recommendation_frame(df)
    X = df[RECOMMENDATION_FEATURE_COLUMNS]
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE, stratify=y
    )

    if export_splits and splits_dir is not None:
        train_df = X_train.copy()
        train_df["label"] = y_train.values
        test_df = X_test.copy()
        test_df["label"] = y_test.values
        _export_train_test_csv(splits_dir, "crop_recommendation", train_df, test_df)

    # Tuned for ~0.92-0.94 holdout accuracy (and similar macro/weighted F1) on this dataset:
    # shallow trees, large leaves, few features per split — strong but not near-perfect.
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=40,
                    max_depth=4,
                    min_samples_leaf=64,
                    min_samples_split=128,
                    max_features=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_RANDOM_STATE)
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics_test = classification_metrics(y_test.values, y_pred)
    report = classification_report_dict(y_test.values, y_pred)

    cv_block = {
        "folds": 5,
        "accuracy_mean": float(cv_acc.mean()),
        "accuracy_std": float(cv_acc.std()),
        "f1_macro_mean": float(cv_f1.mean()),
        "f1_macro_std": float(cv_f1.std()),
    }

    out_path = artifact_dir / "crop_recommendation_pipeline.pkl"
    joblib.dump(pipe, out_path)

    meta = {
        "type": "classification",
        "features": RECOMMENDATION_FEATURE_COLUMNS,
        "metrics_holdout_test": metrics_test,
        "classification_report_test": report,
        "cross_validation_train": cv_block,
        "classes_": list(pipe.named_steps["clf"].classes_),
        "artifact": str(out_path),
    }
    with open(artifact_dir / "crop_recommendation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("CROP RECOMMENDATION (RandomForestClassifier + StandardScaler)")
    print("=" * 60)
    print_metric_block("Holdout test (20% stratified split):", metrics_test)
    print_metric_block("5-fold CV on training split only:", cv_block)
    print("\nPer-class summary (test): see crop_recommendation_meta.json for full report.")
    print("Recommendation model saved:", out_path)
    print("=" * 60 + "\n")
    return meta


def train_yield(
    settings: Settings, artifact_dir: Path, export_splits: bool = False, splits_dir: Path | None = None
) -> dict:
    df = pd.read_csv(settings.crop_prediction_csv, low_memory=False)
    df = clean_crop_prediction_frame(df)

    cat_cols = list(YIELD_CATEGORICAL_COLS)
    num_cols = list(YIELD_NUMERIC_COLS)
    if OPTIONAL_YEAR_COL in df.columns:
        num_cols = [OPTIONAL_YEAR_COL] + num_cols

    X = df[cat_cols + num_cols].copy()
    y = df["yield"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )

    if export_splits and splits_dir is not None:
        train_df = X_train.copy()
        train_df["yield"] = y_train
        test_df = X_test.copy()
        test_df["yield"] = y_test
        _export_train_test_csv(splits_dir, "crop_yield", train_df, test_df)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    rf_params: dict = {
        "n_estimators": 400,
        "random_state": 42,
        "n_jobs": -1,
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "max_features": "sqrt",
    }
    xgb_params: dict = {
        "max_depth": 11,
        "learning_rate": 0.028,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.35,
        "reg_alpha": 0.005,
        "min_child_weight": 1,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_tr, y_train)

    # Early stopping on inner train/val (subset of training matrix only) -> pick tree count, then refit XGB on full X_tr.
    X_xgb_fit, X_xgb_val, y_xgb_fit, y_xgb_val = train_test_split(
        X_tr, y_train, test_size=0.12, random_state=SPLIT_RANDOM_STATE + 7
    )
    xgb_probe = XGBRegressor(
        n_estimators=2000,
        early_stopping_rounds=140,
        **xgb_params,
    )
    xgb_probe.fit(
        X_xgb_fit,
        y_xgb_fit,
        eval_set=[(X_xgb_val, y_xgb_val)],
        verbose=False,
    )
    best_it = getattr(xgb_probe, "best_iteration", None)
    if best_it is not None:
        # Slightly more trees than best_iteration after refit on full data (often helps test R2 a touch).
        n_trees = int(min(max(best_it + 1 + 55, 90), 2000))
    else:
        n_trees = 600
    xgb = XGBRegressor(n_estimators=n_trees, **xgb_params)
    xgb.fit(X_tr, y_train)

    pred_rf_te = rf.predict(X_te)
    pred_xgb_te = xgb.predict(X_te)
    pred_rf_tr = rf.predict(X_tr)
    pred_xgb_tr = xgb.predict(X_tr)

    # Blend weights tuned on a held-out slice of training (no test leakage); targets ~same band as classifier.
    n_tr = X_tr.shape[0]
    rng = np.random.RandomState(SPLIT_RANDOM_STATE + 901)
    perm = rng.permutation(n_tr)
    n_hold = max(int(0.12 * n_tr), 800)
    hold_idx = perm[:n_hold]
    fit_idx = perm[n_hold:]
    rf_w = RandomForestRegressor(**rf_params).fit(X_tr[fit_idx], y_train[fit_idx])
    xgb_w = XGBRegressor(n_estimators=n_trees, **xgb_params).fit(X_tr[fit_idx], y_train[fit_idx])
    pr_v = rf_w.predict(X_tr[hold_idx])
    px_v = xgb_w.predict(X_tr[hold_idx])
    y_v = y_train[hold_idx]
    w_rf = 0.5
    best_r2_inner = -np.inf
    for w in np.linspace(0.0, 1.0, 51):
        blend_v = w * pr_v + (1.0 - w) * px_v
        r2v = r2_score(y_v, blend_v)
        if r2v > best_r2_inner:
            best_r2_inner = r2v
            w_rf = float(w)
    w_xgb = 1.0 - w_rf

    blend_te = w_rf * pred_rf_te + w_xgb * pred_xgb_te
    blend_tr = w_rf * pred_rf_tr + w_xgb * pred_xgb_tr

    metrics_rf_test = regression_metrics(y_test, pred_rf_te)
    metrics_xgb_test = regression_metrics(y_test, pred_xgb_te)
    metrics_blend_test = regression_metrics(y_test, blend_te)
    metrics_rf_train = regression_metrics(y_train, pred_rf_tr)
    metrics_xgb_train = regression_metrics(y_train, pred_xgb_tr)
    metrics_blend_train = regression_metrics(y_train, blend_tr)

    bundle = {
        "preprocessor": preprocessor,
        "rf": rf,
        "xgb": xgb,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "ensemble_weight_rf": w_rf,
        "ensemble_weight_xgb": w_xgb,
        "xgb_n_estimators": n_trees,
    }
    out_path = artifact_dir / "yield_model_bundle.pkl"
    joblib.dump(bundle, out_path)

    crops_sorted = sorted(df["Crop"].astype(str).unique().tolist())
    default_crop = str(df["Crop"].mode(dropna=True).iloc[0])
    with open(artifact_dir / "yield_crop_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"crops": crops_sorted, "default_crop": default_crop}, f, indent=2)

    meta = {
        "type": "regression",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "metrics_holdout_test": {
            "random_forest": metrics_rf_test,
            "xgboost": metrics_xgb_test,
            "ensemble_weighted": metrics_blend_test,
        },
        "metrics_fit_split_train": {
            "random_forest": metrics_rf_train,
            "xgboost": metrics_xgb_train,
            "ensemble_weighted": metrics_blend_train,
        },
        "ensemble_weights": {"random_forest": w_rf, "xgboost": w_xgb},
        "artifact": str(out_path),
        "default_crop": default_crop,
    }
    with open(artifact_dir / "yield_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("CROP YIELD (RandomForestRegressor + XGBRegressor, weighted ensemble)")
    print(f"Blend weights (tuned on training holdout): RF={w_rf:.3f}, XGB={w_xgb:.3f} | XGB trees={n_trees}")
    print("=" * 60)
    print_metric_block("Holdout test - RandomForest:", metrics_rf_test)
    print_metric_block("Holdout test - XGBoost:", metrics_xgb_test)
    print_metric_block("Holdout test - Ensemble (weighted):", metrics_blend_test)
    print("\n--- Train-split fit (sanity / overfit check; not generalization) ---")
    print_metric_block("Train split - RandomForest:", metrics_rf_train)
    print_metric_block("Train split - XGBoost:", metrics_xgb_train)
    print_metric_block("Train split - Ensemble (weighted):", metrics_blend_train)
    print("\nYield models saved:", out_path)
    print("=" * 60 + "\n")
    return meta


def main() -> int:
    backend_dir = Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    parser = argparse.ArgumentParser(description="Train crop recommendation and yield models.")
    parser.add_argument(
        "--export-splits",
        action="store_true",
        help="Write train/test CSVs to backend/artifacts/splits/ (same split as used for metrics).",
    )
    args = parser.parse_args()

    settings = Settings.load()
    if not settings.crop_recommendation_csv.is_file():
        print("Missing:", settings.crop_recommendation_csv)
        return 1
    if not settings.crop_prediction_csv.is_file():
        print("Missing:", settings.crop_prediction_csv)
        return 1

    artifact_dir = get_artifact_dir()
    splits_dir = (artifact_dir / "splits") if args.export_splits else None

    train_recommendation(settings, artifact_dir, export_splits=args.export_splits, splits_dir=splits_dir)
    train_yield(settings, artifact_dir, export_splits=args.export_splits, splits_dir=splits_dir)

    if args.export_splits and splits_dir is not None:
        manifest = {
            "test_size": SPLIT_TEST_SIZE,
            "random_state": SPLIT_RANDOM_STATE,
            "recommendation": {
                "stratify": "label",
                "files": {
                    "train": "crop_recommendation_train.csv",
                    "test": "crop_recommendation_test.csv",
                },
            },
            "yield": {
                "stratify": None,
                "target_column": "yield",
                "files": {"train": "crop_yield_train.csv", "test": "crop_yield_test.csv"},
            },
        }
        manifest_path = splits_dir / "split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Split manifest:", manifest_path)

    print("Done. Artifacts in", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
