"""Metrics and helpers to read saved evaluation artefacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def regression_metrics(y_true, y_pred) -> dict:
    """MAE & RMSE are in the target unit (lakhs INR); R2 is unit-free."""
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def load_reports(reports_dir: Path = REPORTS):
    """Return (metrics_df, test_predictions_df, importance_df, meta) or None if training has not run."""
    try:
        return (pd.read_csv(reports_dir / "model_comparison.csv"),
                pd.read_csv(reports_dir / "test_predictions.csv"),
                pd.read_csv(reports_dir / "feature_importance.csv"),
                json.loads((reports_dir / "metadata.json").read_text()))
    except (FileNotFoundError, ValueError):
        return None
