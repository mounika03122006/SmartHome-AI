"""Train, compare and save the house price model.   Usage: python -m src.train_model"""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from src.data_preprocessing import (CLEAN_PATH, RAW_FEATURES, TARGET, build_preprocessor, clean_data,
                                    feature_step, load_raw)
from src.evaluate_model import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def make_models(seed: int = SEED, fast: bool = False) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=30 if fast else 200, min_samples_leaf=2,
                                               n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=30 if fast else 300, max_depth=4,
                                                       learning_rate=0.05, subsample=0.8, random_state=seed),
    }


def make_pipeline(reg) -> Pipeline:
    """Feature engineering -> preprocessing -> regressor on log(price). One saved artefact."""
    wrapped = TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("features", feature_step()), ("prep", build_preprocessor()), ("model", wrapped)])


def train(df: pd.DataFrame, model_dir: Path, reports_dir: Path, cleaning_report: dict | None = None,
          fast: bool = False, cv: int = 5) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    X, y = df[RAW_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rows, fitted = [], {}
    for name, reg in make_models(SEED, fast).items():
        t = time.time()
        pipe = make_pipeline(reg)  # preprocessing is re-fit inside each CV fold -> no leakage
        cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=("neg_root_mean_squared_error", "r2"))
        pipe.fit(X_tr, y_tr)
        m = regression_metrics(y_te, pipe.predict(X_te))
        rows.append({"Model": name, "CV_RMSE": -cvr["test_neg_root_mean_squared_error"].mean(),
                     "CV_R2": cvr["test_r2"].mean(), "Test_MAE": m["MAE"], "Test_RMSE": m["RMSE"],
                     "Test_R2": m["R2"], "Seconds": round(time.time() - t, 1)})
        fitted[name] = pipe
        print(f"{name:20s} CV RMSE {rows[-1]['CV_RMSE']:.2f}  Test R2 {m['R2']:.3f}")
    comp = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
    best = comp.loc[0, "Model"]  # selected on cross-validation of the TRAIN set only
    pipe = fitted[best]
    pred = pipe.predict(X_te)
    pd.DataFrame({"actual": y_te.values, "predicted": pred, "residual": y_te.values - pred}
                 ).to_csv(reports_dir / "test_predictions.csv", index=False)
    pi = permutation_importance(pipe, X_te, y_te, scoring="r2", n_repeats=5, random_state=SEED, n_jobs=-1)
    pd.DataFrame({"feature": RAW_FEATURES, "importance": pi.importances_mean, "std": pi.importances_std}
                 ).sort_values("importance", ascending=False).to_csv(reports_dir / "feature_importance.csv", index=False)
    comp.to_csv(reports_dir / "model_comparison.csv", index=False)
    meta = {"selected_model": best, "seed": SEED, "cv_folds": cv, "n_train": len(X_tr), "n_test": len(X_te),
            "target_unit": "lakh INR", "locations": sorted(X["location"].dropna().unique().tolist()),
            "area_types": sorted(X["area_type"].unique().tolist()),
            "availability": sorted(X["availability"].unique().tolist()),
            "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in ["total_sqft", "bhk", "bath", "balcony"]},
            "cleaning_report": cleaning_report or {}, "test_metrics": comp.loc[0].to_dict()}
    (reports_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    joblib.dump({"pipeline": pipe, "metadata": meta}, model_dir / "house_price_model.joblib")
    return meta


if __name__ == "__main__":
    clean, rep = clean_data(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True); clean.to_csv(CLEAN_PATH, index=False)
    meta = train(clean, ROOT / "models", ROOT / "reports", rep)
    print("\nSelected model:", meta["selected_model"])
