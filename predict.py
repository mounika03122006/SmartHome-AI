"""Load the saved pipeline, validate user input and predict a price (lakh INR)."""
from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd
from src.data_preprocessing import RAW_FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "house_price_model.joblib"


class ModelNotFoundError(Exception): ...
class ModelLoadError(Exception): ...
class ValidationError(ValueError): ...


def load_model(path: Path | str = MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(f"No trained model at {path}. Run: python -m src.train_model")
    try:
        bundle = joblib.load(path)
        assert "pipeline" in bundle and "metadata" in bundle
        return bundle
    except Exception as e:  # noqa: BLE001
        raise ModelLoadError(f"Model file is corrupt or incompatible: {e}") from e


def validate_input(inp: dict, meta: dict) -> pd.DataFrame:
    """Return a one-row DataFrame or raise ValidationError with a readable message."""
    missing = [c for c in RAW_FEATURES if c not in inp or inp[c] in (None, "")]
    if missing:
        raise ValidationError(f"Missing inputs: {missing}")
    for col, allowed in (("location", meta["locations"]), ("area_type", meta["area_types"]),
                         ("availability", meta["availability"])):
        if inp[col] not in allowed:
            raise ValidationError(f"Unknown {col} '{inp[col]}'. The model has not seen this value.")
    row = {c: inp[c] for c in ("location", "area_type", "availability")}
    for c in ("total_sqft", "bhk", "bath", "balcony"):
        try:
            v = float(inp[c])
        except (TypeError, ValueError):
            raise ValidationError(f"{c} must be a number.") from None
        lo, hi = meta["ranges"][c]
        if not np.isfinite(v) or v < max(lo, 0) or v > hi:
            raise ValidationError(f"{c} must be between {lo:g} and {hi:g} (range seen in training data).")
        row[c] = v
    if row["bath"] > row["bhk"] + 2:
        raise ValidationError("Bathrooms cannot exceed bedrooms + 2 (the cleaning rule used in training).")
    if row["total_sqft"] / row["bhk"] < 300:
        raise ValidationError("Area per bedroom is below 300 sq ft, outside the range the model was trained on.")
    return pd.DataFrame([row], columns=RAW_FEATURES)


def predict_price(inp: dict, bundle: dict) -> dict:
    X = validate_input(inp, bundle["metadata"])
    try:
        lakh = float(bundle["pipeline"].predict(X)[0])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Prediction failed: {e}") from e
    if not np.isfinite(lakh) or lakh <= 0:
        raise RuntimeError("Model produced an invalid prediction; no estimate is shown.")
    return {"price_lakh": lakh, "price_inr": lakh * 1e5, "price_per_sqft": lakh * 1e5 / float(X["total_sqft"][0])}


def format_inr(lakh: float) -> str:
    return f"₹ {lakh / 100:,.2f} Crore" if lakh >= 100 else f"₹ {lakh:,.2f} Lakh"
