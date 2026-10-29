"""Data loading, cleaning and preprocessing pipeline for SmartHome AI."""
from __future__ import annotations
import argparse, re, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "bengaluru_house_prices.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean_data.csv"
DATA_URL = ("https://raw.githubusercontent.com/GowriShaju/House-Price-Prediction-EDA/"
            "main/Bengaluru_House_Data.csv")  # GitHub mirror of the Kaggle "Bengaluru House price data"

REQUIRED_COLUMNS = ["area_type", "availability", "location", "size", "total_sqft", "bath", "balcony", "price"]
CAT_FEATURES = ["location", "area_type", "availability"]
NUM_FEATURES = ["total_sqft", "bhk", "bath", "balcony"]
RAW_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "price"  # in lakhs of INR (1 lakh = 100,000 INR)
MIN_PPSF, MAX_PPSF = 1000, 60000  # plausibility bounds, INR per sq ft


class DataError(Exception):
    """Raised when the dataset is missing, empty or malformed."""


def download_dataset(dest: Path = RAW_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, dest)
    return dest


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found at {path}. Run: python -m src.data_preprocessing --download "
                        "or place the CSV there manually.")
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        raise DataError(f"Could not read CSV: {e}") from e
    if df.empty:
        raise DataError("The dataset is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}")
    return df


def parse_sqft(x) -> float:
    """'1200' -> 1200; '1000 - 1200' -> 1100; unit strings such as '34.46Sq. Meter' -> NaN."""
    try:
        return float(x)
    except (TypeError, ValueError):
        parts = str(x).split("-")
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                pass
    return np.nan


def parse_bhk(x) -> float:
    m = re.match(r"\s*(\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean raw data. Returns (clean_df, report) where the report counts every removal."""
    if df.empty:
        raise DataError("The dataset is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}")
    report = {"raw_rows": len(df), "missing_values": df.isna().sum().to_dict()}
    df = df.copy()
    report["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df["area_type"] = df["area_type"].astype(str).str.split().str.join(" ")
    df["location"] = df["location"].astype("string").str.strip()
    df["availability"] = np.where(df["availability"].astype(str).str.strip() == "Ready To Move",
                                  "Ready To Move", "Under Construction")
    df["total_sqft"] = df["total_sqft"].map(parse_sqft)
    df["bhk"] = df["size"].map(parse_bhk)
    for c in ["bath", "balcony", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n = len(df)
    df = df.dropna(subset=["location", "total_sqft", "bhk", "price"])
    report["dropped_missing_or_unparseable"] = n - len(df)
    n = len(df)
    df = df[(df["price"] > 0) & (df["total_sqft"] > 0) & (df["bhk"] > 0)]
    report["dropped_invalid_nonpositive"] = n - len(df)
    n = len(df)
    ppsf = df["price"] * 1e5 / df["total_sqft"]
    df = df[(df["total_sqft"] / df["bhk"] >= 300) & (ppsf.between(MIN_PPSF, MAX_PPSF))]
    df = df[df["bath"].isna() | (df["bath"] <= df["bhk"] + 2)]
    report["dropped_outliers"] = n - len(df)
    df = df[RAW_FEATURES + [TARGET]].reset_index(drop=True)
    report["clean_rows"] = len(df)
    return df, report


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineered features (applied inside the saved pipeline)."""
    X = X.copy()
    X["log_sqft"] = np.log1p(X["total_sqft"])  # area is heavily right-skewed
    X["log_sqft_per_bhk"] = np.log1p(X["total_sqft"] / X["bhk"].clip(lower=1))
    X["bath_per_bhk"] = X["bath"] / X["bhk"].clip(lower=1)
    return X


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                             sparse_output=False))])
    return ColumnTransformer([("num", num, ["log_sqft", "log_sqft_per_bhk", "bhk", "bath", "balcony", "bath_per_bhk"]),
                              ("cat", cat, CAT_FEATURES)])


def feature_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)


def load_clean() -> tuple[pd.DataFrame, dict]:
    """Cleaned data from the processed CSV if present, otherwise clean the raw file now."""
    if CLEAN_PATH.exists():
        return pd.read_csv(CLEAN_PATH), {}
    return clean_data(load_raw())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download the dataset to data/raw")
    a = ap.parse_args()
    if a.download:
        print("Saved", download_dataset())
    d, r = clean_data(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(CLEAN_PATH, index=False)
    print({k: v for k, v in r.items() if k != "missing_values"}, "\nmissing:", r["missing_values"])
