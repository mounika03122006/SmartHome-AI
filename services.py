"""Cached, shared loaders for the Streamlit app. Caches are keyed by file modification time, so retraining the
model (or regenerating reports) is picked up automatically without a stale model being served."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from src.evaluate_model import REPORTS, load_reports
from src.predict import MODEL_PATH, load_model


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _model(mtime: float):
    return load_model()  # exceptions are not cached, so a missing model is retried on the next run


def cached_model():
    """One loaded model shared by every page and every session."""
    return _model(_mtime(MODEL_PATH))


@st.cache_data(show_spinner=False)
def _reports(mtime: float):
    return load_reports()


def cached_reports():
    return _reports(_mtime(REPORTS / "metadata.json"))
