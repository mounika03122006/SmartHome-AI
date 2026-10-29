"""Tests use a SMALL SYNTHETIC frame (testing only, not real prices) so they run fast without the real dataset."""
import numpy as np, pandas as pd, pytest
from src import history
from src.data_preprocessing import DataError, clean_data, load_raw, parse_bhk, parse_sqft
from src.predict import ModelNotFoundError, ValidationError, ModelLoadError, load_model, predict_price, validate_input
from src.train_model import train


def synthetic_raw(n=400, seed=0):
    r = np.random.default_rng(seed)
    sqft = r.integers(600, 3000, n); bhk = r.integers(1, 5, n)
    loc = r.choice(["A", "B", "C"], n)
    price = sqft * (4 + 2 * (loc == "A")) / 100 + r.normal(0, 3, n) + 10
    return pd.DataFrame({"area_type": r.choice(["Built-up Area", "Plot Area"], n), "availability": "Ready To Move",
                         "location": loc, "size": [f"{b} BHK" for b in bhk], "total_sqft": sqft.astype(str),
                         "bath": np.minimum(bhk, 3), "balcony": r.integers(0, 3, n), "price": price.round(2)})


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    raw = synthetic_raw()
    clean, rep = clean_data(raw)
    train(clean, d, d, rep, fast=True, cv=2)
    return load_model(d / "house_price_model.joblib")


GOOD = dict(location="A", area_type="Built-up Area", availability="Ready To Move", total_sqft=1200, bhk=2, bath=2, balcony=1)


def test_parsers():
    assert parse_sqft("1000 - 1200") == 1100 and parse_sqft("1500") == 1500 and np.isnan(parse_sqft("34Sq. Meter"))
    assert parse_bhk("4 Bedroom") == 4


def test_cleaning_removes_bad_rows():
    raw = synthetic_raw(50)
    bad = raw.iloc[:3].copy(); bad.loc[:, "price"] = [-5, 0, 10]
    out, rep = clean_data(pd.concat([raw, bad, raw.iloc[:2]]))
    assert rep["duplicate_rows"] >= 2
    assert (out.price > 0).all() and not out.duplicated().any()


def test_missing_and_bad_files(tmp_path):
    with pytest.raises(DataError): load_raw(tmp_path / "nope.csv")
    (tmp_path / "e.csv").write_text("a,b\n")
    with pytest.raises(DataError): load_raw(tmp_path / "e.csv")
    (tmp_path / "w.csv").write_text("a,b\n1,2\n")
    with pytest.raises(DataError): load_raw(tmp_path / "w.csv")


def test_missing_model(tmp_path):
    with pytest.raises(ModelNotFoundError): load_model(tmp_path / "x.joblib")


def test_corrupt_model(tmp_path):
    (tmp_path / "bad.joblib").write_text("garbage")
    with pytest.raises(ModelLoadError): load_model(tmp_path / "bad.joblib")


def test_valid_prediction(bundle):
    r = predict_price(GOOD, bundle)
    assert np.isfinite(r["price_lakh"]) and r["price_lakh"] > 0
    assert r["price_per_sqft"] == pytest.approx(r["price_lakh"] * 1e5 / 1200)


def test_prediction_changes_with_input(bundle):
    assert predict_price(GOOD, bundle)["price_lakh"] != predict_price({**GOOD, "total_sqft": 2800, "bhk": 3, "bath": 3}, bundle)["price_lakh"]


@pytest.mark.parametrize("change", [{"location": "Nowhere"}, {"total_sqft": -5}, {"total_sqft": "abc"}, {"bhk": 0}, {"bath": 9}, {"area_type": None}])
def test_invalid_inputs(bundle, change):
    with pytest.raises(ValidationError): predict_price({**GOOD, **change}, bundle)


def test_history(tmp_path):
    db = tmp_path / "h.db"
    history.add(GOOD, 55.5, db)
    h = history.fetch(db); assert len(h) == 1
    history.delete(h.id.tolist(), db); assert history.fetch(db).empty
