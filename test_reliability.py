"""Reliability checks on the REAL trained model (skipped if models/ has not been trained yet)."""
import math, random
import pytest
from src.admin_dashboard import inr
from src.predict import MODEL_PATH, ValidationError, format_inr, load_model, predict_price

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first: python -m src.train_model")
GOOD = dict(location="Whitefield", area_type="Super built-up Area", availability="Ready To Move", total_sqft=1200, bhk=2, bath=2, balcony=1)


@pytest.fixture(scope="module")
def bundle():
    return load_model()


def test_deterministic(bundle):
    assert len({predict_price(GOOD, bundle)["price_lakh"] for _ in range(5)}) == 1


def test_input_key_order_does_not_matter(bundle):
    keys = list(GOOD); random.Random(0).shuffle(keys)
    assert predict_price({k: GOOD[k] for k in keys}, bundle)["price_lakh"] == predict_price(GOOD, bundle)["price_lakh"]


def test_output_finite_and_units_consistent(bundle):
    r = predict_price(GOOD, bundle)
    assert math.isfinite(r["price_lakh"]) and r["price_lakh"] > 0
    assert r["price_inr"] == pytest.approx(r["price_lakh"] * 1e5)
    assert r["price_per_sqft"] == pytest.approx(r["price_inr"] / 1200)
    # customer page (lakh/crore text) and admin pages (full rupees) describe the same number
    assert format_inr(r["price_lakh"]).startswith("₹") and inr(r["price_inr"]).startswith("₹")


def test_bigger_home_costs_more_in_same_area(bundle):
    small, big = predict_price({**GOOD, "total_sqft": 800, "bhk": 2}, bundle), predict_price({**GOOD, "total_sqft": 2400, "bhk": 3, "bath": 3}, bundle)
    assert big["price_lakh"] > small["price_lakh"]


@pytest.mark.parametrize("bad", [{"total_sqft": float("nan")}, {"total_sqft": float("inf")}, {"bhk": -1}, {"balcony": -2},
                                 {"location": "Not A Real Place"}, {"availability": "Soon"}, {"bhk": 3, "total_sqft": 500}])
def test_bad_inputs_rejected(bundle, bad):
    with pytest.raises(ValidationError):
        predict_price({**GOOD, **bad}, bundle)
