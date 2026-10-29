import pytest
from src import history
from src.admin_dashboard import inr

INP = dict(location="Whitefield", area_type="Built-up Area", availability="Ready To Move", total_sqft=1200, bhk=2, bath=2, balcony=1)


def test_inr_format():
    assert inr(4500000) == "₹45,00,000" and inr(999) == "₹999" and inr(12345678) == "₹1,23,45,678" and inr(None) == "—"


@pytest.fixture
def db(tmp_path):
    d = tmp_path / "h.db"
    for i, loc in enumerate(["Whitefield", "Hebbal", "Whitefield", "Indira Nagar"]):
        history.add({**INP, "location": loc, "total_sqft": 1000 + i * 100}, 50 + i * 20, d)
    return d


def test_empty_db_summary(tmp_path):
    s = history.summary(db=tmp_path / "e.db")
    assert s["total"] == 0 and s["avg"] is None and history.recent_count(db=tmp_path / "e.db") == 0


def test_summary_and_recent(db):
    s = history.summary(db=db)
    assert (s["total"], s["min"], s["max"], s["avg"]) == (4, 50, 110, 80) and history.recent_count(7, db) == 4


def test_duplicate_submission_blocked(tmp_path):
    d = tmp_path / "d.db"
    assert history.add(INP, 60.0, d) and not history.add(INP, 60.0, d)
    assert len(history.records(db=d)) == 1


def test_search_sort_paginate(db):
    df, total = history.search("white", db=db); assert total == 2
    df, total = history.search(sort="Price: high to low", limit=2, offset=0, db=db)
    assert total == 4 and df.price_lakh.tolist() == [110, 90]
    df, _ = history.search(sort="Price: high to low", limit=2, offset=2, db=db); assert df.price_lakh.tolist() == [70, 50]


def test_sql_injection_safe(db):
    assert history.search("'; DROP TABLE history; --", db=db)[1] == 0
    assert history.search(sort="id; DROP TABLE history", db=db)[1] == 4  # unknown sort falls back to default
    assert history.summary(db=db)["total"] == 4


def test_date_filter_over_time_get_delete(db):
    assert history.summary("2999-01-01", None, db)["total"] == 0
    assert history.over_time("Day", db=db).predictions.sum() == 4
    r = history.get(1, db); assert r["location"] == "Whitefield"
    history.delete([1], db); assert history.get(1, db) is None and history.summary(db=db)["total"] == 3
