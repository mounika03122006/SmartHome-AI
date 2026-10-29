"""SQLite prediction history. Stores only property inputs + result (no personal data).
All queries are parameterised; sort/period options come from whitelists."""
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

DB = Path(__file__).resolve().parents[1] / "history.db"
FMT = "%Y-%m-%d %H:%M:%S"
SORTS = {"Newest first": "created DESC, id DESC", "Oldest first": "created ASC, id ASC",
         "Price: high to low": "price_lakh DESC, id DESC", "Price: low to high": "price_lakh ASC, id ASC"}
PERIODS = {"Day": "%Y-%m-%d", "Week": "%Y-W%W", "Month": "%Y-%m"}
COLS = "id, created, location, area_type, availability, total_sqft, bhk, bath, balcony, price_lakh"


_READY: set = set()  # databases whose schema has already been checked in this process


def _run(db, sql, params=(), many=False, fetch=None):
    with closing(sqlite3.connect(db)) as c:
        key = str(db)
        if key not in _READY or not Path(db).exists():  # DDL only once per process (and if the file was deleted)
            c.execute("""CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT,
                location TEXT, area_type TEXT, availability TEXT, total_sqft REAL, bhk REAL, bath REAL,
                balcony REAL, price_lakh REAL)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_history_created ON history(created)")
            _READY.add(key)
        cur = c.executemany(sql, params) if many else c.execute(sql, params)
        out = cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None
        c.commit()
        return out


def _where(search=None, start=None, end=None):
    cl, p = [], []
    if search and search.strip():
        s = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cl.append("(location LIKE ? ESCAPE '\\' OR area_type LIKE ? ESCAPE '\\' OR CAST(id AS TEXT) = ?)")
        p += [f"%{s}%", f"%{s}%", search.strip()]
    if start:
        cl.append("date(created) >= date(?)"); p.append(str(start))
    if end:
        cl.append("date(created) <= date(?)"); p.append(str(end))
    return ("WHERE " + " AND ".join(cl)) if cl else "", p


def add(inp: dict, price_lakh: float, db=DB) -> bool:
    """Save a prediction. Returns False (and saves nothing) if an identical one was saved in the last 10 s."""
    now = datetime.now()
    vals = (inp["location"], inp["area_type"], inp["availability"], inp["total_sqft"], inp["bhk"], inp["bath"], inp["balcony"], price_lakh)
    dup = _run(db, "SELECT 1 FROM history WHERE location=? AND area_type=? AND availability=? AND total_sqft=? AND bhk=? AND bath=? "
                   "AND balcony=? AND price_lakh=? AND created>=?", vals + ((now - timedelta(seconds=10)).strftime(FMT),), fetch="one")
    if dup:
        return False
    _run(db, "INSERT INTO history(created,location,area_type,availability,total_sqft,bhk,bath,balcony,price_lakh) VALUES(?,?,?,?,?,?,?,?,?)",
         (now.strftime(FMT),) + vals)
    return True


def fetch(db=DB) -> pd.DataFrame:
    return records(db=db).sort_values("id", ascending=False)


def records(start=None, end=None, db=DB) -> pd.DataFrame:
    w, p = _where(None, start, end)
    rows = _run(db, f"SELECT {COLS} FROM history {w}", p, fetch="all")
    return pd.DataFrame(rows, columns=COLS.replace(" ", "").split(","))


def search(search=None, start=None, end=None, sort="Newest first", limit=10, offset=0, db=DB):
    w, p = _where(search, start, end)
    total = _run(db, f"SELECT COUNT(*) FROM history {w}", p, fetch="one")[0]
    rows = _run(db, f"SELECT {COLS} FROM history {w} ORDER BY {SORTS.get(sort, SORTS['Newest first'])} LIMIT ? OFFSET ?",
                p + [int(limit), int(offset)], fetch="all")
    return pd.DataFrame(rows, columns=COLS.replace(" ", "").split(",")), total


def summary(start=None, end=None, db=DB) -> dict:
    """One SQL aggregation: count/avg/max/min in the range, plus the all-time last-7-days count."""
    w, p = _where(None, start, end)
    since = (datetime.now() - timedelta(days=7)).strftime(FMT)
    r = _run(db, "SELECT COUNT(*), AVG(price_lakh), MAX(price_lakh), MIN(price_lakh), "
                 f"(SELECT COUNT(*) FROM history WHERE created >= ?) FROM history {w}", [since] + p, fetch="one")
    return {"total": r[0], "avg": r[1], "max": r[2], "min": r[3], "recent": r[4]}


def version(db=DB) -> tuple:
    """Cheap fingerprint (row count, highest id). It changes after every insert or delete, so cached
    dashboard data keyed on it is never stale."""
    return _run(db, "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM history", fetch="one")


def recent_count(days=7, db=DB) -> int:
    since = (datetime.now() - timedelta(days=days)).strftime(FMT)
    return _run(db, "SELECT COUNT(*) FROM history WHERE created >= ?", (since,), fetch="one")[0]


def over_time(period="Day", start=None, end=None, db=DB) -> pd.DataFrame:
    w, p = _where(None, start, end)
    rows = _run(db, f"SELECT strftime(?, created) AS period, COUNT(*) FROM history {w} GROUP BY period ORDER BY period",
                [PERIODS.get(period, PERIODS["Day"])] + p, fetch="all")
    return pd.DataFrame(rows, columns=["period", "predictions"])


def get(pid: int, db=DB):
    r = _run(db, f"SELECT {COLS} FROM history WHERE id = ?", (int(pid),), fetch="one")
    return dict(zip(COLS.replace(" ", "").split(","), r)) if r else None


def delete(ids: list[int], db=DB):
    _run(db, "DELETE FROM history WHERE id=?", [(int(i),) for i in ids], many=True)


def clear(db=DB):
    _run(db, "DELETE FROM history")
