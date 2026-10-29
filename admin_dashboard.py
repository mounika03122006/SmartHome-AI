"""Admin dashboard pages for SmartHome AI (Streamlit). Every figure comes from history.db, the saved model
reports, or the raw dataset file. Nothing here retrains the model or modifies data."""
from __future__ import annotations
import html, math, re, sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd, streamlit as st
from src import auth, history, services
from src.data_preprocessing import RAW_FEATURES, RAW_PATH, DataError, load_raw
from src.predict import MODEL_PATH, ModelLoadError, ModelNotFoundError

APP_NAME, APP_VERSION = "SmartHome AI", "1.0.0"
CUSTOMER_SITE = "Customer Site"


class _LazyPlotly:
    """plotly.express is imported on first chart use, so customers (who never see a chart) don't pay for it."""
    def __getattr__(self, name):
        import plotly.express as _px
        return getattr(_px, name)


px = _LazyPlotly()
NAVY, TEAL, SLATE, INK, BORDER = "#142B4A", "#0F9D91", "#718096", "#172B4D", "#E2E8F0"

ICONS = {  # Feather-style SVG paths (stroke = currentColor)
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "up": '<line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/>',
    "down": '<line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
}


def svg(name, size=20):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


ADMIN_CSS = """<style>
.stApp{background:#F4F7FB;color:#172B4D;font-family:'Segoe UI',Inter,system-ui,-apple-system,sans-serif}
.stApp button,.stApp input,.stApp textarea{font-family:inherit}
.stApp .block-container{max-width:100%;padding:1.8rem 2.6rem 3rem 2.6rem}
header[data-testid="stHeader"]{background:transparent}
/* sidebar */
[data-testid="stSidebar"]{background:#142B4A;border-right:none}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:.5rem}
[data-testid="stSidebar"] *{color:#CBD5E1}
.brand-box{display:flex;align-items:center;gap:.8rem;padding:.9rem .4rem 1.2rem .4rem}
.brand-logo{width:42px;height:42px;border-radius:12px;background:#0F9D91;display:flex;align-items:center;justify-content:center;color:#fff!important}
.brand-logo svg{stroke:#fff}
.brand-name{font-weight:700;font-size:1.15rem;color:#fff!important;line-height:1.2}
.brand-sub{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#7F95B2!important}
.nav-label{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#7F95B2!important;padding:.4rem .6rem}
.nav-divider{height:1px;background:rgba(255,255,255,.10);margin:.9rem .4rem .6rem .4rem}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button{background:transparent;border:0;border-left:3px solid transparent;color:#CBD5E1;
  border-radius:12px;padding:.55rem .85rem;font-weight:500;justify-content:flex-start;box-shadow:none;transition:background .15s,color .15s;min-height:2.6rem}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button *{color:inherit}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button>div{justify-content:flex-start;width:100%}
.stApp [data-testid="stSidebar"] [class*="st-key-nav_"] button:hover{background:rgba(255,255,255,.08);color:#fff}
.stApp [data-testid="stSidebar"] .st-key-nav_logout button{color:#F4B6B6}
/* page header */
.ph{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:.4rem}
.ph .hi{color:#0F9D91;font-weight:600;font-size:.9rem;margin-bottom:.15rem}
.ph h1{margin:0;font-size:2rem;font-weight:700;color:#172B4D;letter-spacing:-.02em;line-height:1.15;padding:0}
.ph .sub{color:#718096;font-size:.95rem;margin-top:.25rem}
.chip{display:inline-flex;align-items:center;gap:.55rem;background:#fff;border:1px solid #E2E8F0;border-radius:999px;padding:.3rem .9rem .3rem .35rem;font-weight:600;font-size:.88rem;color:#172B4D}
.chip i{font-style:normal;width:30px;height:30px;border-radius:50%;background:#142B4A;color:#fff;display:flex;align-items:center;justify-content:center;font-size:.8rem}
/* cards */
[class*="st-key-card"]{background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:1.2rem 1.4rem;box-shadow:0 1px 3px rgba(20,43,74,.05)}
.ct{font-weight:600;font-size:1.05rem;color:#172B4D}.cs{color:#718096;font-size:.85rem;margin-bottom:.4rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin:1rem 0 1.1rem 0}
.kpi{background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:1.15rem 1.3rem;box-shadow:0 1px 3px rgba(20,43,74,.05);transition:transform .18s,box-shadow .18s}
.kpi:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(20,43,74,.10)}
.kpi-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem}
.kpi-label{font-size:.72rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#718096}
.kpi-ico{width:38px;height:38px;border-radius:11px;background:#EEF2F8;color:#142B4A;display:flex;align-items:center;justify-content:center}
.kpi-accent .kpi-ico{background:#E3F6F4;color:#0F9D91}
.kpi-val{font-size:clamp(1.35rem,1.9vw,1.85rem);font-weight:700;color:#172B4D;letter-spacing:-.02em;white-space:nowrap;line-height:1.2}
.kpi-sub{color:#718096;font-size:.82rem;margin-top:.3rem}
.strip{display:flex;flex-wrap:wrap;gap:1.2rem 2.6rem;align-items:center}
.strip .it small{display:block;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:#718096;font-weight:600}
.strip .it b{font-size:1rem;color:#172B4D}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:.4rem}
.dot.ok{background:#0F9D91;box-shadow:0 0 0 3px rgba(15,157,145,.2)}.dot.bad{background:#E53E3E}
.tbl-wrap{overflow-x:auto;border-radius:12px;border:1px solid #EDF2F7}
table.t{width:100%;min-width:900px;border-collapse:collapse;font-size:.9rem}
table.t th{background:#F8FAFC;color:#718096;font-weight:600;font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;text-align:left;padding:.85rem 1rem;border-bottom:1px solid #E2E8F0}
table.t td{padding:.95rem 1rem;border-bottom:1px solid #EDF2F7;color:#172B4D;white-space:nowrap}
table.t tr:last-child td{border-bottom:none}table.t tbody tr:hover td{background:#F8FBFD}
table.t .r{text-align:right}table.t td.price{font-weight:700;color:#0F9D91}
.empty{text-align:center;color:#718096;padding:2.2rem 1rem;font-size:.95rem}
/* widgets */
.stApp div.stButton>button,.stApp div[data-testid="stFormSubmitButton"]>button{border-radius:10px;font-weight:600;border:1px solid #E2E8F0;background:#fff;color:#172B4D;transition:all .15s}
.stApp div.stButton>button:hover{border-color:#0F9D91;color:#0F9D91}
.stApp div.stButton>button[kind="primary"],.stApp div[data-testid="stFormSubmitButton"]>button[kind="primary"]{background:#0F9D91;border-color:#0F9D91;color:#fff}
.stApp div.stButton>button[kind="primary"]:hover{background:#0B857A;color:#fff}
.stApp button[data-testid="stBaseButton-primaryFormSubmit"]{background:#0F9D91!important;border-color:#0F9D91!important;color:#fff!important}
.stApp button[data-testid="stBaseButton-primaryFormSubmit"]:hover{background:#0B857A!important}
.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="input"]{border-radius:10px;border-color:#E2E8F0;background:#fff}
[data-testid="stMetric"]{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:.9rem 1.1rem}
[data-testid="stForm"]{background:#fff;border:1px solid #E2E8F0;border-radius:16px}
.kv{background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:1.1rem 1.3rem;line-height:1.9}
@media(max-width:800px){.stApp .block-container{padding:1rem}.ph h1{font-size:1.5rem}}
</style>"""


WORKSPACE = [("Dashboard Overview", ":material/dashboard:"), ("Prediction Analytics", ":material/monitoring:"),
             ("Prediction History", ":material/history:")]
ACCOUNT = [("Admin Settings", ":material/settings:"), (CUSTOMER_SITE, ":material/storefront:")]
_IDX = {n: i for i, (n, _) in enumerate(WORKSPACE + ACCOUNT)}


def _goto(page: str):
    st.session_state.admin_page = page


def _logout_cb():
    st.session_state.update(admin=False, view="Home")


def sidebar_nav() -> str:
    """Sidebar: brand + WORKSPACE + ACCOUNT. One navigation system: st.session_state.admin_page (set by button callbacks)."""
    st.session_state.setdefault("admin_page", WORKSPACE[0][0])
    cur = st.session_state.admin_page
    if cur in _IDX:
        st.sidebar.markdown(f'<style>.stApp [data-testid="stSidebar"] .st-key-nav_{_IDX[cur]} button,'
                            f'.stApp [data-testid="stSidebar"] .st-key-nav_{_IDX[cur]} button:hover'
                            '{background:rgba(15,157,145,.22);color:#fff;border-left-color:#0F9D91}'
                            f'.stApp [data-testid="stSidebar"] .st-key-nav_{_IDX[cur]} button span{{color:#2DD4BF}}</style>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f'<div class="brand-box"><div class="brand-logo">{svg("home", 22)}</div><div><div class="brand-name">SmartHome AI</div>'
                    '<div class="brand-sub">Real Estate Intelligence</div></div></div><div class="nav-label">Workspace</div>', unsafe_allow_html=True)
        for name, icon in WORKSPACE:
            st.button(name, key=f"nav_{_IDX[name]}", icon=icon, on_click=_goto, args=(name,), width="stretch")
        st.markdown('<div class="nav-divider"></div><div style="height:max(.5rem,calc(100vh - 590px))"></div><div class="nav-label">Account</div>', unsafe_allow_html=True)
        for name, icon in ACCOUNT:
            st.button(name, key=f"nav_{_IDX[name]}", icon=icon, on_click=_goto, args=(name,), width="stretch")
        st.button("Log out", key="nav_logout", icon=":material/logout:", on_click=_logout_cb, width="stretch")
    return cur


def inr(rupees) -> str:
    """Indian digit grouping: 4500000 -> ₹45,00,000."""
    if rupees is None or (isinstance(rupees, float) and math.isnan(rupees)):
        return "—"
    n = int(round(rupees)); s = str(abs(n))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return f"{'-' if n < 0 else ''}₹{s}"


def money(lakh):
    """(display, full). Full Indian format; compact crore form only from ₹10 crore up (full value kept for the tooltip)."""
    if lakh is None:
        return "—", ""
    full = inr(lakh * 1e5)
    return (f"₹{lakh / 100:,.2f} Cr", full) if lakh * 1e5 >= 1e8 else (full, full)


def _style(fig, h=320):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=0, r=0, t=10, b=0), font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12),
                      hoverlabel=dict(bgcolor="#fff", font_size=12), showlegend=False)
    fig.update_xaxes(gridcolor="#EDF2F7", zeroline=False, title_font=dict(color=SLATE, size=12), linecolor=BORDER)
    fig.update_yaxes(gridcolor="#EDF2F7", zeroline=False, title_font=dict(color=SLATE, size=12))
    return fig


_fig = _style  # backwards-compatible name used by the model page


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except sqlite3.Error:
        st.error("The prediction database could not be read. Please try again.")
        st.stop()


def header(title: str, subtitle: str = "", greeting: bool = True):
    name = html.escape(auth.get_username())
    a, b, c = st.columns([7, 1.3, 1.9], vertical_alignment="bottom")
    hi = f'<div class="hi">Welcome back, {name}</div>' if greeting else ""
    a.markdown(f'<div class="ph"><div>{hi}<h1>{title}</h1><div class="sub">{html.escape(subtitle)}'
               f'{" · " if subtitle else ""}{date.today():%A, %d %B %Y}</div></div></div>', unsafe_allow_html=True)
    if b.button("Refresh", key=f"refresh_{title}", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear(); st.rerun()
    c.markdown(f'<div style="text-align:right"><span class="chip"><i>{html.escape(name[:1].upper())}</i>{name}</span></div>', unsafe_allow_html=True)
    st.write("")


@contextmanager
def card(key, title="", sub=""):
    with st.container(key=f"card_{key}"):
        if title:
            st.markdown(f'<div class="ct">{title}</div><div class="cs">{sub}</div>', unsafe_allow_html=True)
        yield


def date_filter(key: str):
    """Compact toolbar: preset + start + end. Returns (start, end) or (None, None) for all time."""
    today = date.today()
    with card(f"toolbar_{key}"):
        c1, c2, c3, _ = st.columns([2, 2, 2, 4], vertical_alignment="bottom")
        preset = c1.selectbox("Date range", ["All Time", "Today", "Last 7 Days", "Last 30 Days", "This Month", "Custom Range"], key=f"{key}_preset")
        rng = {"Today": (today, today), "Last 7 Days": (today - timedelta(days=6), today),
               "Last 30 Days": (today - timedelta(days=29), today), "This Month": (today.replace(day=1), today)}.get(preset)
        if preset == "Custom Range":
            s = c2.date_input("Start date", today - timedelta(days=30), key=f"{key}_s", max_value=today)
            e = c3.date_input("End date", today, key=f"{key}_e", max_value=today)
            if s > e:
                st.error("Start date must be on or before the end date."); st.stop()
            return s, e
        if rng:
            c2.date_input("Start date", rng[0], disabled=True, key=f"{key}_s_{preset}")
            c3.date_input("End date", rng[1], disabled=True, key=f"{key}_e_{preset}")
            return rng
        c2.text_input("Start date", "All records", disabled=True, key=f"{key}_sa"); c3.text_input("End date", "All records", disabled=True, key=f"{key}_ea")
    return None, None


def empty_state(msg="No predictions recorded for this period yet. They appear here as customers use the Predict page."):
    st.markdown(f'<div class="empty">{html.escape(msg)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ shared pieces
def kpi(label, value, sub, icon, tip="", accent=False):
    return (f'<div class="kpi{" kpi-accent" if accent else ""}" title="{html.escape(tip)}"><div class="kpi-top"><span class="kpi-label">{label}</span>'
            f'<span class="kpi-ico">{svg(icon)}</span></div><div class="kpi-val">{html.escape(value)}</div><div class="kpi-sub">{sub}</div></div>')


def table_html(df: pd.DataFrame) -> str:
    cols = [("ID", "id", ""), ("Date & time", "when", ""), ("Location", "location", ""), ("Area type", "area_type", ""),
            ("Availability", "availability", ""), ("Area (sq ft)", "sqft", "r"), ("Beds", "bhk", "r"), ("Baths", "bath", "r"),
            ("Balconies", "balcony", "r"), ("Predicted price", "price", "r price")]
    d = pd.DataFrame({"id": "#" + df.id.astype(str), "when": pd.to_datetime(df.created).dt.strftime("%d %b %Y, %I:%M %p"),
                      "location": df.location, "area_type": df.area_type, "availability": df.availability,
                      "sqft": df.total_sqft.map(lambda v: f"{v:,.0f}"), "bhk": df.bhk.astype(int), "bath": df.bath.astype(int),
                      "balcony": df.balcony.astype(int), "price": df.price_lakh.map(lambda v: inr(v * 1e5))})
    th = "".join(f'<th class="{c.split()[0] if c else ""}">{h}</th>' for h, _, c in cols)
    rows = "".join("<tr>" + "".join(f'<td class="{c}">{html.escape(str(r[k]))}</td>' for _, k, c in cols) + "</tr>" for _, r in d.iterrows())
    return f'<div class="tbl-wrap"><table class="t"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>'


def _display(df: pd.DataFrame) -> pd.DataFrame:  # plain table for the History page (sortable)
    return pd.DataFrame({"ID": df.id, "Date & time": df.created, "Location": df.location, "Area type": df.area_type,
                         "Availability": df.availability, "Area (sq ft)": df.total_sqft.round(0).astype(int),
                         "Bedrooms": df.bhk.astype(int), "Bathrooms": df.bath.astype(int), "Balconies": df.balcony.astype(int),
                         "Predicted price": df.price_lakh.map(lambda v: inr(v * 1e5))})


def chart_activity(rec, period="Day", start=None, end=None):
    t = pd.to_datetime(rec.created)
    if period == "Day":
        s = t.dt.normalize().value_counts().sort_index()
        lo, hi = pd.Timestamp(start) if start else s.index.min(), pd.Timestamp(end) if end else s.index.max()
        s = s.reindex(pd.date_range(lo, hi), fill_value=0)
        df = pd.DataFrame({"date": s.index, "predictions": s.values})
    else:
        s = t.dt.strftime(history.PERIODS[period]).value_counts().sort_index()
        df = pd.DataFrame({"date": s.index, "predictions": s.values})
    f = px.area(df, x="date", y="predictions", line_shape="spline" if period == "Day" else "linear", markers=len(df) < 40,
                labels={"date": period, "predictions": "Predictions"})
    f.update_traces(line_color=TEAL, fillcolor="rgba(15,157,145,.14)", marker_color=TEAL)
    f = _style(f).update_yaxes(rangemode="tozero", dtick=1 if df.predictions.max() < 6 else None)
    if period == "Day":
        f.update_xaxes(type="date", tickformat="%d %b", dtick="D1" if len(df) < 15 else None)
        if len(df) == 1:  # a single day: pad the range so the axis shows whole days, not sub-second ticks
            d0 = pd.Timestamp(df.date.iloc[0]); f.update_xaxes(range=[d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=1)])
    return f


def chart_dist(rec):
    f = px.histogram(rec, x="price_lakh", nbins=24, labels={"price_lakh": "Predicted price (₹ lakh)", "count": "Predictions"})
    f.update_traces(marker_color=NAVY, marker_line_width=0)
    return _style(f).update_yaxes(title="Predictions")


def chart_location(rec):
    g = rec.groupby("location").price_lakh.agg(["mean", "count"]).reset_index().sort_values("count", ascending=False).head(10).sort_values("mean")
    f = px.bar(g, x="mean", y="location", orientation="h", hover_data={"count": True}, labels={"mean": "Average price (₹ lakh)", "location": "", "count": "Predictions"})
    f.update_traces(marker_color=TEAL)
    return _style(f)


def chart_scatter(rec):
    f = px.scatter(rec, x="total_sqft", y="price_lakh", hover_data=["location", "bhk"], opacity=.75,
                   labels={"total_sqft": "Area (sq ft)", "price_lakh": "Predicted price (₹ lakh)", "bhk": "Bedrooms"})
    f.update_traces(marker=dict(color=TEAL, size=9, line=dict(width=1, color="#fff")))
    return _style(f)


def chart_bedrooms(rec):
    bed = rec.bhk.astype(int).value_counts().sort_index().reset_index(); bed.columns = ["Bedrooms", "Predictions"]
    f = px.bar(bed, x="Bedrooms", y="Predictions"); f.update_traces(marker_color=NAVY)
    return _style(f).update_xaxes(type="category")


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ------------------------------------------------------------------ Dashboard Overview
def _open_history():
    st.session_state.admin_page = "Prediction History"


@st.cache_data(show_spinner=False, max_entries=32)
def _overview_data(start, end, ver, today):
    """Everything the overview needs, computed once per (date range, data version, day). `ver` changes on every
    insert/delete, so a new prediction is reflected immediately and nothing stale is shown."""
    s = history.summary(start, end)
    rec = history.records(start, end)
    if rec.empty:
        return s, None
    figs = {"activity": chart_activity(rec, "Day", start, end), "dist": chart_dist(rec),
            "loc": chart_location(rec), "scat": chart_scatter(rec)}
    return s, {"figs": figs, "table": table_html(rec.sort_values(["created", "id"], ascending=False).head(5))}


@st.cache_data(show_spinner=False, max_entries=32)
def _analytics_data(start, end, ver, period, today):
    rec = history.records(start, end)
    if rec.empty:
        return None
    return {"activity": chart_activity(rec, period, start, end), "dist": chart_dist(rec), "scat": chart_scatter(rec),
            "loc": chart_location(rec), "bed": chart_bedrooms(rec)}


def overview():
    header("Dashboard Overview", "Monitor property predictions and model insights.")
    start, end = date_filter("ov")
    s, extra = _safe(lambda: _overview_data(start, end, history.version(), date.today()))
    has = extra is not None
    avg, mx, mn = (money(s[k]) for k in ("avg", "max", "min"))
    st.markdown('<div class="kpis">' + kpi("Total predictions", f"{s['total']:,}", "All recorded predictions" if not start else "In selected range", "home")
                + kpi("Average predicted price", avg[0], "Mean of recorded estimates", "trend", avg[1])
                + kpi("Highest predicted price", mx[0], "Largest estimate", "up", mx[1], True)
                + kpi("Lowest predicted price", mn[0], "Smallest estimate", "down", mn[1])
                + kpi("Recent activity", f"{s['recent']:,}", "Predictions in the last 7 days", "clock", accent=True) + '</div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([3, 2])
    with a, card("activity", "Prediction activity", "Daily predictions in the selected range"):
        show(extra["figs"]["activity"]) if has else empty_state()
    with b, card("dist", "House price distribution", "How predicted prices are spread"):
        show(extra["figs"]["dist"]) if has else empty_state()
    st.write("")
    a, b = st.columns(2)
    with a, card("loc", "Average predicted price by location", "Top 10 locations by number of predictions"):
        show(extra["figs"]["loc"]) if has else empty_state()
    with b, card("scat", "Area vs predicted price", "Each point is one prediction"):
        show(extra["figs"]["scat"]) if has else empty_state()
    st.write("")
    with card("latest"):
        a, b = st.columns([5, 1], vertical_alignment="center")
        a.markdown('<div class="ct">Latest predictions</div><div class="cs">Recently generated property price estimates</div>', unsafe_allow_html=True)
        b.button("View all", key="viewall", on_click=_open_history, icon=":material/arrow_forward:", icon_position="right", width="stretch")
        st.markdown(extra["table"], unsafe_allow_html=True) if has else empty_state()


# ------------------------------------------------------------------ Prediction Analytics
def analytics():
    header("Prediction Analytics", "Explore the predictions customers have requested.")
    start, end = date_filter("an")
    period = st.session_state.get("an_period", "Day")
    d = _safe(lambda: _analytics_data(start, end, history.version(), period, date.today()))
    if d is None:
        with card("empty"):
            empty_state()
        return
    st.caption("Prices in ₹ lakh (1 lakh = ₹1,00,000). Charts use only recorded predictions.")
    a, b = st.columns(2)
    with a, card("a1", "Predictions over time"):
        st.selectbox("Group by", list(history.PERIODS), key="an_period", label_visibility="collapsed")
        show(d["activity"])
    with b, card("a2", "House price distribution"):
        show(d["dist"])
    st.write("")
    a, b = st.columns(2)
    with a, card("a3", "Area vs predicted price"):
        show(d["scat"])
    with b, card("a4", "Average price by location", "Top 10 by volume"):
        show(d["loc"])
    st.write("")
    with card("a5", "Bedroom distribution"):
        show(d["bed"])


# ------------------------------------------------------------------ Prediction History
@st.dialog("Prediction details")
def details_dialog(pid: int):
    r = _safe(history.get, pid)
    if not r:
        st.error("That record no longer exists."); return
    st.metric("Predicted price", inr(r["price_lakh"] * 1e5))
    st.caption(f"≈ {inr(r['price_lakh'] * 1e5 / r['total_sqft'])} per sq ft")
    st.table(pd.DataFrame({"Field": ["Prediction ID", "Date & time", "Location", "Area type", "Availability", "Area (sq ft)",
                                     "Bedrooms", "Bathrooms", "Balconies"],
                           "Value": [r["id"], r["created"], r["location"], r["area_type"], r["availability"], f"{r['total_sqft']:,.0f}",
                                     int(r["bhk"]), int(r["bath"]), int(r["balcony"])]}).astype(str).set_index("Field"))


@st.dialog("Confirm deletion")
def delete_dialog(ids: list | None):
    st.warning("Delete ALL prediction records? This cannot be undone." if ids is None
               else f"Delete prediction #{ids[0]}? This cannot be undone.")
    a, b = st.columns(2)
    if a.button("Yes, delete", type="primary", width="stretch"):
        _safe(history.clear) if ids is None else _safe(history.delete, ids)
        st.rerun()
    if b.button("Cancel", width="stretch"):
        st.rerun()


def history_page():
    header("Prediction History", "Search, filter and review every prediction.", greeting=False)
    c1, c2 = st.columns([3, 2])
    q = c1.text_input("Search", placeholder="Location, area type or prediction ID")
    sort = c2.selectbox("Sort by", list(history.SORTS))
    start, end = date_filter("hi")
    size = st.selectbox("Rows per page", [10, 25, 50], key="hi_size")
    total = _safe(history.search, q, start, end, sort, 1, 0)[1]
    if total == 0:
        empty_state("No prediction records match your filters." if (q or start) else
                    "No predictions have been recorded yet."); return
    pages = max(1, math.ceil(total / size))
    page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
    df, _ = _safe(history.search, q, start, end, sort, size, (page - 1) * size)
    st.caption(f"Showing {(page - 1) * size + 1}–{min(page * size, total)} of {total:,} records")
    st.dataframe(_display(df), hide_index=True, width="stretch")
    pid = st.selectbox("Select a prediction ID", df.id.tolist())
    a, b, _ = st.columns([1, 1, 3])
    if a.button("View Details", type="primary", width="stretch"):
        details_dialog(int(pid))
    if b.button("Delete…", width="stretch"):
        delete_dialog([int(pid)])
    with st.expander("Danger zone"):
        if st.button("Delete all prediction history"):
            delete_dialog(None)


# ------------------------------------------------------------------ ML Model Information
def model_page():
    header("Model Performance", "Evaluation of the trained price model.", greeting=False)
    rep = services.cached_reports()
    if rep is None:
        st.info("No trained model reports found. Train the model with `python -m src.train_model`."); return
    comp, pred, imp, meta = rep
    try:
        services.cached_model(); status = "✅ Loaded and ready"
    except (ModelNotFoundError, ModelLoadError):
        status = "❌ Model file missing or unreadable"
    trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%d %b %Y, %H:%M") if MODEL_PATH.exists() else "Not recorded"
    a, b = st.columns(2)
    a.markdown(f'<div class="kv"><b>Model:</b> {meta["selected_model"]}<br><b>Algorithm:</b> scikit-learn pipeline (feature engineering → '
               f'one-hot/scaling → {meta["selected_model"]} on log price)<br><b>Status:</b> {status}<br>'
               f'<b>Model file last modified:</b> {trained}<br><b>Version:</b> Not recorded</div>', unsafe_allow_html=True)
    b.markdown(f'<div class="kv"><b>Training data:</b> Bengaluru House Data (cleaned)<br><b>Training rows:</b> {meta["n_train"]:,}<br>'
               f'<b>Test rows:</b> {meta["n_test"]:,}<br><b>Random seed:</b> {meta["seed"]}<br>'
               f'<b>Input features:</b> location, area type, availability, area (sq ft), bedrooms, bathrooms, balconies</div>', unsafe_allow_html=True)
    st.subheader("Evaluation metrics (held-out test set)")
    if pred.empty:
        st.info("Not calculated")
    else:
        mse = float(np.mean(pred.residual ** 2)); mae = float(np.mean(np.abs(pred.residual)))
        ss = float(((pred.actual - pred.actual.mean()) ** 2).sum()); r2 = 1 - float((pred.residual ** 2).sum()) / ss
        m = st.columns(4)
        m[0].metric("MAE", f"{mae:,.2f} lakh"); m[1].metric("MSE", f"{mse:,.1f} lakh²")
        m[2].metric("RMSE", f"{math.sqrt(mse):,.2f} lakh"); m[3].metric("R²", f"{r2:.3f}")
        st.caption("Computed from the saved test-set predictions of the selected model; the model was not retrained.")
    st.subheader("Model comparison")
    st.dataframe(comp.round(3), hide_index=True, width="stretch")
    a, b = st.columns(2)
    lim = max(pred.actual.max(), pred.predicted.max())
    f = px.scatter(pred, x="actual", y="predicted", opacity=.5, log_x=True, log_y=True, title="Actual vs predicted (test set)",
                   labels={"actual": "Actual (₹ lakh)", "predicted": "Predicted (₹ lakh)"})
    f.add_shape(type="line", x0=10, y0=10, x1=lim, y1=lim, line=dict(color="#1E3A8A", dash="dash"))
    a.plotly_chart(_fig(f), width="stretch")
    b.plotly_chart(_fig(px.histogram(pred, x="residual", nbins=80, title="Residuals (actual − predicted)",
                                     labels={"residual": "Residual (₹ lakh)"})), width="stretch")
    st.plotly_chart(_fig(px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                                title="Feature importance (permutation, drop in R²)")), width="stretch")
    st.caption("Importance shows what the model relies on, not what causes price changes.")


# ------------------------------------------------------------------ Dataset Overview
@st.cache_data(show_spinner="Reading dataset…")
def _dataset(mtime: float):
    df = load_raw()
    cols = pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values, "Missing": df.isna().sum().values,
                         "Missing %": (df.isna().mean() * 100).round(1).values, "Unique values": df.nunique().values})
    return len(df), df.shape[1], int(df.duplicated().sum()), cols, df.describe().T.round(2)


def dataset_page():
    header("Dataset Overview", "Structure and quality of the training data.", greeting=False)
    if not RAW_PATH.exists():
        st.info("The dataset file is not available to the app. Place it at `data/raw/bengaluru_house_prices.csv`."); return
    try:
        rows, ncols, dups, cols, desc = _dataset(RAW_PATH.stat().st_mtime)
    except DataError as e:
        st.error(str(e)); return
    c = st.columns(4)
    c[0].metric("Dataset", "Bengaluru House Data"); c[1].metric("Rows", f"{rows:,}")
    c[2].metric("Columns", ncols); c[3].metric("Duplicate rows", f"{dups:,}")
    st.caption(f"Raw file last modified: {datetime.fromtimestamp(RAW_PATH.stat().st_mtime):%d %b %Y, %H:%M}. Prices are in lakh ₹.")
    st.subheader("Columns"); st.dataframe(cols, hide_index=True, width="stretch")
    st.subheader("Numerical statistics"); st.dataframe(desc, width="stretch")
    st.caption("This page reads the raw file only and never modifies it.")


# ------------------------------------------------------------------ Admin Settings
def settings_page():
    header("Admin Settings", "Profile, password and session.", greeting=False)
    a, b = st.columns(2)
    a.markdown(f'<div class="kv"><b>Administrator:</b> {auth.get_username()}<br><b>Role:</b> Administrator</div>', unsafe_allow_html=True)
    b.markdown(f'<div class="kv"><b>Application:</b> {APP_NAME}<br><b>Version:</b> {APP_VERSION}</div>', unsafe_allow_html=True)
    st.subheader("Change password")
    with st.form("pw"):
        old = st.text_input("Current password", type="password")
        new = st.text_input(f"New password (min {auth.MIN_PASSWORD} characters)", type="password")
        new2 = st.text_input("Repeat new password", type="password")
        if st.form_submit_button("Update password", type="primary"):
            try:
                if len(new) < auth.MIN_PASSWORD: raise ValueError(f"New password must be at least {auth.MIN_PASSWORD} characters.")
                if new != new2: raise ValueError("New passwords do not match.")
                if new == old: raise ValueError("New password must differ from the current one.")
                auth.change_password(old, new); st.success("Password updated.")
            except (ValueError, PermissionError) as e:
                st.error(str(e))
    st.subheader("Session")
    if st.button("Log out"):
        logout()


def logout():
    st.session_state.update(admin=False, view="Home"); st.rerun()


NAV = {"Dashboard Overview": overview, "Prediction Analytics": analytics, "Prediction History": history_page,
       "Admin Settings": settings_page}  # model_page / dataset_page remain available but are not in the menu


def render(page: str):
    if not st.session_state.get("admin"):  # server-side guard: never render admin pages for non-admins
        st.error("Please sign in as admin."); st.stop()
    NAV.get(page, overview)()