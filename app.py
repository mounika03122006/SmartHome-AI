"""SmartHome AI - Streamlit app.  Run: streamlit run app.py
Customers see only Home + Predict. Everything else is behind the single-admin login (see src/auth.py)."""
import pandas as pd, streamlit as st
from src import admin_dashboard, auth, history, services
from src.data_preprocessing import DataError, load_clean
from src.evaluate_model import load_reports
from src.predict import (ModelLoadError, ModelNotFoundError, ValidationError, format_inr, load_model, predict_price)

st.set_page_config(page_title="SmartHome AI", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")
ADMIN_PAGES = list(admin_dashboard.NAV)
CUSTOMER_SITE = "Customer Site"
MAX_ATTEMPTS = 5
st.session_state.setdefault("admin", False)
st.session_state.setdefault("view", "Home")
st.session_state.setdefault("attempts", 0)

CSS = """<style>
.block-container{padding-top:1.5rem;max-width:1100px}
header[data-testid="stHeader"]{background:transparent}
.brand{font-size:1.5rem;font-weight:800;color:#0F766E;letter-spacing:-.5px}
.hero{background:linear-gradient(135deg,#0F766E,#134E4A);border-radius:24px;padding:3rem 2.2rem;margin:1rem 0 1.2rem 0}
.hero h1{margin:0 0 .6rem 0;font-size:2.6rem;color:#fff;line-height:1.15}
.hero p{color:#CCFBF1;font-size:1.15rem;margin:0;max-width:640px}
.card{background:#fff;border:1px solid #E2E8E5;border-radius:16px;padding:1.2rem 1.4rem;box-shadow:0 2px 10px rgba(15,60,50,.06);height:100%}
.card h4{margin:0 0 .4rem 0;color:#0F766E}.card p{margin:0;color:#4B635F;font-size:.95rem}
.price{background:linear-gradient(135deg,#0F766E,#134E4A);border-radius:18px;padding:1.6rem;text-align:center;margin:.5rem 0 1rem 0}
.price h2{margin:.2rem 0;font-size:2.6rem;color:#fff}.price span{color:#CCFBF1}
div.stButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:600;padding:.6rem 1.4rem}
[data-testid="stForm"]{background:#fff;border:1px solid #E2E8E5;border-radius:18px;padding:1.4rem}
[data-testid="stMetric"]{background:#fff;border:1px solid #E2E8E5;border-radius:14px;padding:.8rem 1rem}
@media(max-width:700px){.hero h1{font-size:1.8rem}.hero{padding:2rem 1.2rem}}
</style>"""
HIDE_SIDEBAR = "<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],[data-testid='collapsedControl']{display:none}</style>"
st.markdown(CSS, unsafe_allow_html=True)
if not st.session_state.admin:
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
else:
    st.markdown(admin_dashboard.ADMIN_CSS, unsafe_allow_html=True)


def card(title, text):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def get_model():
    return services.cached_model()  # shared, cached by model-file mtime


@st.cache_data
def get_data():
    return load_clean()[0]


def plotly_dark(fig):  # name kept; styles charts for the light theme
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def need_model():
    try:
        return get_model()
    except (ModelNotFoundError, ModelLoadError):
        st.error("The price estimator is temporarily unavailable. Please try again later.")
        if st.session_state.admin:
            st.info("Admin: train the model first with `python -m src.train_model`.")
        return None


def go(view):
    st.session_state.view = view
    st.session_state.admin_page = CUSTOMER_SITE


# ---- Routing -----------------------------------------------------------------------------------
if st.session_state.admin:
    choice = admin_dashboard.sidebar_nav()
    page = st.session_state.view if choice == CUSTOMER_SITE else choice
else:
    page = st.session_state.view

if page in ("Home", "Predict House Price", "Admin Login"):
    c1, c2, c3 = st.columns([3, 5, 2])
    c1.markdown('<div class="brand">🏠 SmartHome AI</div>', unsafe_allow_html=True)
    if c3.button("Home" if page != "Home" else "🔒 Admin", use_container_width=True) if not st.session_state.admin else False:
        go("Home" if page != "Home" else "Admin Login"); st.rerun()

if page == "Home":
    st.markdown('<div class="hero"><h1>Find out what a home in Bengaluru is worth</h1>'
                '<p>Tell us about the property and get an instant price estimate.</p></div>', unsafe_allow_html=True)
    st.button("Get my price estimate →", type="primary", on_click=go, args=("Predict House Price",))

elif page == "Admin Login":
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div style="text-align:center;margin-top:1rem"><div class="brand" style="font-size:1.8rem">🏠 SmartHome AI</div>'
                    '<p style="color:#4B635F">Admin sign in</p></div>', unsafe_allow_html=True)
        if not auth.is_configured():
            st.warning("No admin account exists yet. Create it once by running `python -m src.auth` in PowerShell, then reload this page.")
        elif st.session_state.attempts >= MAX_ATTEMPTS:
            st.error("Too many failed attempts. Close this browser tab and reopen the app to try again.")
        else:
            with st.form("login"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    if auth.verify(user, pw):
                        st.session_state.update(admin=True, attempts=0, admin_page=ADMIN_PAGES[0], view="Home"); st.rerun()
                    else:
                        st.session_state.attempts += 1
                        st.error(f"Incorrect username or password. {MAX_ATTEMPTS - st.session_state.attempts} attempts left.")

elif page == "Predict House Price":
    st.subheader("Tell us about the property")
    bundle = need_model()
    if bundle:
        meta = bundle["metadata"]; r = meta["ranges"]
        with st.form("predict"):
            a, b = st.columns(2)
            loc = a.selectbox("Location", meta["locations"], index=meta["locations"].index("Whitefield") if "Whitefield" in meta["locations"] else 0)
            atype = b.selectbox("Area type", meta["area_types"])
            avail = a.selectbox("Availability", meta["availability"])
            sqft = b.number_input("Total area (sq ft)", min_value=float(r["total_sqft"][0]), max_value=float(r["total_sqft"][1]), value=1200.0, step=50.0)
            bhk = a.number_input("Bedrooms (BHK)", int(r["bhk"][0]), int(r["bhk"][1]), 2)
            bath = b.number_input("Bathrooms", int(r["bath"][0]), int(r["bath"][1]), 2)
            balc = a.number_input("Balconies", int(r["balcony"][0]), int(r["balcony"][1]), 1)
            go_btn = st.form_submit_button("Predict Price", type="primary")
        if go_btn:
            inp = dict(location=loc, area_type=atype, availability=avail, total_sqft=sqft, bhk=bhk, bath=bath, balcony=balc)
            try:
                res = predict_price(inp, bundle)
                history.add(inp, res["price_lakh"])
                st.markdown(f'<div class="price"><span>Estimated price</span><h2>{format_inr(res["price_lakh"])}</h2>'
                            f'<span>≈ ₹ {res["price_inr"]:,.0f} &nbsp;·&nbsp; ₹ {res["price_per_sqft"]:,.0f} per sq ft</span></div>', unsafe_allow_html=True)
                st.markdown("**Your property**")
                st.dataframe(pd.DataFrame([inp]).rename(columns=str.title), hide_index=True, width="stretch")
                st.info("This is an estimate based on past listings, not a guaranteed selling price. Actual prices can differ.")
            except ValidationError as e:
                st.error(f"Please check your input: {e}")
            except Exception:  # noqa: BLE001
                st.error("Sorry, we couldn't generate an estimate. Please try again.")

else:
    admin_dashboard.render(page)
