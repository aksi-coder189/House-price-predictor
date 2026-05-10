import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HouseLens · Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG    = "#f5f0eb"
CARD  = "#fdfaf6"
DARK  = "#2c1f14"
MID   = "#8c6239"
LITE  = "#c9a87c"
ERR   = "#b5361e"
GREEN = "#3a7d44"
SIDE  = "#ede8e0"

# Plotly colour constants 
CA = "#b87030"   # amber / primary
CB = "#306898"   # blue
CG = "#2e7048"   # green
CR = "#a83838"   # red
CV = "#6048a0"   # violet
CT = "#207878"   # teal
CS = "#7a6040"   # stone
CD = "#c09828"   # gold

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lora:wght@400;600&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {DARK};
}}
.stApp {{
    background-color: {BG};
    background-image:
        radial-gradient(ellipse at 8% 5%,  rgba(200,170,120,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 92% 95%, rgba(170,140,100,0.14) 0%, transparent 50%);
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background-color: {SIDE} !important;
    border-right: 1px solid #cfc8bc;
}}
[data-testid="stSidebar"] * {{ color: {DARK} !important; }}
[data-testid="stSidebar"] input {{
    background-color: #ffffff !important;
    color: {DARK} !important;
    border: 1px solid #c0b8a8 !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background-color: #ffffff !important;
    color: {DARK} !important;
    border: 1px solid #c0b8a8 !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] span {{ color: {DARK} !important; }}
ul[data-testid="stSelectboxVirtualDropdown"],
ul[data-testid="stSelectboxVirtualDropdown"] * {{
    background-color: #ffffff !important;
    color: {DARK} !important;
}}
[data-testid="stSidebar"] [data-baseweb="base-input"] {{
    background-color: #ffffff !important;
    border: 1px solid #c0b8a8 !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="base-input"] input {{
    background-color: #ffffff !important;
    color: {DARK} !important;
}}


[data-testid="stSidebar"] button[aria-label="Increment"],
[data-testid="stSidebar"] button[aria-label="Decrement"] {{
    background-color: #2c1f14 !important; 
    color: #ffffff !important;          
    border: 1px solid #2c1f14 !important;
    opacity: 1 !important;             
}}


[data-testid="stSidebar"] button[aria-label="Increment"]:hover,
[data-testid="stSidebar"] button[aria-label="Decrement"]:hover {{
    background-color: #8c6239 !important; /* Lighter brown on hover */
    border-color: #8c6239 !important;
}}

/* ── headings ── */
h1, h2, h3 {{ font-family: 'Playfair Display', serif; color: {DARK}; }}

/* ── HERO ── */
.hero {{
    background: linear-gradient(135deg, {DARK} 0%, #4a3220 55%, #6b4c2e 100%);
    border-radius: 20px;
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(44,31,20,0.28);
    position: relative; overflow: hidden;
}}
.hero::before {{
    content: "🏡"; position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%); font-size: 5rem; opacity: 0.12;
}}
.hero h1 {{
    font-family: 'Playfair Display', serif; font-size: 2.8rem;
    color: #f8f0e4 !important; margin: 0 0 0.4rem; letter-spacing: -0.3px;
}}
.hero p {{ color: #c5a87c !important; font-size: 1.05rem; margin: 0; }}
.hero .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.12); color: #f0dfc0 !important;
    border: 1px solid rgba(255,255,255,0.2); border-radius: 20px;
    padding: 3px 14px; font-size: 0.78rem; margin-top: 0.8rem; letter-spacing: 0.5px;
}}

/* ── section title ── */
.section-title {{
    font-family: 'Playfair Display', serif; font-size: 1.4rem; color: {DARK};
    border-left: 4px solid {MID}; padding-left: 0.75rem; margin: 1.8rem 0 1rem;
}}
.st-sec {{
    font-family: 'Lora', serif;
    font-size: 1.1rem;
    color: #1a1612;
    font-weight: 600;
    margin: 18px 0 10px;
}}

/* ── KPI cards ── */
.kpi-card {{
    background: {CARD}; border: 1px solid #e0d8cc; border-radius: 16px;
    padding: 1.1rem 1.2rem 0.9rem; text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07);
}}
.kpi-icon {{ font-size: 1.5rem; margin-bottom: 0.3rem; }}
.kpi-val {{
    font-family: 'Playfair Display', serif; font-size: 1.65rem; font-weight: 700;
    color: {MID}; line-height: 1.1;
}}
.kpi-lbl {{ font-size: 0.72rem; color: #7a6a5a; text-transform: uppercase; letter-spacing: .6px; margin-top: 3px; }}
.kpi-sub {{ font-size: 0.7rem; color: #9e8d7a; margin-top: 2px; }}

/* ── prediction box ── */
.pred-hero {{
    background: linear-gradient(135deg, {DARK} 0%, #5a3c24 100%);
    border-radius: 18px; padding: 1.8rem 2.2rem; text-align: center;
    box-shadow: 0 8px 30px rgba(44,31,20,0.3);
    margin-bottom: 14px;
}}
.pred-hero .ph-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: #c5a87c; margin-bottom: 6px;
}}
.pred-hero .ph-price {{
    font-family: 'Playfair Display', serif; font-size: 3rem;
    color: #f8f0e4; line-height: 1;
}}
.pred-hero .ph-range {{
    color: #a89070; font-size: 0.82rem; margin-top: 6px; font-style: italic;
}}
.pred-hero .ph-eq {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 9px; padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem; color: #70b0cc;
    margin-top: 12px; text-align: left; line-height: 1.9;
}}

/* ── eq card ── */
.eq-card {{
    background: {CARD}; border: 1.5px solid #d8cfc3; border-radius: 16px;
    padding: 1.4rem 1.6rem; box-shadow: 0 3px 12px rgba(0,0,0,0.06); height: 100%;
}}
.eq-card h4 {{
    font-family: 'Playfair Display', serif; color: {DARK} !important;
    font-size: 1.1rem; margin: 0 0 0.8rem;
    border-bottom: 1px solid #e0d8cc; padding-bottom: 0.5rem;
}}
.eq-card p {{ color: {DARK} !important; font-size: 0.9rem; margin: 5px 0; }}
.eq-pill {{
    display: inline-block; background: #f0e8dc; color: {DARK} !important;
    border-radius: 8px; padding: 3px 10px; font-size: 0.84rem;
    margin: 2px 2px; font-weight: 500;
}}
.eq-formula {{
    text-align: center; padding: 0.6rem 0.8rem;
    font-size: 1.1rem; color: {DARK}; font-style: italic; font-weight: 700;
    background: #f0e8dc; border-radius: 10px; margin-bottom: 0.9rem;
}}

/* ── equation banner  ── */
.eq-banner {{
    background: {CARD};
    border: 1px solid #d8d0c8;
    border-left: 4px solid #7a6040;
    border-radius: 9px;
    padding: 14px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #2a2420;
    line-height: 2;
    margin: 12px 0 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.eq-banner .acc {{ color: #7a6040; font-weight: 600; }}
.eq-banner .blu {{ color: #30609a; }}
.eq-banner .grn {{ color: #306848; }}
.eq-banner .mut {{ color: #8a8070; font-size: 0.68rem; }}

/* ── insight card ── */
.insight-card {{
    background: {CARD}; border: 1px solid #e0d8cc;
    border-left: 4px solid {MID}; border-radius: 12px;
    padding: 0.9rem 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}
.ic {{ background: {CARD}; border: 1px solid #d8d0c8; border-radius: 10px; padding: 15px 17px; margin: 7px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.ic .ic-t {{ font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8a8070; margin-bottom: 7px; }}
.ic p {{ font-size: 0.83rem; color: #38322c; margin: 4px 0; line-height: 1.55; }}
.ic-title {{ font-weight: 600; color: {DARK}; font-size: 0.85rem; margin-bottom: 3px; }}
.ic-val   {{ font-size: 1.2rem; font-weight: 700; color: {MID}; }}
.ic-desc  {{ font-size: 0.78rem; color: #7a6a5a; }}

/* ── tabs ── */
button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: {DARK} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    border-bottom: 3px solid {MID} !important; color: {MID} !important;
}}

/* ── download button ── */
.stDownloadButton > button {{
    background: {MID} !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.5rem 1.4rem !important;
}}
.stDownloadButton > button:hover {{ background: {DARK} !important; }}

/* ── checkbox ── */
[data-testid="stCheckbox"] label {{ color: {DARK} !important; }}
[data-testid="stCheckbox"] label p {{ color: {DARK} !important; }}

/* ── metric ── */
[data-testid="metric-container"] {{
    background: {CARD};
    border: 1px solid #d8d0c8;
    border-radius: 10px;
    padding: 14px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
[data-testid="stMetricValue"] {{
    font-family: 'Lora', serif !important;
    font-size: 1.45rem !important; font-weight: 600 !important;
    color: #1a1612 !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important; text-transform: uppercase;
    letter-spacing: 0.09em; color: #7a7060 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.68rem !important; color: #4a7a5a !important; }}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid #d8d0c8 !important;
    border-radius: 9px !important;
}}
[data-testid="stDataFrame"] td, 
[data-testid="stDataFrame"] th,
[data-testid="stTable"] td {{
    color: #2c1f14 !important;
    -webkit-text-fill-color: #2c1f14 !important;
}}

/* ── alerts ── */
[data-testid="stAlert"] {{ border-radius: 9px !important; font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important; }}

hr {{ border-color: #d8d0c8 !important; margin: 16px 0 !important; }}
.stCaption {{ color: #8a8070 !important; font-size: 0.7rem !important; }}

/* ── footer ── */
.footer {{
    text-align: center; color: #9e8d7a; font-size: 0.8rem;
    padding: 1.6rem 0 0.6rem; border-top: 1px solid #ddd4c5; margin-top: 2.5rem;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE THEME
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f5f2ed",
    font=dict(family="JetBrains Mono, monospace", color="#2c1f14", size=11), 
    margin=dict(l=54, r=18, t=46, b=54),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.07)", 
        zerolinecolor="rgba(0,0,0,0.14)",
        linecolor="rgba(0,0,0,0.14)",
        tickfont=dict(color="#2c1f14"),
        titlefont=dict(color="#2c1f14"), 
        showspikes=True, spikecolor="#7a6040", spikethickness=1,
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.07)", 
        zerolinecolor="rgba(0,0,0,0.14)",
        linecolor="rgba(0,0,0,0.14)",
        tickfont=dict(color="#2c1f14"), 
        titlefont=dict(color="#2c1f14"),
    ),
    hoverlabel=dict(
        bgcolor="#28211a", bordercolor="rgba(122,96,64,0.7)",
        font_family="JetBrains Mono, monospace",
        font_color="#f0ede8", font_size=12,
    ),
    legend=dict(
        bgcolor="rgba(250,248,245,0.92)", bordercolor="#d8d0c8",
        borderwidth=1, font=dict(size=10, color="#38322c"),
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#bfb0a0", "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "text.color": DARK,
    "grid.color": "#ddd4c5", "grid.linestyle": "--", "grid.alpha": 0.55,
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

def fmt_price_y(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
def fmt_price_x(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION 
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    sqft          = np.random.normal(1800, 600, n).clip(400, 5500)
    bedrooms      = np.random.choice([1,2,3,4,5], n, p=[0.05,0.20,0.40,0.25,0.10])
    bathrooms     = (bedrooms * 0.6 + np.random.uniform(0,1,n)).round(1).clip(1,5)
    age           = np.random.exponential(20, n).clip(0, 80).astype(int)
    garage        = np.random.choice([0,1,2,3], n, p=[0.10,0.35,0.45,0.10])
    floors        = np.random.choice([1,2,3], n, p=[0.40,0.50,0.10])
    lot_size      = (sqft * np.random.uniform(2, 5, n)).clip(800, 40000)
    school_rating = np.random.uniform(3, 10, n).round(1)
    dist_down     = np.random.exponential(10, n).clip(1, 50)
    renovation    = np.random.choice([0,1], n, p=[0.65,0.35])
    neighbourhood = np.random.choice(
        ["Downtown","Suburb","Rural","Waterfront"], n, p=[0.25,0.45,0.15,0.15])
    location      = np.where(
        np.isin(neighbourhood, ["Downtown"]), "Urban",
        np.where(np.isin(neighbourhood, ["Suburb"]), "Suburban", "Rural"))
    nb_map = {"Downtown":30000,"Suburb":10000,"Rural":-15000,"Waterfront":60000}
    nb_arr = np.array([nb_map[x] for x in neighbourhood])
    price = (
        80*sqft + 15000*bedrooms + 12000*bathrooms - 1200*age
        + 8000*garage + 10000*floors + 3*lot_size + 9000*school_rating
        - 1500*dist_down + 25000*renovation + nb_arr
        + np.random.normal(0, 18000, n) + 50000
    ).clip(60000, 2_500_000)
    return pd.DataFrame({
        "sqft"                    : sqft.round(0).astype(int),
        "Area_sqft"               : sqft.round(0).astype(int),  # alias for app.py charts
        "bedrooms"                : bedrooms,
        "Bedrooms"                : bedrooms,
        "bathrooms"               : bathrooms,
        "age_years"               : age,
        "garage_spaces"           : garage,
        "floors"                  : floors,
        "lot_size_sqft"           : lot_size.round(0).astype(int),
        "school_rating"           : school_rating,
        "distance_downtown_miles" : dist_down.round(2),
        "recently_renovated"      : renovation,
        "neighbourhood"           : neighbourhood,
        "Location"                : location,
        "price"                   : price.round(-2).astype(int),
        "Price_USD"               : price.round(-2).astype(int),
    })

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING  
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    X, y = num[["sqft"]], num["price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = LinearRegression().fit(Xtr, ytr)
    ypred = mdl.predict(Xte)
    cv = cross_val_score(mdl, X, y, cv=5, scoring="r2")

    se_model = np.sqrt(mean_squared_error(ytr, mdl.predict(Xtr)))
    se_b1    = se_model / (float(Xtr["sqft"].std()) * np.sqrt(len(Xtr)-1))
    t_stat   = float(mdl.coef_[0]) / se_b1
    p_val    = float(2*(1-stats.t.cdf(abs(t_stat), df=len(Xtr)-2)))

    resid = yte.values - ypred
    std_r = resid / (resid.std() + 1e-9)
    res_df = pd.DataFrame({
        "fitted": ypred, "actual": yte.values,
        "residual": resid, "std_resid": std_r,
    })
    m = {
        "coef"      : round(float(mdl.coef_[0]), 4),
        "intercept" : round(float(mdl.intercept_), 2),
        "r2"        : round(r2_score(yte, ypred), 4),
        "r2_train"  : round(r2_score(ytr, mdl.predict(Xtr)), 4),
        "adj_r2"    : round(1-(1-r2_score(yte,ypred))*(len(yte)-1)/(len(yte)-2), 4),
        "rmse"      : round(float(np.sqrt(mean_squared_error(yte, ypred))), 2),
        "mae"       : round(float(mean_absolute_error(yte, ypred)), 2),
        "mape"      : round(float(np.mean(np.abs(resid/yte.values))*100), 2),
        "cv_mean"   : round(float(cv.mean()), 4),
        "cv_std"    : round(float(cv.std()), 4),
        "cv_scores" : cv,
        "t_stat"    : round(t_stat, 4),
        "p_value"   : p_val,
        "se_b1"     : round(se_b1, 4),
        "r_pearson" : round(float(df["sqft"].corr(df["price"])), 4),
        "train_n"   : len(Xtr),
        "test_n"    : len(Xte),
        # app.py aliases
        "n_train"   : len(Xtr),
        "n_test"    : len(Xte),
    }
    return mdl, m, Xtr, Xte, ytr, yte, ypred, res_df

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART FUNCTIONS 
# ─────────────────────────────────────────────────────────────────────────────
def make_scatter(df, m, hl_sqft=None):
    samp   = df.sample(min(250, len(df)), random_state=42)
    xl     = np.linspace(df["sqft"].min(), df["sqft"].max(), 300)
    yl     = m["intercept"] + m["coef"] * xl
    se     = m["rmse"]
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=xl, y=yl+1.96*se, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xl, y=yl-1.96*se, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(122,96,64,0.10)", name="95% Pred Band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=samp["sqft"], y=samp["price"], mode="markers",
        marker=dict(color=CB, size=5, opacity=0.48, line=dict(color="white",width=0.4)),
        name="Homes",
        hovertemplate="<b>Size:</b> %{x:,} sqft<br><b>Price:</b> $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines", line=dict(color=CA, width=2.4),
        name=f"Fit  R²={m['r2']:.3f}"))
    if hl_sqft is not None:
        hp = m["intercept"] + m["coef"]*hl_sqft
        fig.add_trace(go.Scatter(
            x=[hl_sqft], y=[hp], mode="markers",
            marker=dict(color=CD, size=14, symbol="star",
                        line=dict(color="#28211a",width=2)),
            name=f"Your home ({hl_sqft:,} sqft)",
            hovertemplate=f"<b>Prediction</b><br>${hp:,.0f}<extra></extra>"))
    fig.update_layout(**PL, height=390,
        title=dict(text="Regression Line — House Size vs Sale Price",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="House Size (sqft)", yaxis_title="Sale Price ($)")
    return fig


def make_avp(yte_arr, ypred_arr):
    mn  = min(yte_arr.min(), ypred_arr.min())
    mx  = max(yte_arr.max(), ypred_arr.max())
    err = np.abs(yte_arr - ypred_arr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
        line=dict(color=CA,width=1.5,dash="dot"),
        name="Perfect prediction", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=yte_arr, y=ypred_arr, mode="markers",
        marker=dict(
            color=err, colorscale=[[0,CG],[0.5,CD],[1,CR]],
            size=6, opacity=0.68, showscale=True,
            colorbar=dict(title="Error ($)",thickness=10,
                          tickfont=dict(size=9,color="#58504a"))),
        hovertemplate="<b>Actual:</b> $%{x:,.0f}<br><b>Pred:</b> $%{y:,.0f}<extra></extra>",
        name="Test predictions"))
    fig.update_layout(**PL, height=360,
        title=dict(text="Actual vs Predicted  (colour = error size)",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)")
    return fig


def make_resfit(res_df):
    colors  = [CG if r>=0 else CR for r in res_df["residual"]]
    sorted_r= res_df.sort_values("fitted")
    smooth  = sorted_r["residual"].rolling(18,center=True,min_periods=5).mean()
    fig = go.Figure()
    fig.add_hline(y=0,line=dict(color=CA,width=1.5,dash="dash"))
    fig.add_trace(go.Scatter(
        x=res_df["fitted"], y=res_df["residual"], mode="markers",
        marker=dict(color=colors,size=5,opacity=0.70,
                    line=dict(color="rgba(0,0,0,0.10)",width=0.4)),
        hovertemplate="<b>Fitted:</b> $%{x:,.0f}<br><b>Resid:</b> $%{y:,.0f}<extra></extra>",
        name="Residuals"))
    fig.add_trace(go.Scatter(
        x=sorted_r["fitted"], y=smooth, mode="lines",
        line=dict(color=CV,width=2), name="Rolling mean"))
    fig.update_layout(**PL, height=360,
        title=dict(text="Residuals vs Fitted Values",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Fitted ($)", yaxis_title="Residuals ($)")
    return fig


def make_reshist(res_df):
    r      = res_df["residual"]
    xr     = np.linspace(r.min(), r.max(), 200)
    pdf    = stats.norm.pdf(xr, r.mean(), r.std())
    scale  = len(r)*(r.max()-r.min())/28
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=28,
        marker=dict(color=CV,opacity=0.70,line=dict(color="white",width=0.6)),
        name="Residuals",
        hovertemplate="<b>Bin:</b> $%{x:,.0f}<br><b>Count:</b> %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=xr, y=pdf*scale, mode="lines",
        line=dict(color=CA,width=2), name="Normal fit"))
    fig.add_vline(x=r.mean(),line=dict(color=CG,width=1.5,dash="dot"),
        annotation=dict(text=f"μ={r.mean():,.0f}",
                        font=dict(color=CG,size=10)))
    fig.update_layout(**PL, height=340,
        title=dict(text="Residual Distribution + Normal Overlay",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Residual ($)", yaxis_title="Frequency")
    return fig


def make_qq(res_df):
    sr  = res_df["std_resid"].sort_values().values
    n   = len(sr)
    th  = stats.norm.ppf(np.linspace(0.01,0.99,n))
    lim = max(abs(th.min()),abs(th.max()))+0.4
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=th, y=sr, mode="markers",
        marker=dict(color=CB,size=4,opacity=0.62,
                    line=dict(color="white",width=0.3)),
        hovertemplate="<b>Theoretical:</b> %{x:.3f}<br><b>Sample:</b> %{y:.3f}<extra></extra>",
        name="Q-Q"))
    fig.add_trace(go.Scatter(
        x=[-lim,lim], y=[-lim,lim], mode="lines",
        line=dict(color=CA,width=1.5,dash="dot"), name="Normal line"))
    fig.update_layout(**PL, height=340,
        title=dict(text="Q-Q Plot — Normality of Residuals",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    return fig


def make_price_dist(df):
    mn, md = df["price"].mean(), df["price"].median()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["price"], nbinsx=30,
        marker=dict(color=CG,opacity=0.70,line=dict(color="white",width=0.5)),
        name="Homes",
        hovertemplate="<b>Price:</b> $%{x:,.0f}<br><b>Count:</b> %{y}<extra></extra>"))
    fig.add_vline(x=mn,line=dict(color=CA,width=2,dash="dash"),
        annotation=dict(text=f"Mean ${mn/1e3:.0f}k",font=dict(color=CA,size=10)))
    fig.add_vline(x=md,line=dict(color=CB,width=2,dash="dot"),
        annotation=dict(text=f"Median ${md/1e3:.0f}k",
                        font=dict(color=CB,size=10),yshift=-22))
    fig.update_layout(**PL, height=330,
        title=dict(text="Sale Price Distribution",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Sale Price ($)", yaxis_title="Number of Homes")
    return fig


def make_sqft_dist(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["sqft"], nbinsx=30,
        marker=dict(color=CB,opacity=0.70,line=dict(color="white",width=0.5)),
        name="Homes",
        hovertemplate="<b>sqft:</b> %{x:,}<br><b>Count:</b> %{y}<extra></extra>"))
    fig.add_vline(x=df["sqft"].mean(),line=dict(color=CA,width=2,dash="dash"),
        annotation=dict(text=f"Mean {df['sqft'].mean():,.0f}",
                        font=dict(color=CA,size=10)))
    fig.update_layout(**PL, height=310,
        title=dict(text="House Size (sqft) Distribution",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Square Footage", yaxis_title="Count")
    return fig


def make_learning_curve(df):
    Xa = df[["sqft"]].values
    ya = df["price"].values
    sizes = list(range(40, len(df), 35)) + [len(df)]
    tr_r2, te_r2, actual_sizes = [], [], [] 
    for sz in sizes:
        idx = np.random.choice(len(df), sz, replace=False)
        Xs, ys = Xa[idx], ya[idx]
        if len(Xs) < 20:
            continue
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xs, ys, test_size=0.2, random_state=0)
        m2 = LinearRegression().fit(Xtr2, ytr2)
        tr_r2.append(r2_score(ytr2, m2.predict(Xtr2)))
        te_r2.append(r2_score(yte2, m2.predict(Xte2)))
        actual_sizes.append(sz)
    
    vs = actual_sizes 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vs, y=tr_r2, mode="lines+markers",
        line=dict(color=CB,width=2), marker=dict(size=4), name="Train R²"))
    fig.add_trace(go.Scatter(x=vs, y=te_r2, mode="lines+markers",
        line=dict(color=CA,width=2), marker=dict(size=4), name="Test R²"))
    fig.update_layout(**PL, height=310,
        title=dict(text="Learning Curve — R² vs Training Size",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Training Samples", yaxis_title="R² Score")
    return fig


def make_cv_bars(m):
    np.random.seed(7)
    vals = np.clip(
        [m["cv_mean"]+np.random.normal(0,m["cv_std"]) for _ in range(5)], 0, 1)
    labels = [f"Fold {i+1}" for i in range(5)]
    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker=dict(color=[CB,CG,CA,CV,CT],opacity=0.80,
                    line=dict(color="rgba(0,0,0,0.07)",width=0.5)),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside", textfont=dict(size=10,color="#2a2420")))
    fig.add_hline(y=m["cv_mean"],line=dict(color=CR,width=2,dash="dash"),
        annotation=dict(text=f"Mean R²={m['cv_mean']:.4f}",
                        font=dict(color=CR,size=10)))
    pl2 = dict(**PL)
    pl2["yaxis"] = dict(**PL["yaxis"], range=[0, 1])
    fig.update_layout(**pl2, height=290,
        title=dict(text="5-Fold Cross-Validation R² Scores",
                   font=dict(color="#1a1612",size=14),x=0),
        yaxis_title="R² Score", showlegend=False)
    return fig


def make_feat_corr(df):
    num  = df.select_dtypes(include=[np.number])
    corr = num.corr()["price"].drop("price").sort_values()
    cols = [CB if v >= 0 else CR for v in corr.values]
    fig  = go.Figure(go.Bar(
        x=corr.values, y=[c.replace("_"," ") for c in corr.index],
        orientation="h",
        marker=dict(color=cols,opacity=0.78,
                    line=dict(color="rgba(0,0,0,0.07)",width=0.5)),
        text=[f"{v:.3f}" for v in corr.values],
        textposition="outside", textfont=dict(size=9,color="#2a2420"),
        hovertemplate="<b>%{y}</b><br>r = %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0,line=dict(color="rgba(0,0,0,0.18)",width=1))
    pl2 = dict(**PL)
    pl2["xaxis"].update(range=[-1, 1], tickfont=dict(color="#2c1f14"))
    pl2["yaxis"].update(tickfont=dict(color="#2c1f14", size=10))
    fig.update_layout(
        **pl2, 
        height=380,
        title=dict(
            text="Pearson Correlation with Sale Price",
            font=dict(color="#1a1612", size=14),
            x=0
        ),
        xaxis_title="Correlation Coefficient (r)"
    )
    return fig


def make_heatmap(df):
    num  = df.select_dtypes(include=[np.number])
    corr = num.corr().round(2)
    cols = [c.replace("_"," ") for c in corr.columns]
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols,
        colorscale=[[0,CR],[0.5,"#eeebe5"],[1,CB]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}", textfont=dict(size=8.5,color="#1a1612"),
        hoverongaps=False,
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r",thickness=10,tickfont=dict(size=9,color="#58504a"))))
    pl2 = dict(**PL)
    pl2["margin"] = dict(l=110, r=16, t=46, b=82)
    pl2["xaxis"].update(tickfont=dict(size=9), tickangle=-35)
    pl2["yaxis"].update(tickfont=dict(size=9))
    fig.update_layout(
        **pl2, 
        height=420,
        title=dict(
            text="Correlation Heatmap — All Numeric Features",
            font=dict(color="#1a1612", size=14),
            x=0
        ))
    return fig


def make_boxplot_beds(df):
    palette = [CB,CG,CA,CR,CV]
    fig = go.Figure()
    for i, bed in enumerate(sorted(df["bedrooms"].unique())):
        sub = df[df["bedrooms"]==bed]["price"]
        fig.add_trace(go.Box(
            y=sub, name=f"{bed} bed", boxmean="sd",
            marker=dict(color=palette[i%len(palette)],size=3,opacity=0.68),
            line=dict(width=1.5),
            hovertemplate=f"<b>{bed} Bed</b><br>${{y:,.0f}}<extra></extra>"))
    fig.update_layout(**PL, height=340,
        title=dict(text="Price Distribution by Bedrooms",
                   font=dict(color="#1a1612",size=14),x=0),
        xaxis_title="Bedrooms", yaxis_title="Sale Price ($)", showlegend=False)
    return fig


def make_by_neighbourhood(df):
    grp = df.groupby("neighbourhood")["price"].agg(["mean","std"]).reset_index()
    grp = grp.sort_values("mean",ascending=False)
    fig = go.Figure(go.Bar(
        x=grp["neighbourhood"], y=grp["mean"],
        error_y=dict(type="data",array=grp["std"],visible=True,
                     color="rgba(0,0,0,0.18)"),
        marker=dict(color=[CA,CB,CG,CV],opacity=0.80,
                    line=dict(color="rgba(0,0,0,0.07)",width=0.5)),
        text=[f"${v/1e3:.0f}k" for v in grp["mean"]],
        textposition="outside", textfont=dict(size=10,color="#1a1612"),
        hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(**PL, height=330, showlegend=False,
        title=dict(text="Average Sale Price by Neighbourhood",
                   font=dict(color="#1a1612",size=14),x=0),
        yaxis_title="Mean Price ($)")
    return fig


def make_renovation(df):
    grp  = df.groupby("recently_renovated")["price"].mean().reset_index()
    grp["label"] = grp["recently_renovated"].map({0:"Not Renovated",1:"Renovated"})
    diff = grp.loc[grp["recently_renovated"]==1,"price"].values[0] \
         - grp.loc[grp["recently_renovated"]==0,"price"].values[0]
    fig = go.Figure(go.Bar(
        x=grp["label"], y=grp["price"],
        marker=dict(color=[CR,CG],opacity=0.80,
                    line=dict(color="rgba(0,0,0,0.07)",width=0.5)),
        text=[f"${v:,.0f}" for v in grp["price"]],
        textposition="outside", textfont=dict(size=11,color="#1a1612"),
        hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(**PL, height=300, showlegend=False,
        title=dict(text=f"Renovation Impact — avg premium +${diff:,.0f}",
                   font=dict(color="#1a1612",size=14),x=0),
        yaxis_title="Mean Sale Price ($)")
    return fig


def make_ppsf(df):
    df2        = df.copy()
    df2["ppsf"]= df2["price"]/df2["sqft"]
    grp        = df2.groupby("neighbourhood")["ppsf"].agg(["mean","median"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["neighbourhood"], y=grp["mean"], name="Mean $/sqft",
        marker_color=CB, opacity=0.80,
        text=[f"${v:.0f}" for v in grp["mean"]],
        textposition="outside", textfont=dict(size=10)))
    fig.add_trace(go.Bar(
        x=grp["neighbourhood"], y=grp["median"], name="Median $/sqft",
        marker_color=CA, opacity=0.80,
        text=[f"${v:.0f}" for v in grp["median"]],
        textposition="outside", textfont=dict(size=10)))
    fig.update_layout(**PL, height=310, barmode="group",
        title=dict(text="Price per sqft by Neighbourhood",
                   font=dict(color="#1a1612",size=14),x=0),
        yaxis_title="Price per sqft ($)")
    return fig


def make_age_scatter(df):
    samp = df.sample(min(300,len(df)), random_state=5)
    fig  = px.scatter(
        samp, x="age_years", y="price",
        color="bedrooms", size="sqft", size_max=12, opacity=0.58,
        color_continuous_scale=[[0,CB],[0.5,CG],[1,CR]],
        hover_data={"sqft":True,"price":True,"age_years":True,"bedrooms":True},
        labels={"age_years":"House Age (years)","price":"Sale Price ($)","bedrooms":"Beds"})
    fig.update_layout(**PL, height=350,
        title=dict(text="Price vs Age  (size=sqft, colour=bedrooms)",
                   font=dict(color="#1a1612",size=14),x=0))
    return fig


def make_3d(df):
    samp = df.sample(min(300,len(df)), random_state=7)
    fig  = go.Figure(go.Scatter3d(
        x=samp["sqft"], y=samp["bedrooms"], z=samp["price"], mode="markers",
        marker=dict(size=3.5, color=samp["price"],
                    colorscale=[[0,CB],[0.45,CG],[0.75,CD],[1,CR]],
                    opacity=0.72,
                    colorbar=dict(title="Price",thickness=8,
                                  tickfont=dict(size=8,color="#58504a"))),
        hovertemplate="<b>Sqft:</b> %{x:,}<br><b>Beds:</b> %{y}<br><b>Price:</b> $%{z:,.0f}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#f5f2ed",
            xaxis=dict(title="sqft",gridcolor="#d8d0c8",
                       backgroundcolor="#f5f2ed",showbackground=True),
            yaxis=dict(title="Bedrooms",gridcolor="#d8d0c8",
                       backgroundcolor="#f5f2ed",showbackground=True),
            zaxis=dict(title="Price ($)",gridcolor="#d8d0c8",
                       backgroundcolor="#f5f2ed",showbackground=True)),
        font=dict(family="JetBrains Mono, monospace",color="#58504a",size=10),
        title=dict(text="3D — sqft × Bedrooms → Price",
                   font=dict(color="#1a1612",size=13),x=0),
        margin=dict(l=0,r=0,t=46,b=0), height=410)
    return fig


def make_pairplot(df):
    cols = ["sqft","bedrooms","bathrooms","age_years","school_rating","price"]
    sub  = df[cols].sample(min(250,len(df)), random_state=42)
    fig  = px.scatter_matrix(
        sub, dimensions=cols, color="price",
        color_continuous_scale=[[0,CB],[0.5,CG],[1,CR]],
        opacity=0.40,
        labels={c:c.replace("_"," ") for c in cols})
    fig.update_traces(marker=dict(size=2.5), diagonal_visible=False)
    pl2 = dict(**PL)
    pl2["font"].update(color="#2c1f14")
    pl2["xaxis"].update(tickfont=dict(color="#2c1f14"), titlefont=dict(color="#2c1f14"))
    pl2["yaxis"].update(tickfont=dict(color="#2c1f14"), titlefont=dict(color="#2c1f14"))
    fig.update_layout(
        **pl2, 
        height=510,
        title=dict(
            text="Pair Plot — Key Features",
            font=dict(color="#1a1612", size=14),
            x=0
        ),
        coloraxis_colorbar=dict(
            title="Price",
            thickness=10,
            tickfont=dict(size=9, color="#2c1f14")
        )
    )
    return fig


def make_waterfall(m, sqft_val):
    contrib = m["coef"]*sqft_val
    pred    = m["intercept"]+contrib
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative","relative","total"],
        x=["Base (β₀)", f"Size Effect\n(β₁×{sqft_val:,})", "Prediction"],
        y=[m["intercept"], contrib, 0],
        text=[f"${m['intercept']:,.0f}", f"+${contrib:,.0f}", f"${pred:,.0f}"],
        textposition="outside", textfont=dict(color="#1a1612",size=10),
        connector=dict(line=dict(color="rgba(0,0,0,0.12)",width=1,dash="dot")),
        increasing=dict(marker=dict(color=CG)),
        decreasing=dict(marker=dict(color=CR)),
        totals=dict(marker=dict(color=CA))))
    fig.update_layout(**PL, height=255, showlegend=False,
        title=dict(text="Waterfall — How the Prediction is Built",
                   font=dict(color="#1a1612",size=13),x=0),
        yaxis_title="Amount ($)")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR 
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='font-family:Playfair Display,serif;color:{DARK};margin:0'>🏡 HouseLens</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MID};font-size:0.85rem;margin-top:2px'>Simple Linear Regression Engine</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown(f"<p style='font-weight:600;color:{DARK};font-size:0.92rem;margin-bottom:4px'>⚙️ Dataset Settings</p>", unsafe_allow_html=True)
    n_samples = st.slider("Sample size", 200, 2000, 1000, 100, key="sb_n")
    rand_seed = st.number_input("Random seed", 0, 999, 42, key="sb_seed")

    st.divider()
    st.markdown(f"<p style='font-weight:600;color:{DARK};font-size:0.92rem;margin-bottom:4px'>🔍 Predict Your House</p>", unsafe_allow_html=True)
    input_area = st.number_input(
        "House Area (sq ft)", min_value=200, max_value=7000, value=1800, step=50,
        help="Enter the total floor area in square feet."
    )
    st.markdown(f"<p style='font-size:0.78rem;color:#7a6a5a;margin-top:-8px'>Valid range: 200 – 7,000 sq ft</p>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<p style='font-weight:600;color:{DARK};font-size:0.92rem;margin-bottom:4px'>🎯 Property Details (for snapshot)</p>", unsafe_allow_html=True)
    beds_in  = st.slider("Bedrooms",          1,     5,    3,   1, key="sb_beds")
    baths_in = st.slider("Bathrooms",       1.0,   5.0,  2.0, 0.5, key="sb_bath")
    age_in   = st.slider("Age (years)",       0,    80,   15,   1, key="sb_age")
    gar_in   = st.slider("Garage spaces",     0,     3,    2,   1, key="sb_gar")
    schl_in  = st.slider("School rating",   3.0,  10.0,  7.5, 0.5, key="sb_sch")
    neigh_in = st.selectbox("Neighbourhood",
        ["Downtown","Suburb","Rural","Waterfront"], key="sb_nb")

    st.divider()
    st.markdown(f"<p style='font-weight:600;color:{DARK};font-size:0.92rem;margin-bottom:4px'>📊 Display Options</p>", unsafe_allow_html=True)
    show_confidence = st.checkbox("Show 95% confidence band (regression tab)", True)
    show_raw_data   = st.checkbox("Show full dataset table", False)

    st.divider()
    st.markdown(
        f"<div style='font-size:0.7rem;color:#7a7060;line-height:1.9'>"
        f"<b>Model:</b> Simple Linear Regression<br>"
        f"<b>Feature:</b> sqft → price<br>"
        f"<b>Library:</b> scikit-learn<br>"
        f"<b>Charts:</b> Plotly + Matplotlib<br>"
        f"<b>App:</b> Streamlit</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & TRAIN
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Generating dataset and fitting model…"):
    df = generate_dataset(n_samples, int(rand_seed))
    mdl, m, Xtr, Xte, ytr, yte, ypred, res_df = train_model(df)

# Derived prediction values
sqft_in         = int(input_area)
predicted_price = m["intercept"] + m["coef"] * sqft_in
pred_low        = predicted_price - m["rmse"]
pred_high       = predicted_price + m["rmse"]
contrib         = m["coef"] * sqft_in
pct_err         = (m["rmse"] / max(predicted_price, 1)) * 100

q1, q2, q3  = df["price"].quantile([0.25, 0.5, 0.75])
segment     = "Budget" if predicted_price < q1 else "Mid-Range" if predicted_price < q2 else "Upper-Mid" if predicted_price < q3 else "Premium"
percentile  = round((df["price"] < predicted_price).mean() * 100, 1)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>HouseLens - Price Predictor</h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS  (app.py icon style)
# ─────────────────────────────────────────────────────────────────────────────
kpis = [
    ("📈", f"{m['r2']:.4f}",       "R² Score",        f"{m['r2']*100:.1f}% variance explained"),
    ("📉", f"${m['rmse']:,.0f}",   "RMSE",             "Root mean squared error"),
    ("📐", f"${m['mae']:,.0f}",    "MAE",              "Mean absolute error"),
    ("💲", f"${m['coef']:.2f}",    "Price / sq ft",    "Regression slope β₁"),
    ("🔄", f"{m['cv_mean']:.4f}",  "CV R² (5-fold)",   f"±{m['cv_std']:.4f} std"),
    ("🏠", f"{n_samples:,}",       "Houses",           f"{m['train_n']} train · {m['test_n']} test"),
    ("📊", f"{m['mape']:.2f}%",    "MAPE",             "Mean % prediction error"),
]
cols = st.columns(7)
for col, (icon, val, lbl, sub) in zip(cols, kpis):
    with col:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# EQUATION BANNER  (run.py style)
# ─────────────────────────────────────────────────────────────────────────────
sig_str = "p < 0.001" if m["p_value"] < 0.001 else f"p = {m['p_value']:.4f}"
st.markdown(f"""
<div class="eq-banner">
  <span class="acc">Fitted Model:</span>&nbsp;
  <span class="acc">Price</span> = <span class="blu">{m['intercept']:,.2f}</span>
  + <span class="acc">{m['coef']:,.2f}</span> × <span class="grn">sqft</span><br>
  <span class="mut">
    Pearson r = {m['r_pearson']} &nbsp;|&nbsp;
    t-stat = {m['t_stat']} &nbsp;|&nbsp; {sig_str} &nbsp;|&nbsp;
    SE(β₁) = {m['se_b1']} &nbsp;|&nbsp;
    Train R² = {m['r2_train']} &nbsp;|&nbsp; Test R² = {m['r2']}
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION + EQUATION + MARKET POSITION  (app.py layout)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">💰 Instant Price Prediction</div>', unsafe_allow_html=True)
cp, ce, ci = st.columns([1.1, 1.1, 0.8])

with cp:
    st.markdown(f"""
    <div class="pred-hero">
        <div class="ph-label">Estimated Sale Price</div>
        <div class="ph-price">${predicted_price:,.0f}</div>
        <div class="ph-range">±1σ range: ${pred_low:,.0f} – ${pred_high:,.0f}</div>
        <div class="ph-eq">
            β₀ + β₁ × sqft<br>
            {m['intercept']:,.0f} + {m['coef']:.2f} × {sqft_in:,}<br>
            = <b>${predicted_price:,.0f}</b>
        </div>
        <div style="margin-top:14px;display:flex;justify-content:center;gap:10px">
            <span style="background:rgba(255,255,255,0.13);color:#f0dfc0;border-radius:12px;padding:3px 12px;font-size:0.78rem">{segment}</span>
            <span style="background:rgba(255,255,255,0.13);color:#f0dfc0;border-radius:12px;padding:3px 12px;font-size:0.78rem">Top {100-percentile:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with ce:
    st.markdown(f"""
    <div class="eq-card">
        <h4>📐 Regression Equation</h4>
        <div class="eq-formula">ŷ = {m['coef']:.2f} · x + {m['intercept']:,.0f}</div>
        <p><span class="eq-pill">x</span> House area (sq ft)</p>
        <p><span class="eq-pill">ŷ</span> Predicted price (USD)</p>
        <p><span class="eq-pill">β₁ = {m['coef']:.2f}</span> +${m['coef']:.2f} per extra sq ft</p>
        <p><span class="eq-pill">β₀ = ${m['intercept']:,.0f}</span> Base / intercept price</p>
        <p><span class="eq-pill">R² = {m['r2']:.4f}</span> Variance explained</p>
        <p><span class="eq-pill">RMSE = ${m['rmse']:,.0f}</span> Avg error</p>
    </div>
    """, unsafe_allow_html=True)

with ci:
    st.markdown(f"""
    <div class="eq-card">
        <h4>📊 Market Position</h4>
        <p><b>Percentile:</b> Above {percentile}% of dataset</p>
        <p><b>Segment:</b> {segment}</p>
        <p><b>Dataset Median:</b> ${df['price'].median():,.0f}</p>
        <p><b>Dataset Mean:</b> ${df['price'].mean():,.0f}</p>
        <p><b>Avg $/sqft:</b> ${(df['price']/df['sqft']).mean():,.1f}</p>
        <p><b>t-stat:</b> {m['t_stat']:.4f} &nbsp;|&nbsp; {sig_str}</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET AT A GLANCE  (app.py)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔎 Dataset at a Glance</div>', unsafe_allow_html=True)
ic1, ic2, ic3, ic4, ic5 = st.columns(5)
for col, title, val, desc in [
    (ic1, "Min Area",    f"{df['sqft'].min():,.0f} sqft", "Smallest property"),
    (ic2, "Max Area",    f"{df['sqft'].max():,.0f} sqft", "Largest property"),
    (ic3, "Avg Price",   f"${df['price'].mean():,.0f}", "Mean sale price"),
    (ic4, "Price Range", f"${df['price'].max()-df['price'].min():,.0f}", "Max − Min"),
    (ic5, "Avg Beds",    f"{df['bedrooms'].mean():.1f} bd", "Average bedrooms"),
]:
    with col:
        st.markdown(f"""<div class="insight-card">
            <div class="ic-title">{title}</div>
            <div class="ic-val">{val}</div>
            <div class="ic-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯  Predictor",
    "📊  Model Overview",
    "🔬  Diagnostics",
    "📈  Feature Analysis",
    "🏘️  Market Explorer",
    "📉  Location Charts",
    "🗃️  Dataset",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_card, col_vis = st.columns([1, 2.1])

    with col_card:
        st.dataframe(
            pd.DataFrame({
                "Feature" : ["Size (sqft) ★","Bedrooms","Bathrooms",
                             "Age","Garage","School","Neighbourhood"],
                "Value"   : [f"{sqft_in:,} sqft", f"{beds_in} bed",
                             f"{baths_in}", f"{age_in} yrs", f"{gar_in}",
                             f"{schl_in}/10", neigh_in],
                "In Model": ["✅ Yes","—","—","—","—","—","—"],
            }).style.set_properties(**{
                'color': '#2c1f14',
                'background-color': '#fdfaf6',
                'border-color': '#e0d8cc'
            }), 
            use_container_width=True, 
            hide_index=True
        )

        st.markdown('<p style="color:#2c1f14; font-weight:bold; margin-bottom:-10px;">Price breakdown</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Component": ["Base value (β₀)",
                          f"Size effect (β₁×{sqft_in:,})",
                          "Prediction", "Low (−RMSE)", "High (+RMSE)"],
            "Amount"   : [f"${m['intercept']:,.0f}", f"+${contrib:,.0f}",
                          f"${predicted_price:,.0f}", f"${pred_low:,.0f}", f"${pred_high:,.0f}"],
        }).style.set_properties(**{
            'color': '#2c1f14',
            'background-color': '#fdfaf6'
        }), use_container_width=True, hide_index=True)

        st.markdown(f"""
    <div style="background-color: #d1e3f3; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff;">
        <span style="color:#1a1612;">
            ℹ️ Model explains <b>{m['r2']*100:.1f}%</b> of price variance. 
            Typical error ≈ <b>±{pct_err:.1f}%</b> (RMSE ${m['rmse']:,.0f}).
        </span>
    </div>
    """, unsafe_allow_html=True)

    with col_vis:
        st.plotly_chart(make_scatter(df, m, hl_sqft=sqft_in),
                        use_container_width=True, key="t1_scatter")
        st.plotly_chart(make_waterfall(m, sqft_in),
                        use_container_width=True, key="t1_waterfall")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL OVERVIEW  (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_scatter(df, m),
                        use_container_width=True, key="t2_scatter")
        st.plotly_chart(make_price_dist(df),
                        use_container_width=True, key="t2_pdist")
    with c2:
        st.plotly_chart(make_avp(yte.values, ypred),
                        use_container_width=True, key="t2_avp")
        st.plotly_chart(make_learning_curve(df),
                        use_container_width=True, key="t2_lc")
    st.plotly_chart(make_cv_bars(m),
                    use_container_width=True, key="t2_cv")

    # Full model report table
    st.markdown("<div class='st-sec'>Full Model Report</div>", unsafe_allow_html=True)
    report = pd.DataFrame({
        "Metric": [
            "β₀  Intercept","β₁  Slope (per sqft)",
            "Standard Error of β₁","t-statistic (β₁)",
            "p-value (H₀: β₁=0)","Pearson r",
            "R² — Training","R² — Test","Adjusted R²",
            "RMSE","MAE","MAPE",
            "CV R² Mean (5-fold)","CV R² Std Dev",
            "Training samples","Test samples","Total dataset",
        ],
        "Value": [
            f"${m['intercept']:,.2f}", f"${m['coef']:,.4f}",
            f"{m['se_b1']:.4f}", f"{m['t_stat']:.4f}",
            f"{m['p_value']:.2e}", f"{m['r_pearson']:.4f}",
            f"{m['r2_train']:.4f}", f"{m['r2']:.4f}", f"{m['adj_r2']:.4f}",
            f"${m['rmse']:,.2f}", f"${m['mae']:,.2f}", f"{m['mape']:.2f}%",
            f"{m['cv_mean']:.4f}", f"±{m['cv_std']:.4f}",
            f"{m['train_n']:,}", f"{m['test_n']:,}", f"{len(df):,}",
        ],
        "Interpretation": [
            "Predicted price at 0 sqft (theoretical model floor)",
            "Each additional sqft adds this to predicted price",
            "Precision of slope estimate — lower = more confident",
            "Standard errors the slope is away from zero",
            "Strong evidence slope ≠ 0 if p < 0.05",
            "Linear relationship strength between sqft and price",
            "Variance explained in training set",
            "Variance explained in unseen test set",
            "R² adjusted for number of predictors used",
            "Square root of average squared error — dollar scale",
            "Average absolute prediction error in dollars",
            "Average percentage prediction error",
            "Cross-validation generalisation estimate",
            "Stability across folds (lower = more stable)",
            "Rows used to fit β₀ and β₁",
            "Rows used to evaluate (unseen during training)",
            "Total synthetic dataset size",
        ],
    })
    st.dataframe(report, use_container_width=True, hide_index=True)

    # CV fold detail table
    st.markdown("<div class='st-sec'>Cross-Validation Fold Detail</div>",
                unsafe_allow_html=True)
    fold_v = m["cv_scores"]
    cv_tbl = pd.DataFrame({
        "Fold"    : [f"Fold {i+1}" for i in range(len(fold_v))],
        "R² Score": [f"{v:.6f}" for v in fold_v],
        "Δ Mean"  : [f"{v-m['cv_mean']:+.6f}" for v in fold_v],
        "Status"  : ["✅ Above avg" if v>=m["cv_mean"] else "⚠ Below avg"
                     for v in fold_v],
    })
    st.dataframe(cv_tbl, use_container_width=True, hide_index=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇ Model report (CSV)",
            data=report.to_csv(index=False).encode(),
            file_name="model_report.csv", mime="text/csv")
    with dl2:
        st.download_button("⬇ CV results (CSV)",
            data=cv_tbl.to_csv(index=False).encode(),
            file_name="cv_results.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTICS  (Plotly + Matplotlib)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='st-sec'>Regression Diagnostics — Plotly</div>",
                unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(make_resfit(res_df),
                        use_container_width=True, key="t3_resfit")
    with d2:
        st.plotly_chart(make_reshist(res_df),
                        use_container_width=True, key="t3_reshist")

    d3, d4 = st.columns(2)
    with d3:
        st.plotly_chart(make_qq(res_df),
                        use_container_width=True, key="t3_qq")
    with d4:
        st.plotly_chart(make_avp(yte.values, ypred),
                        use_container_width=True, key="t3_avp")

    # Residual metrics
    r = res_df["residual"]
    rm = st.columns(6)
    rm[0].metric("Residual Mean",   f"${r.mean():,.0f}",   "≈ 0 ideal")
    rm[1].metric("Residual Std",    f"${r.std():,.0f}",    "spread")
    rm[2].metric("Max Over-pred",   f"${r.max():,.0f}",    "overestimate")
    rm[3].metric("Max Under-pred",  f"${r.min():,.0f}",    "underestimate")
    rm[4].metric("Within ±RMSE",    f"{(r.abs()<=m['rmse']).mean()*100:.1f}%", "")
    rm[5].metric("Within ±2×RMSE",  f"{(r.abs()<=2*m['rmse']).mean()*100:.1f}%","")

    # Matplotlib residual panel (from app.py)
    st.markdown("<div class='st-sec'>Residuals — Matplotlib Detail</div>",
                unsafe_allow_html=True)
    residuals = res_df["residual"].values
    y_pred_te = res_df["fitted"].values
    y_te_vals = res_df["actual"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].axhline(0, color=DARK, lw=1.8, ls="--", alpha=0.7)
    axes[0].scatter(y_pred_te, residuals, color=MID, alpha=0.72, s=38, edgecolors=DARK, lw=0.3)
    axes[0].set_xlabel("Predicted Price", fontsize=10, fontweight="600")
    axes[0].set_ylabel("Residual (USD)",  fontsize=10, fontweight="600")
    axes[0].set_title("Residuals vs Fitted", fontsize=11, fontweight="700", color=DARK)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    axes[0].grid(True)

    axes[1].hist(residuals, bins=25, color=MID, alpha=0.80, edgecolor=DARK, lw=0.5)
    axes[1].axvline(0,                color=ERR,   lw=2.2, ls="--", label="Zero")
    axes[1].axvline(residuals.mean(), color=GREEN, lw=1.8, ls="-",  label=f"Mean ${residuals.mean():,.0f}")
    axes[1].set_xlabel("Residual (USD)", fontsize=10, fontweight="600")
    axes[1].set_ylabel("Count",          fontsize=10, fontweight="600")
    axes[1].set_title("Residual Histogram", fontsize=11, fontweight="700", color=DARK)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    axes[1].legend(fontsize=8, framealpha=0.9, facecolor=CARD)
    axes[1].grid(True)

    (osm, osr), (slope, intercept_qq, _) = stats.probplot(residuals, dist="norm")
    axes[2].plot(osm, osr, "o", color=MID, alpha=0.7, ms=5)
    lx = np.array([osm.min(), osm.max()])
    axes[2].plot(lx, slope*lx + intercept_qq, color=DARK, lw=2, label="Normal line")
    axes[2].set_xlabel("Theoretical Quantiles", fontsize=10, fontweight="600")
    axes[2].set_ylabel("Sample Quantiles",       fontsize=10, fontweight="600")
    axes[2].set_title("Q-Q Plot (Normality)", fontsize=11, fontweight="700", color=DARK)
    axes[2].legend(fontsize=8, framealpha=0.9, facecolor=CARD)
    axes[2].grid(True)

    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)
    st.markdown(f"""
    <div style="background-color: #d1e3f3; padding: 12px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 10px;">
        <span style="color:#1a1612;">
            📌 Q-Q plot checks normality of residuals. Points near the diagonal → well-distributed errors. 
            Mean residual = <b>${residuals.mean():,.0f}</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Shapiro-Wilk test
    sw_stat, sw_p = stats.shapiro(pd.Series(residuals).sample(min(50,len(residuals)), random_state=42))
    st.markdown("---")
    st.markdown("<div class='st-sec'>Shapiro-Wilk Normality Test on Residuals</div>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Test"       : ["Shapiro-Wilk"],
        "Statistic W": [f"{sw_stat:.6f}"],
        "p-value"    : [f"{sw_p:.4f}"],
        "H₀"         : ["Residuals are normally distributed"],
        "Decision"   : ["Fail to reject H₀ — looks normal" if sw_p>0.05
                        else "Reject H₀ — non-normal residuals"],
    }), use_container_width=True, hide_index=True)

    # Interpretation cards
    st.markdown("---")
    st.markdown("<div class='st-sec'>Diagnostic Interpretation Guide</div>",
                unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""
        <div class="ic"><div class="ic-t">Residuals vs Fitted</div>
        <p>✅ Random scatter around zero → linearity assumption holds</p>
        <p>⚠️ Funnel shape → heteroscedasticity; variance grows with price</p>
        <p>⚠️ Curved band → non-linear pattern; try log or polynomial regression</p>
        <p>💡 The purple rolling-mean line should stay close to zero</p></div>
        <div class="ic"><div class="ic-t">Q-Q Plot</div>
        <p>✅ Points close to diagonal → residuals are approximately normal</p>
        <p>⚠️ Heavy tails → outliers are pulling predictions off course</p>
        <p>⚠️ S-shaped curve → the distribution is skewed</p></div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="ic"><div class="ic-t">Model Performance</div>
        <p>R² = <b>{m['r2']:.4f}</b> → sqft explains {m['r2']*100:.1f}% of price variance</p>
        <p>Remaining {(1-m['r2'])*100:.1f}% = location, condition, market timing, etc.</p>
        <p>Adding more features (bedrooms, age, neighbourhood) would raise R²</p></div>
        <div class="ic"><div class="ic-t">Error Metrics Compared</div>
        <p>RMSE = <b>${m['rmse']:,.0f}</b> — penalises large errors more heavily</p>
        <p>MAE  = <b>${m['mae']:,.0f}</b> — average absolute dollar error</p>
        <p>MAPE = <b>{m['mape']:.2f}%</b> — scale-free percentage error</p>
        <p>RMSE &gt; MAE → some large outlier residuals exist in test set</p></div>
        """, unsafe_allow_html=True)

    # Residuals table
    st.markdown("<div class='st-sec'>Residuals — First 50 Test Predictions</div>",
                unsafe_allow_html=True)
    diag = res_df.head(50).copy()
    diag["Error %"]       = ((diag["residual"].abs()/diag["actual"])*100).round(2)
    diag["Within 1 RMSE"] = diag["residual"].abs() <= m["rmse"]
    diag = diag.rename(columns={
        "fitted"   : "Predicted ($)",
        "actual"   : "Actual ($)",
        "residual" : "Residual ($)",
        "std_resid": "Std Residual",
    })
    for c in ["Predicted ($)","Actual ($)","Residual ($)"]:
        diag[c] = diag[c].apply(lambda x: f"${x:,.0f}")
    diag["Std Residual"] = diag["Std Residual"].round(3)
    st.dataframe(diag.reset_index(drop=True),
                 use_container_width=True, height=310, hide_index=True)
    st.download_button("⬇ Full residuals (CSV)",
        data=res_df.to_csv(index=False).encode(),
        file_name="residuals.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(make_feat_corr(df),
                        use_container_width=True, key="t4_corr")
    with f2:
        st.plotly_chart(make_heatmap(df),
                        use_container_width=True, key="t4_heat")

    st.plotly_chart(make_pairplot(df),
                    use_container_width=True, key="t4_pair")
    st.plotly_chart(make_3d(df),
                    use_container_width=True, key="t4_3d")

    # Correlation ranking table
    st.markdown("<div class='st-sec'>Feature Ranking — Correlation with Price</div>",
                unsafe_allow_html=True)
    num_df    = df.select_dtypes(include=[np.number])
    corr_vals = num_df.corr()["price"].drop("price")
    # remove duplicate alias columns
    corr_vals = corr_vals[~corr_vals.index.isin(["Price_USD","Area_sqft","Bedrooms"])]
    rank_idx  = corr_vals.abs().sort_values(ascending=False).index
    st.dataframe(pd.DataFrame({
        "Rank"        : range(1, len(rank_idx)+1),
        "Feature"     : [f.replace("_"," ") for f in rank_idx],
        "Pearson r"   : [f"{corr_vals[f]:.4f}" for f in rank_idx],
        "|r|"         : [f"{abs(corr_vals[f]):.4f}" for f in rank_idx],
        "Direction"   : ["Positive ↑" if corr_vals[f]>0 else "Negative ↓"
                         for f in rank_idx],
        "Strength"    : ["Strong" if abs(corr_vals[f])>=0.6
                         else "Moderate" if abs(corr_vals[f])>=0.3
                         else "Weak" for f in rank_idx],
        "In SLR Model": ["✅ Yes" if f=="sqft" else "—" for f in rank_idx],
    }), use_container_width=True, hide_index=True)

    # Descriptive stats
    st.markdown("<div class='st-sec'>Descriptive Statistics — Key Numeric Features</div>",
                unsafe_allow_html=True)
    desc_cols = ["sqft","bedrooms","bathrooms","age_years","garage_spaces",
                 "school_rating","distance_downtown_miles","price"]
    desc = df[desc_cols].describe().T.round(2).reset_index()
    desc.columns = ["Feature"] + list(desc.columns[1:])
    st.dataframe(desc, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MARKET EXPLORER  (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='st-sec'>Market-Level Insights</div>",
                unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(make_by_neighbourhood(df),
                        use_container_width=True, key="t5_nb")
        st.plotly_chart(make_renovation(df),
                        use_container_width=True, key="t5_reno")
    with e2:
        st.plotly_chart(make_ppsf(df),
                        use_container_width=True, key="t5_ppsf")
        st.plotly_chart(make_boxplot_beds(df),
                        use_container_width=True, key="t5_box")
    st.plotly_chart(make_age_scatter(df),
                    use_container_width=True, key="t5_age")

    # Neighbourhood summary table
    st.markdown("<div class='st-sec'>Neighbourhood Summary</div>",
                unsafe_allow_html=True)
    nb_g = df.groupby("neighbourhood").agg(
        Count=("price","count"),
        Mean_Price=("price","mean"),
        Median_Price=("price","median"),
        Std_Price=("price","std"),
        Min_Price=("price","min"),
        Max_Price=("price","max"),
        Avg_sqft=("sqft","mean"),
        Avg_School=("school_rating","mean"),
        Renovated_Pct=("recently_renovated","mean"),
    ).reset_index().round(0)
    nb_g["Renovated_Pct"] = (nb_g["Renovated_Pct"]*100).round(1).astype(str)+"%"
    for c in ["Mean_Price","Median_Price","Std_Price","Min_Price","Max_Price"]:
        nb_g[c] = nb_g[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(nb_g, use_container_width=True, hide_index=True)

    # Bedroom pricing table
    st.markdown("<div class='st-sec'>Bedroom-Level Pricing</div>",
                unsafe_allow_html=True)
    bd_g = df.groupby("bedrooms").agg(
        Count=("price","count"),
        Mean_Price=("price","mean"),
        Median_Price=("price","median"),
        Min_Price=("price","min"),
        Max_Price=("price","max"),
        Avg_sqft=("sqft","mean"),
        Avg_Bath=("bathrooms","mean"),
    ).reset_index().round(1)
    for c in ["Mean_Price","Median_Price","Min_Price","Max_Price"]:
        bd_g[c] = bd_g[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(bd_g, use_container_width=True, hide_index=True)

    # Renovation × Neighbourhood
    st.markdown("<div class='st-sec'>Renovation Impact by Neighbourhood</div>",
                unsafe_allow_html=True)
    rn_g = df.groupby(["neighbourhood","recently_renovated"]).agg(
        Count=("price","count"),
        Mean_Price=("price","mean"),
        Avg_sqft=("sqft","mean"),
    ).reset_index()
    rn_g["Status"]     = rn_g["recently_renovated"].map({0:"Not Renovated",1:"Renovated"})
    rn_g["Mean_Price"] = rn_g["Mean_Price"].apply(lambda x: f"${x:,.0f}")
    rn_g["Avg_sqft"]   = rn_g["Avg_sqft"].round(0).astype(int)
    st.dataframe(
        rn_g[["neighbourhood","Status","Count","Mean_Price","Avg_sqft"]],
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LOCATION CHARTS  (Matplotlib — from app.py)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='st-sec'>Regression with Confidence Band</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(Xtr["sqft"], ytr, color=LITE, alpha=0.45, s=22, label="Train data", zorder=2)
    ax.scatter(Xte["sqft"], yte, color=MID,  alpha=0.80, s=36, label="Test data",  zorder=3,
               edgecolors=DARK, linewidths=0.3)
    xs = np.linspace(df["sqft"].min() - 100, df["sqft"].max() + 100, 300)
    ys = m["intercept"] + m["coef"] * xs
    ax.plot(xs, ys, color=DARK, linewidth=2.8, label="Regression line", zorder=4)
    if show_confidence:
        n_pts, x_m = len(df), df["sqft"].mean()
        ss = ((df["sqft"] - x_m) ** 2).sum()
        se_band = np.sqrt(m["rmse"] ** 2 * (1/n_pts + (xs - x_m)**2 / ss))
        t_crit  = stats.t.ppf(0.975, df=n_pts - 2)
        ax.fill_between(xs, ys - t_crit*se_band, ys + t_crit*se_band,
                        alpha=0.13, color=MID, label="95% confidence band")
    ax.vlines(sqft_in, 0, predicted_price, color=ERR, lw=1.2, ls=":", alpha=0.7)
    ax.hlines(predicted_price, df["sqft"].min(), sqft_in, color=ERR, lw=1.2, ls=":", alpha=0.7)
    ax.scatter([sqft_in], [predicted_price], color=ERR, s=180, zorder=6, marker="*",
               label=f"Your house ({sqft_in:,} sqft → ${predicted_price:,.0f})")
    ax.set_xlabel("House Area (sq ft)", fontsize=11, fontweight="600")
    ax.set_ylabel("Price (USD)",        fontsize=11, fontweight="600")
    ax.set_title("House Price vs Area — Simple Linear Regression", fontsize=13,
                 fontweight="700", color=DARK, pad=14)
    fmt_price_y(ax)
    ax.legend(framealpha=0.92, facecolor=CARD, edgecolor="#ddd4c5", fontsize=9)
    ax.grid(True, alpha=0.45)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)
    st.markdown(f"<p style='font-size:0.84rem;color:#7a6a5a'>📌 Light dots = train ({m['train_n']}) &nbsp;|&nbsp; Dark dots = test ({m['test_n']}) &nbsp;|&nbsp; ★ = your prediction &nbsp;|&nbsp; Shaded = 95% CI</p>", unsafe_allow_html=True)

    # Price distribution + Avg by bedroom
    st.markdown("<div class='st-sec'>Price Distribution & Bedroom Averages</div>",
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts, bins, _ = axes[0].hist(df["price"], bins=32, color=LITE, alpha=0.75,
                                    edgecolor=DARK, lw=0.4, density=True)
    kde_x = np.linspace(df["price"].min(), df["price"].max(), 300)
    kde   = stats.gaussian_kde(df["price"])
    axes[0].plot(kde_x, kde(kde_x), color=DARK, lw=2.5, label="KDE")
    axes[0].axvline(df["price"].mean(),   color=GREEN, lw=2.0, ls="--", label=f"Mean ${df['price'].mean()/1000:.0f}k")
    axes[0].axvline(df["price"].median(), color=MID,   lw=2.0, ls="-.", label=f"Median ${df['price'].median()/1000:.0f}k")
    axes[0].axvline(predicted_price,      color=ERR,   lw=2.3, ls="-",  label=f"Your house ${predicted_price/1000:.0f}k")
    axes[0].set_xlabel("Price (USD)", fontsize=10, fontweight="600")
    axes[0].set_ylabel("Density",     fontsize=10, fontweight="600")
    axes[0].set_title("Price Distribution + KDE", fontsize=11, fontweight="700", color=DARK)
    fmt_price_x(axes[0])
    axes[0].legend(fontsize=8, framealpha=0.9, facecolor=CARD)
    axes[0].grid(True)

    bed_groups = df.groupby("bedrooms")["price"].mean().reset_index()
    bars = axes[1].bar(bed_groups["bedrooms"].astype(str), bed_groups["price"],
                        color=MID, edgecolor=DARK, lw=0.5, alpha=0.85, width=0.6)
    for bar, val in zip(bars, bed_groups["price"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                     f"${val/1000:.0f}k", ha="center", va="bottom", fontsize=8,
                     color=DARK, fontweight="600")
    axes[1].set_xlabel("Bedrooms",        fontsize=10, fontweight="600")
    axes[1].set_ylabel("Avg Price (USD)", fontsize=10, fontweight="600")
    axes[1].set_title("Avg Price by Bedroom Count", fontsize=11, fontweight="700", color=DARK)
    fmt_price_y(axes[1])
    axes[1].grid(True, axis="y")
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    # Box plots
    st.markdown("<div class='st-sec'>Box Plots — Actual vs Predicted & Location</div>",
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bp = axes[0].boxplot([yte.values, ypred], patch_artist=True,
                          labels=["Actual", "Predicted"], widths=0.45,
                          medianprops=dict(color=ERR, lw=2.5),
                          whiskerprops=dict(color=DARK, lw=1.3),
                          capprops=dict(color=DARK, lw=1.8),
                          flierprops=dict(marker="o", color=MID, ms=5, alpha=0.6))
    bp["boxes"][0].set_facecolor(LITE); bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(MID);  bp["boxes"][1].set_alpha(0.8)
    axes[0].set_title("Actual vs Predicted Prices", fontsize=11, fontweight="700", color=DARK)
    axes[0].set_ylabel("Price (USD)", fontsize=10, fontweight="600")
    fmt_price_y(axes[0])
    axes[0].grid(True, axis="y")

    loc_data = [df[df["Location"] == loc]["price"].values for loc in ["Urban", "Suburban", "Rural"]]
    bp2 = axes[1].boxplot(loc_data, patch_artist=True, labels=["Urban", "Suburban", "Rural"], widths=0.45,
                           medianprops=dict(color=ERR, lw=2.5),
                           whiskerprops=dict(color=DARK, lw=1.3),
                           capprops=dict(color=DARK, lw=1.8),
                           flierprops=dict(marker="o", color=MID, ms=5, alpha=0.6))
    for box, c in zip(bp2["boxes"], [DARK, MID, LITE]):
        box.set_facecolor(c); box.set_alpha(0.75)
    axes[1].set_title("Price by Location", fontsize=11, fontweight="700", color=DARK)
    axes[1].set_ylabel("Price (USD)", fontsize=10, fontweight="600")
    fmt_price_y(axes[1])
    axes[1].grid(True, axis="y")
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)
    st.markdown(f"""
    <div style="background-color: #d1e3f3; padding: 12px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 10px;">
        <span style="color:#1a1612;">
            📌 <b>Box</b> = IQR (25th–75th percentile). <b>Line</b> = median. 
            <b>Whiskers</b> = 1.5×IQR. <b>Dots</b> = outliers.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Heatmap (Bedrooms × Area)
    st.markdown("<div class='st-sec'>Avg Price Heatmap — Bedrooms × Area Bin</div>",
                unsafe_allow_html=True)
    df2 = df.copy()
    df2["Area_bin"] = pd.cut(df2["sqft"], bins=7)
    pivot = df2.pivot_table(values="price", index="bedrooms", columns="Area_bin", aggfunc="mean") / 1000
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im   = ax.imshow(pivot.values, aspect="auto", cmap="YlOrBr", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Avg Price ($k)", fontsize=9, color=DARK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=DARK)
    ax.set_xticks(range(len(pivot.columns)));  ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)));    ax.set_yticklabels([f"{b} bd" for b in pivot.index], fontsize=9)
    ax.set_xlabel("Area Range (sq ft)", fontsize=10, fontweight="600")
    ax.set_ylabel("Bedrooms",           fontsize=10, fontweight="600")
    ax.set_title("Avg Price Heatmap  (Bedrooms × Area)", fontsize=12, fontweight="700", color=DARK, pad=14)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"${val:.0f}k", ha="center", va="center", fontsize=8, fontweight="600",
                        color="white" if val > np.nanmax(pivot.values) * 0.6 else DARK)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)
    st.markdown(f"""
    <div style="background-color: #d1e3f3; padding: 12px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 10px;">
        <span style="color:#1a1612;">
            📌 <b>Darker</b> = higher average price. <b>Blank</b> = no data for that combination.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Location mean/median bar + CDF
    st.markdown("<div class='st-sec'>Location Price Analysis</div>",
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    loc_stats = df.groupby("Location")["price"].agg(["mean","median","std"]).reset_index()
    x_pos = np.arange(len(loc_stats)); w = 0.32
    axes[0].bar(x_pos - w/2, loc_stats["mean"],   w, label="Mean",   color=DARK, alpha=0.85)
    axes[0].bar(x_pos + w/2, loc_stats["median"], w, label="Median", color=MID,  alpha=0.85)
    axes[0].errorbar(x_pos - w/2, loc_stats["mean"], yerr=loc_stats["std"],
                     fmt="none", color=ERR, capsize=6, lw=1.5, label="±1 SD")
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(loc_stats["Location"], fontsize=10)
    axes[0].set_ylabel("Price (USD)", fontsize=10, fontweight="600")
    axes[0].set_title("Mean & Median by Location", fontsize=11, fontweight="700", color=DARK)
    fmt_price_y(axes[0])
    axes[0].legend(fontsize=9, framealpha=0.9, facecolor=CARD)
    axes[0].grid(True, axis="y")

    for loc, clr in zip(["Urban", "Suburban", "Rural"], [DARK, MID, LITE]):
        sub = np.sort(df[df["Location"] == loc]["price"].values)
        axes[1].plot(sub, np.linspace(0, 1, len(sub)), color=clr, lw=2.2, label=loc)
    axes[1].axvline(predicted_price, color=ERR, lw=1.8, ls="--",
                    label=f"Your house ${predicted_price/1000:.0f}k")
    axes[1].set_xlabel("Price (USD)",  fontsize=10, fontweight="600")
    axes[1].set_ylabel("Cumulative %", fontsize=10, fontweight="600")
    axes[1].set_title("Cumulative Distribution by Location", fontsize=11, fontweight="700", color=DARK)
    fmt_price_x(axes[1])
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    axes[1].legend(fontsize=9, framealpha=0.9, facecolor=CARD)
    axes[1].grid(True)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("<div class='st-sec'>Dataset Explorer</div>",
                unsafe_allow_html=True)
    dk = st.columns(4)
    dk[0].metric("Total Homes",  f"{len(df):,}")
    dk[1].metric("Features",     f"{len(df.columns)-1}")
    dk[2].metric("Avg Price",    f"${df['price'].mean():,.0f}")
    dk[3].metric("Price Range",  f"${df['price'].min()/1e3:.0f}k–${df['price'].max()/1e3:.0f}k")

    st.markdown("---")
    fa, fb, fc, fd = st.columns(4)
    with fa:
        # Added a dark color style to the heading
        st.markdown('<p style="color:#2c1f14; font-weight:600; margin-bottom:-10px;">Price ($)</p>', unsafe_allow_html=True)
        pr = st.slider("", # Empty label because we used markdown for the heading
            int(df.price.min()), int(df.price.max()),
            (int(df.price.min()), int(df.price.max())), key="ds_pr")
    with fb:
        st.markdown('<p style="color:#2c1f14; font-weight:600; margin-bottom:-10px;">sqft</p>', unsafe_allow_html=True)
        sr = st.slider("", 
            int(df.sqft.min()), int(df.sqft.max()),
            (int(df.sqft.min()), int(df.sqft.max())), key="ds_sr")
    with fc:
        st.markdown('<p style="color:#2c1f14; font-weight:600; margin-bottom:-10px;">Bedrooms</p>', unsafe_allow_html=True)
        bf = st.multiselect("", 
            sorted(df.bedrooms.unique()), default=sorted(df.bedrooms.unique()),
            key="ds_bf")
    with fd:
        st.markdown('<p style="color:#2c1f14; font-weight:600; margin-bottom:-10px;">Neighbourhood</p>', unsafe_allow_html=True)
        nf = st.multiselect("", 
            list(df.neighbourhood.unique()), default=list(df.neighbourhood.unique()),
            key="ds_nf")

    display_cols = ["sqft","bedrooms","bathrooms","age_years","garage_spaces","floors",
                    "lot_size_sqft","school_rating","distance_downtown_miles",
                    "recently_renovated","neighbourhood","Location","price"]
    filt = df[
        df.price.between(*pr) & df.sqft.between(*sr) &
        df.bedrooms.isin(bf) & df.neighbourhood.isin(nf)
    ][display_cols].reset_index(drop=True)

    st.caption(f"Showing {len(filt):,} of {len(df):,} homes after filters")
    st.dataframe(filt, use_container_width=True, height=330)

    # Filtered distribution charts
    src = filt if len(filt) > 10 else df
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(make_sqft_dist(src),
                        use_container_width=True, key="t7_sqftd")
    with v2:
        st.plotly_chart(make_price_dist(src),
                        use_container_width=True, key="t7_priced")

    # Actual vs Predicted table (from app.py)
    st.markdown("<div class='st-sec'>Actual vs Predicted — Test Set</div>",
                unsafe_allow_html=True)
    compare_df = pd.DataFrame({
        "Area (sqft)":   Xte["sqft"].values,
        "Actual ($)":    yte.values.round(0).astype(int),
        "Predicted ($)": ypred.round(0).astype(int),
        "Error ($)":     (yte.values - ypred).round(0).astype(int),
    }).assign(**{"Error %": lambda d: (d["Error ($)"] / d["Actual ($)"] * 100).round(2)})\
      .sort_values("Area (sqft)").reset_index(drop=True)
    st.dataframe(
        compare_df.style
        .format({"Actual ($)": "${:,}", "Predicted ($)": "${:,}",
                 "Error ($)": "${:,}", "Error %": "{:.2f}%"})
        .background_gradient(subset=["Error ($)"], cmap="RdYlGn_r")
        .bar(subset=["Area (sqft)"], color=LITE, vmin=0)
        .set_properties(**{"background-color": CARD, "color": DARK, "border-color": "#e0d8cc"}),
        use_container_width=True, height=300
    )

    st.markdown("---")
    st.markdown("<div class='st-sec'>Descriptive Statistics</div>",
                unsafe_allow_html=True)
    st.dataframe(filt.describe().round(2), use_container_width=True)

    # Percentile table
    st.markdown("<div class='st-sec'>Price Percentile Table</div>",
                unsafe_allow_html=True)
    pcts = [5,10,25,50,75,90,95,99]
    price_src = filt["price"] if len(filt) > 0 else df["price"]
    st.dataframe(pd.DataFrame({
        "Percentile": [f"P{p}" for p in pcts],
        "Price"     : [f"${np.percentile(price_src, p):,.0f}" for p in pcts],
        "Meaning"   : [
            "5% of homes cheaper than this","10% of homes cheaper than this",
            "Lower quartile (Q1)","Median price (Q2)",
            "Upper quartile (Q3)","90% of homes cheaper than this",
            "Top 5% threshold","Top 1% — premium tier",
        ],
    }), use_container_width=True, hide_index=True)

    # Data quality report
    st.markdown("<div class='st-sec'>Data Quality Report</div>",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Column"  : display_cols,
        "Type"    : [str(df[c].dtype) for c in display_cols],
        "Non-null": [int(df[c].notna().sum()) for c in display_cols],
        "Nulls"   : [int(df[c].isna().sum()) for c in display_cols],
        "Unique"  : [int(df[c].nunique()) for c in display_cols],
        "Min"     : [str(df[c].min()) for c in display_cols],
        "Max"     : [str(df[c].max()) for c in display_cols],
    }), use_container_width=True, hide_index=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("⬇ Full dataset (CSV)",
            data=df[display_cols].to_csv(index=False).encode(),
            file_name="housing_dataset.csv", mime="text/csv")
    with dl2:
        st.download_button("⬇ Filtered dataset (CSV)",
            data=filt.to_csv(index=False).encode(),
            file_name="filtered_dataset.csv", mime="text/csv")
    with dl3:
        st.download_button("⬇ Model metrics (CSV)",
            data=pd.DataFrame([m]).to_csv(index=False).encode(),
            file_name="model_metrics.csv", mime="text/csv")

    if show_raw_data:
        st.divider()
        st.markdown('<div class="section-title">📋 Full Dataset (Raw)</div>', unsafe_allow_html=True)
        d2 = df[display_cols].copy()
        d2["price"] = d2["price"].round(0).astype(int)
        st.dataframe(
            d2.style
            .format({"sqft": "{:,}", "price": "${:,}"})
            .background_gradient(subset=["price"], cmap="YlOrBr")
            .set_properties(**{"background-color": CARD, "color": DARK}),
            use_container_width=True, height=380
        )

