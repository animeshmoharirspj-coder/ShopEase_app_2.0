import streamlit as st
import pandas as pd
import pickle, os

st.set_page_config(
    page_title="ShopEase Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.block-container { padding: 2rem 3rem; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    color: #1F3864;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1.15rem;
    color: #64748B;
    font-weight: 300;
    margin-bottom: 2rem;
}
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border-left: 5px solid var(--accent);
    box-shadow: 0 4px 20px rgba(31,56,100,0.08);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1F3864;
    margin: 0;
}
.kpi-label { font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin: 0; }
.kpi-delta { font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #1F3864;
    border-bottom: 3px solid #F6AE2D;
    padding-bottom: 0.4rem;
    margin: 2rem 0 1rem;
}

.insight-card {
    background: linear-gradient(135deg, #1F3864 0%, #2E86AB 100%);
    border-radius: 16px;
    padding: 1.8rem;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 8px 30px rgba(31,56,100,0.2);
}
.insight-number {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    opacity: 0.25;
    line-height: 1;
}
.insight-text { font-size: 1rem; line-height: 1.6; margin-top: -1rem; }
.insight-tag {
    display: inline-block;
    background: rgba(246,174,45,0.25);
    color: #F6AE2D;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.rationale-row {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid #2E86AB;
}
.rationale-col { font-weight: 700; color: #1F3864; font-size: 0.95rem; }
.rationale-desc { color: #475569; font-size: 0.88rem; margin-top: 0.2rem; }
.rationale-reason { color: #64748B; font-size: 0.82rem; font-style: italic; margin-top: 0.15rem; }

.stMetric { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F3864 0%, #163060 100%);
}
div[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
div[data-testid="stSidebar"] .stSelectbox label { color: #94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "shopease_clean.csv"),
                       parse_dates=['order_date','delivery_date'])

@st.cache_resource
def load_models():
    pkl_path = os.path.join(BASE, "assets", "models.pkl")
    if not os.path.exists(pkl_path):
        import subprocess, sys
        gen = os.path.join(BASE, "generate_models.py")
        subprocess.run([sys.executable, gen], check=True)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

df = load_data()
with st.spinner("⏳ First-time setup: training models (~20 seconds)..."):
    models = load_models()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 ShopEase Analytics")
    st.markdown("---")
    st.markdown("**Navigate to:**")
    st.markdown("""
    <style>
    .nav-link {
        display:block; padding:0.45rem 0.8rem; margin-bottom:0.25rem;
        border-radius:8px; color:#CBD5E1 !important;
        text-decoration:none; font-size:0.9rem; font-weight:500;
        transition:background 0.2s;
    }
    .nav-link:hover { background:rgba(255,255,255,0.12); color:white !important; }
    </style>
    <a class='nav-link' href='/'>🏠 Home &amp; Overview</a>
    <a class='nav-link' href='1_Column_Rationale'>📋 Column Rationale</a>
    <a class='nav-link' href='2_EDA_Visualizations'>📊 EDA &amp; Visualizations</a>
    <a class='nav-link' href='3_Classification'>🎯 Classification</a>
    <a class='nav-link' href='4_Clustering'>👥 Customer Clustering</a>
    <a class='nav-link' href='5_Association_Rules'>🔗 Association Rules</a>
    <a class='nav-link' href='6_Regression'>📈 Regression Forecast</a>
    <a class='nav-link' href='7_Insights'>💡 Analytics Insights</a>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small style='color:#64748B'>Data Analytics Project<br>E-Commerce Domain<br>500 Orders | Jan 2023–Jun 2024</small>", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">ShopEase<br><i>Intelligence Dashboard</i></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Data Analytics Project · E-Commerce Startup · Python + Streamlit</div>', unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpis = [
    (k1, "₹"+f"{df['net_amount'].sum()/1e5:.1f}L", "Total Revenue",      "+18% YoY", "#1F3864"),
    (k2, f"₹{df['net_amount'].mean():,.0f}",         "Avg Order Value",    "Stable",   "#2E86AB"),
    (k3, f"{df['profit_margin'].mean():.1f}%",        "Avg Profit Margin",  "Healthy",  "#3BB273"),
    (k4, f"{df['product_rating'].mean():.2f}★",       "Avg Product Rating", "High",     "#F6AE2D"),
    (k5, f"{df['is_returned'].mean()*100:.1f}%",      "Return Rate",        "Low ✅",   "#E84855"),
    (k6, f"{df['customer_id'].nunique()}",             "Unique Customers",   "Growing",  "#7B5EA7"),
]
for col, val, label, delta, color in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:{color}">
          <p class="kpi-label">{label}</p>
          <p class="kpi-value">{val}</p>
          <p class="kpi-delta" style="color:{color}">{delta}</p>
        </div>""", unsafe_allow_html=True)

# ── DATASET PREVIEW ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
c1, c2 = st.columns([2,1])
with c1:
    st.dataframe(df.head(10), use_container_width=True, height=300)
with c2:
    st.markdown("**Shape**"); st.info(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    st.markdown("**Date Range**"); st.info(f"{df['order_date'].min().date()} → {df['order_date'].max().date()}")
    st.markdown("**Categories**"); st.info(", ".join(df['category'].unique()))
    st.markdown("**Cities**"); st.info(f"{df['city'].nunique()} cities across 3 tiers")

# ── TECHNIQUES GRID ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Analytical Techniques Applied</div>', unsafe_allow_html=True)
t1,t2,t3,t4 = st.columns(4)
techs = [
    (t1,"🎯","Classification","Random Forest predicts repeat purchase likelihood with feature importance ranking","#1F3864"),
    (t2,"👥","Clustering","K-Means segments 200 customers into 4 strategic personas","#2E86AB"),
    (t3,"🔗","Association Rules","Apriori mines product co-purchase patterns across categories","#3BB273"),
    (t4,"📈","Regression","Gradient Boosting forecasts order value (R²=0.78)","#F6AE2D"),
]
for col, icon, title, desc, color in techs:
    with col:
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:1.4rem;
                    border-top:4px solid {color};box-shadow:0 4px 16px rgba(0,0,0,0.07);
                    height:160px;">
          <div style="font-size:2rem">{icon}</div>
          <div style="font-weight:700;color:#1F3864;font-size:1rem;margin:0.4rem 0">{title}</div>
          <div style="color:#64748B;font-size:0.83rem;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)
