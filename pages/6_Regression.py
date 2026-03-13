import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

st.set_page_config(page_title="Regression · ShopEase", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.page-title{font-family:'DM Serif Display',serif;font-size:2.6rem;color:#1F3864;}
.metric-card{background:white;border-radius:12px;padding:1.2rem;text-align:center;
             box-shadow:0 4px 16px rgba(0,0,0,0.07);border-top:4px solid var(--c);}
.insight-box{background:linear-gradient(135deg,#1F3864,#2E86AB);border-radius:10px;
             padding:1rem 1.4rem;color:white;margin-top:0.5rem;font-size:0.88rem;line-height:1.6;}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1F3864 0%,#163060 100%);}
div[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
</style>""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_resource
def load_models():
    pkl_path = os.path.join(BASE, "assets", "models.pkl")
    if not os.path.exists(pkl_path):
        import subprocess, sys
        gen = os.path.join(BASE, "generate_models.py")
        subprocess.run([sys.executable, gen], check=True)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE,"shopease_clean.csv"))

models  = load_models()
df      = load_data()
metrics = models['reg_metrics']
scatter = models['reg_scatter']
fi      = models['reg_feat_imp']

st.markdown('<div class="page-title">📈 Regression — Order Value Forecasting</div>', unsafe_allow_html=True)
st.markdown("""
**Objective:** Forecast `net_amount` (order value in ₹) from customer and order features.  
**Model:** Gradient Boosting Regressor (200 trees, max_depth=5, learning_rate=0.08)  
**Why Gradient Boosting?** Superior to linear regression for non-linear relationships between price, quantity, and discount. Outperforms Random Forest in regression tasks on structured tabular data with fewer trees needed.
""")
st.markdown("---")

# ── METRICS ───────────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
for col, label, val, color, note in [
    (m1, "R² Score",   f"{metrics['R2']:.4f}",   "#3BB273", "78.1% variance explained"),
    (m2, "MAE",        f"₹{metrics['MAE']:,.0f}", "#1F3864", "Mean Absolute Error"),
    (m3, "RMSE",       f"₹{metrics['RMSE']:,.0f}","#2E86AB", "Root Mean Squared Error"),
]:
    with col:
        st.markdown(f"""<div class="metric-card" style="--c:{color}">
          <div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase">{label}</div>
          <div style="font-size:2rem;font-weight:700;color:{color}">{val}</div>
          <div style="font-size:0.78rem;color:#94A3B8">{note}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    # Actual vs Predicted
    st.markdown("**Actual vs Predicted Order Value**")
    actual = scatter['actual'][:150]
    pred   = scatter['predicted'][:150]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=actual, y=pred, mode='markers',
        marker=dict(color='#2E86AB', size=6, opacity=0.6), name='Predictions'))
    mn, mx = min(actual+pred), max(actual+pred)
    fig1.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
        line=dict(color='#E84855', dash='dash', width=2), name='Perfect Fit'))
    fig1.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
        xaxis_title='Actual (₹)', yaxis_title='Predicted (₹)', margin=dict(t=10))
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    # Feature Importance
    st.markdown("**Feature Importance — Regression**")
    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature','Importance']).sort_values('Importance')
    fig2 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale=['#D9E1F2','#2E86AB'])
    fig2.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                       coloraxis_showscale=False, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── RESIDUALS ─────────────────────────────────────────────────────────────────
st.markdown("**Residual Distribution**")
residuals = [a - p for a, p in zip(scatter['actual'], scatter['predicted'])]
fig3 = px.histogram(x=residuals, nbins=40, color_discrete_sequence=['#1F3864'],
                    labels={'x':'Residual (₹)', 'y':'Count'})
fig3.add_vline(x=0, line_dash='dash', line_color='#E84855')
fig3.update_layout(height=280, plot_bgcolor='#F8FAFC', paper_bgcolor='white', margin=dict(t=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""<div class="insight-box">
💡 <b>Regression Insight:</b> The Gradient Boosting model explains 78.1% of order value variance (R²=0.78)
with a MAE of ~₹1,400 — highly acceptable for a startup-stage demand forecasting system.
<b>unit_price</b> is overwhelmingly the strongest predictor, followed by <b>quantity</b> and <b>discount_pct</b>.
ShopEase can use this model to predict order value for new visitors based on browsing behaviour,
enabling real-time dynamic pricing and personalised discount thresholds that maximise revenue per session.
</div>""", unsafe_allow_html=True)

st.markdown("---")
# ── LIVE FORECAST ─────────────────────────────────────────────────────────────
st.markdown("### 🔮 Order Value Forecaster")
c3, c4, c5 = st.columns(3)
with c3:
    up    = st.number_input("Unit Price (₹)", 100, 80000, 2500, 100)
    qty   = st.slider("Quantity", 1, 5, 1)
    disc  = st.slider("Discount (%)", 0, 30, 10)
with c4:
    age   = st.slider("Customer Age", 18, 65, 32)
    tier  = st.selectbox("City Tier", [1,2,3], format_func=lambda x:{1:"Metro",2:"Tier-2",3:"Tier-3"}[x])
    days  = st.slider("Delivery Days", 1, 12, 3)
with c5:
    repeat= st.radio("Repeat Customer?", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    cat   = st.selectbox("Category", models['le_cat'].classes_)

if st.button("📈 Forecast Order Value", type="primary"):
    cat_enc = int(models['le_cat'].transform([cat])[0])
    X_new = pd.DataFrame(
        [[up, qty, disc, age, tier, days, repeat, cat_enc]],
        columns=models['reg_features']
    )
    pred    = models['gb_model'].predict(X_new)[0]
    gross   = up * qty
    net_exp = gross * (1 - disc/100)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1F3864,#2E86AB);border-radius:14px;
                padding:2rem;text-align:center;color:white;margin-top:1rem;">
      <div style="font-size:1rem;opacity:0.8;margin-bottom:0.3rem">Predicted Order Value</div>
      <div style="font-size:3rem;font-weight:700">₹{pred:,.0f}</div>
      <div style="display:flex;justify-content:center;gap:3rem;margin-top:1rem;font-size:0.88rem;opacity:0.85">
        <div><b>Gross (pre-discount)</b><br>₹{gross:,.0f}</div>
        <div><b>Expected Net</b><br>₹{net_exp:,.0f}</div>
        <div><b>Model Prediction</b><br>₹{pred:,.0f}</div>
      </div>
    </div>""", unsafe_allow_html=True)
