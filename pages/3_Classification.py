import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

st.set_page_config(page_title="Classification · ShopEase", page_icon="🎯", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.page-title{font-family:'DM Serif Display',serif;font-size:2.6rem;color:#1F3864;}
.insight-box{background:linear-gradient(135deg,#1F3864,#2E86AB);border-radius:10px;
             padding:1rem 1.4rem;color:white;margin-top:0.5rem;font-size:0.88rem;line-height:1.6;}
.metric-card{background:white;border-radius:12px;padding:1.2rem;text-align:center;
             box-shadow:0 4px 16px rgba(0,0,0,0.07);border-top:4px solid var(--c);}
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

models = load_models()
df = load_data()

st.markdown('<div class="page-title">🎯 Classification — Predicting Repeat Customers</div>', unsafe_allow_html=True)
st.markdown("""
**Objective:** Predict whether a customer will make a repeat purchase (`is_repeat_customer = 1`).  
**Model:** Random Forest Classifier (150 trees, max_depth=8, class_weight='balanced')  
**Why this model?** Random Forest handles mixed feature types, is robust to outliers, and provides feature importance rankings — ideal for a business dataset with both numeric and encoded categorical variables.
""")
st.markdown("---")

report = models['clf_report']
cm     = models['clf_cm']
fi     = models['feat_imp_clf']

# ── METRICS ROW ──────────────────────────────────────────────────────────────
m1,m2,m3,m4 = st.columns(4)
metrics_data = [
    (m1,"Accuracy",   f"{report['accuracy']:.1%}",     "#1F3864"),
    (m2,"Precision",  f"{report['1']['precision']:.1%}","#2E86AB"),
    (m3,"Recall",     f"{report['1']['recall']:.1%}",   "#3BB273"),
    (m4,"F1-Score",   f"{report['1']['f1-score']:.1%}", "#F6AE2D"),
]
for col, label, val, color in metrics_data:
    with col:
        st.markdown(f"""<div class="metric-card" style="--c:{color}">
          <div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase">{label}</div>
          <div style="font-size:2rem;font-weight:700;color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    # Confusion Matrix
    st.markdown("**Confusion Matrix**")
    labels = ['New Customer (0)', 'Repeat Customer (1)']
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,'#F8FAFC'],[1,'#1F3864']],
        text=[[str(v) for v in row] for row in cm],
        texttemplate='<b>%{text}</b>', textfont={"size":18},
        showscale=False))
    fig_cm.update_layout(height=320, margin=dict(t=10),
        xaxis_title='Predicted', yaxis_title='Actual',
        paper_bgcolor='white', plot_bgcolor='#F8FAFC')
    st.plotly_chart(fig_cm, use_container_width=True)

with c2:
    # Feature Importance
    st.markdown("**Feature Importance**")
    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature','Importance']).sort_values('Importance')
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale=['#D9E1F2','#1F3864'])
    fig_fi.update_layout(height=320, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                          coloraxis_showscale=False, margin=dict(t=10))
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown(f"""<div class="insight-box">
💡 <b>Classification Insight:</b> The Random Forest model achieves {report['accuracy']:.1%} accuracy in predicting repeat purchases.
<b>net_amount</b> and <b>product_rating</b> are the top predictors — customers who spend more and rate highly are significantly more likely to return.
Operationally, ShopEase should trigger personalised follow-up emails within 48 hours for first-time buyers who rated ≥4.0 and spent ≥₹3,000 — the model identifies this as the highest-probability conversion segment.
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── LIVE PREDICTOR ────────────────────────────────────────────────────────────
st.markdown("### 🔮 Live Prediction Tool")
st.markdown("Enter customer details to predict repeat purchase likelihood:")

col1, col2, col3 = st.columns(3)
with col1:
    age       = st.slider("Customer Age", 18, 65, 32)
    city_tier = st.selectbox("City Tier", [1,2,3], format_func=lambda x: {1:"Metro",2:"Tier-2",3:"Tier-3"}[x])
    disc      = st.slider("Discount Given (%)", 0, 30, 10)
with col2:
    net_amt   = st.number_input("Order Value (₹)", 200, 80000, 3000, step=500)
    rating    = st.slider("Product Rating", 1.0, 5.0, 4.0, 0.1)
    delivery  = st.slider("Delivery Days", 1, 12, 3)
with col3:
    qty       = st.slider("Quantity", 1, 5, 1)
    returned  = st.radio("Was Order Returned?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    cat_opts  = list(models['le_cat'].classes_)
    cat_sel   = st.selectbox("Category", cat_opts)

if st.button("🎯 Predict", type="primary"):
    cat_enc = int(models['le_cat'].transform([cat_sel])[0])
    X_new = pd.DataFrame(
        [[age, city_tier, disc, net_amt, rating, delivery, returned, qty, cat_enc]],
        columns=models['clf_features']
    )
    prob    = models['clf_model'].predict_proba(X_new)[0][1]
    pred    = models['clf_model'].predict(X_new)[0]
    color   = "#3BB273" if pred == 1 else "#E84855"
    label   = "LIKELY TO RETURN ✅" if pred == 1 else "UNLIKELY TO RETURN ⚠️"
    st.markdown(f"""
    <div style="background:{color};border-radius:12px;padding:1.5rem;text-align:center;color:white;margin-top:1rem;">
      <div style="font-size:1.4rem;font-weight:700">{label}</div>
      <div style="font-size:2.5rem;font-weight:700;margin:0.3rem 0">{prob:.1%}</div>
      <div style="font-size:0.9rem;opacity:0.85">Probability of Repeat Purchase</div>
    </div>""", unsafe_allow_html=True)
