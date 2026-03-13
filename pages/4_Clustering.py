import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

st.set_page_config(page_title="Clustering · ShopEase", page_icon="👥", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.page-title{font-family:'DM Serif Display',serif;font-size:2.6rem;color:#1F3864;}
.persona-card{border-radius:14px;padding:1.4rem;color:white;margin-bottom:0.8rem;
              box-shadow:0 6px 20px rgba(0,0,0,0.12);}
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

models = load_models()
cust   = models['cluster_df']
summ   = models['cluster_summary']
colors = models['cluster_colors']
labels = models['cluster_labels']

PALETTE = ["#F6AE2D","#1F3864","#2E86AB","#3BB273"]

st.markdown('<div class="page-title">👥 Customer Clustering — Persona Segments</div>', unsafe_allow_html=True)
st.markdown("""
**Objective:** Segment ShopEase's 200 customers into behavioural personas to enable targeted marketing.  
**Model:** K-Means Clustering (k=4, StandardScaler normalisation, n_init=10)  
**Why K-Means?** Fast, interpretable, and well-suited for customer segmentation where we want distinct, non-overlapping groups for marketing strategy design.
""")
st.markdown("---")

# ── PERSONA CARDS ─────────────────────────────────────────────────────────────
persona_meta = {
    "Bargain Hunters":      ("🏷️","Buy frequently but heavily discount-dependent; low AOV; moderate return rate","Run flash sales and bundle offers; avoid full-price campaigns","#F6AE2D"),
    "Loyal High-Spenders":  ("💎","High total spend, strong brand loyalty, low return rate","VIP programme, early access to launches, premium packaging","#1F3864"),
    "Occasional Browsers":  ("🔍","Infrequent purchases, medium spend, low engagement","Re-engagement campaigns, personalised recommendations, exit-intent offers","#2E86AB"),
    "Premium Champions":    ("🏆","Highest AOV, excellent ratings, virtually zero returns","Co-create products, brand ambassador programme, referral incentives","#3BB273"),
}

c1, c2 = st.columns(2)
for i, (persona, (icon, desc, action, color)) in enumerate(persona_meta.items()):
    col = c1 if i % 2 == 0 else c2
    row = summ[summ['persona'] == persona]
    count = int(row['count'].values[0]) if len(row) > 0 else 0
    avg_s = float(row['avg_spent'].values[0]) if len(row) > 0 else 0
    with col:
        st.markdown(f"""
        <div class="persona-card" style="background:{color}">
          <div style="font-size:2rem">{icon}</div>
          <div style="font-size:1.2rem;font-weight:700;margin:0.3rem 0">{persona}</div>
          <div style="font-size:0.85rem;opacity:0.9;margin-bottom:0.8rem">{desc}</div>
          <div style="display:flex;gap:1.5rem;margin-bottom:0.6rem">
            <div><div style="font-size:1.4rem;font-weight:700">{count}</div><div style="font-size:0.72rem;opacity:0.8">Customers</div></div>
            <div><div style="font-size:1.4rem;font-weight:700">₹{avg_s:,.0f}</div><div style="font-size:0.72rem;opacity:0.8">Avg Spend</div></div>
          </div>
          <div style="background:rgba(255,255,255,0.2);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.82rem">
            💡 <b>Action:</b> {action}
          </div>
        </div>""", unsafe_allow_html=True)

# ── CLUSTER VISUALISATION ─────────────────────────────────────────────────────
st.markdown("---")
c3, c4 = st.columns(2)
with c3:
    st.markdown("**Cluster Scatter — Spend vs Orders**")
    fig1 = px.scatter(cust, x='total_orders', y='total_spent', color='persona',
                      color_discrete_sequence=PALETTE, size='avg_order_val',
                      hover_data=['customer_id','avg_rating','return_rate'],
                      labels={'total_orders':'Total Orders','total_spent':'Total Spend (₹)'})
    fig1.update_layout(height=380, plot_bgcolor='#F8FAFC', paper_bgcolor='white', margin=dict(t=10))
    st.plotly_chart(fig1, use_container_width=True)

with c4:
    st.markdown("**Cluster Radar — Persona Profiles**")
    radar_metrics = ['avg_spent','avg_orders','avg_discount','avg_rating']
    radar_labels  = ['Avg Spend','Avg Orders','Avg Discount','Avg Rating']
    fig2 = go.Figure()
    for i, row in summ.iterrows():
        vals = [float(row[m]) for m in radar_metrics]
        maxv = [summ[m].max() for m in radar_metrics]
        norm = [v/mx if mx>0 else 0 for v,mx in zip(vals,maxv)]
        norm.append(norm[0])
        fig2.add_trace(go.Scatterpolar(
            r=norm, theta=radar_labels+[radar_labels[0]],
            fill='toself', name=str(row['persona']),
            line=dict(color=PALETTE[i % len(PALETTE)]), opacity=0.7))
    fig2.update_layout(height=380, polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                       showlegend=True, paper_bgcolor='white', margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
st.markdown("**Cluster Summary Table**")
display_summ = summ.copy()
display_summ['avg_spent'] = display_summ['avg_spent'].apply(lambda x: f"₹{x:,.0f}")
display_summ['return_rate'] = display_summ['return_rate'].apply(lambda x: f"{x*100:.1f}%")
display_summ['avg_rating'] = display_summ['avg_rating'].apply(lambda x: f"{x:.2f}★")
st.dataframe(display_summ, use_container_width=True, hide_index=True)

st.markdown("""<div class="insight-box">
💡 <b>Clustering Insight:</b> K-Means reveals 4 distinct customer personas with very different profit contributions.
<b>Premium Champions</b> and <b>Loyal High-Spenders</b> together account for the majority of revenue with near-zero return rates —
ShopEase should protect these segments with VIP programmes and dedicated account management.
<b>Bargain Hunters</b> represent a conversion opportunity: personalised nudges toward full-price purchases during non-sale periods
can gradually shift them toward higher-margin buying behaviour.
</div>""", unsafe_allow_html=True)
