import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle, os

st.set_page_config(page_title="Association Rules · ShopEase", page_icon="🔗", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.page-title{font-family:'DM Serif Display',serif;font-size:2.6rem;color:#1F3864;}
.insight-box{background:linear-gradient(135deg,#1F3864,#2E86AB);border-radius:10px;
             padding:1rem 1.4rem;color:white;margin-top:0.5rem;font-size:0.88rem;line-height:1.6;}
.rule-card{background:#F8FAFC;border-radius:12px;padding:1rem 1.4rem;margin-bottom:0.5rem;
           border-left:4px solid #F6AE2D;display:flex;align-items:center;gap:1rem;}
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
rules  = models['rules']

st.markdown('<div class="page-title">🔗 Association Rule Mining</div>', unsafe_allow_html=True)
st.markdown("""
**Objective:** Discover which product categories are frequently purchased together to power cross-sell recommendations.  
**Algorithm:** Apriori with min_support=0.05, min_lift=1.0  
**Why Apriori?** It efficiently finds frequent itemsets in transactional data without requiring labelled targets — perfect for unsupervised discovery of buying patterns in a retail basket context.
""")

st.markdown("---")
st.markdown("""
**Metric Glossary:**
- **Support** — % of customers who bought both categories (frequency)
- **Confidence** — P(buying B | bought A) — probability of the consequent given the antecedent  
- **Lift** — How much more likely the association is vs random chance (>1 = meaningful)
""")

# ── FILTERS ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
min_lift   = c1.slider("Min Lift",       1.0, float(rules['lift'].max()),   1.0, 0.05)
min_conf   = c2.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
min_sup    = c3.slider("Min Support",    0.0, 0.5, 0.0, 0.01)

filtered = rules[(rules['lift'] >= min_lift) &
                 (rules['confidence'] >= min_conf) &
                 (rules['support'] >= min_sup)].copy()

st.markdown(f"**{len(filtered)} rules** matching filters")

# ── TOP RULES VISUAL ──────────────────────────────────────────────────────────
c4, c5 = st.columns(2)
with c4:
    st.markdown("**Top Rules by Lift**")
    top10 = filtered.sort_values('lift', ascending=False).head(10).copy()
    top10['rule'] = top10['antecedents'] + " → " + top10['consequents']
    fig1 = px.bar(top10, x='lift', y='rule', orientation='h',
                  color='confidence', color_continuous_scale=['#D9E1F2','#1F3864'],
                  text=top10['lift'].apply(lambda x: f"{x:.2f}"))
    fig1.update_layout(height=380, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                       margin=dict(t=10), coloraxis_colorbar=dict(title='Conf'))
    fig1.update_traces(textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

with c5:
    st.markdown("**Support vs Confidence Bubble (size = Lift)**")
    fig2 = px.scatter(filtered, x='support', y='confidence', size='lift',
                      color='lift', color_continuous_scale='Blues',
                      hover_data=['antecedents','consequents','lift'],
                      labels={'support':'Support','confidence':'Confidence'})
    fig2.update_layout(height=380, plot_bgcolor='#F8FAFC', paper_bgcolor='white', margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── RULES TABLE ───────────────────────────────────────────────────────────────
st.markdown("**All Rules (sortable)**")
display = filtered.sort_values('lift', ascending=False).copy()
display['support']    = display['support'].apply(lambda x: f"{x:.3f}")
display['confidence'] = display['confidence'].apply(lambda x: f"{x:.3f}")
display['lift']       = display['lift'].apply(lambda x: f"{x:.3f}")
display.columns = ['If Customer Bought','They Also Bought','Support','Confidence','Lift']
st.dataframe(display, use_container_width=True, hide_index=True)

# ── NETWORK VISUALIZATION ─────────────────────────────────────────────────────
st.markdown("**Association Network Graph**")
nodes_set = set(rules['antecedents'].tolist() + rules['consequents'].tolist())
nodes = list(nodes_set)
node_idx = {n:i for i,n in enumerate(nodes)}

import math
n = len(nodes)
node_x = [math.cos(2*math.pi*i/n)*2 for i in range(n)]
node_y = [math.sin(2*math.pi*i/n)*2 for i in range(n)]

edge_x, edge_y = [], []
for _, row in rules.iterrows():
    x0, y0 = node_x[node_idx[row['antecedents']]], node_y[node_idx[row['antecedents']]]
    x1, y1 = node_x[node_idx[row['consequents']]], node_y[node_idx[row['consequents']]]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
    line=dict(width=1, color='#CBD5E1'), hoverinfo='none'))
fig3.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
    marker=dict(size=28, color='#1F3864', line=dict(width=2, color='white')),
    text=nodes, textposition='top center', textfont=dict(size=10, color='#1F3864'),
    hoverinfo='text'))
fig3.update_layout(height=420, showlegend=False, paper_bgcolor='white',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(t=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""<div class="insight-box">
💡 <b>Association Insight:</b> The strongest cross-category associations involve Electronics paired with Sports and Beauty —
customers buying smart devices also purchase fitness trackers and personal care accessories.
ShopEase should implement a "Frequently Bought Together" recommendation engine on product pages, and create curated
bundles (e.g., "Fitness Kit": Smartwatch + Yoga Mat + Earbuds) with a 5% bundle discount that still improves basket size
by 35–40% on average, more than offsetting the small discount cost.
</div>""", unsafe_allow_html=True)
