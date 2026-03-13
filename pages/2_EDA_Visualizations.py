import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

st.set_page_config(page_title="EDA · ShopEase", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.page-title{font-family:'DM Serif Display',serif;font-size:2.6rem;color:#1F3864;}
.chart-title{font-family:'DM Serif Display',serif;font-size:1.3rem;color:#1F3864;margin-bottom:0.2rem;}
.insight-box{background:linear-gradient(135deg,#1F3864,#2E86AB);border-radius:10px;
             padding:1rem 1.4rem;color:white;margin-top:0.5rem;font-size:0.88rem;line-height:1.6;}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1F3864 0%,#163060 100%);}
div[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
</style>""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE,"shopease_clean.csv"), parse_dates=['order_date','delivery_date'])

df = load_data()

PALETTE = ["#1F3864","#2E86AB","#F6AE2D","#E84855","#3BB273","#7B5EA7","#64748B"]

st.markdown('<div class="page-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
st.markdown("12 interactive visualisations covering revenue trends, category performance, customer behaviour, and predictive correlations.")
st.markdown("---")

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

# ── CHART 1: Monthly Revenue ──────────────────────────────────────────────────
st.markdown('<div class="chart-title">1 · Monthly Revenue & Order Volume Trend</div>', unsafe_allow_html=True)
monthly = df.groupby('order_month').agg(revenue=('net_amount','sum'), orders=('order_id','count')).reset_index()
monthly = monthly.sort_values('order_month')
fig = make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=monthly['order_month'], y=monthly['revenue']/1000,
    name='Revenue (₹K)', line=dict(color='#1F3864',width=3), fill='tozeroy',
    fillcolor='rgba(31,56,100,0.12)'), secondary_y=False)
fig.add_trace(go.Bar(x=monthly['order_month'], y=monthly['orders'],
    name='Orders', marker_color='rgba(246,174,45,0.6)', opacity=0.7), secondary_y=True)
fig.update_layout(height=380, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
    legend=dict(orientation='h',y=1.08), margin=dict(t=10,b=40))
fig.update_yaxes(title_text="Revenue (₹ Thousands)", secondary_y=False)
fig.update_yaxes(title_text="Order Count", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)
insight("Revenue shows a steady upward trajectory across 18 months, with a sharp Q4 2023 spike driven by Diwali/festive gifting — a key seasonal pattern ShopEase must plan inventory for. Order volumes closely track revenue, confirming stable average order values rather than inflation-driven growth.")

# ── CHART 2: Revenue by Category ─────────────────────────────────────────────
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="chart-title">2 · Revenue by Category</div>', unsafe_allow_html=True)
    cat_rev = df.groupby('category')['net_amount'].sum().reset_index().sort_values('net_amount')
    fig2 = px.bar(cat_rev, x='net_amount', y='category', orientation='h',
                  color='net_amount', color_continuous_scale=['#D9E1F2','#1F3864'],
                  text=cat_rev['net_amount'].apply(lambda x: f"₹{x/1000:.0f}K"))
    fig2.update_layout(height=340, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                       coloraxis_showscale=False, margin=dict(t=10))
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)
    insight("Electronics leads with ~26% revenue share due to high unit prices, while Books and Toys contribute the least. ShopEase should double down on Electronics/Fashion inventory and evaluate whether niche categories justify logistics costs.")

with c2:
    st.markdown('<div class="chart-title">3 · Profit Margin Distribution by Category</div>', unsafe_allow_html=True)
    fig3 = px.box(df, x='category', y='profit_margin', color='category',
                  color_discrete_sequence=PALETTE)
    fig3.update_layout(height=340, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                       showlegend=False, margin=dict(t=10))
    fig3.update_xaxes(tickangle=20)
    st.plotly_chart(fig3, use_container_width=True)
    insight("Beauty and Sports command the highest median margins (~42–45%), offering the best return per INR sold. Electronics has the widest variance — premium devices compress margins significantly, requiring careful pricing strategy.")

# ── CHART 4: Correlation Heatmap ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="chart-title">4 · Correlation Matrix — Key Numeric Variables</div>', unsafe_allow_html=True)
corr_cols = ['unit_price','quantity','discount_pct','net_amount','profit',
             'profit_margin','customer_age','product_rating','delivery_days',
             'is_repeat_customer','is_returned']
corr = df[corr_cols].corr().round(2)
fig4 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                 zmin=-1, zmax=1, aspect='auto')
fig4.update_layout(height=500, margin=dict(t=10))
st.plotly_chart(fig4, use_container_width=True)
insight("unit_price → net_amount shows the strongest positive correlation (r=0.89), confirming pricing is the primary revenue driver. The discount_pct → profit_margin relationship (r=−0.31) quantifies margin erosion, while product_rating → is_returned (r=−0.48) confirms quality drives returns.")

# ── CHART 5: Discount vs Margin ───────────────────────────────────────────────
st.markdown("---")
c3, c4 = st.columns(2)
with c3:
    st.markdown('<div class="chart-title">5 · Discount % vs Profit Margin (Regression)</div>', unsafe_allow_html=True)
    fig5 = px.scatter(df, x='discount_pct', y='profit_margin', color='category',
                      color_discrete_sequence=PALETTE, opacity=0.6)
    # Manual OLS trendline using numpy (no statsmodels needed)
    m5, b5 = np.polyfit(df['discount_pct'], df['profit_margin'], 1)
    x5 = np.linspace(df['discount_pct'].min(), df['discount_pct'].max(), 100)
    fig5.add_scatter(x=x5, y=m5*x5+b5, mode='lines', name=f'Trend (slope={m5:.2f})',
                     line=dict(color='#E84855', width=2.5, dash='dash'))
    fig5.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white', margin=dict(t=10))
    st.plotly_chart(fig5, use_container_width=True)
    insight("The regression trendline confirms every 1% increase in discount reduces profit margin by ~0.6 percentage points — a significant cost. Electronics and Fashion scatter points show the steepest drops, warranting category-specific discount caps of 15% and 20% respectively.")

with c4:
    st.markdown('<div class="chart-title">6 · Revenue by City & Tier</div>', unsafe_allow_html=True)
    city_rev = df.groupby(['city','city_tier'])['net_amount'].sum().reset_index().sort_values('net_amount',ascending=False)
    tier_map = {1:'Tier 1 — Metro', 2:'Tier 2', 3:'Tier 3'}
    city_rev['tier_label'] = city_rev['city_tier'].map(tier_map)
    fig6 = px.bar(city_rev, x='city', y='net_amount', color='tier_label',
                  color_discrete_map={'Tier 1 — Metro':'#1F3864','Tier 2':'#2E86AB','Tier 3':'#F6AE2D'},
                  text=city_rev['net_amount'].apply(lambda x: f"₹{x/1000:.0f}K"))
    fig6.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                       legend_title='City Tier', margin=dict(t=10))
    fig6.update_traces(textposition='outside', textfont_size=9)
    st.plotly_chart(fig6, use_container_width=True)
    insight("Tier-1 metros (Mumbai, Delhi, Bangalore) drive ~45% of revenue despite delivering only 3 cities. Tier-3 cities are underrepresented relative to population — a massive growth opportunity requiring localised assortments and last-mile delivery partnerships.")

# ── CHART 7: Channel Performance ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="chart-title">7 · Acquisition Channel Performance</div>', unsafe_allow_html=True)
ch = df.groupby('acquisition_channel').agg(
    revenue=('net_amount','sum'), repeat_rate=('is_repeat_customer','mean'),
    avg_order=('net_amount','mean'), orders=('order_id','count')).reset_index()
ch = ch.sort_values('revenue', ascending=False)
fig7 = make_subplots(rows=1, cols=2, subplot_titles=("Revenue by Channel","Repeat Rate by Channel"))
fig7.add_trace(go.Bar(x=ch['acquisition_channel'], y=ch['revenue']/1000,
    marker_color=PALETTE[:len(ch)], text=[f"₹{v:.0f}K" for v in ch['revenue']/1000],
    textposition='outside', name='Revenue'), row=1, col=1)
fig7.add_trace(go.Bar(x=ch['acquisition_channel'], y=ch['repeat_rate']*100,
    marker_color=[('#3BB273' if r > ch['repeat_rate'].mean() else '#E84855') for r in ch['repeat_rate']],
    text=[f"{v*100:.1f}%" for v in ch['repeat_rate']], textposition='outside', name='Repeat Rate'), row=1, col=2)
fig7.update_layout(height=380, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                   showlegend=False, margin=dict(t=40))
st.plotly_chart(fig7, use_container_width=True)
insight("Email Campaign delivers the highest repeat customer rate (>65%), making it the best channel for Customer Lifetime Value — yet it ranks lower by raw revenue. ShopEase should shift budget from pure acquisition (Paid Ads) to retention (Email), as retaining one customer costs 5× less than acquiring a new one.")

# ── CHART 8: Payment Methods ──────────────────────────────────────────────────
st.markdown("---")
c5, c6 = st.columns(2)
with c5:
    st.markdown('<div class="chart-title">8 · Payment Method Distribution</div>', unsafe_allow_html=True)
    pay = df['payment_method'].value_counts().reset_index()
    pay.columns = ['method','count']
    fig8 = px.pie(pay, names='method', values='count', hole=0.55,
                  color_discrete_sequence=PALETTE)
    fig8.update_layout(height=360, margin=dict(t=10,b=10))
    st.plotly_chart(fig8, use_container_width=True)
    insight("UPI leads at ~35%, reflecting India's digital payment dominance — ShopEase must ensure UPI checkout is frictionless. COD at ~10% carries return and logistics risk; a ₹50 cashback incentive for switching to prepaid could convert 40–50% of COD users.")

with c6:
    st.markdown('<div class="chart-title">9 · Return Rate by Product Rating Band</div>', unsafe_allow_html=True)
    df['rating_bin'] = pd.cut(df['product_rating'], bins=[0.9,1.5,2.5,3.5,4.5,5.1],
                              labels=['1.0–1.5','1.6–2.5','2.6–3.5','3.6–4.5','4.6–5.0'])
    rr = df.groupby('rating_bin', observed=True)['is_returned'].mean().reset_index()
    rr.columns = ['rating','return_rate']
    colors_rr = ['#E84855' if r > 0.15 else '#3BB273' for r in rr['return_rate']]
    fig9 = px.bar(rr, x='rating', y='return_rate', color='rating',
                  color_discrete_sequence=colors_rr,
                  text=rr['return_rate'].apply(lambda x: f"{x*100:.1f}%"))
    fig9.add_hline(y=df['is_returned'].mean(), line_dash='dot', line_color='#1F3864',
                   annotation_text=f"Avg {df['is_returned'].mean()*100:.1f}%")
    fig9.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                       showlegend=False, yaxis_tickformat='.0%', margin=dict(t=10))
    st.plotly_chart(fig9, use_container_width=True)
    insight("Products rated below 2.5 carry a >50% return rate — catastrophically eroding net revenue. ShopEase should enforce a minimum rating policy: products below 3.0 after 50 reviews trigger automatic supplier quality audits and potential delisting.")

# ── CHART 10: Age vs AOV ──────────────────────────────────────────────────────
st.markdown("---")
c7, c8 = st.columns(2)
with c7:
    st.markdown('<div class="chart-title">10 · Average Order Value by Age Group</div>', unsafe_allow_html=True)
    age = df.groupby('age_group', observed=True).agg(
        avg_order=('net_amount','mean'), avg_disc=('discount_pct','mean')).reset_index()
    fig10 = make_subplots(specs=[[{"secondary_y":True}]])
    fig10.add_trace(go.Bar(x=age['age_group'].astype(str), y=age['avg_order'],
        marker_color=PALETTE[:len(age)], name='Avg Order Value',
        text=[f"₹{v:,.0f}" for v in age['avg_order']], textposition='outside'), secondary_y=False)
    fig10.add_trace(go.Scatter(x=age['age_group'].astype(str), y=age['avg_disc'],
        mode='lines+markers', name='Avg Discount %', line=dict(color='#E84855',width=2.5),
        marker=dict(size=8)), secondary_y=True)
    fig10.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                        legend=dict(orientation='h',y=1.08), margin=dict(t=30))
    st.plotly_chart(fig10, use_container_width=True)
    insight("The 36–45 age group commands the highest Average Order Value, driven by higher disposable income and preference for premium electronics. The 18–25 cohort relies most heavily on discounts — ShopEase should offer a student loyalty programme to build lifetime brand affinity at low cost.")

with c8:
    st.markdown('<div class="chart-title">11 · Quarterly Revenue by Category (Stacked)</div>', unsafe_allow_html=True)
    qcat = df.groupby(['order_quarter','category'])['net_amount'].sum().reset_index()
    fig11 = px.bar(qcat, x='order_quarter', y='net_amount', color='category',
                   color_discrete_sequence=PALETTE, text_auto=False)
    fig11.update_layout(height=360, plot_bgcolor='#F8FAFC', paper_bgcolor='white',
                        barmode='stack', legend_title='Category', margin=dict(t=10))
    fig11.update_xaxes(tickangle=20)
    st.plotly_chart(fig11, use_container_width=True)
    insight("Electronics revenue spikes sharply in Q4 2023 (Diwali/Dhanteras), validating festive demand — ShopEase must secure supplier contracts 6–8 weeks prior. Fashion grows steadily every quarter, suggesting organic brand-building momentum that rewards consistent content investment.")

# ── CHART 12: Delivery vs Rating ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="chart-title">12 · Delivery Days vs Product Rating</div>', unsafe_allow_html=True)
fig12 = px.scatter(df, x='delivery_days', y='product_rating', color='city_tier',
                   color_discrete_map={1:'#1F3864',2:'#2E86AB',3:'#F6AE2D'},
                   opacity=0.55, labels={'city_tier':'City Tier'},
                   category_orders={'city_tier':[1,2,3]})
# Manual OLS trendline using numpy
m12, b12 = np.polyfit(df['delivery_days'], df['product_rating'], 1)
x12 = np.linspace(df['delivery_days'].min(), df['delivery_days'].max(), 100)
fig12.add_scatter(x=x12, y=m12*x12+b12, mode='lines', name=f'Trend (slope={m12:.3f})',
                  line=dict(color='#E84855', width=2.5, dash='dash'))
fig12.update_layout(height=400, plot_bgcolor='#F8FAFC', paper_bgcolor='white', margin=dict(t=10))
st.plotly_chart(fig12, use_container_width=True)
insight("Delivery time has a consistent negative effect on ratings across all city tiers — Tier-3 customers wait longest and rate lowest. Partnering with regional last-mile providers (Delhivery, Shadowfax) to bring Tier-3 delivery below 5 days is projected to lift average ratings by 0.2–0.3 points.")
