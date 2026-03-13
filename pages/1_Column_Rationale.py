import streamlit as st

st.set_page_config(page_title="Column Rationale · ShopEase", page_icon="📋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.page-title{font-family:'DM Serif Display',serif;font-size:2.6rem;color:#1F3864;}
.group-header{font-weight:700;font-size:1.05rem;color:white;padding:0.6rem 1rem;
              border-radius:8px;margin:1.2rem 0 0.4rem;}
.col-card{background:#F8FAFC;border-radius:10px;padding:0.9rem 1.2rem;
          margin-bottom:0.45rem;border-left:4px solid var(--c);}
.col-name{font-weight:700;color:#1F3864;font-size:0.95rem;}
.col-type{display:inline-block;background:#E2E8F0;color:#475569;font-size:0.72rem;
          padding:0.15rem 0.5rem;border-radius:20px;margin-left:0.4rem;}
.col-desc{color:#475569;font-size:0.87rem;margin-top:0.2rem;}
.col-why{color:#64748B;font-size:0.82rem;font-style:italic;margin-top:0.15rem;}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#1F3864 0%,#163060 100%);}
div[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📋 Dataset Column Rationale</div>', unsafe_allow_html=True)
st.markdown("Every column in the ShopEase dataset was deliberately designed to serve a specific analytical purpose. Below is the full data dictionary with rationale.", unsafe_allow_html=True)
st.markdown("---")

groups = {
    "🧾 Order Identifiers": {
        "color": "#1F3864",
        "bg": "#1F3864",
        "cols": [
            ("order_id",       "String",  "Unique identifier for each transaction",
             "Primary key — enables deduplication, order tracking, and join operations across tables."),
            ("order_date",     "Date",    "Date the order was placed",
             "Enables time-series analysis, monthly/quarterly trends, and seasonality detection."),
            ("delivery_date",  "Date",    "Expected delivery date",
             "Paired with order_date to compute delivery_days, a key customer satisfaction metric."),
            ("delivery_days",  "Integer", "Number of days between order and delivery",
             "Proxy for logistics efficiency. Correlates with product ratings and city tier — used in regression and EDA."),
        ]
    },
    "👤 Customer Demographics": {
        "color": "#2E86AB",
        "bg": "#2E86AB",
        "cols": [
            ("customer_id",      "String",  "Anonymised unique customer identifier",
             "Links multiple orders to the same customer — enables RFM analysis and repeat purchase tracking."),
            ("customer_age",     "Integer", "Age of the customer in years",
             "Demographic segmentation variable. Helps identify high-AOV age brackets for targeted campaigns."),
            ("gender",           "Category","Customer gender (Male/Female/Other)",
             "Enables gender-based product affinity analysis and personalised marketing strategy."),
            ("city",             "String",  "City of delivery",
             "Geographic dimension — used for city-level revenue breakdowns and logistics planning."),
            ("city_tier",        "Integer", "1=Metro, 2=Tier-2, 3=Tier-3",
             "Encodes urban hierarchy numerically. Correlates with delivery speed, AOV, and acquisition cost — essential regression feature."),
            ("age_group",        "Category","Binned age: 18-25, 26-35, 36-45, 46-55, 56-65",
             "Engineered feature. Enables clean demographic bar charts without continuous-age noise."),
        ]
    },
    "🛍️ Product Information": {
        "color": "#3BB273",
        "bg": "#3BB273",
        "cols": [
            ("category",         "Category","Product category (7 types)",
             "Primary product grouping. Used in every analytical technique — group-by revenue, association rules, clustering."),
            ("product_name",     "String",  "Name of the product purchased",
             "Granular product label — useful for product-level analysis and association mining at SKU level."),
            ("unit_price",       "Float",   "Selling price per unit (INR)",
             "Primary pricing variable. Strongest predictor of net_amount in regression (r=0.89). Basis for margin calculation."),
            ("cost_price",       "Float",   "Cost/COGS per unit (INR)",
             "Enables genuine profitability calculation. Without COGS, profit_margin cannot be computed accurately."),
            ("quantity",         "Integer", "Number of units in the order",
             "Volume metric. Multiplied with unit_price for gross_amount. Identifies bulk-buying behaviour segments."),
        ]
    },
    "💰 Financial Metrics": {
        "color": "#F6AE2D",
        "bg": "#B8860B",
        "cols": [
            ("discount_pct",     "Float",   "Discount applied as percentage",
             "Critical pricing lever. Negatively correlated with profit_margin — key variable in regression and scatter analysis."),
            ("gross_amount",     "Float",   "unit_price × quantity (pre-discount)",
             "Represents full potential revenue. Difference from net_amount reveals revenue lost to discounts."),
            ("discount_amount",  "Float",   "Monetary value of discount given",
             "Converts discount_pct to absolute INR — more intuitive for finance team reporting and P&L analysis."),
            ("net_amount",       "Float",   "Actual revenue received per order",
             "The core revenue KPI. Regression target variable. Used in all aggregations (by city, category, channel)."),
            ("profit",           "Float",   "net_amount − (cost_price × quantity)",
             "True business profitability per order. More meaningful than revenue for strategic decision-making."),
            ("profit_margin",    "Float",   "profit / net_amount × 100 (%)",
             "Normalised profitability metric. Allows fair comparison across categories with different price points."),
        ]
    },
    "📣 Marketing & Behaviour": {
        "color": "#7B5EA7",
        "bg": "#7B5EA7",
        "cols": [
            ("payment_method",       "Category","Mode of payment used",
             "Identifies digital vs cash preference. COD correlates with higher return rates — actionable operational insight."),
            ("acquisition_channel",  "Category","Marketing channel that brought the customer",
             "ROI measurement variable. Reveals which channels drive revenue vs retention — crucial for budget allocation."),
            ("is_repeat_customer",   "Binary",  "1 = returning customer, 0 = new",
             "Classification target variable. Indicates customer loyalty and LTV potential — central to CRM strategy."),
            ("product_rating",       "Float",   "Customer rating (1.0–5.0)",
             "Quality perception metric. Strong negative correlation with is_returned (r=−0.48). Drives product curation policy."),
            ("is_returned",          "Binary",  "1 = order returned, 0 = kept",
             "Operational risk metric. Correlated with low ratings and COD payments. Impacts net revenue calculations."),
        ]
    },
    "🗓️ Engineered Time Features": {
        "color": "#E84855",
        "bg": "#C0392B",
        "cols": [
            ("order_month",    "String",  "Year-Month of order (e.g., 2023-09)",
             "Enables monthly trend aggregation without writing complex date logic. Powers time-series visualisations."),
            ("order_quarter",  "String",  "Year-Quarter (e.g., 2023Q3)",
             "Quarterly granularity captures seasonality (Q4 Diwali spike) better than monthly noise."),
        ]
    },
}

for group_name, group_data in groups.items():
    st.markdown(f'<div class="group-header" style="background:{group_data["bg"]}">{group_name}</div>',
                unsafe_allow_html=True)
    for col_name, dtype, desc, why in group_data["cols"]:
        st.markdown(f"""
        <div class="col-card" style="--c:{group_data['color']}">
          <span class="col-name">{col_name}</span>
          <span class="col-type">{dtype}</span>
          <div class="col-desc">{desc}</div>
          <div class="col-why">📌 Why included: {why}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"**Total: 28 columns** across 6 functional groups — each serving a distinct analytical or business purpose.")
