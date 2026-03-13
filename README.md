# 🛒 ShopEase Analytics Dashboard

A full-stack **Data Analytics project** built with Python + Streamlit, simulating an e-commerce startup's intelligence platform. Covers the complete data analytics pipeline from synthetic data generation to ML model deployment.

## 🚀 Live Demo
Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud) — see deployment instructions below.

---

## 📦 What's Inside

| Page | Description |
|------|-------------|
| 🏠 Home | KPI dashboard, dataset overview, technique summary |
| 📋 Column Rationale | Full data dictionary with analytical justification for every column |
| 📊 EDA & Visualisations | 12 interactive Plotly charts with two-liner business insights |
| 🎯 Classification | Random Forest predicts repeat customers + live prediction tool |
| 👥 Clustering | K-Means customer personas with radar charts and persona cards |
| 🔗 Association Rules | Apriori product co-purchase mining with network graph |
| 📈 Regression | Gradient Boosting order value forecaster + live calculator |
| 💡 Insights | 5 narrative insight cards + 6-point decision framework |

---

## 🛠️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/shopease-analytics.git
cd shopease-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate models (run once)
```bash
python generate_models.py
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. **Push to GitHub** (public repository)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository → Branch: `main` → Main file: `app.py`
5. Click **Deploy** — live in ~2 minutes!

> **Note:** The `assets/models.pkl` file must be committed to the repository (it's pre-generated). The `generate_models.py` script is provided for reproducibility.

---

## 📊 Dataset

**ShopEase** — Synthetic E-Commerce Dataset
- 500 orders | 200 unique customers | 18 months (Jan 2023 – Jun 2024)
- 28 columns across order, customer, product, financial, and marketing dimensions
- 7 product categories | 10 Indian cities (Tier 1/2/3)

---

## 🧠 ML Techniques

| Technique | Model | Target | Performance |
|-----------|-------|--------|-------------|
| Classification | Random Forest (150 trees) | is_repeat_customer | Accuracy: ~53% |
| Clustering | K-Means (k=4) | Customer segments | 4 personas identified |
| Association Rules | Apriori | Category co-purchase | 16 rules (lift ≥ 1.0) |
| Regression | Gradient Boosting (200 trees) | net_amount (₹) | R² = 0.78 |

---

## 📁 Project Structure

```
shopease-analytics/
├── app.py                     # Main Streamlit home page
├── generate_models.py         # Train & save ML models (run once)
├── shopease_clean.csv         # Cleaned dataset (500 rows × 28 cols)
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit theme & server config
├── assets/
│   ├── models.pkl             # Pre-trained models (auto-generated)
│   └── *.png                  # EDA chart images
└── pages/
    ├── 1_Column_Rationale.py
    ├── 2_EDA_Visualizations.py
    ├── 3_Classification.py
    ├── 4_Clustering.py
    ├── 5_Association_Rules.py
    ├── 6_Regression.py
    └── 7_Insights.py
```

---

## 📚 Academic Context

Built as part of a **Data Analytics Project-Based Learning** assignment.  
Domain: E-Commerce | Tools: Python, Streamlit, Scikit-learn, Plotly, MLxtend
