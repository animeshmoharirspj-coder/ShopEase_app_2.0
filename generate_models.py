"""
Run once to pre-compute all model artefacts and save to assets/.
"""
import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score, mean_squared_error)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "shopease_clean.csv"))

# ── 1. CLASSIFICATION ─────────────────────────────────────────────────────────
# Target: will the customer buy again? (is_repeat_customer)
clf_features = ['customer_age','city_tier','discount_pct','net_amount',
                'product_rating','delivery_days','is_returned','quantity']
le_cat = LabelEncoder()
df['category_enc'] = le_cat.fit_transform(df['category'])
clf_features.append('category_enc')

X_clf = df[clf_features].fillna(0)
y_clf = df['is_repeat_customer']

X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')
rf.fit(X_tr, y_tr)
y_pred_clf = rf.predict(X_te)

clf_report = classification_report(y_te, y_pred_clf, output_dict=True)
clf_cm     = confusion_matrix(y_te, y_pred_clf).tolist()
feat_imp   = dict(zip(clf_features, rf.feature_importances_.tolist()))

# ── 2. CLUSTERING ─────────────────────────────────────────────────────────────
cust = df.groupby('customer_id').agg(
    total_orders   = ('order_id',   'count'),
    total_spent    = ('net_amount', 'sum'),
    avg_order_val  = ('net_amount', 'mean'),
    avg_discount   = ('discount_pct','mean'),
    avg_rating     = ('product_rating','mean'),
    return_rate    = ('is_returned', 'mean'),
    avg_delivery   = ('delivery_days','mean'),
    unique_cats    = ('category',   'nunique'),
).reset_index()

scaler = StandardScaler()
X_clust = scaler.fit_transform(cust[['total_orders','total_spent','avg_order_val',
                                     'avg_discount','avg_rating','return_rate']])
km = KMeans(n_clusters=4, random_state=42, n_init=10)
cust['cluster'] = km.fit_predict(X_clust)

cluster_labels = {0:"Bargain Hunters", 1:"Loyal High-Spenders",
                  2:"Occasional Browsers", 3:"Premium Champions"}
cluster_colors = {0:"#F6AE2D", 1:"#1F3864", 2:"#2E86AB", 3:"#3BB273"}
cust['persona'] = cust['cluster'].map(cluster_labels)

cluster_summary = cust.groupby('persona').agg(
    count=('customer_id','count'),
    avg_spent=('total_spent','mean'),
    avg_orders=('total_orders','mean'),
    avg_discount=('avg_discount','mean'),
    avg_rating=('avg_rating','mean'),
    return_rate=('return_rate','mean'),
).round(2).reset_index()

# ── 3. ASSOCIATION RULES ──────────────────────────────────────────────────────
basket = df.groupby(['customer_id','category'])['order_id'].count().unstack(fill_value=0)
basket_bool = basket.map(lambda x: True if x > 0 else False)

freq_items = apriori(basket_bool, min_support=0.05, use_colnames=True)
rules_df   = association_rules(freq_items, metric="lift", min_threshold=1.0)
rules_df   = rules_df.sort_values('lift', ascending=False).head(20)
rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
rules_save = rules_df[['antecedents','consequents','support','confidence','lift']].round(4)

# ── 4. REGRESSION ─────────────────────────────────────────────────────────────
# Forecast net_amount (order value)
reg_features = ['unit_price','quantity','discount_pct','customer_age',
                'city_tier','delivery_days','is_repeat_customer','category_enc']
X_reg = df[reg_features].fillna(0)
y_reg = df['net_amount']
X_rtr, X_rte, y_rtr, y_rte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
gb.fit(X_rtr, y_rtr)
y_pred_reg = gb.predict(X_rte)

reg_metrics = {
    "MAE":  round(float(mean_absolute_error(y_rte, y_pred_reg)), 2),
    "RMSE": round(float(np.sqrt(mean_squared_error(y_rte, y_pred_reg))), 2),
    "R2":   round(float(r2_score(y_rte, y_pred_reg)), 4),
}
reg_scatter = {"actual": y_rte.tolist(), "predicted": y_pred_reg.tolist()}
reg_feat_imp = dict(zip(reg_features, gb.feature_importances_.tolist()))

# ── SAVE ARTEFACTS ────────────────────────────────────────────────────────────
assets = os.path.join(BASE, "assets")
os.makedirs(assets, exist_ok=True)

artefacts = {
    "clf_model":       rf,
    "clf_report":      clf_report,
    "clf_cm":          clf_cm,
    "clf_features":    clf_features,
    "feat_imp_clf":    feat_imp,
    "le_cat":          le_cat,
    "cluster_model":   km,
    "cluster_scaler":  scaler,
    "cluster_df":      cust,
    "cluster_summary": cluster_summary,
    "cluster_labels":  cluster_labels,
    "cluster_colors":  cluster_colors,
    "rules":           rules_save,
    "freq_items":      freq_items,
    "gb_model":        gb,
    "reg_metrics":     reg_metrics,
    "reg_scatter":     reg_scatter,
    "reg_features":    reg_features,
    "reg_feat_imp":    reg_feat_imp,
}
with open(os.path.join(assets, "models.pkl"), "wb") as f:
    pickle.dump(artefacts, f)

print("✅ Models trained and saved.")
print(f"   Classification Accuracy : {clf_report['accuracy']:.3f}")
print(f"   Regression R²           : {reg_metrics['R2']}")
print(f"   Clusters formed         : {cust['persona'].nunique()}")
print(f"   Association rules found : {len(rules_save)}")
