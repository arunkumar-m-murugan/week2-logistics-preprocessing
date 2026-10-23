"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Dataset : DataCo Smart Supply Chain (order-item level logistics data, 2015-2018)
Pipeline: audit -> simulated collection faults -> cleaning -> imputation (scored against ground truth)
          -> outlier handling -> normalisation -> encoding -> validation
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sns.set_style("whitegrid")
R = {}                      # key numbers, reused when writing the report
SEED = 42
rng = np.random.default_rng(SEED)

# =============================================================== 1. DATA COLLECTION
URL = ("https://raw.githubusercontent.com/create-sourav/"
       "supply-chain-analysis-python-sqlite3-project/main/supply_chain_clean.csv.gz")
df_raw = pd.read_csv(URL, low_memory=False)
df = df_raw.copy()
R["rows"], R["cols"] = df_raw.shape
print("== SHAPE =="); print(df_raw.shape)
print("\n== DTYPES =="); print(df_raw.dtypes.value_counts().to_string())
R["n_orders"] = int(df_raw["Order Id"].nunique())
R["n_customers"] = int(df_raw["Customer Id"].nunique())
R["n_products"] = int(df_raw["Product Name"].str.strip().nunique())
R["n_markets"] = int(df_raw["Market"].nunique())
R["n_countries_order"] = int(df_raw["Order Country"].nunique())
print("orders", R["n_orders"], "| items", len(df_raw), "| customers", R["n_customers"],
      "| products", R["n_products"], "| order countries", R["n_countries_order"])

# =============================================================== 2. AUDIT
print("\n== AUDIT ==")
R["missing_total_raw"] = int(df_raw.isna().sum().sum())
R["dup_rows"] = int(df_raw.duplicated().sum())
R["dup_item_id"] = int(df_raw["Order Item Id"].duplicated().sum())
print("Missing cells:", R["missing_total_raw"], "| duplicate rows:", R["dup_rows"],
      "| duplicate Order Item Id:", R["dup_item_id"])

constant_cols = [c for c in df_raw.columns if df_raw[c].nunique() == 1]
print("Constant columns:", constant_cols)

def find_redundant(d):
    cols = [c for c in d.columns if d[c].nunique() > 1]
    nun = {c: d[c].nunique() for c in cols}
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if nun[a] != nun[b]:
                continue
            if pd.api.types.is_numeric_dtype(d[a]) and pd.api.types.is_numeric_dtype(d[b]):
                same = np.allclose(d[a].to_numpy(float), d[b].to_numpy(float), atol=1e-6)
            else:
                same = (d[a].astype(str).to_numpy() == d[b].astype(str).to_numpy()).all()
            if same:
                pairs.append((a, b))
    return pairs
pairs = find_redundant(df_raw)
R["redundant_pairs"] = pairs
print("Redundant (identical) column pairs:"); [print("  ", a, "==", b) for a, b in pairs]

txt_cols = df_raw.select_dtypes(include=["object", "string"]).columns
ws = {c: int((df_raw[c] != df_raw[c].str.strip()).sum()) for c in txt_cols}
ws = {k: v for k, v in ws.items() if v > 0}
R["whitespace"] = ws
print("Text columns with leading/trailing spaces:", ws)
dbl = {c: int(df_raw[c].str.contains(r"\s{2,}", regex=True).sum()) for c in txt_cols}
dbl = {k: v for k, v in dbl.items() if v > 0}
R["double_space"] = dbl
print("Text columns with repeated internal spaces:", dbl)

garbled_state = int(df_raw["Order State"].str.contains("?", regex=False).sum())
garbled_city = int(df_raw["Order City"].str.contains("?", regex=False).sum())
R["garbled_state"], R["garbled_city"] = garbled_state, garbled_city
print("Garbled '?' characters  Order State:", garbled_state, "| Order City:", garbled_city)
print("Customer Country labels:", df_raw["Customer Country"].unique().tolist())
R["pii_masked"] = {c: df_raw[c].unique().tolist() for c in ["Customer Email", "Customer Password"]}

def in_us_pr(lat, lon):
    return lat.between(17, 72) & lon.between(-170, -60)
bad_coord = ~in_us_pr(df_raw["Latitude"], df_raw["Longitude"])
R["bad_coord"] = int(bad_coord.sum())
R["bad_coord_pos_lon"] = int((bad_coord & (df_raw["Longitude"] > 0)).sum())
print("Customer coordinates outside the US/Puerto Rico bounding box:", R["bad_coord"],
      "| of which positive longitude:", R["bad_coord_pos_lon"])

od = pd.to_datetime(df_raw["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
sd = pd.to_datetime(df_raw["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M")
R["date_min"], R["date_max"] = str(od.min().date()), str(od.max().date())
R["ship_before_order"] = int((sd < od).sum())
day_diff = (sd.dt.normalize() - od.dt.normalize()).dt.days
R["days_mismatch"] = int((day_diff != df_raw["Days for shipping (real)"]).sum())
print("Order dates:", R["date_min"], "to", R["date_max"], "| shipped before ordered:", R["ship_before_order"],
      "| 'Days for shipping (real)' disagrees with the dates:", R["days_mismatch"])

canc = df_raw["Delivery Status"].eq("Shipping canceled")
R["cancelled_rows"] = int(canc.sum())
print("Rows with 'Shipping canceled':", R["cancelled_rows"],
      "| Order Status values on those rows:", df_raw.loc[canc, "Order Status"].unique().tolist())
R["late_rate_raw_all"] = float(df_raw["Late_delivery_risk"].mean())
R["late_rate_raw_excl_cancel"] = float(df_raw.loc[~canc, "Late_delivery_risk"].mean())
print("Late rate (all rows): %.4f | excluding cancelled: %.4f" % (R["late_rate_raw_all"], R["late_rate_raw_excl_cancel"]))
print("Negative-profit rows:", int((df_raw["Order Profit Per Order"] < 0).sum()))

# =============================================================== 3. SIMULATED COLLECTION FAULTS
# The public copy has no nulls, so missing data is injected completely at random (MCAR).
# The true values are kept so every imputation can be scored.
faults = {"Days for shipping (real)": 0.05, "Order Item Discount": 0.04,
          "Shipping Mode": 0.02, "Customer Segment": 0.03}
truth, inj_idx = {}, {}
for col, frac in faults.items():
    idx = df.index[rng.random(len(df)) < frac]
    inj_idx[col] = idx
    truth[col] = df_raw.loc[idx, col].copy()
    df.loc[idx, col] = np.nan
miss = df.isna().sum(); miss = miss[miss > 0]
R["injected"] = {c: int(n) for c, n in miss.items()}
print("\n== SIMULATED MISSING VALUES =="); print(pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)}).to_string())

fig, ax = plt.subplots(figsize=(7, 3))
mp = (miss / len(df) * 100).sort_values()
ax.barh(mp.index, mp.values, color="#2E5C8A")
for i, v in enumerate(mp.values):
    ax.text(v + 0.1, i, f"{v:.1f}%", va="center")
ax.set_xlabel("Missing values (%)"); ax.set_title("Simulated Missing Data (MCAR, seed 42)")
ax.set_xlim(0, 6.5); plt.tight_layout(); plt.savefig("fig1_missing.png", dpi=150); plt.close()

# =============================================================== 4. STRUCTURAL CLEANING
# 4.1 text hygiene
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)
df["Customer Country"] = df["Customer Country"].replace({"EE. UU.": "United States"})

# 4.2 drop constant, redundant and personal-data columns
redundant_drop = ["Benefit per order", "Sales per customer", "Order Customer Id",
                  "Order Item Cardprod Id", "Product Category Id", "Product Price"]
in_pairs = {x for p in pairs for x in p}
assert set(redundant_drop) <= in_pairs
pii_drop = ["Customer Fname", "Customer Lname", "Customer Street", "Product Image"]
df = df.drop(columns=constant_cols + redundant_drop + pii_drop)
R["cols_after_drop"] = df.shape[1]
R["dropped"] = {"constant": constant_cols, "redundant": redundant_drop, "personal": pii_drop}
print("\nColumns after dropping:", df.shape[1])

# 4.3 types
df["order_dt"] = pd.to_datetime(df["order date (DateOrders)"], format="%m/%d/%Y %H:%M")
df["ship_dt"] = pd.to_datetime(df["shipping date (DateOrders)"], format="%m/%d/%Y %H:%M")
df = df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)"])
df["Customer Zipcode"] = df["Customer Zipcode"].astype(int).astype(str).str.zfill(5)

# 4.4 invalid coordinates: flip lost minus sign, otherwise blank and fill from the same city/state
lat, lon = df["Latitude"].copy(), df["Longitude"].copy()
bad = ~in_us_pr(lat, lon)
flip = bad & (lon > 0) & in_us_pr(lat, -lon)
lon[flip] = -lon[flip]
still_bad = ~in_us_pr(lat, lon)
R["coord_flipped"], R["coord_blanked"] = int(flip.sum()), int(still_bad.sum())
lat[still_bad], lon[still_bad] = np.nan, np.nan
key_city = df["Customer State"] + "|" + df["Customer City"]
for s, name in [(lat, "Latitude"), (lon, "Longitude")]:
    s[:] = s.fillna(s.groupby(key_city).transform("median"))
    s[:] = s.fillna(s.groupby(df["Customer State"]).transform("median"))
df["Latitude"], df["Longitude"] = lat, lon
print("Coordinates: sign flipped =", R["coord_flipped"], "| blanked and refilled =", R["coord_blanked"],
      "| remaining invalid =", int((~in_us_pr(df['Latitude'], df['Longitude'])).sum()))

# 4.5 flags (kept, not deleted)
df["is_cancelled"] = df["Delivery Status"].eq("Shipping canceled").astype(int)
df["garbled_location"] = (df["Order State"].str.contains("?", regex=False) |
                          df["Order City"].str.contains("?", regex=False)).astype(int)

# =============================================================== 5. MISSING VALUES
print("\n== IMPUTATION (scored against the true values) ==")
# 5.1 Shipping Mode: deterministic from the scheduled window
known = df["Shipping Mode"].notna()
nun_per_sched = df[known].groupby("Days for shipment (scheduled)")["Shipping Mode"].nunique()
R["mode_lookup_unique"] = bool((nun_per_sched == 1).all())
mode_map = df[known].groupby("Days for shipment (scheduled)")["Shipping Mode"].agg(lambda s: s.mode().iloc[0])
R["mode_map"] = {int(k): v for k, v in mode_map.items()}
idx = inj_idx["Shipping Mode"]
pred_rule = df.loc[idx, "Days for shipment (scheduled)"].map(mode_map)
pred_mode = pd.Series(df.loc[known, "Shipping Mode"].mode().iloc[0], index=idx)
R["sm_acc_rule"] = float((pred_rule == truth["Shipping Mode"]).mean())
R["sm_acc_mode"] = float((pred_mode == truth["Shipping Mode"]).mean())
print("Shipping Mode  accuracy  rule lookup: %.4f | most-frequent: %.4f" % (R["sm_acc_rule"], R["sm_acc_mode"]))
df.loc[idx, "Shipping Mode"] = pred_rule

# 5.2 Days for shipping (real): compare four strategies
col = "Days for shipping (real)"; idx = inj_idx[col]; obs = df[col].notna()
t = truth[col]
cands = {}
cands["Mean"] = pd.Series(df.loc[obs, col].mean(), index=idx)
cands["Median"] = pd.Series(df.loc[obs, col].median(), index=idx)
g1 = df[obs].groupby("Shipping Mode")[col].median()
cands["Median by Shipping Mode"] = df.loc[idx, "Shipping Mode"].map(g1)
g2 = df[obs].groupby(["Shipping Mode", "Late_delivery_risk"])[col].median()
cands["Median by Mode x Late flag"] = pd.Series(
    [g2[(m, l)] for m, l in zip(df.loc[idx, "Shipping Mode"], df.loc[idx, "Late_delivery_risk"])], index=idx)
days_scores = {k: {"MAE": float((v - t).abs().mean()), "RMSE": float(np.sqrt(((v - t) ** 2).mean()))} for k, v in cands.items()}
R["days_scores"] = days_scores
print(pd.DataFrame(days_scores).T.round(3).to_string())
df.loc[idx, col] = cands["Median by Shipping Mode"]          # chosen: non-circular option

# 5.3 Order Item Discount: compare statistics with a business rule (rate x sales)
col = "Order Item Discount"; idx = inj_idx[col]; obs = df[col].notna(); t = truth[col]
cd = {"Mean": pd.Series(df.loc[obs, col].mean(), index=idx),
      "Median": pd.Series(df.loc[obs, col].median(), index=idx),
      "Rate x Sales rule": df.loc[idx, "Order Item Discount Rate"] * df.loc[idx, "Sales"]}
disc_scores = {k: {"MAE": float((v - t).abs().mean()), "RMSE": float(np.sqrt(((v - t) ** 2).mean()))} for k, v in cd.items()}
R["disc_scores"] = disc_scores
print(pd.DataFrame(disc_scores).T.round(3).to_string())
df.loc[idx, col] = cd["Rate x Sales rule"]

# 5.4 Customer Segment: no structure to exploit -> explicit 'Unknown'
idx = inj_idx["Customer Segment"]; obs = df["Customer Segment"].notna()
seg_mode = df.loc[obs, "Customer Segment"].mode().iloc[0]
R["seg_mode"] = seg_mode
R["seg_acc_mode"] = float((truth["Customer Segment"] == seg_mode).mean())
print("Customer Segment: filling with the mode '%s' would be right only %.1f%% of the time" % (seg_mode, R["seg_acc_mode"] * 100))
df["Customer Segment"] = df["Customer Segment"].fillna("Unknown")
R["missing_after_imputation"] = int(df.drop(columns=[]).isna().sum().sum())
print("Missing cells after imputation:", R["missing_after_imputation"])

# effect on the distribution
R["days_mean_true"], R["days_std_true"] = float(df_raw["Days for shipping (real)"].mean()), float(df_raw["Days for shipping (real)"].std())
R["days_mean_imp"], R["days_std_imp"] = float(df["Days for shipping (real)"].mean()), float(df["Days for shipping (real)"].std())
R["disc_mean_true"], R["disc_std_true"] = float(df_raw["Order Item Discount"].mean()), float(df_raw["Order Item Discount"].std())
R["disc_mean_imp"], R["disc_std_imp"] = float(df["Order Item Discount"].mean()), float(df["Order Item Discount"].std())
print("Days real mean/std  true %.3f/%.3f  imputed %.3f/%.3f" % (R["days_mean_true"], R["days_std_true"], R["days_mean_imp"], R["days_std_imp"]))
print("Discount  mean/std  true %.3f/%.3f  imputed %.3f/%.3f" % (R["disc_mean_true"], R["disc_std_true"], R["disc_mean_imp"], R["disc_std_imp"]))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, sc, title in [(axs[0], days_scores, "Days for shipping (real)"), (axs[1], disc_scores, "Order Item Discount")]:
    s = pd.DataFrame(sc).T["MAE"]
    bars = ax.barh(s.index, s.values, color=["#BFBFBF"] * (len(s) - 1) + ["#2E5C8A"] if title.startswith("Order") else ["#BFBFBF", "#BFBFBF", "#2E5C8A", "#E0A030"])
    for i, v in enumerate(s.values): ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    ax.set_title(title); ax.set_xlabel("Mean absolute error (lower is better)"); ax.set_xlim(0, s.max() * 1.25)
plt.tight_layout(); plt.savefig("fig2_imputation.png", dpi=150); plt.close()

# =============================================================== 6. OUTLIERS
print("\n== OUTLIERS ==")
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - k * i, q3 + k * i
out_cols = ["Order Profit Per Order", "Sales", "Order Item Discount", "Order Item Product Price", "Order Item Profit Ratio"]
ot = []
for c in out_cols:
    lo, hi = iqr_bounds(df[c]); z = (df[c] - df[c].mean()) / df[c].std()
    row = dict(col=c, lo=float(lo), hi=float(hi), iqr=int(((df[c] < lo) | (df[c] > hi)).sum()),
               z3=int((z.abs() > 3).sum()), skew=float(df[c].skew()), mn=float(df[c].min()), mx=float(df[c].max()))
    ot.append(row); print(row)
R["outliers"] = ot
prof = df["Order Profit Per Order"]
lo, hi = iqr_bounds(prof)
flag_low, flag_high = int((prof < lo).sum()), int((prof > hi).sum())
R["profit_low"], R["profit_high"] = flag_low, flag_high
print("Profit IQR-flagged: below fence", flag_low, "| above fence", flag_high)
worst = df.loc[prof.nsmallest(3).index, ["Product Name", "Order Item Product Price", "Order Item Quantity",
                                         "Order Item Discount", "Order Item Profit Ratio", "Order Profit Per Order"]]
R["worst"] = worst.round(2).to_dict("records")
print(worst.round(2).to_string())
R["neg_profit_rows"] = int((prof < 0).sum()); R["neg_profit_share"] = float((prof < 0).mean())
low_df = df[prof < lo]
R["low_median_price"] = float(low_df["Order Item Product Price"].median())
R["all_median_price"] = float(df["Order Item Product Price"].median())
print("Median price of the large-loss rows %.2f vs all rows %.2f" % (R["low_median_price"], R["all_median_price"]))

p1, p99 = prof.quantile([.01, .99])
df["Profit_w"] = prof.clip(p1, p99)
df["Sales_log"] = np.log1p(df["Sales"])
df["Discount_log"] = np.log1p(df["Order Item Discount"])
R["p1"], R["p99"] = float(p1), float(p99)
R["skew_profit_raw"], R["skew_profit_w"] = float(prof.skew()), float(df["Profit_w"].skew())
R["skew_sales_raw"], R["skew_sales_log"] = float(df["Sales"].skew()), float(df["Sales_log"].skew())
R["skew_disc_raw"], R["skew_disc_log"] = float(df["Order Item Discount"].skew()), float(df["Discount_log"].skew())
print("Skew  profit %.2f -> %.2f | sales %.2f -> %.2f | discount %.2f -> %.2f" % (
    R["skew_profit_raw"], R["skew_profit_w"], R["skew_sales_raw"], R["skew_sales_log"], R["skew_disc_raw"], R["skew_disc_log"]))

fig, axs = plt.subplots(2, 2, figsize=(10, 5))
sns.boxplot(x=prof, ax=axs[0, 0], color="#C0504D"); axs[0, 0].set_title("Profit per order (raw)")
sns.boxplot(x=df["Profit_w"], ax=axs[0, 1], color="#2E5C8A"); axs[0, 1].set_title("Profit per order (winsorised 1-99%)")
sns.boxplot(x=df["Sales"], ax=axs[1, 0], color="#C0504D"); axs[1, 0].set_title("Sales (raw)")
sns.boxplot(x=df["Sales_log"], ax=axs[1, 1], color="#2E5C8A"); axs[1, 1].set_title("Sales (log1p)")
for a in axs.ravel(): a.set_xlabel("")
plt.tight_layout(); plt.savefig("fig3_outliers.png", dpi=150); plt.close()

# =============================================================== 7. NORMALISATION
print("\n== NORMALISATION COMPARISON (Sales) ==")
x = df[["Sales"]]
comp = {}
for name, arr in {"Raw": x["Sales"].to_numpy(),
                  "Min-Max": MinMaxScaler().fit_transform(x).ravel(),
                  "Z-score": StandardScaler().fit_transform(x).ravel(),
                  "Robust": RobustScaler().fit_transform(x).ravel(),
                  "log1p + Z-score": StandardScaler().fit_transform(df[["Sales_log"]]).ravel()}.items():
    s = pd.Series(arr)
    comp[name] = dict(mean=float(s.mean()), std=float(s.std()), mn=float(s.min()), mx=float(s.max()),
                      skew=float(s.skew()), share_low=float((s < s.min() + 0.1 * (s.max() - s.min())).mean()))
R["norm"] = comp
print(pd.DataFrame(comp).T.round(3).to_string())
fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
for ax, (name, arr, colr) in zip(axs, [("Raw Sales", x["Sales"], "#C0504D"),
                                       ("Min-Max", MinMaxScaler().fit_transform(x).ravel(), "#E0A030"),
                                       ("Z-score", StandardScaler().fit_transform(x).ravel(), "#E0A030"),
                                       ("log1p + Z-score", StandardScaler().fit_transform(df[["Sales_log"]]).ravel(), "#2E5C8A")]):
    ax.hist(arr, bins=50, color=colr); ax.set_title(name, fontsize=10); ax.set_yticks([])
plt.tight_layout(); plt.savefig("fig4_scaling.png", dpi=150); plt.close()

# =============================================================== 8. FEATURES, ENCODING, FINAL SCALING
df["order_hour"] = df["order_dt"].dt.hour
df["order_dow"] = df["order_dt"].dt.dayofweek
df["order_month"] = df["order_dt"].dt.month
df["is_weekend"] = (df["order_dow"] >= 5).astype(int)
df["processing_days"] = (df["ship_dt"].dt.normalize() - df["order_dt"].dt.normalize()).dt.days   # post-shipment: not a model input

num_feats = ["Order Item Quantity", "Order Item Discount Rate", "Sales_log", "Discount_log", "Profit_w",
             "order_hour", "order_dow", "order_month", "is_weekend"]
cat_feats = ["Shipping Mode", "Market", "Type", "Customer Segment", "Department Name"]
X = pd.get_dummies(df[num_feats + cat_feats], columns=cat_feats, drop_first=True, dtype=int)
y = df["Late_delivery_risk"]
tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(X.loc[tr_idx, num_feats])            # fitted on TRAIN only
Xs = X.copy()
Xs[num_feats] = scaler.transform(X[num_feats])
R["n_features"] = Xs.shape[1]
R["train_rows"], R["test_rows"] = len(tr_idx), len(te_idx)
print("\nModel-ready matrix:", Xs.shape, "| train", len(tr_idx), "| test", len(te_idx))
print("Train means of scaled numeric columns:", Xs.loc[tr_idx, num_feats].mean().round(3).to_dict())

# smoke test + leakage demonstration
def auc_of(cols, extra=None):
    A = Xs[cols].copy()
    if extra is not None: A[extra.name] = extra
    m = LogisticRegression(max_iter=2000).fit(A.loc[tr_idx], y.loc[tr_idx])
    return float(roc_auc_score(y.loc[te_idx], m.predict_proba(A.loc[te_idx])[:, 1]))
R["auc_clean"] = auc_of(list(Xs.columns))
leak = (df["Days for shipping (real)"] - df["Days for shipping (real)"].loc[tr_idx].mean()) / df["Days for shipping (real)"].loc[tr_idx].std()
leak.name = "days_real_scaled"
R["auc_leaky"] = auc_of(list(Xs.columns), leak)
print("Logistic regression AUC  order-time features only: %.3f | with post-shipment 'Days for shipping (real)': %.3f" % (R["auc_clean"], R["auc_leaky"]))

# =============================================================== 9. VALIDATION
print("\n== VALIDATION ==")
R["final_rows"], R["final_cols_clean"] = df.shape
print("Rows raw/clean:", len(df_raw), len(df), "| missing cells:", int(df.isna().sum().sum()), "| duplicate Order Item Id:", int(df["Order Item Id"].duplicated().sum()))
kp = pd.DataFrame({"raw": df_raw.groupby("Shipping Mode")["Late_delivery_risk"].mean(),
                   "clean": df.groupby("Shipping Mode")["Late_delivery_risk"].mean()})
R["late_by_mode"] = kp.round(4).to_dict()
print(kp.round(4).to_string())
dm = pd.DataFrame({"raw": df_raw.groupby("Shipping Mode")["Days for shipping (real)"].mean(),
                   "clean": df.groupby("Shipping Mode")["Days for shipping (real)"].mean()})
R["days_by_mode"] = dm.round(3).to_dict()
print(dm.round(3).to_string())
R["sales_total_raw"], R["sales_total_clean"] = float(df_raw["Sales"].sum()), float(df["Sales"].sum())
R["profit_total_raw"], R["profit_total_clean"] = float(df_raw["Order Profit Per Order"].sum()), float(df["Order Profit Per Order"].sum())
R["disc_total_raw"], R["disc_total_clean"] = float(df_raw["Order Item Discount"].sum()), float(df["Order Item Discount"].sum())
print("Total sales raw/clean: %.2f / %.2f | total discount raw/clean: %.2f / %.2f" % (
    R["sales_total_raw"], R["sales_total_clean"], R["disc_total_raw"], R["disc_total_clean"]))

fig, ax = plt.subplots(figsize=(7, 3.3))
order = ["Standard Class", "Second Class", "Same Day", "First Class"]
kp = kp.loc[order]; xx = np.arange(len(order))
ax.bar(xx - 0.2, kp["raw"] * 100, 0.4, label="Raw", color="#BFBFBF")
ax.bar(xx + 0.2, kp["clean"] * 100, 0.4, label="After pipeline", color="#2E5C8A")
ax.set_xticks(xx); ax.set_xticklabels(order); ax.set_ylabel("Late delivery rate (%)")
ax.set_title("Late Delivery Rate by Shipping Mode: Raw vs Cleaned"); ax.legend()
plt.tight_layout(); plt.savefig("fig5_validation.png", dpi=150); plt.close()

# =============================================================== 10. SAVE
keep_clean = [c for c in df.columns if c not in ("processing_days",)]
df[keep_clean].to_csv("dataco_cleaned.csv.gz", index=False, compression="gzip")
out = Xs.copy(); out.insert(0, "Order Item Id", df["Order Item Id"]); out["Late_delivery_risk"] = y
out["split"] = np.where(df.index.isin(tr_idx), "train", "test")
out.to_csv("dataco_model_ready.csv.gz", index=False, compression="gzip")
json.dump(R, open("results.json", "w"), indent=1, default=str)
print("\nSaved dataco_cleaned.csv.gz and dataco_model_ready.csv.gz")
