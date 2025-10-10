import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. SETUP AND CONFIGURATION ---
print("Step 1: Setting up paths and parameters...")
set_random_seed(123)
np.random.seed(123)
data_path = "data/Python"
results_path = "results"
os.makedirs(results_path, exist_ok=True)
colors = ["#0077b6", "#00b4d8", "#90e0ef", "#adb5bd", "#6c757d"]
sns.set_palette(sns.color_palette(colors))

# --- 2. DATA LOADING AND PREPROCESSING ---
print("Step 2: Loading and preprocessing data...")
try:
    den = pd.read_csv(f"{data_path}/ds1.csv")    # dengue cases
    ips = pd.read_csv(f"{data_path}/ds3.csv", encoding="latin1")  # infra/health
    cli = pd.read_csv(f"{data_path}/ds2.csv")    # climate
except FileNotFoundError as e:
    print(f"Error: Data file not found. Make sure '{e.filename}' is in the '{data_path}' directory.")
    exit()

for df, col in zip([den, ips, cli], ['municipio', 'Código IBGE', 'munic_code']):
    df['cod6'] = df[col].astype(str).str[:6]

fixed_features = pd.merge(
    ips,
    cli.drop(columns=['municipio', 'estado', 'regiao', 'koppen', 'munic_code'], errors="ignore"),
    on="cod6", how="left"
)
fixed_features = fixed_features.loc[:, ['cod6'] + [
    c for c in fixed_features.columns 
    if fixed_features[c].dtype in [np.float64, np.int64]
]]

years = [str(a) for a in range(2013, 2026)]
for year in years:
    if year in den.columns:
        den[year] = pd.to_numeric(den[year].replace("-", 0), errors="coerce").fillna(0)

den_long = (
    den.melt(
        id_vars=['cod6', 'municipio'],
        value_vars=years,
        var_name='year',
        value_name='dengue_cases'
    )
    .merge(fixed_features, on='cod6', how='left')
    .dropna()
)

den_long['risk_real'] = 0
for yr in den_long['year'].unique():
    med = den_long.loc[den_long['year']==yr, 'dengue_cases'].median()
    den_long.loc[den_long['year']==yr, 'risk_real'] = (
        den_long['dengue_cases'] > med
    ).astype(int)

n_min = den_long['risk_real'].value_counts().min()
data_balanced = pd.concat([
    den_long.query('risk_real==0').sample(n=n_min, random_state=123),
    den_long.query('risk_real==1').sample(n=n_min, random_state=123)
]).sample(frac=1, random_state=123)

features = [
    c for c in data_balanced.columns 
    if c not in ['cod6','municipio','dengue_cases','year','risk_real']
]
X = data_balanced[features].values
y = data_balanced['risk_real'].values
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)

# --- 3. TRAIN INITIAL LSTM ---
print("Step 3: Training initial LSTM model...")
X_lstm = X_scaled.reshape((-1, 1, X_scaled.shape[1]))
model_full = Sequential([
    Input(shape=(1, X_scaled.shape[1])),
    LSTM(16),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])
model_full.compile(loss="binary_crossentropy", optimizer="adam")
callbacks = [EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)]
model_full.fit(X_lstm, y, epochs=12, batch_size=32, verbose=0, callbacks=callbacks)

# --- 4. FEATURE IMPORTANCE WITH SHAP (FIXED) ---
print("Step 4: Calculating feature importance with SHAP...")
def predict_fn(z):
    z3 = z.reshape((z.shape[0], 1, z.shape[1]))
    return model_full.predict(z3, verbose=0).flatten()

background = shap.kmeans(X_scaled, 30)
explainer = shap.KernelExplainer(predict_fn, background, nsamples=100)
idx = np.random.choice(X_scaled.shape[0], min(100, X_scaled.shape[0]), replace=False)
X_sample = X_scaled[idx]

print("  Running SHAP (this may take a minute)...")
shap_vals = explainer.shap_values(X_sample)

if isinstance(shap_vals, list):
    arr = np.array(shap_vals[0])
else:
    arr = np.array(shap_vals)

if arr.ndim == 3:
    arr2 = arr.reshape(-1, arr.shape[-1])
elif arr.ndim == 2:
    arr2 = arr
else:
    raise ValueError(f"Unexpected SHAP array shape {arr.shape}")

importance_vector = np.mean(np.abs(arr2), axis=0).flatten()

n_feat = len(features)
if importance_vector.shape[0] != n_feat:
    vec = np.zeros(n_feat)
    vec[:importance_vector.shape[0]] = importance_vector[:n_feat]
    importance_vector = vec

feature_importance = pd.DataFrame({
    'variable': features,
    'importance': importance_vector
}).sort_values('importance', ascending=False)

feature_importance.to_csv(f"{results_path}/fim.csv", index=False)
relevant_vars = feature_importance.query("importance>0")['variable'].tolist()
if not relevant_vars:
    relevant_vars = feature_importance['variable'].tolist()[:3]
print(f"  Selected {len(relevant_vars)} variables.")

# --- 5. TRAIN LSTM WITH RELEVANT FEATURES ---
print("Step 5: Training LSTM on selected features...")
X_rel = data_balanced[relevant_vars].values
scaler_rel = MinMaxScaler().fit(X_rel)
X_rel_s = scaler_rel.transform(X_rel).reshape((-1, 1, len(relevant_vars)))
model_final = Sequential([
    Input(shape=(1, len(relevant_vars))),
    LSTM(16),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])
model_final.compile(loss="binary_crossentropy", optimizer="adam")
model_final.fit(X_rel_s, y, epochs=12, batch_size=32, verbose=0, callbacks=callbacks)

# --- 5b. K-FOLD CROSS-VALIDATION (STRATIFIED, k=5) ---
print("Step 5b: 5-Fold Stratified Cross-Validation on LSTM...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
cv_aucs, cv_accs = [], []

for i, (train_idx, test_idx) in enumerate(kfold.split(X_rel, y), start=1):
    X_train, X_test = X_rel[train_idx], X_rel[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scaler_cv = MinMaxScaler().fit(X_train)
    X_train_s = scaler_cv.transform(X_train).reshape((-1, 1, len(relevant_vars)))
    X_test_s  = scaler_cv.transform(X_test).reshape((-1, 1, len(relevant_vars)))
    model_cv = Sequential([
        Input(shape=(1, len(relevant_vars))),
        LSTM(16),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])
    model_cv.compile(loss="binary_crossentropy", optimizer="adam")
    model_cv.fit(X_train_s, y_train, epochs=12, batch_size=32, verbose=0, callbacks=callbacks)
    y_pred_prob = model_cv.predict(X_test_s, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    cv_aucs.append(auc)
    cv_accs.append(acc)
    print(f"  Fold {i}: AUC={auc:.3f}, ACC={acc:.3f}")

cv_df = pd.DataFrame({
    "fold": np.arange(1, 6),
    "auc": cv_aucs,
    "accuracy": cv_accs
})
cv_df.to_csv(f"{results_path}/kfold.csv", index=False)
print(f"  Mean AUC: {np.mean(cv_aucs):.3f} | Mean ACC: {np.mean(cv_accs):.3f}")
print(f"K-Fold results saved to {results_path}/kfold.csv")

# --- 6. PREDICTION & EVALUATION ---
print("Step 6: Predictions and evaluation...")
X_full_rel = scaler_rel.transform(den_long[relevant_vars].values)
probs = model_final.predict(X_full_rel.reshape(-1,1,len(relevant_vars)), verbose=0).flatten()
df_probs = den_long[['cod6','municipio','year']].copy()
df_probs['prob'] = probs
df_probs.to_csv(f"{results_path}/prb.csv", index=False)

test_probs = model_final.predict(X_rel_s, verbose=0).flatten()
test_preds = (test_probs>0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, test_preds).ravel()
metrics = {
    'auc': roc_auc_score(y,test_probs),
    'accuracy': accuracy_score(y,test_preds),
    'sensitivity': tp/(tp+fn),
    'specificity': tn/(tn+fp)
}
pd.DataFrame([metrics]).to_csv(f"{results_path}/met.csv", index=False)

# --- 7. BOOTSTRAP & PLOTS ---
print("Step 7: Bootstrap AUC and plots...")
rng = np.random.RandomState(123)
boots = []
for _ in range(1000):
    i = rng.choice(len(test_probs), len(test_probs), replace=True)
    if len(np.unique(y[i]))<2: continue
    boots.append(roc_auc_score(y[i], test_probs[i]))
plt.figure(figsize=(7,4))
sns.histplot(boots, kde=True, color=colors[0])
plt.xlabel("AUC (bootstrap)"); plt.ylabel("Density")
plt.tight_layout(); plt.savefig(f"{results_path}/auc.pdf"); plt.close()

plt.figure(figsize=(7,4))
sns.histplot(df_probs['prob'], kde=True, color=colors[1])
plt.xlabel("Predicted Probability"); plt.ylabel("Frequency")
plt.tight_layout(); plt.savefig(f"{results_path}/his.pdf"); plt.close()

# --- 8. CLUSTERING ---
print("Step 8: KMeans clustering...")
meanp = df_probs.groupby('cod6')['prob'].mean().reset_index()
vars_mean = den_long.groupby('cod6')[relevant_vars].mean().reset_index()
cluster_in = meanp.merge(vars_mean, on='cod6').dropna()
dist, ks = [], range(2,7)
for k in ks:
    dist.append(KMeans(n_clusters=k, random_state=123, n_init=10)
                .fit(cluster_in.drop('cod6',axis=1)).inertia_)
plt.figure(figsize=(7,4))
plt.plot(ks,dist,'o-'); plt.xlabel("k"); plt.ylabel("Inertia")
plt.tight_layout(); plt.savefig(f"{results_path}/elb.pdf"); plt.close()
best_k=4
clusters = KMeans(n_clusters=best_k,random_state=123,n_init=10).fit_predict(cluster_in.drop('cod6',axis=1))
cluster_in['clu']=clusters
cluster_in.to_csv(f"{results_path}/clu.csv", index=False)

# --- 9. FINAL AGGREGATION ---
print("Step 9: Final aggregation and saving acsm.csv...")
den_long['prob'] = probs
den_long['ps'] = den_long['cod6'].str[:2]
top_states = den_long.groupby('ps')['prob'].mean().nlargest(10).index
final = []
for ps in top_states:
    sub = den_long[den_long['ps']==ps]
    top_m = sub.groupby('municipio')['prob'].mean().nlargest(30).index
    for m in top_m:
        rows = sub[sub['municipio']==m]
        entry = {
            'ps': ps, 'mun': m,
            'per':'2013-2025','prob_mean': rows['prob'].mean()
        }
        for v in relevant_vars[:10]:
            entry[f"{v}_mean"] = rows[v].mean()
        final.append(entry)
pd.DataFrame(final).to_csv(f"{results_path}/acsm.csv", index=False)

# Top 10 municipalities
top10 = den_long.groupby('municipio').agg(
    prob_mean=('prob','mean'),
    cases_sum=('dengue_cases','sum')
).nlargest(10,'prob_mean')
top10.to_csv(f"{results_path}/top10.csv")

print("\nAnalysis complete. Results in 'results/' directory.")