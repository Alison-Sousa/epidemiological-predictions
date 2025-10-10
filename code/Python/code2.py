# FILE MAPPING:
# ds1.csv = dengue cases (was datasus.csv)
# ds2.csv = IBGE/climate (was ibge.csv)
# ds3.csv = infrastructure/health (was ips.csv, ONLY THIS ONE NEEDS encoding="latin1")
# All files are in data/Python/. Output always in results/ with short file names.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import set_random_seed
from scipy.stats import gaussian_kde

# --- CONFIG ---
set_random_seed(123)
np.random.seed(123)
data_path = "data/Python"
results_path = "results"
os.makedirs(results_path, exist_ok=True)

# Academic color scheme
colors = {
    "main": "#003f5c",
    "sec": "#ffa600",
    "tri": "#bc5090",
    "lt": "#f8f9fa",
    "dk": "#212529",
    "grid": "#ced4da"
}

# --- DATA LOADING ---
print("Loading and preprocessing data...")

den = pd.read_csv(f"{data_path}/ds1.csv")  # dengue cases
cli = pd.read_csv(f"{data_path}/ds2.csv")  # climate
ips = pd.read_csv(f"{data_path}/ds3.csv", encoding="latin1")  # infra/health

den['cod6'] = den['municipio'].astype(str).str[:6]
ips['cod6'] = ips['Código IBGE'].astype(str).str[:6]
cli['cod6'] = cli['munic_code'].astype(str).str[:6]

fixed = pd.merge(
    ips,
    cli.drop(columns=['municipio', 'estado', 'regiao', 'koppen', 'munic_code'], errors="ignore"),
    on="cod6", how="left"
)
fixed = fixed.loc[:, ['cod6'] + [c for c in fixed.columns if fixed[c].dtype in [np.float64, np.int64]]]

years = [str(y) for y in range(2013, 2026)]
for year in years:
    if year in den.columns:
        den[year] = pd.to_numeric(den[year].replace("-", 0), errors="coerce").fillna(0)

def run_lstm(X, y, X_pred, epochs=15, batch_size=32):
    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
    X_pred_lstm = X_pred.values.reshape((X_pred.shape[0], 1, X_pred.shape[1]))
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, X.shape[1]), return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(X_lstm, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model.predict(X_pred_lstm, verbose=0).flatten()

def process_years_and_pyramid():
    print("Processing years and AUC pyramid...")
    pos = {
        2013: (0,1), 2014: (0,2), 2015: (0,3),
        2016: (1,0), 2017: (1,1), 2018: (1,2), 2019: (1,3), 2020: (1,4),
        2021: (2,0), 2022: (2,1), 2023: (2,2), 2024: (2,3), 2025: (2,4)
    }
    allp = pd.DataFrame()
    pdf_path = f"{results_path}/map.pdf"
    with PdfPages(pdf_path) as pp:
        fig, axs = plt.subplots(3, 5, figsize=(18, 12))
        plt.subplots_adjust(wspace=0.36, hspace=0.32)
        for ax in axs.flatten(): ax.set_visible(False)
        for year in years:
            yint = int(year)
            print(f"  Year {year}...")
            yd = den[['cod6', 'municipio', year]].rename(columns={year: 'dengue'})
            yd = yd.merge(fixed, on='cod6', how='left').dropna()
            med = np.median(yd['dengue'])
            yd['risk_real'] = (yd['dengue'] > med).astype(int)
            nmin = yd['risk_real'].value_counts().min()
            bal = pd.concat([
                yd[yd['risk_real']==0].sample(n=nmin, random_state=123),
                yd[yd['risk_real']==1].sample(n=nmin, random_state=123)
            ])
            X = bal.drop(columns=['cod6','municipio','dengue','risk_real'])
            yv = bal['risk_real'].values
            Xp = yd.drop(columns=['cod6','municipio','dengue','risk_real'])
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            Xp = pd.DataFrame(scaler.transform(Xp), columns=X.columns)
            probs = run_lstm(X, yv, Xp)
            y_true = yd['risk_real'].values
            res = pd.DataFrame({
                'cod6': yd['cod6'],
                'mun': yd['municipio'],
                'yr': year,
                'prob': probs,
                'risk': y_true
            })
            allp = pd.concat([allp, res])
            n_boot = 1000
            rng = np.random.RandomState(123)
            idx = np.arange(len(y_true))
            boots = []
            for _ in range(n_boot):
                bi = rng.choice(idx, size=len(idx), replace=True)
                try:
                    auc = roc_auc_score(y_true[bi], probs[bi])
                    boots.append(auc)
                except:
                    pass
            boots = np.array(boots)
            row, col = pos[yint]
            ax = axs[row, col]
            ax.set_visible(True)
            ax.hist(boots, bins=20, density=True, color=colors["sec"],
                   edgecolor="white", alpha=0.7)
            dens = gaussian_kde(boots)
            xs = np.linspace(min(boots), max(boots), 200)
            ax.plot(xs, dens(xs), color=colors["main"], lw=2)
            mean_auc = np.mean(boots)
            ax.axvline(x=mean_auc, color=colors["tri"], linestyle='--', lw=1.5)
            ax.text(mean_auc, ax.get_ylim()[1]*0.9, f"{mean_auc:.3f}",
                   color=colors["tri"], ha='center', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
            ax.set_title(str(year), fontsize=20, pad=10)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlabel("AUC", fontsize=12)
            ax.grid(alpha=0.15, linestyle='--')
            for s in ax.spines.values(): s.set_visible(True)
        plt.tight_layout()
        pp.savefig(fig)
        plt.close(fig)
    allp.to_csv(f"{results_path}/zen.csv", index=False)
    print(f"Saved all results to {results_path}/zen.csv and pyramid to {pdf_path}")
    return allp

def cluster_analysis(allp):
    print("Analyzing risk by cluster...")
    muni = allp.groupby('cod6')['prob'].mean().reset_index()
    try:
        clus = pd.read_csv(f"{results_path}/clu.csv", dtype={'cod6': str})
        df = muni.merge(clus[['cod6', 'cluster']], on='cod6', how='left')
    except:
        df = muni.copy()
        df['cluster'] = 0
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['a','b','c','d','e']
    df['pbin'] = pd.cut(df['prob'], bins=bins, labels=labels, include_lowest=True)
    plt.figure(figsize=(10,6))
    plt.hist(df['prob'].dropna(), bins=30, color=colors["sec"], edgecolor=colors["dk"], alpha=0.8)
    if len(df) > 1:
        density = gaussian_kde(df['prob'].dropna())
        x = np.linspace(0, 1, 1000)
        plt.plot(x, density(x) * len(df) / 30, color=colors["main"], lw=2)
    plt.xlabel('Prob', fontsize=14)
    plt.ylabel('N', fontsize=14)
    plt.grid(axis='y', alpha=0.15, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{results_path}/fox.pdf")
    plt.close()
    if 'cluster' in df.columns:
        cnt = df.groupby(['cluster', 'pbin']).size().unstack(fill_value=0)
        cnt.to_csv(f"{results_path}/owl.csv")
    tot = df['pbin'].value_counts().sort_index()
    tot.to_csv(f"{results_path}/rug.csv")
    print(f"Saved: owl.csv, rug.csv, zen.csv, map.pdf, fox.pdf in {results_path}/")

if __name__ == "__main__":
    print("Starting dengue risk analysis...")
    allp = process_years_and_pyramid()
    cluster_analysis(allp)
    print("Done.")