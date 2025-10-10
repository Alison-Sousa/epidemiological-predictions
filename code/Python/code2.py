# FILE MAPPING:
# ds1.csv: Dengue cases per municipality.
# ds2.csv: Geographic, demographic (IBGE), and climate data.
# ds3.csv: Socioeconomic and infrastructure data (requires latin1 encoding).
# All source files are expected in the 'data/Python/' directory.
# All outputs will be saved to the 'results/' directory.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import set_random_seed
from scipy.stats import gaussian_kde

# --- 1. CONFIGURATION AND SETUP ---
set_random_seed(123)
np.random.seed(123)
data_path = "data/Python"
results_path = "results"
os.makedirs(results_path, exist_ok=True)

# Academic color scheme for plots
colors = {
    "main": "#003f5c",
    "sec": "#ffa600",
    "tri": "#bc5090",
    "lt": "#f8f9fa",
    "dk": "#212529",
    "grid": "#ced4da"
}

# --- 2. DATA LOADING AND PREPROCESSING ---
print("Step 1: Loading and preprocessing data...")

# Load datasets
den = pd.read_csv(f"{data_path}/ds1.csv")  # Dengue cases
cli = pd.read_csv(f"{data_path}/ds2.csv")  # Climate
ips = pd.read_csv(f"{data_path}/ds3.csv", encoding="latin1")  # Infrastructure/Health

# Standardize municipality codes to 6 digits
den['cod6'] = den['municipio'].astype(str).str[:6]
ips['cod6'] = ips['Código IBGE'].astype(str).str[:6]
cli['cod6'] = cli['munic_code'].astype(str).str[:6]

# Merge static features (socioeconomic and climate)
fixed = pd.merge(
    ips,
    cli.drop(columns=['municipio', 'estado', 'regiao', 'koppen', 'munic_code'], errors="ignore"),
    on="cod6", how="left"
)
fixed = fixed.loc[:, ['cod6'] + [c for c in fixed.columns if fixed[c].dtype in [np.float64, np.int64]]]

# Process time-series data for dengue cases
years = [str(y) for y in range(2013, 2026)]
for year in years:
    if year in den.columns:
        den[year] = pd.to_numeric(den[year].replace("-", 0), errors="coerce").fillna(0)

# --- 3. MODEL AND ANALYSIS FUNCTIONS ---

def run_lstm(X, y, X_pred, epochs=15, batch_size=32):
    """Trains an LSTM model and returns predictions."""
    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
    X_pred_lstm = X_pred.values.reshape((X_pred.shape[0], 1, X_pred.shape[1]))
    
    model = Sequential([
        LSTM(32, input_shape=(1, X.shape[1]), return_sequences=False),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(X_lstm, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model.predict(X_pred_lstm, verbose=0).flatten()

def process_years_and_pyramid():
    """
    Iterates through each year to train a model, evaluate performance,
    and generate the AUC distribution pyramid plot.
    """
    print("Step 2: Processing each year and generating AUC pyramid...")
    
    # Grid positions for the pyramid plot
    pos = {
        2013: (0,1), 2014: (0,2), 2015: (0,3),
        2016: (1,0), 2017: (1,1), 2018: (1,2), 2019: (1,3), 2020: (1,4),
        2021: (2,0), 2022: (2,1), 2023: (2,2), 2024: (2,3), 2025: (2,4)
    }
    
    all_predictions = pd.DataFrame()
    yearly_metrics = []
    pdf_path = f"{results_path}/map.pdf"
    
    with PdfPages(pdf_path) as pp:
        fig, axs = plt.subplots(3, 5, figsize=(22, 14)) # Increased figure size for larger fonts
        plt.subplots_adjust(wspace=0.4, hspace=0.4) # Adjusted spacing
        
        for ax in axs.flatten(): 
            ax.set_visible(False)
            
        for year in years:
            yint = int(year)
            print(f"  - Processing year {year}...")
            
            # Prepare data for the current year
            yd = den[['cod6', 'municipio', year]].rename(columns={year: 'dengue'})
            yd = yd.merge(fixed, on='cod6', how='left').dropna()
            
            # Define risk based on the year's median incidence
            med = np.median(yd['dengue'])
            yd['risk_real'] = (yd['dengue'] > med).astype(int)
            
            # Balance classes using undersampling for training
            nmin = yd['risk_real'].value_counts().min()
            bal = pd.concat([
                yd[yd['risk_real']==0].sample(n=nmin, random_state=123),
                yd[yd['risk_real']==1].sample(n=nmin, random_state=123)
            ])
            
            X_train = bal.drop(columns=['cod6','municipio','dengue','risk_real'])
            y_train = bal['risk_real'].values
            X_predict = yd.drop(columns=['cod6','municipio','dengue','risk_real'])
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_predict_scaled = pd.DataFrame(scaler.transform(X_predict), columns=X_predict.columns)
            
            # Run model and get predictions for the entire year's dataset
            probs = run_lstm(X_train_scaled, y_train, X_predict_scaled)
            y_true = yd['risk_real'].values
            
            # Calculate and store yearly performance metrics
            y_pred_class = (probs > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            yearly_metrics.append({
                'year': year,
                'auc': roc_auc_score(y_true, probs),
                'accuracy': accuracy_score(y_true, y_pred_class),
                'sensitivity': sensitivity,
                'specificity': specificity
            })

            # Store predictions
            res = pd.DataFrame({
                'cod6': yd['cod6'],
                'mun': yd['municipio'],
                'yr': year,
                'prob': probs,
                'risk': y_true
            })
            all_predictions = pd.concat([all_predictions, res])
            
            # Bootstrap AUC for robust evaluation
            n_boot = 1000
            rng = np.random.RandomState(123)
            idx = np.arange(len(y_true))
            boot_aucs = []
            for _ in range(n_boot):
                boot_indices = rng.choice(idx, size=len(idx), replace=True)
                if len(np.unique(y_true[boot_indices])) < 2: continue
                boot_aucs.append(roc_auc_score(y_true[boot_indices], probs[boot_indices]))

            boot_aucs = np.array(boot_aucs)
            
            # Plotting the AUC distribution for the current year
            row, col = pos[yint]
            ax = axs[row, col]
            ax.set_visible(True)
            ax.hist(boot_aucs, bins=20, density=True, color=colors["sec"], edgecolor="white", alpha=0.7)
            
            # Density curve
            dens = gaussian_kde(boot_aucs)
            xs = np.linspace(min(boot_aucs), max(boot_aucs), 200)
            ax.plot(xs, dens(xs), color=colors["main"], lw=2)
            
            # Mean AUC line and text
            mean_auc = np.mean(boot_aucs)
            ax.axvline(x=mean_auc, color=colors["tri"], linestyle='--', lw=2.5)
            ax.text(mean_auc, ax.get_ylim()[1]*0.9, f"{mean_auc:.3f}",
                   color=colors["tri"], ha='center', fontsize=20, weight='bold', # Increased font size
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # --- FONT SIZE ADJUSTMENTS ---
            ax.set_title(str(year), fontsize=26, pad=15, weight='bold')
            ax.tick_params(axis='both', labelsize=18) 
            ax.set_xlabel("AUC", fontsize=20)
            ax.set_ylabel("Density", fontsize=20)
            ax.grid(alpha=0.2, linestyle='--')
            for s in ax.spines.values(): s.set_visible(True)

        plt.tight_layout()
        pp.savefig(fig)
        plt.close(fig)
    
    # Save the calculated yearly metrics to a new CSV file
    pd.DataFrame(yearly_metrics).to_csv(f"{results_path}/yearly_metrics.csv", index=False)
    print(f"\nSaved yearly performance metrics to {results_path}/yearly_metrics.csv")
        
    all_predictions.to_csv(f"{results_path}/zen.csv", index=False)
    print(f"Saved all yearly predictions to {results_path}/zen.csv")
    print(f"Saved AUC pyramid plot to {pdf_path}")
    return all_predictions

def cluster_analysis(all_predictions):
    """Performs cluster analysis based on model predictions."""
    print("\nStep 3: Analyzing risk by cluster...")
    
    muni_risk_profile = all_predictions.groupby('cod6')['prob'].mean().reset_index()
    
    try:
        clusters = pd.read_csv(f"{results_path}/clu.csv", dtype={'cod6': str})
        # FIX: Rename the 'clu' column from code1's output to the expected 'cluster'
        if 'clu' in clusters.columns:
            clusters.rename(columns={'clu': 'cluster'}, inplace=True)
        df = muni_risk_profile.merge(clusters[['cod6', 'cluster']], on='cod6', how='left')
    except FileNotFoundError:
        print("  - Cluster file not found. Proceeding without cluster data.")
        df = muni_risk_profile.copy()
        df['cluster'] = 'N/A'
        
    # Binning probabilities for analysis
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['risk_category'] = pd.cut(df['prob'], bins=bins, labels=labels, include_lowest=True)
    
    # Plotting histogram of mean probabilities
    plt.figure(figsize=(10,6))
    plt.hist(df['prob'].dropna(), bins=30, color=colors["sec"], edgecolor=colors["dk"], alpha=0.8)
    if len(df['prob'].dropna()) > 1:
        density = gaussian_kde(df['prob'].dropna())
        x = np.linspace(0, 1, 1000)
        # Scale density to match histogram counts
        bin_width = (df['prob'].max() - df['prob'].min()) / 30
        density_scale = len(df['prob'].dropna()) * bin_width
        plt.plot(x, density(x) * density_scale, color=colors["main"], lw=2.5)
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Number of Municipalities', fontsize=14)
    plt.grid(axis='y', alpha=0.15, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{results_path}/fox.pdf")
    plt.close()
    
    # Save cross-tabulation of clusters and risk categories
    if 'cluster' in df.columns and df['cluster'].nunique() > 1:
        contingency_table = pd.crosstab(df['cluster'], df['risk_category'])
        contingency_table.to_csv(f"{results_path}/owl.csv")
    
    risk_category_counts = df['risk_category'].value_counts().sort_index()
    risk_category_counts.to_csv(f"{results_path}/rug.csv")
    print(f"Saved analysis files: owl.csv, rug.csv, fox.pdf in '{results_path}/'")

# --- 4. SCRIPT EXECUTION ---
if __name__ == "__main__":
    print("Starting yearly dengue risk analysis pipeline...")
    all_predictions_df = process_years_and_pyramid()
    cluster_analysis(all_predictions_df)
    print("\nPipeline finished successfully.")

