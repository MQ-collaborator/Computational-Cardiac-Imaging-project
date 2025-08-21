#program to use linear regression on latent space and original phenotypes to find out how much each phenotype variable effects SBP
#Find variables most impactful on prediction by combining lantent space regression coefficients and correlation between latent variable and feature
#Target relationship: Feature_importance(x_j) = sum (abs(coef[i])) * abs(correlation(z_j, x_j))

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load encoded dataset (expects a CSV created by the pipeline)
BASE = Path(__file__).parent
ENCODED_CSV = BASE / "latent_and_pca_data" / "encoded_data.csv"
OUTPUT_FOLDER = BASE / "latent_and_pca_data"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUTPUT_FOLDER / "feature_importance_from_latent.png"
CSV_PATH = OUTPUT_FOLDER / "feature_importance_from_latent.csv"

if not ENCODED_CSV.exists():
    raise FileNotFoundError(f"Expected encoded data at {ENCODED_CSV} - adjust path or generate encoded_data.csv")

df = pd.read_csv(ENCODED_CSV)

# find target column (SBP)
sbp_cols = [c for c in df.columns if re.search(r"SBP", c, re.I)]
if not sbp_cols:
    raise RuntimeError("Could not find SBP target column in encoded_data.csv (expecting name containing 'SBP').")
target_col = sbp_cols[0]

y = df[target_col].to_numpy().ravel()

# detect latent columns by name pattern
latent_pattern = re.compile(r"^(z\d+|latent|enc|emb|dim|pc|component)", re.I)
latent_cols = [c for c in df.columns if latent_pattern.search(c)]

# fallback: if no columns match pattern, try to detect numeric columns that are not the target and that likely represent embeddings
if not latent_cols:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    candidate = [c for c in numeric_cols if re.search(r"\b0\b|\b1\b|_\d+$|\d+$", c)]
    latent_cols = candidate if candidate else numeric_cols

# original phenotype columns are numeric columns excluding latent cols and target
original_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in latent_cols and c != target_col]

# If detection failed, try reading saved column names from data/dataset_info/dl_column_names.csv
if len(original_cols) == 0:
    dataset_info_csv = BASE.parent.parent / "data" / "dataset_info" / "dl_column_names.csv"
    if dataset_info_csv.exists():
        try:
            cols_df = pd.read_csv(dataset_info_csv)
            # pick first column of that file (common formats vary)
            candidate_list = cols_df.iloc[:, 0].astype(str).tolist()
            original_cols = [c for c in candidate_list if c in df.columns and c != target_col and c not in latent_cols]
        except Exception:
            original_cols = []

# final fallback: take numeric columns that don't look like latent names (more permissive)
if len(original_cols) == 0:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    permissive = [c for c in numeric_cols if not latent_pattern.search(c)]
    if permissive:
        original_cols = permissive

if len(latent_cols) == 0:
    raise RuntimeError("No latent columns detected in encoded_data.csv - please include latent embeddings with recognisable names.")

if len(original_cols) == 0:
    # try to recover original phenotype columns from the project's preprocessing
    recovered = []
    try:
        # prefer package-relative import, fall back to absolute
        try:
            from . import utils as dl_utils
        except Exception:
            from deep_learning import utils as dl_utils

        proc_df = dl_utils.preprocess()
        # numeric phenotype candidates excluding target and latent names
        proc_numeric = [c for c in proc_df.select_dtypes(include=[np.number]).columns if c != target_col and c not in latent_cols]
        if proc_numeric:
            # require that processed dataframe aligns with encoded CSV (same length and SBP order)
            if len(proc_df) == len(df) and np.allclose(proc_df[target_col].to_numpy(), df[target_col].to_numpy(), equal_nan=True):
                recovered = proc_numeric
            else:
                # lengths/order mismatch — do not silently misalign rows
                print("Warning: processed phenotype dataframe does not align with encoded_data.csv (length or SBP ordering differs).")
                print(f"processed_df rows: {len(proc_df)}, encoded rows: {len(df)}")
                # still provide candidate list to user for manual inspection
                recovered = proc_numeric
    except Exception:
        recovered = []

    # if we recovered candidates, accept them (but warn if alignment is uncertain)
    if recovered:
        original_cols = recovered
        print(f"Recovered original phenotype columns from preprocessing: {len(original_cols)} columns found.")
        if len(proc_df) != len(df) or not np.allclose(proc_df[target_col].to_numpy(), df[target_col].to_numpy(), equal_nan=True):
            print("WARNING: Recovered phenotype dataframe did not match encoded_data.csv row ordering/length. "
                  "Please verify row alignment (e.g. by including an ID column in encoded_data.csv) before trusting importances.")
    else:
        available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        raise RuntimeError(
            "No original phenotype numeric columns detected (after excluding latent columns and target). "
            f"Detected latent columns: {latent_cols}. Available numeric columns: {available_numeric}. "
            "If your encoded_data.csv doesn't contain original phenotype columns, regenerate it including phenotype columns "
            "or make dl_column_names.csv available in data/dataset_info. As a fallback the script attempted to read the project's "
            "preprocessed phenotype dataframe but could not recover usable columns."
        )

Z = df[latent_cols].to_numpy()
X = df[original_cols].to_numpy()

# fit linear regression from latent -> SBP
lr = LinearRegression()
lr.fit(Z, y)
coef = lr.coef_  # shape (n_latent,)
abs_coef = np.abs(coef)

# compute absolute Pearson correlations between each latent dim (columns of Z) and each original feature (columns of X)
# produce matrix shape (n_latent, n_original)
corr_matrix = np.zeros((Z.shape[1], X.shape[1]), dtype=float)
for i in range(Z.shape[1]):
    for j in range(X.shape[1]):
        zi = Z[:, i]
        xj = X[:, j]
        # if constant column, corr is zero
        if np.std(zi) == 0 or np.std(xj) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(zi, xj)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        corr_matrix[i, j] = abs(corr)

# Feature importance: for each original feature j, importance_j = sum_i abs(coef_i) * abs(corr_{i,j})
importances = (abs_coef.reshape(-1, 1) * corr_matrix).sum(axis=0)

# create a DataFrame with results, sort descending
importance_df = pd.DataFrame({
    "feature": original_cols,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

# save CSV
importance_df.to_csv(CSV_PATH, index=False)

# plot horizontal bar chart (top 30 or all if fewer)
top_n = min(30, len(importance_df))
plot_df = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal bar order

plt.close("all")
fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * top_n)))
ax.barh(plot_df["feature"], plot_df["importance"], color="C0")
ax.set_xlabel("Importance (|latent_coef| * |corr| summed over latent dims)")
ax.set_title("Feature importance derived from latent -> SBP regression")
plt.tight_layout()
fig.savefig(FIG_PATH, dpi=200)
plt.show()

print("Saved feature importances to:", CSV_PATH)
print("Saved plot to:", FIG_PATH)