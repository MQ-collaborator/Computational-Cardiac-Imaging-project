import sys
from pathlib import Path
import numpy as np
import pandas as pd
import conventional_utils
from conventional_utils import home_directory
from sklearn.linear_model import Lasso
import statsmodels.api as sm

"""
lin_reg_v4.py
- Use Lasso (sklearn) for variable selection, then refit OLS (statsmodels) on selected features
  to obtain unbiased beta estimates, 95% CI, p-values and R-squared.
- Saves coefficients, 95% CI, p-values, and R-squared to CSV in data/regression_results.
- Uses conventional_utils.preprocess and conventional_utils.list_data to load data.
"""

coefficients_folder = home_directory / "data" / "regression_results"
coefficients_folder.mkdir(parents=True, exist_ok=True)

# Lasso penalty (same convention as other files)
ALPHA = 5e-5


def _map_columns_by_correlation(df, X_train, target_name="SBP_at_MRI"):
    """Attempt to recover column names for X_train by matching numeric df columns to X columns."""
    p = X_train.shape[1]
    candidate_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_name]
    if len(candidate_cols) < p:
        return None
    abs_corr = np.zeros((len(candidate_cols), p), dtype=float)
    for j, cname in enumerate(candidate_cols):
        col_vals = df[cname].to_numpy()
        for k in range(p):
            xcol = X_train[:, k]
            if np.nanstd(col_vals) == 0 or np.nanstd(xcol) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(col_vals, xcol)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            abs_corr[j, k] = abs(corr)
    available = set(range(len(candidate_cols)))
    mapped = []
    for k in range(p):
        masked = np.full(len(candidate_cols), -np.inf)
        for j in available:
            masked[j] = abs_corr[j, k]
        jbest = int(np.argmax(masked))
        mapped.append(candidate_cols[jbest])
        available.remove(jbest)
    return mapped


def run_lasso_then_ols(datamode="f", save=True, lasso_alpha=ALPHA):
    """
    Fit Lasso once for variable selection, then refit OLS on selected features to obtain
    coefficient estimates, 95% CI, p-values and R-squared. Saves CSVs to regression_results.
    """
    # load dataframe and design matrix (conventional_utils.list_data should return X in row order matching df)
    df = conventional_utils.preprocess(datamode=datamode)
    X, y, input_size = conventional_utils.list_data(df)

    # Fit Lasso once (no manual iteration)
    lasso = Lasso(alpha=lasso_alpha, max_iter=10000)
    lasso.fit(X, y)
    lasso_coefs = lasso.coef_.astype(float)

    # Determine selected features (non-zero by Lasso)
    selected_mask = np.abs(lasso_coefs) > 0.0
    n_selected = int(selected_mask.sum())

    # If Lasso selects none, fall back to OLS on all features
    if n_selected == 0:
        X_sel = X
        sel_idx = np.arange(input_size)
        print("Lasso selected no features; fitting OLS on all features.")
    else:
        X_sel = X[:, selected_mask]
        sel_idx = np.nonzero(selected_mask)[0]
        print(f"Lasso selected {n_selected}/{input_size} features; refitting OLS on selected set.")

    # Add constant (intercept) and fit OLS on selected features
    X_design = sm.add_constant(X_sel, has_constant='add')
    model = sm.OLS(y, X_design)
    results = model.fit()

    # Extract stats from OLS
    params = results.params  # intercept + coefficients for selected features
    conf = results.conf_int(alpha=0.05)  # (p_sel+1, 2)
    pvals = results.pvalues
    rsq = results.rsquared
    adj_rsq = results.rsquared_adj

    # Prepare full-length arrays aligned with X order
    coef_full = np.array(lasso_coefs, dtype=float)  # default to Lasso estimate (zero for unselected)
    ci_lower_full = np.full(input_size, np.nan, dtype=float)
    ci_upper_full = np.full(input_size, np.nan, dtype=float)
    pval_full = np.full(input_size, np.nan, dtype=float)

    # Fill selected indices with OLS estimates and stats (drop intercept entry)
    if n_selected == 0:
        # use OLS results for all features
        coef_full = params[1:]
        ci_lower_full = conf[1:, 0]
        ci_upper_full = conf[1:, 1]
        pval_full = pvals[1:]
    else:
        ols_coefs = params[1:]
        ols_ci_lower = conf[1:, 0]
        ols_ci_upper = conf[1:, 1]
        ols_pvals = pvals[1:]
        coef_full[selected_mask] = ols_coefs
        ci_lower_full[selected_mask] = ols_ci_lower
        ci_upper_full[selected_mask] = ols_ci_upper
        pval_full[selected_mask] = ols_pvals
        # For unselected features keep Lasso coef (likely 0). Leave CI/pval as NaN.

    # Map column names to X order (prefer explicit df column order if it matches)
    try:
        candidate_cols = [c for c in df.columns if c != 'SBP_at_MRI']
    except Exception:
        candidate_cols = []

    if len(candidate_cols) == input_size:
        column_names = candidate_cols
    else:
        mapped = _map_columns_by_correlation(df, X, target_name='SBP_at_MRI')
        if mapped is not None and len(mapped) == input_size:
            column_names = mapped
        else:
            column_names = [f"feature_{i}" for i in range(input_size)]

    # Assemble output DataFrame
    out_df = pd.DataFrame({
        "Column": column_names,
        "Coefficient": coef_full,
        "CI_0.025": ci_lower_full,
        "CI_0.975": ci_upper_full,
        "p_value": pval_full
    })

    # Save results and sorted version
    if save:
        coefficients_path = coefficients_folder / f"coefficients_{datamode}.csv"
        sorted_path = coefficients_folder / f"sorted_coefficients_{datamode}.csv"
        out_df.to_csv(coefficients_path, index=False)
        out_df['Abs_Coefficient'] = out_df['Coefficient'].abs()
        out_df.sort_values(by='Abs_Coefficient', ascending=False).to_csv(sorted_path, index=False)

    # Print summary
    print(f"R-squared (OLS on selected features): {rsq:.4f}, Adjusted R-squared: {adj_rsq:.4f}")
    print(f"Saved coefficients to: {coefficients_path}")
    return results, out_df


if __name__ == "__main__":
    datamode = sys.argv[1] if len(sys.argv) > 1 else "r"
    run_results, df_out = run_lasso_then_ols(datamode=datamode, save=True)
