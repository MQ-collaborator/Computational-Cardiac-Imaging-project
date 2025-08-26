import conventional_utils as conventional_utils
import sys
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from conventional_utils import home_directory
import warnings

# Implementation of linear regression with interaction terms
# Single fit (iterations removed) + bootstrap 95% CI on coefficients

# alpha determines the degree of regularization (must be positive)
ALPHA = 5e-5

"""File paths"""
coefficients_folder = home_directory / "data" / "regression_results"


def _map_columns_by_correlation(df, X_train, target_name="SBP_at_MRI"):
    """
    Try to recover column names for X_train by matching dataframe numeric columns
    to X_train columns using absolute Pearson correlation. Returns list of names
    length = X_train.shape[1].
    """
    p = X_train.shape[1]
    candidate_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_name]
    if len(candidate_cols) < p:
        return None  # not enough candidates
    abs_corr = np.zeros((len(candidate_cols), p), dtype=float)
    for j, cname in enumerate(candidate_cols):
        col_vals = df[cname].to_numpy()
        for k in range(p):
            xcol = X_train[:, k]
            if np.std(col_vals) == 0 or np.std(xcol) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(col_vals, xcol)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            abs_corr[j, k] = abs(corr)
    # greedy assignment to avoid duplicate mapping
    available = set(range(len(candidate_cols)))
    mapped = []
    for k in range(p):
        masked = np.full(len(candidate_cols), -np.inf)
        for j in available:
            masked[j] = abs_corr[j, k]
        jbest = int(np.argmax(masked))
        if masked[jbest] <= 1e-8:
            # mapping uncertain but continue
            pass
        mapped.append(candidate_cols[jbest])
        available.remove(jbest)
    return mapped


def linear_regression(regression_type, save, iterations, datamode="r",
                      n_bootstrap=500, n_jobs=1, subsample_frac=1.0):
    """
    Single fit linear regression (Lasso or Ridge) and bootstrap 95% CIs.

    n_bootstrap: number of bootstrap resamples (default 500).
    n_jobs: number of parallel workers for bootstrap (joblib). Use -1 for all cores.
    subsample_frac: fraction of rows used per bootstrap resample (<=1.0). Use 1.0 for full-size resamples.
    """
    # load data
    df = conventional_utils.preprocess(datamode=datamode)
    X_train, y_train, input_size = conventional_utils.list_data(df)
    # --------------------------------------------------------------

    # Basic sanity checks / defensive programming
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # iterations argument ignored (kept for CLI compatibility)
    if iterations is None or iterations <= 0:
        print("Warning: 'iterations' was not > 0; using a single fit.")
    elif iterations != 1:
        print(f"Note: iterations={iterations} ignored — running a single fit for speed/consistency.")

    # ensure output directory exists
    coefficients_folder.mkdir(parents=True, exist_ok=True)

    # Single model fit
    if regression_type == 1:
        model = Lasso(alpha=ALPHA, max_iter=5000)
    else:
        model = Ridge(alpha=ALPHA)

    model.fit(X_train, y_train)
    coefficients = model.coef_.astype(float)
    y_pred = model.predict(X_train)
    average_rmse = np.sqrt(mean_squared_error(y_train, y_pred))

    # check for NaNs/Infs before saving
    if np.isnan(coefficients).any() or np.isinf(coefficients).any():
        raise RuntimeError("Coefficients contain NaN or Inf; aborting save. Check training data.")

    # Compute bootstrap CIs by refitting model on resampled rows
    n_samples = X_train.shape[0]
    bs_size = max(1, int(n_samples * float(subsample_frac)))

    def _one_boot(seed):
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, n_samples, bs_size)
        Xb = X_train[idx]
        yb = y_train[idx]
        if regression_type == 1:
            m = Lasso(alpha=ALPHA, max_iter=5000)
        else:
            m = Ridge(alpha=ALPHA)
        try:
            m.fit(Xb, yb)
            return m.coef_.astype(float)
        except Exception:
            return np.full(input_size, np.nan, dtype=float)

    # generate reproducible seeds
    master_rng = np.random.default_rng(0)
    seeds = master_rng.integers(0, 2**31 - 1, size=n_bootstrap)

    # run bootstrap in parallel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs)(delayed(_one_boot)(s) for s in seeds)

    coef_samples = np.vstack(results) if len(results) > 0 else np.full((0, input_size), np.nan)
    valid_mask = ~np.isnan(coef_samples).all(axis=1)
    n_valid = int(valid_mask.sum())
    if n_valid < max(10, n_bootstrap // 10):
        print(f"Warning: only {n_valid}/{n_bootstrap} successful bootstrap fits; CIs may be unreliable.")

    # percentiles (ignore NaNs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if coef_samples.size == 0:
            ci_lower = np.full(input_size, np.nan)
            ci_upper = np.full(input_size, np.nan)
        else:
            ci_lower = np.nanpercentile(coef_samples, 2.5, axis=0)
            ci_upper = np.nanpercentile(coef_samples, 97.5, axis=0)

    # save coefficients + CIs to CSV, ensuring column names align with X_train order
    if save == 0:
        # attempt direct mapping if df columns (excluding target) match input size
        try:
            candidate_cols = [c for c in df.columns if c != 'SBP_at_MRI']
        except Exception:
            candidate_cols = []

        if len(candidate_cols) == input_size:
            column_names = candidate_cols
        else:
            mapped = _map_columns_by_correlation(df, X_train, target_name='SBP_at_MRI')
            if mapped is not None and len(mapped) == input_size:
                column_names = mapped
            else:
                # fallback generic names but warn
                print("Warning: could not unambiguously map dataframe column names to X_train columns.")
                column_names = [f"feature_{i}" for i in range(input_size)]

        coef_df = pd.DataFrame({
            'Column': column_names,
            'Coefficient': coefficients,
            'CI_0.025': ci_lower,
            'CI_0.975': ci_upper
        })

        coefficients_path = coefficients_folder / (f"coefficients_{datamode}.csv")
        coef_df.to_csv(coefficients_path, index=False)

        sorted_coefficients_path = coefficients_folder / (f"sorted_coefficients_{datamode}.csv")
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        sorted_coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
        sorted_coef_df.to_csv(sorted_coefficients_path, index=False)

    # display results
    print("Root mean squared error: " + str(average_rmse))
    print(f"Bootstrap CIs saved (n_bootstrap={n_bootstrap}, n_valid={n_valid})")

    return 0


if __name__ == "__main__":
    status = 0
    if len(sys.argv) != 5:
        print("Usage: python lin_reg_vn.py type save datamode iterations")
        print("Regression types: (1=Lasso or 2=Ridge).")
        print("Save modes: 0 to save coefficients, 1 to not save coefficients.")
        print("Preprocess datamode: 'r' for reduced dataset, 'f' for full dataset.")
        print("Iterations: ignored, single fit will run.")
        status = 1

    regression_type = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if regression_type != 1 and regression_type != 2:
        print("The integer passed does not correspond to a mode that has been implemented")
        status = 1
    save = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    datamode = sys.argv[3] if len(sys.argv) > 3 else "r"
    iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    if status == 0:
        # defaults: 500 bootstrap samples, single job. Adjust n_bootstrap and n_jobs for speed/accuracy.
        linear_regression(regression_type, save, iterations, datamode=datamode,
                          n_bootstrap=500, n_jobs=1, subsample_frac=1.0)