import conventional_utils as conventional_utils
import sys
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from conventional_utils import home_directory

# new imports
from sklearn.utils import resample
import warnings

#Implementation of linear regression with interaction terms
#Iterative process to average coefficients over multiple iterations
#This is holdout CV being run iteratively with random train test split. This is less rigorous than k-fold cross validation but allows for faster results
#Techincally it may be better to use k-fold to save computation time but mathematically speaking this is equivalent

""""List of variables affecting style of training"""
#alpha determines the degree of regularization (must be positive)
ALPHA = 5e-5

"""File paths"""
coefficients_folder = home_directory / "data" / "regression_results" 

def linear_regression(regression_type, save, iterations, datamode = "r", ci_method="ols", n_bootstrap=200):
    """
    ci_method: "ols" (uses statsmodels OLS conf_int) or "bootstrap" (bootstrap model refit CI; works for Lasso/Ridge)
    n_bootstrap: number of bootstrap samples when ci_method="bootstrap"
    """
    #load data
    df = conventional_utils.preprocess(datamode=datamode)
    X_train, y_train, input_size = conventional_utils.list_data(df)
    # --------------------------------------------------------------

    # Basic sanity checks / defensive programming
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    if iterations is None or iterations <= 0:
        raise ValueError(f"Invalid iterations value: {iterations}. Must be > 0")

    # ensure output directory exists
    coefficients_folder.mkdir(parents=True, exist_ok=True)

    #Iterative regression. Average values stored in coefficients_path
    coefficients = np.zeros(input_size, dtype=float)
    average_rmse = 0.0

    if regression_type == 1:
        for i in range(iterations):
            lasso = Lasso(alpha=ALPHA)
            lasso.fit(X_train, y_train)
            y_pred = lasso.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            coefficients += lasso.coef_
            average_rmse += rmse
            if i % max(1, iterations//5) == 0:
                print(f"iter {i}: rmse={rmse:.6f}, coef sample={lasso.coef_[:3]}")
    else:
        for i in range(iterations):
            ridge = Ridge(alpha=ALPHA)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            coefficients += ridge.coef_
            average_rmse += rmse
            if i % max(1, iterations//5) == 0:
                print(f"iter {i}: rmse={rmse:.6f}, coef sample={ridge.coef_[:3]}")

    # finalise averages (safe division guaranteed because we validated iterations)
    coefficients /= iterations
    average_rmse /= iterations

    # check for NaNs/Infs before saving
    if np.isnan(coefficients).any() or np.isinf(coefficients).any():
        raise RuntimeError("Coefficients contain NaN or Inf; aborting save. Check training data and 'iterations' value.")

    # compute 95% confidence intervals
    ci_lower = np.full(input_size, np.nan)
    ci_upper = np.full(input_size, np.nan)

    if ci_method == "bootstrap":
        # bootstrap refit of chosen model (works for Lasso and Ridge)
        coef_samples = np.zeros((n_bootstrap, input_size), dtype=float)
        rng = np.random.RandomState(0)
        for b in range(n_bootstrap):
            idx = rng.randint(0, X_train.shape[0], X_train.shape[0])
            Xb = X_train[idx]
            yb = y_train[idx]
            if regression_type == 1:
                m = Lasso(alpha=ALPHA, max_iter=5000)
            else:
                m = Ridge(alpha=ALPHA)
            try:
                m.fit(Xb, yb)
                coef_samples[b, :] = m.coef_
            except Exception:
                coef_samples[b, :] = np.nan
        # percentiles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ci_lower = np.nanpercentile(coef_samples, 2.5, axis=0)
            ci_upper = np.nanpercentile(coef_samples, 97.5, axis=0)
    else:
        # default: use OLS conf_int from statsmodels (fast, provides analytic CI for unregularized OLS)
        try:
            import statsmodels.api as sm
            X_ = sm.add_constant(X_train)
            ols = sm.OLS(y_train, X_).fit()
            conf = ols.conf_int(alpha=0.05)  # shape (p+1, 2) including intercept
            if conf.shape[0] - 1 != input_size:
                # unexpected shape; fall back to NaNs but continue
                print("Warning: statsmodels returned unexpected conf shape, skipping OLS CIs.")
            else:
                conf_no_intercept = conf[1:, :]  # drop intercept row
                ci_lower = conf_no_intercept[:, 0]
                ci_upper = conf_no_intercept[:, 1]
        except Exception as e:
            print("Statsmodels OLS CI failed:", e)
            # leave ci arrays as NaN

    #save coefficients to a csv file for use in testing
    if save == 0:
        #join coefficients with column names
        column_names = df.drop(columns=['SBP_at_MRI']).columns
        #create a DataFrame with coefficients and column names
        coef_df = pd.DataFrame({
            'Column': column_names,
            'Coefficient': coefficients,
            'CI 0.025': ci_lower,
            'CI 0.975': ci_upper
        })

        #save coefficients to a csv file in the order they appear in the dataset
        #seprate coefficient for mode r and mode f regressions (different input data)
        coefficients_path = coefficients_folder / (f"coefficients_{datamode}.csv")
        coef_df.to_csv(coefficients_path, index=False)

        sorted_coefficients_path = coefficients_folder / (f"sorted_coefficients_{datamode}.csv")

        #sort by coefficient absolute value (most impactful features first)
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        sorted_coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
        sorted_coef_df.to_csv(sorted_coefficients_path, index=False)

    #display results
    print("Average Root mean squared error: " + str(average_rmse))

    return 0
    #implementation of a unit test on a patient remains to be done

if __name__ == "__main__":
    status = 0
    if len(sys.argv) != 5:
        print("Usage: python lin_reg_vn.py type save datamode iterations  ")
        print("Regression types: (1 or 2).")
        print("Save modes: 0 to save coefficients, 1 to not save coefficients.")
        
        print("Preprocess datamode: 'r' for reduced dataset, 'f' for full dataset.")
        print("Iterations: number of iterations to run the regression for. Default is 100.")
        
        status = 1
    regression_type = int(sys.argv[1])
    if regression_type!= 1 and regression_type != 2:
        print("The integer passed does not correspond to a mode that has been implemented")
        status = 1
    save = int(sys.argv[2])
    datamode = sys.argv[3] if len(sys.argv) > 3 else "r"

    iterations = int(sys.argv[4])

    if status == 0:
        # example: use OLS CIs (fast). For bootstrap set ci_method="bootstrap" and adjust n_bootstrap.
        linear_regression(regression_type, save, iterations, datamode=datamode, ci_method="ols")