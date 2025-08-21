import  utils
import sys
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from deep_learning.utils import columns_path


""""List of variables affecting style of training"""
#alpha determines the degree of regularization (must be positive)
ALPHA = 0

"""File paths"""
coefficients_path = r"P:\\MQ_Summer_Student\\regression_results\\regression_coefficients.csv"

sorted_coefficients_path = r"P:\\MQ_Summer_Student\\regression_results\\sorted_regression_coefficients.csv"


def main():
    if len(sys.argv) != 3:
        print("Usage: python lin_reg.py mode save")
        print("Mode is the type of regression (1 or 2).")
        print("Save modes: 0 to save coefficients, 1 to not save coefficients.")
        return 1

    mode = int(sys.argv[1])
    save = int(sys.argv[2])
    if mode!= 1 and mode != 2:
        print("The integer passed does not correspond to a mode that has been implemented")
        return 1
    #load data

    df = utils.preprocess(depth = 1)
    X_train, X_test, Y_train, Y_test = utils.split_and_normalize(df)
    #implementation of L1 regression
    if mode == 1:
        lasso = Lasso(alpha=ALPHA)
        lasso.fit(X_train, Y_train)

        #store y values predicted by regression
        y_pred = lasso.predict(X_test)
        #print root mean squared error(an easy to understand measure of error)
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
        #TODO
    
    #implementation of L2 regression
    else:
        ridge = Ridge(alpha=ALPHA)
        ridge.fit(X_train, Y_train)
        y_pred = ridge.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

    #save coefficients to a csv file for use in testing
    if save == 0:
        

        # For L1 regression
        if mode == 1:
            coef = lasso.coef_
        # For L2 regression
        else:
            coef = ridge.coef_
        pd.Series(coef).to_csv(coefficients_path, index=False, header=False)

        #join coefficients with column names
        column_names = pd.read_csv(columns_path, header=None)
        coef_df = pd.DataFrame({
            'Column': column_names[0],
            'Coefficient': coef
        })

        #save coefficients to a csv file in the order they appear in the dataset
        coef_df.to_csv(coefficients_path, index=False)

        #sort by coefficient absolute value (most impactful features first)
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        sorted_coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
        sorted_coef_df.to_csv(sorted_coefficients_path, index=False)

        
    #display results
    print("Root mean squared error: " + str(rmse))
    return 0
    #implementation of a unit test on a patient remains to be done

if __name__ == "__main__":
    main()