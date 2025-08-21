import conventional_scripts.conventional_utils as conventional_utils
import sys
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from deep_learning.utils import columns_path

#Implementation of linear regression with interaction terms
#Iterative process to average coefficients over multiple iterations
#This is holdout CV being run iteratively with random train test split. This is less rigorous than k-fold cross validation but allows for faster results
#Techincally it may be better to use k-fold to save computation time but mathematically speaking this is equivalent

""""List of variables affecting style of training"""
#alpha determines the degree of regularization (must be positive)
ALPHA = 0.2

"""File paths"""
coefficients_path = r"P:\\MQ_Summer_Student\\regression_results\\averaged_regression_coefficients.csv"

sorted_coefficients_path = r"P:\\MQ_Summer_Student\\regression_results\\average_sorted_regression_coefficients.csv"


def linear_regression(regression_type, save, iterations, datamode, interaction_mode = 0):
    print("Running linear regression iteratively)")

    
    #load data

    df = conventional_utils.preprocess(depth = 1, datamode = datamode)

    #add interaction terms to the dataset if specified
    if interaction_mode == 0:
        #add interaction terms to the dataset
        df['age*sex'] = df['age_at_MRI'] * df['Sex']

        df['age^2'] = df['age_at_MRI']**2
    #for mode 1 no changes needed - interaction terms are not added
    elif interaction_mode == 2:
        #only interaction terms
        df = df[["SBP_at_MRI", "age_at_MRI", "Sex"]].copy()
        df['age*sex'] = df['age_at_MRI'] * df['Sex']

        df['age^2'] = df['age_at_MRI']**2


    X_train, X_test, Y_train, Y_test = conventional_utils.split_and_normalize(df)

    
        
    #Iterative regression. Average values stored in coefficients_path
    coefficients = np.zeros(X_train.shape[1])
    average_rmse = 0
    
    if regression_type == 1:
        for i in range(iterations):
            lasso = Lasso(alpha=ALPHA)
            lasso.fit(X_train, Y_train)

            #store y values predicted by regression
            y_pred = lasso.predict(X_test)
            #print root mean squared error(an easy to understand measure of error)
            rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

            coefficients += lasso.coef_
            average_rmse += rmse
            print(i)
    else:
        for i in range(iterations):
            ridge = Ridge(alpha=ALPHA)
            ridge.fit(X_train, Y_train)
            y_pred = ridge.predict(X_test)
     
            rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
            coefficients += ridge.coef_
            average_rmse += rmse
            print(i)
    coefficients /= iterations
    average_rmse /= iterations
    #save coefficients to a csv file for use in testing
    if save == 0:


        #join coefficients with column names
        column_names = df.drop(columns=['SBP_at_MRI']).columns
        #create a DataFrame with coefficients and column names
        coef_df = pd.DataFrame({
            'Column': column_names,
            'Coefficient': coefficients
        })

        #save coefficients to a csv file in the order they appear in the dataset
        coef_df.to_csv(coefficients_path, index=False)

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
    if len(sys.argv) != 6:
        print("Usage: python lin_reg_vn.py type save datamode iterations interaction_mode ")
        print("Regression types: (1 or 2).")
        print("Save modes: 0 to save coefficients, 1 to not save coefficients.")
        
        print("Preprocess datamode: 'r' for reduced dataset, 'f' for full dataset.")
        print("Iterations: number of iterations to run the regression for. Default is 100.")
        print("Interaction mode: 0 for dataset and interaction, 1 for dataset without interaction, 2 for only interaction)")
        status = 1
    regression_type = int(sys.argv[1])
    if regression_type!= 1 and regression_type != 2:
        print("The integer passed does not correspond to a mode that has been implemented")
        status = 1
    save = int(sys.argv[2])
    datamode = sys.argv[3] if len(sys.argv) > 3 else "r"

    iterations = int(sys.argv[4])

    interaction_mode = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    if status == 0:
        linear_regression(regression_type, save, iterations, datamode=datamode, interaction_mode = interaction_mode)