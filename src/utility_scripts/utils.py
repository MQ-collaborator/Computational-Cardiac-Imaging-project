#program with tools which will be needed frequently over studentship for analysing data 
#use pandas to import and manage dataset
import pandas as pd

#perform requisite math operations with numpy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""All relevant file paths are here and easy to alter"""

#UK Biobank file cotaining raw data
data_path= r"\\isd_netapp\Cardiac$\UKBB_40616\Phenotypes\phenotypes_48k_99CMR_13other.csv"    

#file to store column titles for later use
columns_path = r"P:\\MQ_Summer_Student\\dataset_info\\column_names.csv"


average_path = r"P:\\MQ_Summer_Student\\dataset_info\\averages.csv"

#file to store relevant information regarding missing data
missing_path = r"P:\\MQ_Summer_Student\\dataset_info\\missing_values_report.csv"


#import tabular data from cardiac$
def preprocess(depth = 1, datamode = "f"):
    """ Datamode determinses the size of input data returned and the type of data returned:
    Mode r is used to produce reduced low dimensional dataset for regression
    Mode f returns the full dataset excluding columns with no predictive power"""
    #imputation is done by mean value
    """Depth parameter determines if statistics on dataset are collected:
     depth = 0: collect statistics on dataset, save column names, averages and missing values
     depth = 1: preprocess data for regression or learning, no statistics collected"""""

      

    df = pd.read_csv(data_path)

    #remove data with no predictive power for blood pressure or administrative labels

    columns_to_drop = ['eid_40616', 'batch',

                'Ell_1','Ell_2','Ell_3','Ell_4','Ell_5','Ell_6', 

                'Ecc_AHA_1','Ecc_AHA_2','Ecc_AHA_3','Ecc_AHA_4','Ecc_AHA_5','Ecc_AHA_6','Ecc_AHA_7','Ecc_AHA_8','Ecc_AHA_9','Ecc_AHA_10','Ecc_AHA_11','Ecc_AHA_12','Ecc_AHA_13','Ecc_AHA_14','Ecc_AHA_15','Ecc_AHA_16',

                'Err_AHA_1','Err_AHA_2','Err_AHA_3','Err_AHA_4','Err_AHA_5','Err_AHA_6','Err_AHA_7','Err_AHA_8','Err_AHA_9','Err_AHA_10','Err_AHA_11','Err_AHA_12','Err_AHA_13','Err_AHA_14','Err_AHA_15','Err_AHA_16',

                'WT_AHA_1','WT_AHA_2','WT_AHA_3','WT_AHA_4','WT_AHA_5','WT_AHA_6','WT_AHA_7','WT_AHA_8','WT_AHA_9','WT_AHA_10','WT_AHA_11','WT_AHA_12','WT_AHA_13','WT_AHA_14','WT_AHA_15','WT_AHA_16',
                                 
                'WT_Max_AHA_1','WT_Max_AHA_2','WT_Max_AHA_3','WT_Max_AHA_4','WT_Max_AHA_5','WT_Max_AHA_6','WT_Max_AHA_7','WT_Max_AHA_8','WT_Max_AHA_9','WT_Max_AHA_10','WT_Max_AHA_11','WT_Max_AHA_12','WT_Max_AHA_13','WT_Max_AHA_14','WT_Max_AHA_15','WT_Max_AHA_16',  
                
                'has_ICD10',
                'DBP_at_MRI','MAP_at_MRI',
                
                'Ethnic_background']
    df = df.drop(columns = columns_to_drop,axis =1 )

    #remove patients whose cardiomyopathy status is unknown - otherwise our understanding of the effect of CM on SBP may be compromised

    df.dropna(subset=['CM'], inplace = True)
    
    #remove patients who lack systloic blood pressure data - this is what we are trying to predict
    df.dropna(subset=['SBP_at_MRI'], inplace = True)


    #Cleanse duplicated
    df.drop_duplicates()


    #convert true false data into binary
    df = df.replace({'True': 1, 'False': 0})
    
    
    
    #collate information about reduced dataset in a file
    if depth ==0:
        print("Saving dataset information to CSV files ...")
        #store the name of each colun and its average value in a csv file
        averages  = df.mean()
        averages.drop("SBP_at_MRI", inplace=True)  # Exclude target variable from averages
        averages.to_csv(average_path, header=False)


        #record prevalence of missing data in a csv file
        missing_counts = df.isna().sum()
        missing_df = missing_counts.reset_index()
        missing_df.columns = ['Column', 'MissingValues']
        missing_df.to_csv(missing_path, index = False)

        print("Information saved to csv")
    

    #remove data not suitable for regression, add interaction terms and include only select number of variables for regression
    if datamode == "r":
        df = df[["SBP_at_MRI", "age_at_MRI", "Sex", "DAo_min_area", "Height", "Weight_at_MRI", "LVEDV"]]
        df["age*sex"] = df["age_at_MRI"] * df["Sex"]
        df["age^2"] = df["age_at_MRI"] ** 2
    #else if datamode is f, return full dataset


    """Approach to data imputation taken:
        remove patients who have criticial information missing (mean imputation would simply make model less responsive to outliers)
         
        use mean imputation where less important information is missing """
    

    

    #remaining data is filled in using median imputation. This is likely to change in future iterations
    df.fillna(df.median(), inplace=True)
    return df



"""Function to split data into training and testing sets in different ways. In addition data is split into inputs(all other phenotype data) and outputsSBP_at_MRI generally.
Function returns a tuple of training and testing dataframes. Additioanlly data is normalised"""
def split_and_normalize(df, test_size=0.2, random_state=None, mode = 0):
    """Modes allow for various types of data splitting defined within one function:
    mode 0:  A/B testing, separate normalisations
    mode 1: Split data and normalize, no subdivision (PCA, visuals, not model training)"""

    ""
    # Split the data into features and target variable
    X = df.drop(columns=['SBP_at_MRI'])
    Y = df['SBP_at_MRI']
    if mode == 0:
        

        # Split the data into training and testing sets
        #with an option to set the random state for reproducibility.
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)


        #scale training data and testing data separately
        scaler = StandardScaler()
        #only scale X datasets

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        return X_train, X_test, Y_train, Y_test
    
    elif mode ==1:
        scaler = StandardScaler()
        # Reshape X to 2D before scaling (in case X is a Series)
        if isinstance(X, pd.Series):
            X_standardised = scaler.fit_transform(X.values.reshape(-1, 1))
        else:
            X_standardised = scaler.fit_transform(X)

        # Reshape Y to 2D before scaling
        Y_standardised = scaler.fit_transform(Y.values.reshape(-1, 1))
        return X_standardised, Y_standardised




if __name__ == "__main__":

    df = preprocess( depth = 0)


    print("Head of dataframe")
    print(df.head)