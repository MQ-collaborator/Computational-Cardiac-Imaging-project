#program with tools which will be needed frequently over studentship for analysing data 

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

imputed_data_path = r"\\isd_netapp\Cardiac$\UKBB_40616\Phenotypes\img_features_all_rf_imputed_48k_with_biochem.csv"

#file to store column titles for later use
dl_columns_path = r"P:\\MQ_Summer_Student\\dataset_info\\dl_column_names.csv"

class features_labels_Dataset(Dataset):
    def __init__(self,X,y,dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype).view(-1,1)

    def __len__(self):
        return len(self.X)
    
    def _getitem_(self, idx):
        return self.X[idx], self.y[idx]
    


#import tabular data from cardiac$
#simplified program
def preprocess():

    df = pd.read_csv(imputed_data_path)

    #remove data with no predictive power for blood pressure or administrative labels
    #This set of colums is different and is unique to imputated dataset
    columns_to_drop = ['eid_40616', 'eid_47602' , 'eid_18545',

                'Ell_1','Ell_2','Ell_3','Ell_4','Ell_5','Ell_6', 

                'Ecc_AHA_1','Ecc_AHA_2','Ecc_AHA_3','Ecc_AHA_4','Ecc_AHA_5','Ecc_AHA_6','Ecc_AHA_7','Ecc_AHA_8','Ecc_AHA_9','Ecc_AHA_10','Ecc_AHA_11','Ecc_AHA_12','Ecc_AHA_13','Ecc_AHA_14','Ecc_AHA_15','Ecc_AHA_16',

                'Err_AHA_1','Err_AHA_2','Err_AHA_3','Err_AHA_4','Err_AHA_5','Err_AHA_6','Err_AHA_7','Err_AHA_8','Err_AHA_9','Err_AHA_10','Err_AHA_11','Err_AHA_12','Err_AHA_13','Err_AHA_14','Err_AHA_15','Err_AHA_16',

                'WT_AHA_1','WT_AHA_2','WT_AHA_3','WT_AHA_4','WT_AHA_5','WT_AHA_6','WT_AHA_7','WT_AHA_8','WT_AHA_9','WT_AHA_10','WT_AHA_11','WT_AHA_12','WT_AHA_13','WT_AHA_14','WT_AHA_15','WT_AHA_16',
                                 
                'WT_Max_AHA_1','WT_Max_AHA_2','WT_Max_AHA_3','WT_Max_AHA_4','WT_Max_AHA_5','WT_Max_AHA_6','WT_Max_AHA_7','WT_Max_AHA_8','WT_Max_AHA_9','WT_Max_AHA_10','WT_Max_AHA_11','WT_Max_AHA_12','WT_Max_AHA_13','WT_Max_AHA_14','WT_Max_AHA_15','WT_Max_AHA_16',  
                

                'DBP_at_MRI','MAP_at_MRI',
                
                'Ethnic_background']
    df = df.drop(columns = columns_to_drop,axis =1 )

 
    
    #remove patients who lack systloic blood pressure data - this is what we are trying to predict
    df.dropna(subset=['SBP_at_MRI'], inplace = True)

    return df

def dataloader(df, test_size = 0.3, random_state = None):
    #split data into features and labels
    Y = df['SBP_at_MRI']
    X = df.drop(columns=['SBP_at_MRI'])

    fullDataset = features_labels_Dataset(df)

    #index dataset by creating an np array of indices
    indices = np.arange(len(fullDataset))
    #initial split into test set and temp(training + validation)
    temp_idx, test_idx = train_test_split(indices, test_size = test_size, random_state = random_state)

    #split temp into training and validation
    train_idx, val_idx = train_test_split

def split_and_normalize(df, mode = 1, test_size=0.2, random_state=None,):
    #split data into features and labels
    Y = df['SBP_at_MRI']
    X = df.drop(columns=['SBP_at_MRI'])
    
    #normalize all data to be encoded
    if mode == 0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()

        return X, Y

    #mode 1 is used to train autoencoder. Returns a validation set, training set and testing set
    elif mode == 1:
        #split data into features and labels
        Y = df['SBP_at_MRI']
        X = df.drop(columns=['SBP_at_MRI'])

        #Normalize all data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Y = scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()

        #Separates (training + validation) and test sets
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        #separate training and validation sets
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=test_size, random_state=random_state)


        #return testing, training and validation sets
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
   


    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    df = preprocess()
    #save column names to a file to check which to remove
    column_names = df.drop(columns=['SBP_at_MRI']).columns
    column_names_df = pd.DataFrame(column_names, columns=['Column Names'])
    column_names_df.to_csv(dl_columns_path, index=False)
    print("Column names saved to:", dl_columns_path)
    dataloader(df)
    print("Data loading complete")