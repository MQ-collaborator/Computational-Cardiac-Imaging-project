#program with tools which will be needed frequently over studentship for analysing data 

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #ideal scaling for ReLU
from pathlib import Path
from joblib import dump, load
imputed_data_path = r"\\isd_netapp\Cardiac$\UKBB_40616\Phenotypes\img_features_all_rf_imputed_48k_with_biochem.csv"

#configure file paths to be relative to current file
home_directory = Path(__file__).parent.parent.parent # outermost directory
dataset_info_directory = home_directory / "data" / "dataset_info"

#file to store column titles for later use
dl_columns_path = dataset_info_directory / "dl_column_names.csv"

#store scalers and indices for reuse
helper_directory = r"./helper_data"

class features_labels_Dataset(Dataset):
    def __init__(self,X,y,dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype).view(-1,1)

        #sanity check for implementation
        assert len(self.X) == len(self.y) #check lengths match
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


#import tabular data from cardiac$
#simplified program
def preprocess(datamode = "r"):

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

    #remove data not suitable for regression, add interaction terms and include only select number of variables for regression
    if datamode == "r":
        df = df[["SBP_at_MRI", "age_at_MRI", "Sex", "DAo_min_area", "Height", "Weight_at_MRI", "LVEDVi"]]
        df["age*sex"] = df["age_at_MRI"] * df["Sex"]
        df["age^2"] = df["age_at_MRI"] ** 2
    #else if datamode is f, return full dataset
    
    #remove patients who lack systloic blood pressure data - this is what we are trying to predict
    df.dropna(subset=['SBP_at_MRI'], inplace = True)

    return df

def dataloader(df, test_size = 0.2, val_size=0.2, random_state = None, generate_indices = False, new_scaler = False):
    #split data into features and labels
    Y = df['SBP_at_MRI']
    X = df.drop(columns=['SBP_at_MRI'])

    #Normalize all data using scalers
    if new_scaler:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X = x_scaler.fit_transform(X)
        Y = y_scaler.fit_transform(Y.values.reshape(-1, 1)).flatten()
        dump(x_scaler, f"{helper_directory}/x_scaler.joblib")
        dump(y_scaler, f"{helper_directory}/y_scaler.joblib")

        #store scalars for later inference
    else:
        x_scaler = load(f"{helper_directory}/x_scaler.joblib")
        y_scaler = load(f"{helper_directory}/y_scaler.joblib")
        X = x_scaler.transform(X)
        Y = y_scaler.transform(Y.values.reshape(-1, 1)).flatten()

    fullDataset = features_labels_Dataset(X,Y)

    if generate_indices:
        print("Generating indices...")
        #index dataset by creating an np array of indices
        indices = np.arange(len(fullDataset))
        #initial split into train and (test+validation)
        train_idx, temp_idx = train_test_split(indices, test_size = (test_size+val_size), random_state = random_state)

        #split temp into validation and testing
        val_idx, test_idx = train_test_split(temp_idx, test_size = (test_size/(test_size + val_size)))

        #save indices that have been generated as npz file
        np.savez("./helper_data/split_indices.npz", train = train_idx, val = val_idx, test = test_idx)

    else:
        #load existing indixes
        split_indices = np.load(f"{helper_directory}/split_indices.npz")
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]


    #create subsets using indices
    train_ds = Subset(fullDataset, train_idx)
    val_ds = Subset(fullDataset, val_idx)
    test_ds = Subset(fullDataset, test_idx)
    print("The sizes of datasets are as follows:")
    print(f"Training: {len(train_idx)}")
    print(f"Testing: {len(test_idx)}")
    print(f"Validation {len(val_idx)}")

    #Dataloaders
    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle= True, num_workers = 0, drop_last = False)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_ds, batch_size = batch_size, num_workers = 0)

    return train_loader , val_loader, test_loader

def list_data(df):
    #load data
    train_loader , val_loader, test_loader = dataloader(df, new_scaler=True, generate_indices=True)

    #get shape of inputs by iterating once through a DataLoader
    for X_batch, _ in train_loader:
        print("Input batch shape", X_batch.shape)
        input_size = X_batch.shape[1]
        break

    # ---- new: collect full training arrays from the DataLoader ----
    X_train_parts = []
    y_train_parts = []
    for Xb, yb in train_loader:
        # ensure tensors moved to cpu and converted to numpy
        if hasattr(Xb, "detach"):
            Xb = Xb.detach().cpu().numpy()
        if hasattr(yb, "detach"):
            yb = yb.detach().cpu().numpy()
        X_train_parts.append(Xb)
        y_train_parts.append(yb)

    for Xb, yb in val_loader:
        if hasattr(Xb, "detach"):
            Xb = Xb.detach().cpu().numpy()
        if hasattr(yb, "detach"):
            yb = yb.detach().cpu().numpy()
        X_train_parts.append(Xb)
        y_train_parts.append(yb)

    for Xb, yb in test_loader:
        if hasattr(Xb, "detach"):
            Xb = Xb.detach().cpu().numpy()
        if hasattr(yb, "detach"):
            yb = yb.detach().cpu().numpy()
        X_train_parts.append(Xb)
        y_train_parts.append(yb)

    if len(X_train_parts) == 0:
        raise ValueError("Training DataLoader is empty")

    all_x_array= np.vstack(X_train_parts)
    all_y_array = np.concatenate(y_train_parts).ravel()

    return all_x_array, all_y_array, input_size

def get_train_phenotypes(variable_name):
    df = preprocess(datamode="f")
    split_indices = np.load("./helper_data/split_indices.npz")
    train_idx = split_indices["train"]

    if variable_name not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Variable '{variable_name}' not found in phenotype data.")
    return df.iloc[train_idx][variable_name].values

if __name__ == "__main__":
    df = preprocess()
    dataloader(df, generate_indices = True, new_scaler = True)
    #save column names to a file to check which to remove
    column_names = df.drop(columns=['SBP_at_MRI']).columns
    column_names_df = pd.DataFrame(column_names, columns=['Column Names'])
    column_names_df.to_csv(dl_columns_path, index=False)
    print("Column names saved to:", dl_columns_path)

    print("Data loading complete")