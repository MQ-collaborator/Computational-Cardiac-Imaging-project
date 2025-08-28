from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from latent_regression import latent_regression_coefficients_path
import utils
from utils import helper_directory
""""Extract the importance of each input feature on:
SBP predictions made my linear regression model on latent space.
Combine impact of input feature on latent space and effect of latent space on SBP (Beta coefficient, chain rule)"""

feature_importance_path = helper_directory / "feature_importance.csv"

def main():
    #extract beta coefficients from linear regression model
    regression_coefficients_df = pd.read_csv(latent_regression_coefficients_path)
    regression_coefficients = regression_coefficients_df['Coefficient'].values

    #correlate latent features with input features to get effect of input features on latent space
    #load unencoded data
    train_loader , val_loader, test_loader = utils.dataloader(utils.preprocess())
    all_data = []
    #put all inputs into a single numpy array
    for X_batch, _ in train_loader:
        all_data.append(X_batch.numpy())
    for X_batch, _ in val_loader:
        all_data.append(X_batch.numpy())
    for X_batch, _ in test_loader:
        all_data.append(X_batch.numpy())
    all_data = np.vstack(all_data)
    print("All data shape:", all_data.shape)

    #load latent embeddings and put them in a single list (in a similar fashion)
    latent_embeddings = np.load("./helper_data/embeddings.npz")
    all_latent_embeddings = np.vstack([latent_embeddings['train_embeddings'], latent_embeddings['val_embeddings'], latent_embeddings['test_embeddings']])
    print("All latent embeddings shape:", all_latent_embeddings.shape)
    input_size = all_data.shape[1]
    latent_size = all_latent_embeddings.shape[1]
    feature_importance = np.zeros((latent_size, input_size))
    for i in range(latent_size):
        for j in range(input_size):
            #find correlation between each input feature and each latent feature
            corr = np.corrcoef(all_latent_embeddings[:, i], all_data[:, j])[0, 1]
            feature_importance[i, j] = corr
    print("Feature importance shape:", feature_importance.shape)
    #combine the two effects using chain rule
    combined_importance = np.zeros((input_size,))
    for j in range(input_size):
        for i in range(latent_size):
            combined_importance[j] += feature_importance[i, j] * regression_coefficients[i]
    print("Combined importance shape:", combined_importance.shape)
    #get feature names
    df = utils.preprocess()
    feature_names = df.drop(columns=['SBP_at_MRI']).columns
    #save to csv
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': combined_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', key=abs, ascending=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print("Feature importance saved to ./helper_data/feature_importance.csv")



    #create a forest plot of feature importance
    font=16
    for _, row in feature_importance_df.iterrows():
        plt.errorbar(x=row['Importance'], y=row['Feature'],
                                         #xerr=[[row['metabolite beta'] - row['CI 0.025']], [row['CI 0.975'] - row['metabolite beta']]],
                                         fmt='o', markersize=7, label=row['Importance'], color='tab:blue', elinewidth=3)
        #errors = f'{np.round(row["metabolite beta"], 2)} ({np.round(row["CI 0.025"], 3)}, {np.round(row["CI 0.975"], 3)})'
                # plt.text(row['metabolite beta'] * 0 + 0.81, row['metabolite'],
                #                  errors,
                #                  ha='left', va='center', size=14, style='normal', color='black')
        plt.axvline(x=0, color='black', linestyle='dashed', alpha=0.6)
        plt.xlabel('Effect on SBP', fontsize=font)
        plt.ylabel('Feature', fontsize=font)
        plt.title('Feature Importance in deep learning prediction model', fontsize=font)
        # plt.grid(which='major', color='#EBEBEB', linewidth=0.8)
        plt.xlim(-0.07, 0.07)
        plt.xticks([-0.1, 0, 0.1], fontsize=font)
        plt.yticks(fontsize=font)
        plt.gca().set_axisbelow(True)
        plt.gca().margins(y=0.03)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        ax = plt.gca()
        ax.tick_params(length = 0)
        

        plt.tight_layout()
    plt.show()

    return 0



main()