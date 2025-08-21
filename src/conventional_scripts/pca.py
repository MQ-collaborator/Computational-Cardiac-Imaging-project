import  conventional_utils as conventional_utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import sys
from conventional_utils import home_directory, get_train_phenotypes
import numpy as np

#create file paths to PCA image folder and a folder to save PCA vectors relative to home directory
pca_vectors_path = home_directory / "data" / "PCA" / "pca_vectors.csv"
pca_image_folder = home_directory / "data" / "PCA"/ "pca_images"

def run_pca( save_vectors = True):
    #SBP is excluded from PCA because we are trying to find the underlying structure of the data (namely the number of variables needed to predict SBP)
    #load data
    df = conventional_utils.preprocess(datamode="f")
    X_train, y_train, input_size = conventional_utils.list_data(df)

    #when running quickly on keep 2 dimensions to generate visuals. No real need to keep more than 2 dimensions for visualisation purposes
    #pca = PCA(n_components=2)

    pca = PCA()

    principalComponents = pca.fit_transform(X_train)

    principal_df = pd.DataFrame(data=principalComponents[:, :2], columns=['PC1', 'PC2'])
    

    #create a dataframe of the df column names excluding SBP to later concatenate with pca vectors
    columns_df = pd.DataFrame(df.columns.drop("SBP_at_MRI"), columns=['Feature'])
    #create a dataframe of the pca vectors
    pca_vectors_df = pd.DataFrame(pca.components_, columns=columns_df['Feature'])
    if save_vectors:
        pca_vectors_df.to_csv(pca_vectors_path, index=False)
        print(f"PCA vectors saved to {pca_vectors_path}")
    #Find the variance explained by each principal component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each principal component: {explained_variance}")
    
    #return the values of PC1 and PC2 for each data point
    return principal_df['PC1'], principal_df['PC2']

    #Currently these is no clustering in the PCA. TO improve we need to remove useless data


def pca_overlay(mapvariable = "CM", save_image = True, show = True):
    #Overlay PCA results with a heatmap of a given phenotype to ascribe meaning to clusters in PCA
    
    #get overlay variable data from data_frame
    overlay_values = get_train_phenotypes(mapvariable)

    PC1, PC2 = run_pca( save_vectors=False)
    #create scaled version of colour map
    plt.figure(figsize=(10, 8))
    plt.scatter(PC1, PC2, c=overlay_values, cmap='viridis', alpha=0.5)
    plt.colorbar(label=mapvariable)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of phenotype] Data')
    if save_image:
        plt.savefig(f"{pca_image_folder}/pca_scatter_plot_" +mapvariable+ ".png")
        print(f"{pca_image_folder}/pca_scatter_plot_" +mapvariable+ ".png")
    if show:
        plt.show()

if __name__ == "__main__":

    #take the variable to overlay as a command line argment. If not specified only show pca with SBP_at_MRI
    if len(sys.argv) > 1:
        mapvariable = sys.argv[1]
    else:
        mapvariable = "SBP_at_MRI"
    pca_overlay(mapvariable, save_image=True, show=True)