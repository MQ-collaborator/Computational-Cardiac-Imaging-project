#program to train PCA on all embeddings then compare results for each dataset and phenotype
from utils import home_directory, dl_columns_path, get_train_phenotypes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import sys
from sklearn.decomposition import PCA
from encode_space import latent_embeddings_path, load_embeddings
import numpy as np

pca_image_folder = r"./embeddings_pca_images"

def main():
    with open(dl_columns_path, 'r') as f:
        #run PCA iteratively with 
        columns = f.read().splitlines()
def run_pca( save_vectors = True):
    #load latent embeddings from npz file
    train_embeddings, validation_embeddings, test_embeddings, train_labels, val_labels, test_labels = load_embeddings()
    

    #load phenotypes

    #when running quickly only keep 2 dimensions to generate visuals. No real need to keep more than 2 dimensions for visualisation purposes
    #pca = PCA(n_components=2)

    pca = PCA()

    principalComponents = pca.fit_transform(train_embeddings)

    principal_df = pd.DataFrame(data=principalComponents[:, :2], columns=['PC1', 'PC2'])
    

    
    
    #Find the variance explained by each principal component
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each principal component: {explained_variance}")
    print(train_embeddings.shape)
    #return the values of PC1 and PC2 for each data point
    return principal_df['PC1'], principal_df['PC2'], train_labels

    #Currently these is no clustering in the PCA. TO improve we need to remove useless data





def pca_overlay(mapvariable = "CM", save_image = False, show = True):
    #Overlay PCA results with a heatmap of a given phenotype to ascribe meaning to clusters in PCA
    
    #get overlay variable data from data_frame
    overlay_values = get_train_phenotypes(mapvariable)

    PC1, PC2, train_labels = run_pca( save_vectors=False)
    #create scaled version of colour map
    plt.figure(figsize=(10, 8))
    plt.scatter(PC1, PC2, c=overlay_values, cmap='viridis', alpha=0.5)
    plt.colorbar(label=mapvariable)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of phenotype] Data')
    if save_image:
        plt.savefig(f"{pca_image_folder}/embedding_pca_" +mapvariable+ ".png")
        print(f"Saved to {pca_image_folder}/embedding_pca_" +mapvariable+ ".png")
    if show:
        plt.show()

#usage python [script] [mapvariable] 
if __name__ == "__main__":

    #take the variable to overlay as a command line argment. If not specified only show pca with SBP_at_MRI
    if len(sys.argv) > 1:
        mapvariable = sys.argv[1]
    else:
        mapvariable = "SBP_at_MRI"
    
    pca_overlay(mapvariable, save_image=True, show=True)