#program to train PCA on all embeddings then compare results for each dataset and phenotype
import dl_utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
#create file paths within the PCA folder to save data

pca_vectors_path = r"P:\\MQ_Summer_Student\\PCA\\pca_vectors.csv"
pca_image_folder = r"P:\\MQ_Summer_Student\\PCA\\pca_images"







def pca_overlay(mapvariable = "CM", save_image = True, show = True):
    #Overlay PCA results with a heatmap of a given phenotype to ascribe meaning to clusters in PCA
    
    #get overlay variable data from data_frame
    df = dl_utils.preprocess
    PC1, PC2 = run_pca( save_vectors=False)
    #create scaled version of colour map
    plt.figure(figsize=(10, 8))
    plt.scatter(PC1, PC2, c=df[mapvariable], cmap='viridis', alpha=0.5)
    plt.colorbar(label=mapvariable)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of phenotype] Data')
    if save_image:
        plt.savefig(f"{pca_image_folder}/latent_pca_plot_" +mapvariable+ ".png")
        print(f"{pca_image_folder}/laente_pca_plot_" +mapvariable+ ".png")
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