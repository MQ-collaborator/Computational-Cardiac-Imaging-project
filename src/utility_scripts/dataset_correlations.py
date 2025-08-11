import utils
import matplotlib.pyplot as plt

#File paths to store files of note
correlation_path = r"P:\\MQ_Summer_Student\\dataset_info\\correlations.csv"

correlation_heatmap_path = r"P:\\MQ_Summer_Student\\analysis_images\\correlation_heatmap.png"

def correlation_map(save = True):
    print("Generating Correlations...")
    df = utils.preprocess()

    #Calculate correlation coefficients (pearson by default)
    corr = df.corr()
    plt.figure(figsize=(16, 14))
    plt.imshow(corr, cmap='coolwarm')

    # Create a heatmap of correlation matrix
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Map of Tabula Data') 


        #save heatmap if save mode is on
    if save:

        corr.to_csv(correlation_path)
        plt.savefig(fname = correlation_heatmap_path)

    plt.show()






if __name__ == "__main__":
    correlation_map()