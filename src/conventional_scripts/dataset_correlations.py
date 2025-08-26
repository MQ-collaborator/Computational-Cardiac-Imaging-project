from conventional_utils import home_directory
import conventional_utils
import matplotlib.pyplot as plt

#File paths to store files of note
correlation_path = home_directory / "data" / "dataset_info" / "correlations.csv"

correlation_heatmap_path = home_directory / "data" / "analysis_images" / "correlation_heatmap.png"

def correlation_map(save = True):
    print("Generating Correlations...")
    df = conventional_utils.preprocess(datamode = "f")
    #remove clutter (phenotypes explained already by other datapoints)

    #Calculate correlation coefficients (pearson by default)
    corr = df.corr()
    plt.figure(figsize=(16, 14))
    plt.imshow(corr, cmap='coolwarm')

    # Create a heatmap of correlation matrix
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Map of Tabular Data') 


        #save heatmap if save mode is on
    if save:

        corr.to_csv(correlation_path)
        plt.savefig(fname = correlation_heatmap_path)

    plt.show()






if __name__ == "__main__":
    correlation_map()