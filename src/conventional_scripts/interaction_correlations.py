import utils
import matplotlib.pyplot as plt

#File paths to store files of note
interaction_correlation_path = r"P:\\MQ_Summer_Student\\dataset_info\\interaction_correlations.csv"

interaction_correlation_heatmap_path = r"P:\\MQ_Summer_Student\\analysis_images\\interaction_correlation_heatmap.png"

def correlation_map(save = True):
    print("Generating Correlations...")
    df = utils.preprocess()

    #Calculate correlation coefficients (pearson by default)
    reduced_df = df[["SBP_at_MRI", "age_at_MRI", "Sex"]].copy()
    reduced_df['age*sex'] = df['age_at_MRI'] * df['Sex']
    reduced_df['age^2*sex'] = (df['age_at_MRI']**2) * df['Sex']
    reduced_df['age^2'] = df['age_at_MRI']**2
    print(reduced_df)

    corr = reduced_df.corr()
    plt.figure(figsize=(16, 14))
    plt.imshow(corr, cmap='coolwarm')

    # Create a heatmap of correlation matrix
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Map of Tabula Data') 


        #save heatmap if save mode is on
    if save:

        corr.to_csv(interaction_correlation_path)
        plt.savefig(fname = interaction_correlation_heatmap_path)

    plt.show()




if __name__ == "__main__":
    correlation_map()