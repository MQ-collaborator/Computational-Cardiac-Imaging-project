import conventional_utils as conventional_utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import sys
from conventional_utils import home_directory
import numpy as np
from pathlib import Path

# create file paths to PCA image folder and a folder to save PCA vectors relative to home directory
pca_vectors_path = home_directory / "data" / "PCA" / "pca_vectors.csv"
pca_image_folder = home_directory / "data" / "PCA" / "pca_images"
pca_image_folder.mkdir(parents=True, exist_ok=True)


def run_pca(save_vectors=True, df=None, datamode="f"):
    """
    Run PCA on the feature matrix constructed directly from the preprocess dataframe.
    Important: build X from df in the same row order so PCA scores keep the same index.
    Returns principal_df (PC1/PC2) indexed by df.index, the dataframe used, and pca_vectors_df.
    """
    if df is None:
        df = conventional_utils.preprocess(datamode=datamode)

    # build X directly from dataframe (preserves index & order)
    if "SBP_at_MRI" not in df.columns:
        raise RuntimeError("Expected 'SBP_at_MRI' in preprocess dataframe.")
    X = df.drop(columns=["SBP_at_MRI"]).to_numpy()

    pca = PCA()
    principalComponents = pca.fit_transform(X)

    # Keep original dataframe index so overlays align exactly
    principal_df = pd.DataFrame(data=principalComponents[:, :2],
                                columns=['PC1', 'PC2'],
                                index=df.index)

    # create dataframe of feature names excluding SBP to save pca vectors
    feature_names = df.columns.drop("SBP_at_MRI")
    pca_vectors_df = pd.DataFrame(pca.components_, columns=feature_names)
    if save_vectors:
        pca_vectors_df.to_csv(pca_vectors_path, index=False)
        print(f"PCA vectors saved to {pca_vectors_path}")

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each principal component: {explained_variance}")

    return principal_df, df, pca_vectors_df


def pca_overlay(mapvariable="SBP_at_MRI", save_image=True, show=True, datamode="f"):
    """
    Overlay PCA PC1/PC2 with a phenotype column taken directly from the same dataframe used
    to build the PCA so values are aligned row-for-row. Categorical overlays use discrete colors.
    """
    principal_df, df, _ = run_pca(save_vectors=True, df=None, datamode=datamode)

    if mapvariable not in df.columns:
        raise ValueError(f"Variable '{mapvariable}' not found in dataframe columns. Available: {list(df.columns)}")

    # take overlay values from the SAME dataframe and align by index
    overlay_values = df[mapvariable].loc[principal_df.index]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8))

    # choose continuous vs categorical plotting and make points transparent
    if pd.api.types.is_numeric_dtype(overlay_values) and overlay_values.nunique() > 10:
        sc = ax.scatter(principal_df['PC1'], principal_df['PC2'],
                        c=overlay_values, cmap='viridis', alpha=0.35, s=30, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(mapvariable)
    else:
        cats = pd.Categorical(overlay_values)
        categories = cats.categories
        palette = plt.get_cmap('tab10')
        color_map = {cat: palette(i % 10) for i, cat in enumerate(categories)}
        for cat in categories:
            mask = cats == cat
            ax.scatter(principal_df['PC1'][mask], principal_df['PC2'][mask],
                       label=str(cat), color=color_map[cat], alpha=0.45, s=40, edgecolors='none')
        ax.legend(title=mapvariable, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(f'PCA colored by {mapvariable}')
    plt.tight_layout()

    if save_image:
        out = pca_image_folder / f"pca_scatter_plot_{mapvariable}.png"
        plt.savefig(out)
        print(f"Saved PCA overlay to {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mapvariable = sys.argv[1]
    else:
        mapvariable = "SBP_at_MRI"
    print(f"Overlaying PCA with {mapvariable}")
    pca_overlay(mapvariable, save_image=True, show=True)