import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial.distance import pdist, jensenshannon
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import seaborn as sns
import os
import pickle
import re
import contextlib
from typing import List
from itertools import combinations

# settings for plotting style
plot_style = "forest_and_sky_academic" if "forest_and_sky_academic" in plt.style.available else "default"
plt.rcParams['svg.fonttype'] = 'none'

# make a color palette for seaborn
color_hex_codes = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
theme_palette = sns.color_palette(color_hex_codes)

#plt.style.use('~/.matplotlib/stylelib/tme_research.mplstyle')

# 1. Load the clustered window data that has the averages of the windows
abm_window_data = pd.read_csv(
    os.path.join(
        "data",
        "abm_windows_clustered_with_state_label.csv"
    )
)

non_feature_columns = ['sim_id', 'start_time_step', 'end_time_step', 'window_index_in_sim', 'hierarchical_label']
feature_columns = [i.replace('avg_', '') for i in list(abm_window_data.drop(non_feature_columns, axis=1).columns)]

# 2. ABM Leiden clustering results: loading the distance and linkage matrix calculated from the Leiden cluster centroids
linkage_matrix = np.load(
    os.path.join(
        "data", 
        "abm_full_features_leiden_clusters_linkage_matrix.npy"
    )
)
distance_matrix = np.load(
    os.path.join(
        "data",
        "abm_full_features_leiden_clusters_distance_matrix.npy"
    )
)
leiden_cluster_means = pd.read_csv(
    os.path.join(
        "data", 
        "abm_leiden_cluster_means.csv"
    )
)

n_states = abm_window_data['hierarchical_label'].unique().max()
cluster_labels = sp.cluster.hierarchy.fcluster(linkage_matrix, n_states, criterion='maxclust')

def collapse_repeated_measures(df):
    """
    Collapse repeated-measure columns by averaging across time points.
    Columns must end with a numeric suffix such as _0, _1, _2, etc.
    """
    # Pattern: capture base name + numeric suffix
    pattern = re.compile(r"(.+)_([0-9]+)$")

    groups = {}  # base name -> list of columns

    for col in df.columns:
        match = pattern.match(col)
        if match:
            base = match.group(1)
            groups.setdefault(base, []).append(col)

    # Build output DataFrame
    collapsed = {}

    for base, cols in groups.items():
        collapsed[base] = df[cols].mean(axis=1)

    # Add all non-repeated-measure columns
    non_repeated = [c for c in df.columns if not pattern.match(c)]
    for c in non_repeated:
        collapsed[c] = df[c]

    return pd.DataFrame(collapsed)

def calculate_mean_feature(df, feature_name):
    # Select columns containing the substring
    grouped_columns = [col for col in df.columns if feature_name in col]

    # Calculate the mean of these columns for each row
    df[f'{feature_name}_avg'] = df[grouped_columns].mean(axis=1)

    return df

for feature in feature_columns:
    calculate_mean_feature(leiden_cluster_means, feature)

#leiden_cluster_centroids_averaged = leiden_cluster_means.filter(like='_avg')
leiden_cluster_centroids_averaged = collapse_repeated_measures(leiden_cluster_means)

# We need the optimal_hac_cluster_labels to color the rows
# These labels correspond to the rows of the original cluster_means DataFrame
# Let's create a color palette based on the number of optimal HAC clusters
def plot_hac_clustergram() -> None:
    """
    Plots a clustermap of Leiden cluster centroids, ordered by hierarchical clustering
    and colored by optimal HAC clusters.

    The clustermap visualizes the scaled feature values for each Leiden cluster,
    with rows representing Leiden clusters and columns representing features.
    Rows are clustered using a pre-calculated linkage matrix, and columns are
    clustered automatically by seaborn. Row colors indicate the optimal
    hierarchical clusters. All feature labels are explicitly shown on the x-axis.
    """
    # Convert the list of colors to a pandas Series with the correct index
    # The index should match the cluster_means index
    row_colors_list: list[str] = [theme_palette[label - 1] for label in cluster_labels]
    row_colors_series: pd.Series = pd.Series(row_colors_list, index=leiden_cluster_means.index)

    # Now call clustermap
    # Pass the linkage matrix for rows to use the pre-calculated clustering
    # Let clustermap cluster columns automatically (default behavior)
    # Use standard_scale=1 to scale features (columns)
    with plt.style.context(plot_style):
        g: sns.matrix.ClusterGrid = sns.clustermap(
            leiden_cluster_centroids_averaged, # Use the original DataFrame
            row_linkage=linkage_matrix, # Use the pre-calculated row linkage
            row_colors=row_colors_series, # Add row colors based on HAC clusters
            cmap='viridis', # Colormap
            cbar_kws={'label': 'Scaled Feature Value'},
            standard_scale=1, # Scale features (columns)
            yticklabels=False, #ignore the Leiden cluster labels
            figsize=(10, 8) # Adjust size as needed
        )

        # Get the column labels in the order they appear on the heatmap after clustering.
        # This ensures all feature labels are present and correctly ordered.
        ordered_col_indices: np.ndarray = g.dendrogram_col.reordered_ind
        ordered_col_names: pd.Index = leiden_cluster_centroids_averaged.columns[ordered_col_indices]
        x_labels_cleaned: list[str] = ordered_col_names#[" ".join(t.split("_")[1:]) for t in ordered_col_names]

        # Set the x-tick locations and labels explicitly to ensure all are shown.
        # Ticks are centered on the heatmap cells.
        g.ax_heatmap.set_xticks(np.arange(len(x_labels_cleaned)) + 0.5)
        g.ax_heatmap.set_xticklabels(x_labels_cleaned)

        plt.setp(
            g.ax_heatmap.xaxis.get_majorticklabels(),
            rotation=45, 
            ha='right',
            fontsize=8
        )
        g.ax_heatmap.tick_params(
            axis='y', 
            which='both', 
            left=False, 
            right=False, 
            labelleft=False
        )
        g.ax_heatmap.set_ylabel('Average Feature Value of Leiden Cluster Centroid (Window)')

        g.fig.suptitle(
            f'Spatial Statistics and Population Measurements Across Identified States', 
            y=1.02
        ) 

        # save the figure
        os.makedirs("output", exist_ok=True)

        fig = plt.gcf()
        fig.savefig(
            fname=os.path.join(
                "output", 
                "figure_2_panel_a.png"
            ), 
            dpi=300
        )

        fig.savefig(
            fname=os.path.join(
                "output", 
                "figure_2_panel_a.svg"
            )
        )

        plt.close()
        
plot_hac_clustergram()