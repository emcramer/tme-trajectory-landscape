import marimo

__generated_with = "0.16.5"
app = marimo.App(
    width="medium",
    layout_file="layouts/02_abm_state_space_analysis.marimo.slides.json",
)


@app.cell
def _():
    import marimo as mo
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

    from scatter_path_animation import animated_scatter_with_path
    from plotting_utilities import save_animation_to_mp4

    # settings for plotting style
    plot_style = "forest_and_sky_academic"
    plt.rcParams['svg.fonttype'] = 'none'

    # make a color palette for seaborn
    color_hex_codes = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
    theme_palette = sns.color_palette(color_hex_codes)

    plt.style.use('~/.matplotlib/stylelib/tme_research.mplstyle')

    def _():
        sns.palplot(theme_palette)
        plt.show()
        plt.close()
    _()
    return (
        BoundaryNorm,
        GridSpec,
        List,
        ListedColormap,
        animated_scatter_with_path,
        color_hex_codes,
        combinations,
        contextlib,
        dendrogram,
        jensenshannon,
        linkage,
        mo,
        mpatches,
        np,
        os,
        pd,
        pdist,
        pickle,
        plot_style,
        plt,
        re,
        sns,
        sp,
        theme_palette,
    )


@app.cell
def _(os, pd):
    # 1. Load the clustered window data that has the averages of the windows
    abm_window_data = pd.read_csv(
        os.path.join(
            "data",
            "abm", 
            "processed",
            "abm_windows_clustered_with_state_label_20251126.csv"
        )
    )

    non_feature_columns = ['sim_id', 'start_time_step', 'end_time_step', 'window_index_in_sim', 'hierarchical_label']
    feature_columns = [i.replace('avg_', '') for i in list(abm_window_data.drop(non_feature_columns, axis=1).columns)]
    return abm_window_data, feature_columns


@app.cell
def _(np, os, pd):
    # 2. ABM Leiden clustering results: loading the distance and linkage matrix calculated from the Leiden cluster centroids
    linkage_matrix = np.load(
        os.path.join(
            "output", 
            "objects", 
            "abm", 
            "abm_full_features_leiden_clusters_linkage_matrix.npy"
        )
    )
    distance_matrix = np.load(
        os.path.join(
            "output", 
            "objects", 
            "abm", 
            "abm_full_features_leiden_clusters_distance_matrix.npy"
        )
    )
    leiden_cluster_means = pd.read_csv(
        os.path.join(
            "data", 
            "abm", 
            "processed", 
            "abm_leiden_cluster_means.csv"
        )
    )
    return leiden_cluster_means, linkage_matrix


@app.function
def calculate_mean_feature(df, feature_name):
    # Select columns containing the substring
    grouped_columns = [col for col in df.columns if feature_name in col]

    # Calculate the mean of these columns for each row
    df[f'{feature_name}_avg'] = df[grouped_columns].mean(axis=1)

    return df


@app.cell
def _(pd, re):
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
    return (collapse_repeated_measures,)


@app.cell
def _(feature_columns, leiden_cluster_means):
    for feature in feature_columns:
        calculate_mean_feature(leiden_cluster_means, feature)
    return


@app.cell
def _(collapse_repeated_measures, leiden_cluster_means):
    #leiden_cluster_centroids_averaged = leiden_cluster_means.filter(like='_avg')
    leiden_cluster_centroids_averaged = collapse_repeated_measures(leiden_cluster_means)
    return (leiden_cluster_centroids_averaged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Clustermap illustrating spatial statistic differences between Leiden cluster centroid windows""")
    return


@app.cell
def _(
    abm_window_data,
    leiden_cluster_centroids_averaged,
    leiden_cluster_means,
    linkage_matrix,
    np,
    os,
    pd,
    plot_style,
    plt,
    sns,
    sp,
    theme_palette,
):

    # Data for clustermap is the cluster_means DataFrame
    # We will use the original cluster_means DataFrame
    # clustermap will handle scaling internally if requested

    n_states = abm_window_data['hierarchical_label'].unique().max()
    cluster_labels = sp.cluster.hierarchy.fcluster(linkage_matrix, n_states, criterion='maxclust')

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

            # Display the ClusterGrid object by getting the current axes
            plt.gca()

            # save the figure
            fig = plt.gcf()
            fig.savefig(
                fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_space_spatial_statistics_clustermap.png"
                ), 
                dpi=300
            )

            fig.savefig(
                fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_space_spatial_statistics_clustermap.svg"
                )
            )

            plt.show()
            plt.close()
    plot_hac_clustergram()
    return (cluster_labels,)


@app.cell
def _(combinations, jensenshannon, np, pd):
    def calculate_group_js_distances(df, group_column, measure_columns):
        """
        Calculate pairwise Jensen-Shannon distances between groups for specified measures.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe
        group_column : str
            Column name to group by (e.g., 'group' with values A, B, C)
        measure_columns : list of str
            Column names containing the measures to compare (e.g., ['x', 'y', 'z'])

        Returns:
        --------
        dict of pandas.DataFrame
            Dictionary where keys are measure names and values are DataFrames
            containing pairwise JS distances between groups
        """

        # Get unique groups
        groups = df[group_column].unique()

        # Initialize results dictionary
        results = {}

        # For each measure column
        for measure in measure_columns:
            # Create empty matrix for pairwise distances
            n_groups = len(groups)
            js_matrix = pd.DataFrame(
                np.zeros((n_groups, n_groups)),
                index=groups,
                columns=groups
            )

            # Calculate pairwise JS distances
            for group1, group2 in combinations(groups, 2):
                # Get distributions for each group
                dist1 = df[df[group_column] == group1][measure].values
                dist2 = df[df[group_column] == group2][measure].values

                # Create probability distributions (histograms)
                # Using common bins for both distributions
                all_values = np.concatenate([dist1, dist2])
                bins = np.histogram_bin_edges(all_values, bins='auto')

                hist1, _ = np.histogram(dist1, bins=bins, density=True)
                hist2, _ = np.histogram(dist2, bins=bins, density=True)

                # Normalize to probability distributions
                hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
                hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

                # Calculate JS distance
                js_dist = jensenshannon(hist1, hist2)

                # Fill symmetric matrix
                js_matrix.loc[group1, group2] = js_dist
                js_matrix.loc[group2, group1] = js_dist

            results[measure] = js_matrix

        return results

    def calculate_group_js_distances_long(df, group_column, measure_columns):
        """
        Returns results in long format instead of matrices.
        """
        results = []
        groups = df[group_column].unique()

        for measure in measure_columns:
            for group1, group2 in combinations(groups, 2):
                dist1 = df[df[group_column] == group1][measure].values
                dist2 = df[df[group_column] == group2][measure].values

                all_values = np.concatenate([dist1, dist2])
                bins = np.histogram_bin_edges(all_values, bins='auto')

                hist1, _ = np.histogram(dist1, bins=bins, density=True)
                hist2, _ = np.histogram(dist2, bins=bins, density=True)

                hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
                hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

                js_dist = jensenshannon(hist1, hist2)

                results.append({
                    'measure': measure,
                    'group1': group1,
                    'group2': group2,
                    'js_distance': js_dist
                })

        return pd.DataFrame(results)

    def rank_features_by_divergence(df, group_column, measure_columns, 
                                      aggregation='mean', return_details=False):
        """
        Rank features by their overall divergence across groups using JS distance.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataframe
        group_column : str
            Column name to group by
        measure_columns : list of str
            Column names containing the measures to compare
        aggregation : str, default='mean'
            How to aggregate pairwise distances: 'mean', 'max', 'sum', 'median'
        return_details : bool, default=False
            If True, return detailed pairwise distances along with rankings

        Returns:
        --------
        pandas.DataFrame
            Features ranked by divergence score (highest to lowest)
        """

        groups = df[group_column].unique()
        divergence_scores = []
        all_pairwise = {}

        for measure in measure_columns:
            pairwise_distances = []

            for group1, group2 in combinations(groups, 2):
                dist1 = df[df[group_column] == group1][measure].values
                dist2 = df[df[group_column] == group2][measure].values

                # Create probability distributions
                all_values = np.concatenate([dist1, dist2])
                bins = np.histogram_bin_edges(all_values, bins='auto')

                hist1, _ = np.histogram(dist1, bins=bins, density=True)
                hist2, _ = np.histogram(dist2, bins=bins, density=True)

                hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
                hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

                js_dist = jensenshannon(hist1, hist2)
                pairwise_distances.append(js_dist)

            # Aggregate pairwise distances
            if aggregation == 'mean':
                score = np.mean(pairwise_distances)
            elif aggregation == 'max':
                score = np.max(pairwise_distances)
            elif aggregation == 'sum':
                score = np.sum(pairwise_distances)
            elif aggregation == 'median':
                score = np.median(pairwise_distances)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            divergence_scores.append({
                'feature': measure,
                'divergence_score': score,
                'min_pairwise': np.min(pairwise_distances),
                'max_pairwise': np.max(pairwise_distances),
                'std_pairwise': np.std(pairwise_distances)
            })

            all_pairwise[measure] = pairwise_distances

        # Create ranked dataframe
        ranked_df = pd.DataFrame(divergence_scores)
        ranked_df = ranked_df.sort_values('divergence_score', ascending=False).reset_index(drop=True)
        ranked_df['rank'] = range(1, len(ranked_df) + 1)

        if return_details:
            return ranked_df, all_pairwise
        else:
            return ranked_df


    # Alternative: Calculate normalized divergence score
    def rank_features_by_divergence_normalized(df, group_column, measure_columns):
        """
        Rank features using coefficient of variation of pairwise JS distances.
        This can help identify features with consistent high divergence vs. 
        features with variable divergence across group pairs.
        """

        groups = df[group_column].unique()
        results = []

        for measure in measure_columns:
            pairwise_distances = []

            for group1, group2 in combinations(groups, 2):
                dist1 = df[df[group_column] == group1][measure].values
                dist2 = df[df[group_column] == group2][measure].values

                all_values = np.concatenate([dist1, dist2])
                bins = np.histogram_bin_edges(all_values, bins='auto')

                hist1, _ = np.histogram(dist1, bins=bins, density=True)
                hist2, _ = np.histogram(dist2, bins=bins, density=True)

                hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
                hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

                js_dist = jensenshannon(hist1, hist2)
                pairwise_distances.append(js_dist)

            mean_dist = np.mean(pairwise_distances)
            std_dist = np.std(pairwise_distances)

            results.append({
                'feature': measure,
                'mean_divergence': mean_dist,
                'std_divergence': std_dist,
                'cv_divergence': std_dist / mean_dist if mean_dist > 0 else 0,
                'max_divergence': np.max(pairwise_distances)
            })

        ranked_df = pd.DataFrame(results)
        ranked_df = ranked_df.sort_values('mean_divergence', ascending=False).reset_index(drop=True)
        ranked_df['rank'] = range(1, len(ranked_df) + 1)

        return ranked_df
    return (rank_features_by_divergence,)


@app.cell
def _(
    cluster_labels,
    leiden_cluster_centroids_averaged,
    rank_features_by_divergence,
):
    clustermap_df = leiden_cluster_centroids_averaged.copy()
    measure_columns = clustermap_df.columns.values
    clustermap_df['state_label'] = cluster_labels

    ranked_measures, state_js_distances = rank_features_by_divergence(
        clustermap_df, 
        'state_label', 
        measure_columns, 
        aggregation='mean', 
        return_details=True
    )
    return


@app.cell
def _(
    cluster_labels,
    leiden_cluster_centroids_averaged,
    leiden_cluster_means,
    linkage_matrix,
    np,
    os,
    pd,
    plot_style,
    plt,
    sns,
    theme_palette,
):
    def plot_hac_clustergram_ranked_features() -> None:
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
                figsize=(14, 8) # Adjust size as needed
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

            # Display the ClusterGrid object by getting the current axes
            plt.gca()

            # save the figure
            fig = plt.gcf()
            fig.savefig(
                fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_space_spatial_statistics_clustermap.png"
                ), 
                dpi=300
            )
            fig.savefig(
                fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_space_spatial_statistics_clustermap.svg"
                )
            )

            plt.show()
            plt.close()
    plot_hac_clustergram_ranked_features()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tracking the evolution of states over time""")
    return


@app.cell
def _(List, contextlib, os, pd, plt):
    # plot for figure 3A
    def plot_state_occupancy_over_time(trajectories_nested: List[List[int]], plot_context=True) -> None:
        """
        Plots the total count of simulations occupying each state over time.

        Args:
            trajectories_nested: A list of lists, where each inner list represents a
                                 simulation trajectory and contains the sequence of
                                 states visited by that simulation. States are
                                 represented by integers from 1 to 6.
        """
        if not trajectories_nested:
            print("No trajectories provided to plot.")
            return

        plot_context = plt.style.context('forest_and_sky_academic') if plot_context else contextlib.nullcontext()

        # Determine the maximum length of any trajectory to set the time axis
        max_len = max(len(traj) for traj in trajectories_nested)

        # Initialize a dictionary to store counts for each state at each time step
        state_counts_over_time = {state: [0] * max_len for state in range(1, 7)}

        # Populate the state counts
        for trajectory in trajectories_nested:
            for time_step, state in enumerate(trajectory):
                if 1 <= state <= 6:  # Ensure state is within the expected range
                    state_counts_over_time[state][time_step] += 1

        # Convert the dictionary to a pandas DataFrame for easier plotting
        df_state_counts = pd.DataFrame(state_counts_over_time)

        # Plotting
        with plot_context:
            plt.figure(figsize=(12, 7))
            for state in range(1, 7):
                plt.plot(df_state_counts.index, df_state_counts[state], label=f'State {state}', marker='o', markersize=4, linestyle='-')

            plt.title('Evolution of TME Tissue States Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Number of Simulations')
            #plt.xticks(df_state_counts.index) # Ensure all time steps are shown as ticks
            plt.legend()
            plt.grid(True)
            plt.gca()

            # save the figure
            fig = plt.gcf()
            fig.savefig(
                fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_occupancy_over_time.png"
                ), 
                dpi=300
            )

            plt.show()
            plt.close()
    return (plot_state_occupancy_over_time,)


@app.cell
def _(abm_window_data, mo):
    mo.ui.dataframe(abm_window_data)
    return


@app.cell
def _(abm_window_data):
    grouped = abm_window_data.groupby('sim_id')

    trajectories_nested = []
    for name, group in grouped:
        trajectories_nested.append(group['hierarchical_label'].values)
    return (trajectories_nested,)


@app.cell
def _(plot_state_occupancy_over_time, trajectories_nested):
    plot_state_occupancy_over_time(trajectories_nested)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Heatmap of trajectories illustrating cannonical progressions through meta-stable states""")
    return


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    dendrogram,
    gridspec,
    linkage,
    mplcm,
    np,
    pdist,
    plt,
):
    def hcluster_order(data):
        """
        Performs hierarchical clustering on the input data and returns the order of the leaves.

        Args:
            data (np.ndarray): The input data for clustering.

        Returns:
            list: A list of indices representing the reordered leaves from the clustering.
        """
        # Calculate pairwise Euclidean distances between rows
        row_distances = pdist(data, metric='hamming') # default to Euclidean
        # Perform hierarchical clustering using the 'ward' method
        linked_matrix = linkage(row_distances, method='ward')
        # Generate dendrogram information without plotting
        dendro_info = dendrogram(linked_matrix, no_plot=True)
        # Get the order of the leaves from the dendrogram
        reordered_indices = dendro_info['leaves']
        return reordered_indices, linked_matrix, dendro_info

    def plot_clustergram(data, sim_ids, reordered_indices):
        """
        Plots a heatmap with a dendrogram showing the hierarchical clustering of simulation trajectories.

        Args:
            data (np.ndarray): The data to be visualized.
            sim_ids (np.ndarray): An array of simulation IDs corresponding to the rows of the data.
            reordered_indices (list): The order of indices determined by hierarchical clustering.
        """
        # Adjust data for plotting (subtract 1 to align with colormap indexing if necessary)
        plot_data = data - 1
        # Get unique values in the data to define the colormap
        unique_values = np.unique(plot_data)

        # Define a colormap for the heatmap
        # Using 'tab10' which has 10 distinct colors
        tab10_cmap = mplcm.get_cmap('tab10')
        # Create custom colors based on the unique values in the data
        custom_colors = [tab10_cmap(i) for i in unique_values]
        # Create a ListedColormap from the custom colors
        categorical_cmap = ListedColormap(custom_colors[:len(unique_values)])
        # Define boundaries for the colormap to ensure discrete color mapping
        bounds = np.arange(len(unique_values) + 1) - 0.5
        norm = BoundaryNorm(bounds, categorical_cmap.N)

        # Calculate the linkage matrix for the dendrogram
        row_distances = pdist(data, metric='euclidean')
        linked_matrix = linkage(row_distances, method='ward')

        # Create a figure and axes for the plot
        # Adjust figsize for better visualization of heatmap and dendrogram
        fig = plt.figure(figsize=(12, 12))
        # Use GridSpec to create a layout with a main heatmap and a dendrogram axis
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

        # Create the dendrogram axis (top-left)
        ax_dendro = fig.add_subplot(gs[0, 0])
        # Plot the dendrogram
        dendrogram(linked_matrix, orientation='top', ax=ax_dendro, no_plot=False)
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        ax_dendro.spines['bottom'].set_visible(False)
        ax_dendro.spines['right'].set_visible(False)
        ax_dendro.spines['top'].set_visible(False)
        ax_dendro.spines['left'].set_visible(False)

        # Create the heatmap axis (bottom-right)
        ax_heatmap = fig.add_subplot(gs[1, 1])
        # Plot the heatmap using pcolor
        im = ax_heatmap.pcolor(
            plot_data[reordered_indices],
            cmap='tab10',
            norm=norm
        )

        # Set the title for the heatmap
        ax_heatmap.set_title('Simulation Trajectories Through State Space')
        # Set the x-axis label
        ax_heatmap.set_xlabel('Time Window (Minutes)')
        # Set the y-axis label
        ax_heatmap.set_ylabel('Simulation ID')

        # Set the y-axis ticks and labels using the reordered simulation IDs
        ax_heatmap.set_yticks(
            ticks=np.arange(len(sim_ids)) + 0.5, # Center ticks on the cells
            labels=sim_ids[reordered_indices],
            fontsize=4,
        )
        # Remove x-axis ticks for clarity
        ax_heatmap.set_xticks([])

        # Adjust the dendrogram to align with the heatmap
        # The dendrogram's x-axis should correspond to the heatmap's y-axis order
        ax_dendro.set_xlim(ax_heatmap.get_ylim())

        # Add a colorbar to the plot
        cbar = fig.colorbar(im, ax=ax_heatmap, ticks=bounds[:-1] + 0.5)
        cbar.set_ticklabels(unique_values+1)
        cbar.set_label('State Value')

        # Display the plot
        plt.gca()
        plt.show()
        plt.close()
    return (hcluster_order,)


@app.cell
def _(
    BoundaryNorm,
    GridSpec,
    ListedColormap,
    color_hex_codes,
    hcluster_order,
    np,
    os,
    plt,
    trajectories_nested,
):
    # Assuming trajectories_nested is already defined and is a list of lists of integers
    # Example: trajectories_nested = [[1, 2, 1, 3], [2, 1, 3, 2], [1, 3, 2, 1]]

    # Convert the nested list to a NumPy array for easier manipulation
    trajectories_array = np.array(trajectories_nested)

    # Get the reordered indices based on hierarchical clustering of the data
    reordered_indices, linked_matrix, dendro_info = hcluster_order(trajectories_array)

    # Sort the data according to the reordered indices
    sorted_data = trajectories_array[reordered_indices]

    unique_values = np.unique(trajectories_array)

    # Define a colormap for the heatmap
    # Using 'tab10' which has 10 distinct colors
    #tab10_cmap = mplcm.get_cmap('tab10')
    #theme_cmap = LinearSegmentedColormap.from_list('forest_and_sky', theme_colors, N=len(theme_colors))
    # Create custom colors based on the unique values in the data
    #custom_colors = [tab10_cmap(i) for i in unique_values]
    # Create a ListedColormap from the custom colors
    categorical_cmap = ListedColormap(color_hex_codes[:len(unique_values)])
    # Define boundaries for the colormap to ensure discrete color mapping
    bounds = np.arange(len(unique_values)+1)-0.5
    norm = BoundaryNorm(bounds, categorical_cmap.N)

    # Create a figure and axes for the plot
    # Adjust figsize for better visualization of heatmap and dendrogram
    with plt.style.context('forest_and_sky_academic'):
        fig = plt.figure(figsize=(12, 12))
        # Use GridSpec to create a layout with a main heatmap and a dendrogram axis
        gs = GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

        # Create the dendrogram axis (top-left)
        #ax_dendro = fig.add_subplot(gs[0, 0])
        # Plot the dendrogram
        #dendrogram(linked_matrix, orientation='top', ax=ax_dendro, no_plot=False)
        #ax_dendro.set_xticks([])
        #ax_dendro.set_yticks([])
        #ax_dendro.spines['bottom'].set_visible(False)
        #ax_dendro.spines['right'].set_visible(False)
        #ax_dendro.spines['top'].set_visible(False)
        #ax_dendro.spines['left'].set_visible(False)

        # Create the heatmap axis (bottom-right)
        ax_heatmap = fig.add_subplot(gs[1, 1])
        # Plot the heatmap using pcolor
        im = ax_heatmap.pcolor(
            sorted_data,
            cmap=categorical_cmap,
            #norm=norm
        )

        # Set the title for the heatmap
        ax_heatmap.set_title('Simulation Trajectories Through State Space \n(sorted by Hamming distance)')
        # Set the x-axis label
        ax_heatmap.set_xlabel('Time Window (Minutes)')
        # Set the y-axis label
        ax_heatmap.set_ylabel('Simulations')

        # Set the y-axis ticks and labels using the reordered simulation IDs
        #ax_heatmap.set_yticks(
        #    ticks=np.arange(len(sim_ids)) + 0.5, # Center ticks on the cells
        #    labels=[sim_ids[i] for i in reordered_indices],
        #    fontsize=4,
        #)
        # Remove x-axis ticks for clarity
        #ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])

        # Adjust the dendrogram to align with the heatmap
        # The dendrogram's x-axis should correspond to the heatmap's y-axis order
        #ax_dendro.set_xlim(ax_heatmap.get_ylim())

        # Add a colorbar to the plot
        cbar = fig.colorbar(im, ax=ax_heatmap)#, ticks=bounds[:-1] + 0.5)
        #cbar.set_ticklabels(unique_values)
        cbar.set_ticks(ticks=[1.4, 2.3, 3.1, 3.9, 4.8, 5.6], labels=unique_values)
        cbar.set_label('State Value')
        #cbar.set_ticks([(b0 + b1) / 2 + 0.5 for b0, b1 in zip(bounds[:-1], bounds[1:])])
        #tick_texts = cbar.ax.set_yticklabels([""] + [str(i) for i in unique_values]+[""])
        #tick_texts[0].set_verticalalignment('top')
        #tick_texts[-1].set_verticalalignment('bottom')
        #cbar.ax.tick_params(length=0)

        # save the figure
        fig = plt.gcf()
        fig.savefig(
            fname=os.path.join(
                "output", 
                "figures", 
                "abm", 
                "state_space_analysis", 
                "abm_simulation_trajectory_heatmap_hamming_distance.png"
            ), 
            dpi=300
        )

        plt.show()
        plt.close()
    return


@app.cell
def panel6a(
    GridSpec,
    ListedColormap,
    color_hex_codes,
    dendrogram,
    linkage,
    mpatches,
    np,
    os,
    pdist,
    pickle,
    plt,
    trajectories_nested,
):
    def _():
        def hcluster_order(data):
            # Perform clustering on rows using Hamming or Euclidean (?) distance
            #row_distances = pdist(data, metric='hamming')
            row_distances = pdist(data, metric='euclidean')
            linked_matrix = linkage(row_distances, method='average')  # average works with Hamming
            dendro_info = dendrogram(linked_matrix, no_plot=True)
            reordered_indices = dendro_info['leaves']
            return reordered_indices, linked_matrix, dendro_info

        def plot_clustergram(data, sim_ids, color_hex_codes):
            # Cluster on ORIGINAL 1-indexed state matrix
            reordered_indices, linked_matrix, dendro_info = hcluster_order(data)

            # Sort data by clustering order
            sorted_data = data[reordered_indices]

            # Sorted unique original state labels
            unique_states = np.unique(data)
            n_states = len(unique_states)

            # Ensure color list cycles correctly even if shorter than n_states
            cmap = ListedColormap([color_hex_codes[i % len(color_hex_codes)] for i in range(n_states)])

            # Remap sorted_data into rank space [0, n_states-1]
            # Each state gets the color corresponding to its sorted position, cycling through cmap
            state_to_rank = {state: i for i, state in enumerate(unique_states)}
            ranked_data = np.vectorize(lambda x: state_to_rank[x])(sorted_data)

            # Plot layout
            fig = plt.figure(figsize=(12, 12))
            gs = GridSpec(1, 2, width_ratios=[1, 4], wspace=0.02)

            # Dendrogram on left
            ax_dendro = fig.add_subplot(gs[0, 0])
            dendrogram(linked_matrix, orientation='left', ax=ax_dendro, link_color_func=lambda k: "black")
            ax_dendro.set_xticks([])
            ax_dendro.set_yticks([])
            for spine in ax_dendro.spines.values():
                spine.set_visible(False)

            # Heatmap in center
            ax_heatmap = fig.add_subplot(gs[0, 1])
            ax_heatmap.pcolor(ranked_data, cmap=cmap)

            # Move row labels to right side
            ax_heatmap.yaxis.tick_right()
            #ax_heatmap.set_yticks(
            #    ticks=np.arange(len(sim_ids)) + 0.5,
            #    labels=sim_ids[reordered_indices],
            #    fontsize=6,
            #)
            ax_heatmap.set_yticks(ticks=[], labels=[])

            # Remove left Y-axis label since labels are now right-aligned
            ax_heatmap.set_ylabel("")
            ax_heatmap.set_xticks([])

            # Build discrete patch legend using original 1-indexed state labels
            legend_patches = [
                mpatches.Patch(color=cmap(state_to_rank[s]), label=str(s))
                for s in unique_states
            ]
            ax_heatmap.legend(
                handles=legend_patches,
                title="State Value",
                loc="center left",
                bbox_to_anchor=(1.15, 0.5),
                frameon=True
            )

            # removing the legend for manuscript panel - comment out to include legend
            # ----------
            legend = ax_heatmap.get_legend()
            if legend is not None:  # Check to avoid AttributeError
                legend.remove()
            # ----------

            ax_heatmap.set_title('Simulation Trajectories Through State Space')
            ax_heatmap.set_xlabel('Time Window')

            #plt.show()
            #plt.close()
            return fig


        # 1. Convert your nested list into a NumPy array (rows = simulations, columns = time windows)
        data = np.array(trajectories_nested)

        # 2. Build a simulation ID array if you want labeled rows (optional but recommended)
        # If you already have sim IDs, skip generating them and use your existing array instead
        sim_ids = np.array([f"Sim_{i}" for i in range(data.shape[0])])

        with open(
            os.path.join(
                'output',
                'figures',
                'paper-figures',
                'figure-5-panels',
                'data',
                'fig_5a_data.pkl'
            ),
            'wb'
        ) as f:
            panel_data = {
                'data': data, 
                'sim_ids': sim_ids,
                'color_hex_codes': color_hex_codes
            }
            pickle.dump(panel_data, file=f)
            print(f'Panel data saved to {f}')

        # 3. Call the plotting function directly with your color palette
        fig = plot_clustergram(data, sim_ids, color_hex_codes)

        plt.show()

        # saveout
        fig.savefig(
            fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_space_trajectory_clustermap_euclidean.svg"
                )
        )
        fig.savefig(
            fname=os.path.join(
                    "output", 
                    "figures", 
                    "abm", 
                    "state_space_analysis", 
                    "abm_state_space_trajectory_clustermap_euclidean.png"
                ),
            dpi=300
        )

        #plt.close()
        return fig
    figure = _()
    return


@app.cell
def _(mo):
    mo.md(r"""### Spatial Statistics Over UMAP""")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Animate Progression of Simulations through States""")
    return


@app.cell
def _(np):
    # load the ABM state space embedding
    abm_embedding = np.load("output/objects/abm/abm_full_features_pc_n647_umap_embedding.npy")
    return (abm_embedding,)


@app.cell
def _(abm_window_data):
    abm_window_data
    return


@app.cell
def _(
    abm_embedding,
    abm_window_data,
    animated_scatter_with_path,
    color_hex_codes,
    pd,
):
    def generate_path_animation(sim_id):

        plot_data = pd.DataFrame({
            "x": abm_embedding[:, 0],
            "y": abm_embedding[:, 1],
            'sim_id':abm_window_data['sim_id'].values,
            'Time (Minutes)':abm_window_data['end_time_step'].values,
            "State": abm_window_data['hierarchical_label'].values
        })

        # -------------------------------
        # Background data (static scatter)
        # -------------------------------
        background_data = plot_data.copy().drop(['sim_id'], axis=1)

        # -------------------------------
        # Foreground data (animated path)
        # -------------------------------

        foreground_data = plot_data.copy()[plot_data['sim_id']==sim_id]

        anim, fig, ax = animated_scatter_with_path(
            background_data=background_data,
            foreground_data=foreground_data,
            background_color_col="State",
            foreground_color_col="Time (Minutes)",
            background_cmap=color_hex_codes[0:6],
            style="seaborn-v0_8",
            interval=150
        )
        return anim
    return (generate_path_animation,)


@app.cell
def _(generate_path_animation):
    animated_path = generate_path_animation(0)
    return


@app.cell
def _():
    #mo.Html(animated_path.to_html5_video())
    return


@app.cell
def _():
    #save_animation_to_mp4(animated_path, 'output/animations/example_trajectory_fixed_animation.mp4')
    return


@app.cell
def _(abm_window_data, mo):
    mo.ui.dataframe(abm_window_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
