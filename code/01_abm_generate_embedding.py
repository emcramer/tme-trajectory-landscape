import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import scipy as sp
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap
    import os
    import pickle

    import leidenalg as la
    import networkx as nx
    import igraph as ig
    from igraph import Graph

    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph
    from sklearn.decomposition import PCA

    from utils import window_trajectory_data, unnest_list, create_pmf, jensen_shannon_divergence, calculate_jsd_baseline, zscore_then_minmax_normalize, enhance_legend_markers, add_suffix_to_repeats, get_today

    np.random.seed(42)

    theme_palette = sns.color_palette(['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c'])# Initialization code that runs before all other cells


    # abm data set directory
    abm_data_dir = os.path.join("data", "abm")

    # set output directory
    output_dir = os.path.join('output')

    plt.rcParams['svg.fonttype'] = 'none'


@app.cell
def _():
    # loading the ABM data
    # loading the simulation trajectories
    def _():
        with open(os.path.join(abm_data_dir, "all_simulations_param_trajectory_20251022.pkl"), 'rb') as f:
            abm_trajectories = pickle.load(f)
        return abm_trajectories
    abm_trajectory_info = _()
    abm_trajectories = [k['trajectory'] for k in abm_trajectory_info]
    sim_initial_conditions = [d["initial_conditions.cell_positions.filename"] for d in abm_trajectory_info]

    # loading the time step summaries for each simulation
    abm_timestep_summaries_df = pd.read_csv(
        #os.path.join(abm_data_dir, "processed", "abm_normed_scaled_spatial_features_local_time.csv")
        os.path.join(abm_data_dir, "processed", "normed_scaled_features_local_time_20251123.csv")
    )

    feat_cols = list(abm_timestep_summaries_df.drop(['time_step', 'sim_id'], axis=1).columns)
    return (
        abm_timestep_summaries_df,
        abm_trajectories,
        feat_cols,
        sim_initial_conditions,
    )


@app.cell
def _(abm_timestep_summaries_df, feat_cols):
    # window the abm feature data
    #feature_columns = list(abm_summaries_df.columns[3:])
    window_size = 50

    # using all of the features
    windowed_summary_data, abm_summaries_df_windowed = window_trajectory_data(abm_timestep_summaries_df, feat_cols, window_size)
    return abm_summaries_df_windowed, windowed_summary_data


@app.cell
def _(abm_summaries_df_windowed, abm_trajectories):
    abm_summaries_df_windowed['state_label'] = unnest_list(abm_trajectories)
    abm_summaries_df_windowed.columns = abm_summaries_df_windowed.columns.str.replace('avg_', '')
    abm_summaries_df_windowed
    return


@app.cell
def _(abm_summaries_df_windowed, windowed_summary_data):
    # save the windowed data
    abm_summaries_df_windowed.to_csv(
        os.path.join(
            'data',
            'abm',
            'processed',
            'abm_window_info_and_averages.csv'
        ),
        index=False
    )

    np.save(
        file=os.path.join(
            'data',
            'abm',
            'processed',
            'abm_time_delay_embedding.npy'
        ),
        arr=windowed_summary_data
    )
    return


@app.cell
def _(abm_summaries_df_windowed):
    avg_time_per_timestep = (abm_summaries_df_windowed['start_time_step']+abm_summaries_df_windowed['start_time_step'])/2
    return (avg_time_per_timestep,)


@app.function
def pca_preprocessing(X):
    """
    Returns the principal components that constitute 95% of the variance of the data.
    """

    # instantiate PCA object
    pca = PCA()

    # Fit PCA to the windowed data
    pca.fit(X)

    # Get the explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # The maximum number of components is the number of features or samples, whichever is smaller
    max_components = min(X.shape[0], X.shape[1])

    # Find the number of components needed for 95% cumulative variance
    n_components_95_var = np.argmax(cumulative_explained_variance >= 0.95) + 1 if cumulative_explained_variance[-1] >= 0.95 else max_components

    # perform dimensionality reduction
    pca_reducer = PCA(
        n_components = n_components_95_var
    )

    X_pcs = pca_reducer.fit_transform(X)

    # save the reducer 
    with open(os.path.join("output","objects","abm", "abm_pca_reducer.pkl"), 'wb') as f:
        pickle.dump(pca_reducer, f)

    return X_pcs


@app.function
def umap_embedding(X, show_plot=True, save_path="", color=None, **kwargs):
    """
    Performs UMAP embedding. Options to show a plot and save the embedding array.
    """

    opts = {
        'n_neighbors':15,
        'min_dist':0.1,
        'n_components':2,
        'random_state':42,
        'metric':'euclidean'
    }
    opts.update(kwargs)
    # umap dimensionality reduction to 2 dimensions 
    reducer = umap.UMAP(**opts)
    X_umap = reducer.fit_transform(X)

    # save the reducer 
    with open(os.path.join("output","objects","abm", "abm_umap_reducer.pkl"), 'wb') as f:
        pickle.dump(reducer, f)

    if show_plot:
        with plt.style.context('forest_and_sky_academic'):
            # plot the result
            fig = plt.figure()
            ax = fig.add_subplot()

            scat = ax.scatter(
                X_umap[:, 0],
                X_umap[:, 1],
                color= color if color is not None else 'grey',
                marker='.',
                alpha=0.6
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP of ABM Windows from Intersection of Features with MIBI Dataset')
            plt.tight_layout()
            plt.show()
            plt.close()

    if save_path != "":
        np.save(
            save_path,
            X_umap,
            allow_pickle=True
        )

    # return the embeddings
    return X_umap


@app.cell
def _(abm_summaries_df_windowed, windowed_summary_data):
    def abm_pca(data):
        """ Wrapper for getting the PCs of the ABM time lagged embedding. Tries to load previously calculated PCs and recalculates them if not. """
        abm_pc_fname = os.path.join(
            "output",
            "objects",
            "abm",
            "abm_full_features_pca.npy" # modify this to specify which PCs to load
        )
        try:
            # load pre-computed principal components
            pcs = np.load(abm_pc_fname)
        except Exception as e:
            print(f"Error: {e}")
            print("Re-calculating principal components.")
            pcs = pca_preprocessing(
                 data
            )
            np.save(abm_pc_fname, arr=pcs, allow_pickle=True)
        return pcs

    def abm_umap(data):
        """
        Wrapper for getting the UMAP manifold of the ABM time lagged embedding. Tries to load previously calculated manifold and recalculates the manifold otherwise.
        """
        # try loading a previous embedding
        abm_umap_fname = os.path.join(
            'output', 
            'objects', 
            'abm', 
            'abm_full_features_pc_n647_umap_embedding.npy' # modify this to specify which umap embedding to load
        )
        try:
            # load pre-computed umap embeddings
            abm_embedding = np.load(abm_umap_fname)
        except Exception as e:
            print(f"Error: {e}")
            print("Re-calculating UMAP manifold.")
            abm_embedding = umap_embedding(
                data, 
                n_neighbors=int(abm_summaries_df_windowed['sim_id'].value_counts()[0]) # number of time points per simulation
            )
            # saveout
            np.save(abm_umap_fname, abm_embedding, allow_pickle=True)
        return abm_embedding

    def generate_abm_pc_umap():
        """
        Wrapper for loading or generating the principal components and UMAP manifold of the time step windows from the ABM output.
        """
        # get the PCs for the full feature set of the ABM data
        pcs = abm_pca(windowed_summary_data) # change this to modify which data to use for PCA

        # get the umap embedding for the full feature set of the ABM data
        umap = abm_umap(pcs)

        # return the PCs and the UMAP embedding
        return pcs, umap
    return (generate_abm_pc_umap,)


@app.cell
def _(generate_abm_pc_umap):
    abm_pcs, abm_umap_embedding = generate_abm_pc_umap()
    return (abm_umap_embedding,)


@app.cell
def _(abm_umap_embedding, avg_time_per_timestep):
    # checking how the time component aligns on the UMAP manifold
    def _():
        #with plt.style.context('forest_and_sky_academic'):
        with plt.style.context('~/.matplotlib/stylelib/tme_research.mplstyle'):
            # plot the result
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot()

            scat = ax.scatter(
                abm_umap_embedding[:, 0],
                abm_umap_embedding[:, 1],
                c=avg_time_per_timestep,
                marker='.',
                alpha=0.6,
                s=5
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('Time-Delay Embedding')
            plt.colorbar(scat, label='Average Time of Window (Minutes)')
            plt.tight_layout()

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm',
                    'abm_time_umap.png'
                ), 
                dpi=300
            )
            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm',
                    'abm_time_umap.svg'
                )
            )
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(abm_umap_embedding, avg_time_per_timestep):
    # Plotting a simplified version of the time component for
    def _():
        with plt.style.context('forest_and_sky_academic'):
            # plot the result
            fig = plt.figure()
            ax = fig.add_subplot()

            scat = ax.scatter(
                abm_umap_embedding[:, 0],
                abm_umap_embedding[:, 1],
                c=avg_time_per_timestep,
                marker='.',
                alpha=0.6,
                s=5
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP of ABM Time-Delay Embedding')
            cbar = plt.colorbar(scat, label='Time', shrink=0.3)
            # Set custom tick positions and labels
            cbar.set_ticks([0, 8000, 16000])
            cbar.set_ticklabels(['Early', 'Middle', 'Late'])
            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(abm_umap_embedding, sim_initial_conditions):
    # checking the role of initial conditions...
    def _():
        tp_initial_conditions = np.concatenate([np.tile(i, 647) for i in sim_initial_conditions])
        pdf = pd.DataFrame({
            'x':abm_umap_embedding[:, 0],
            'y':abm_umap_embedding[:, 1],
            'initial_positions':tp_initial_conditions
        })
        with plt.style.context('forest_and_sky_academic'):
            # plot the result
            fig = plt.figure()
            ax = fig.add_subplot()

            sns.scatterplot(
                pdf,
                x = 'x',
                y = 'y',
                hue='initial_positions',
                palette='tab10',
                marker='.',
                alpha=0.6,
                ax=ax
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP of ABM Time Delay Embedding')
            #plt.colorbar(scat, label='Initial Cell Positions')
            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(abm_summaries_df_windowed, abm_umap_embedding):
    # check to see how the individual simulations arrange on the UMAP manifold
    def _():
        with plt.style.context('forest_and_sky_academic'):
            # plot the result
            fig = plt.figure()
            ax = fig.add_subplot()

            scat = ax.scatter(
                abm_umap_embedding[:, 0],
                abm_umap_embedding[:, 1],
                c=abm_summaries_df_windowed.sim_id,
                cmap='magma',
                marker='.',
                s=5,
                alpha=0.6,

            )
            plt.colorbar(scat, label='Simulation ID')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP of ABM Time Delay Embedding')
            plt.tight_layout()

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'sim_ids_time_delay_embedding_umap.png'
                ), 
                dpi=300
            )

            plt.show()
            plt.close()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Plotting Features Over UMAP""")
    return


@app.cell
def _(abm_summaries_df_windowed):
    abm_summaries_df_windowed.columns
    return


@app.cell
def _(abm_summaries_df_windowed, abm_umap_embedding):
    # Plotting a simplified version of the time component for
    def plot_feature_over_umap(feature, cmap='inferno'):
        with plt.style.context('forest_and_sky_academic'):
            # plot the result
            fig = plt.figure()
            ax = fig.add_subplot()

            scat = ax.scatter(
                abm_umap_embedding[:, 0],
                abm_umap_embedding[:, 1],
                c=abm_summaries_df_windowed[feature],
                marker='.',
                alpha=0.6,
                s=5,
                cmap=cmap
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'UMAP of ABM Time-Delay Embedding: {feature}')
            cbar = plt.colorbar(scat, label='Relative Population (%)')
            plt.tight_layout()

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'feature_umaps', 
                    f'{feature}.png'
                ), 
                dpi=300
            )
            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'feature_umaps', 
                    f'{feature}.svg'
                )
            )

            plt.show()
            plt.close()

    plot_feature_over_umap('malignant_epithelial_cell')
    plot_feature_over_umap('effector_T_cell')
    plot_feature_over_umap('exhausted_T_cell')
    plot_feature_over_umap('malignant_epithelial_cell_degree_centrality')
    return


@app.cell(hide_code=True)
def _():
    mo.md("""## Building State Space""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Build kNN Graph""")
    return


@app.function
def build_knn(data_points, n_neighbors=50, save_path="", **kwargs):
    opts = {
        'mode':'distance', # Use distance as edge weight
        'metric':'euclidean'
    }
    opts.update(kwargs)

    knn_graph_sparse = kneighbors_graph(
        data_points,
        n_neighbors=n_neighbors,
        **opts
    )

    # Convert sparse matrix to igraph
    sources, targets = knn_graph_sparse.nonzero()
    weights = knn_graph_sparse.data
    edges = list(zip(sources, targets))

    # Create igraph graph
    graph = ig.Graph(edges, directed=False) # Assuming undirected graph
    graph.es['weight'] = weights

    if save_path != "":
        with open(save_path, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

    print(f"Built k-NN graph with {graph.vcount()} vertices and {graph.ecount()} edges.")
    return graph


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Leiden Clustering of kNN Graph""")
    return


@app.cell
def _():
    def cluster_leiden(graph, save_path="", **kwargs):
        opts = {
            'resolution': 1.0,
            'random_seed':42
        }
        opts.update(kwargs)

        graph_unweighted = graph.copy()
        graph_unweighted.es['weight'] = 1.0 # Set all weights to 1 for unweighted clustering

        try:
            # Use the standard ModularityVertexPartition for unweighted graphs
            partition = la.find_partition(
                graph_unweighted,
                #la.ModularityVertexPartition,
                la.CPMVertexPartition,
                #weights='weight'
                resolution_parameter = opts['resolution'],
                seed=opts['random_seed']
            )
            leiden_labels = np.array(partition.membership)
            n_leiden_clusters = len(set(leiden_labels))

            if save_path != "":
                graph.vs['leiden_community'] = partition.membership
                graph.write_gml(save_path)

            print(f"Found {n_leiden_clusters} Leiden clusters with resolution {opts['resolution']}.")

        except Exception as e:
            leiden_labels = None
            print(f"Leiden clustering failed: {e}")

        return leiden_labels

    def mean_leiden_clusters():
        pass
    return (cluster_leiden,)


@app.cell(hide_code=True)
def _():
    mo.md("""## Run kNN Graph and Leiden Clustering""")
    return


@app.cell
def _(windowed_summary_data):
    with mo.persistent_cache(os.path.join(".mo.cache")):
        abm_knn_graph = build_knn(
        windowed_summary_data,
        n_neighbors=int(np.sqrt(len(windowed_summary_data))), # typical default value
        save_path=os.path.join('output','objects','abm','abm_full_features_knn.pkl')
    )
    return (abm_knn_graph,)


@app.cell
def _(abm_knn_graph, cluster_leiden):
    with mo.persistent_cache(os.path.join(".mo.cache")):
        abm_leiden_clusters = cluster_leiden(
            graph=abm_knn_graph,
            save_path=os.path.join('output','objects','abm', 'abm_full_features_leiden_partition.gml')
        )
    return (abm_leiden_clusters,)


@app.cell
def _(abm_leiden_clusters, abm_umap_embedding):
    def _():
        # plot the result
        fig = plt.figure()
        ax = fig.add_subplot()

        scat = ax.scatter(
            abm_umap_embedding[:, 0],
            abm_umap_embedding[:, 1],
            c=abm_leiden_clusters,
            cmap='rainbow',
            marker='.',
            alpha=0.6,

        )
        plt.colorbar(scat, label='Leiden Cluster')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP of ABM Windows from Intersection of Features with MIBI Dataset\nLeiden Clustering Results')
        plt.tight_layout()
        plt.show()
        plt.close()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Hieararchical Clustering to Aggregate Leiden Clusters""")
    return


@app.cell
def _(abm_leiden_clusters, feat_cols, windowed_summary_data):
    # calculate cluster means
    window_column_names = add_suffix_to_repeats(np.tile(feat_cols, 50))
    windowed_summary_data2 = pd.DataFrame(
        windowed_summary_data.copy(), 
        columns=window_column_names
    )
    windowed_summary_data2['leiden_label'] = abm_leiden_clusters
    leiden_cluster_means = windowed_summary_data2.groupby('leiden_label').mean()
    return (leiden_cluster_means,)


@app.cell
def _(leiden_cluster_means):
    # save the leiden cluster means to use for visualization
    leiden_cluster_means.to_csv(
        os.path.join(
            "data", 
            "abm", 
            "processed", 
            "abm_leiden_cluster_means.csv"
        ),
        index=False
    )
    return


@app.cell
def _(leiden_cluster_means):
    # hierarchical clustering of the leiden cluster means in the full window feature space

    # Calculate the condensed distance matrix between Leiden cluster means
    # Using 'euclidean' distance
    distance_matrix_leiden = sp.spatial.distance.pdist(leiden_cluster_means.to_numpy(), metric='euclidean')

    # Perform hierarchical clustering using Ward's method
    linkage_matrix_leiden = sp.cluster.hierarchy.linkage(distance_matrix_leiden, method='ward')

    # save the distance and linkage matrices for additional visualization
    np.save(
        os.path.join(
            "output", 
            "objects", 
            "abm", 
            "abm_full_features_leiden_clusters_distance_matrix.npy"
        ),
        distance_matrix_leiden
    )

    np.save(
        os.path.join(
            "output", 
            "objects", 
            "abm", 
            "abm_full_features_leiden_clusters_linkage_matrix.npy"
        ),
        linkage_matrix_leiden
    )
    return (linkage_matrix_leiden,)


@app.cell
def _(leiden_cluster_means, linkage_matrix_leiden):
    # plot the dendrogram
    def _():
        # Plot the dendrogram of Leiden cluster means
        with plt.style.context('forest_and_sky_academic'):
            plt.figure(figsize=(15, 7))
            plt.title('Hierarchical Clustering Dendrogram of Leiden Cluster Means')
            plt.xlabel('Leiden Cluster Index or Cluster Size')
            plt.ylabel('Distance')
            sp.cluster.hierarchy.dendrogram(
                linkage_matrix_leiden,
                labels=leiden_cluster_means.index.tolist(), # Use Leiden cluster IDs as labels
                truncate_mode='lastp',  # Show only the last p merged clusters
                p=30,                   # Show the last 30 merges (adjust as needed)
                leaf_rotation=90.,
                leaf_font_size=8.,
                show_contracted=True,   # To show counts of leaves in brackets
            )
            # Display the plot
            plt.gca()
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(linkage_matrix_leiden):
    hac_cluster_labels = sp.cluster.hierarchy.fcluster(linkage_matrix_leiden, 6, criterion='maxclust')
    return (hac_cluster_labels,)


@app.cell
def _(leiden_cluster_means, linkage_matrix_leiden):
    def compute_wcss(data, linkage_matrix, max_clusters=10):
        """
        Compute Within-Cluster Sum of Squares (WCSS) for different cluster counts.

        Parameters:
            data (ndarray): Original dataset (n_samples x n_features)
            linkage_matrix (ndarray): Linkage matrix from scipy.cluster.hierarchy.linkage
            max_clusters (int): Maximum number of clusters to evaluate

        Returns:
            dict: {k: WCSS value}
        """
        wcss_values = {}
        for k in range(1, max_clusters + 1):
            # Assign cluster labels
            labels = sp.cluster.hierarchy.fcluster(linkage_matrix, k, criterion='maxclust')
            wcss = 0.0
            for cluster_id in np.unique(labels):
                cluster_points = data[labels == cluster_id]
                centroid = np.mean(cluster_points, axis=0)
                # Sum of squared distances to centroid
                wcss += np.sum(np.square(cluster_points - centroid))
            wcss_values[k] = wcss
        return wcss_values

    wcss = compute_wcss(
        data = leiden_cluster_means.to_numpy(),
        linkage_matrix=linkage_matrix_leiden,
    )
    return (wcss,)


@app.cell
def _(wcss):
    def _():
        cluster_numbers, wcssvals = zip(*wcss.items())
        wcss_deltas = np.diff(wcssvals)
        with plt.style.context('forest_and_sky_academic'):
            fig, ax = plt.subplots()
            ax.plot(cluster_numbers[:-1], wcss_deltas, c='black')
            ax.scatter(cluster_numbers[:-1], wcss_deltas, marker='o', c='cornflowerblue')
            ax.axvline(x=6, c='red')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel(r'$\Delta$WCSS')
            ax.set_title('Change to the Within Cluster Sum of Squares')

            fig.savefig(
                os.path.join(
                    'output',
                    'figures', 
                    'abm', 
                    'state_space_identification',
                    'delta_wcss_hierarchical_clustering.png'
                ), 
                dpi=300, 
                bbox_inches='tight'
            )

            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(hac_cluster_labels, leiden_cluster_means):
    # 1. Create a mapping from Leiden cluster ID to hierarchical cluster label
    leiden_to_hierarchical_map = {leiden_id: hier_label 
                                  for leiden_id, hier_label 
                                  in zip(leiden_cluster_means.index, hac_cluster_labels)}
    return (leiden_to_hierarchical_map,)


@app.cell
def _(
    abm_leiden_clusters,
    abm_summaries_df_windowed,
    leiden_to_hierarchical_map,
):
    # 2. Assign hierarchical labels to original data points
    # make a copy of the windowed data
    abm_summaries_df_windowed2 = abm_summaries_df_windowed.copy()

    # assign the hac label as the state label
    abm_summaries_df_windowed2['state_label'] = pd.Series(abm_leiden_clusters).map(leiden_to_hierarchical_map) 
    return (abm_summaries_df_windowed2,)


@app.cell
def _(abm_summaries_df_windowed2):
    # save the data
    abm_summaries_df_windowed2.to_csv(
        os.path.join(
            'data',
            'abm', 
            'processed', 
            f'abm_windows_clustered_with_state_label_{get_today()}.csv'
        ), 
        index=False
    )
    return


@app.cell
def _(abm_summaries_df_windowed2, abm_umap_embedding):
    # check to see how the full-featured state space distributes over the common-features manifold
    def _():
        plot_df = pd.DataFrame({
            'x':abm_umap_embedding[:, 0],
            'y':abm_umap_embedding[:, 1],
            'state_label':abm_summaries_df_windowed2['state_label']
        })
        # plot the result
        #with plt.style.context('forest_and_sky_academic'):
        with plt.style.context('~/.matplotlib/stylelib/tme_research.mplstyle'):
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot()

            scat = sns.scatterplot(
                plot_df,
                x='x',
                y='y',
                hue='state_label',
                palette=theme_palette,
                ax=ax,
                marker='.',
                s=5,
                alpha=0.6,
                edgecolor=None
            )
            legend = plt.legend(bbox_to_anchor=(1.02, 0.5), title='State')
            enhance_legend_markers(ax=ax)
            ax.set_xlabel('UMAP 1') 
            ax.set_ylabel('UMAP 2')
            ax.set_title('ABM Tissue States')
            plt.tight_layout()

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures',
                    'abm', 
                    'abm_state_umap.png'
                ), 
                dpi=300
            )
            fig.savefig(
                os.path.join(
                    'output', 
                    'figures',
                    'abm', 
                    'abm_state_umap.svg'
                )
            )

            plt.show()
            plt.close()

            return plot_df
    abm_umap_embedding_state_labeled = _()
    return (abm_umap_embedding_state_labeled,)


@app.cell
def _(abm_umap_embedding_state_labeled):
    abm_umap_embedding_state_labeled.to_csv(
        os.path.join(
            'data', 
            'abm', 
            'processed', 
            'abm_umap_embedding_state_labeled.csv'
        ), 
        index=False
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
