import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy as sp
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    import os
    import umap

    from utils import zscore_then_minmax_normalize, unnest_list, majority_vote

    # make a color palette
    theme_palette = sns.color_palette(['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c'])

    # loading the trajectories of each simulation 
    def _():
        with open(os.path.join('data', 'abm', "all_simulations_param_trajectory_20251022.pkl"), 'rb') as f:
            abm_trajectories = pickle.load(f)
        return abm_trajectories
    abm_trajectory_info = _()
    abm_trajectories = [k['trajectory'] for k in abm_trajectory_info]

    # loading the normalized time step data for each simulation
    abm_timestep_data = pd.read_csv(
        os.path.join(
            'data',
            'abm',
            'processed',
            "normed_scaled_features_local_time_20251123.csv")
    )

    # loading the window information from the time-delay embedding (includes the window info and the average feature values of the window)
    abm_window_info = pd.read_csv(
        os.path.join(
            'data',
            'abm',
            'processed',
            #'abm_window_info_and_averages.csv'
            'abm_windows_clustered_with_state_label_20251126.csv'
        )
    )
    abm_window_info.columns = abm_window_info.columns.str.replace("avg_", "")

    abm_time_delay_embedding = np.load(
        os.path.join(
            'data',
            'abm',
            'processed',
            'abm_time_delay_embedding.npy'
        )
    )


@app.cell(hide_code=True)
def _():
    mo.md("""# kNN Mapping for Biological Samples to ABM State Space""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""## Functions for Mapping Data Using kNN""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Calculating the Closest _k_ ABM Points to a Sample""")
    return


@app.function
def find_closest_k_rows(a, b, k, metric='cosine'):
    """
    For each row in matrix `a`, find the `k` closest rows in matrix `b`.

    Parameters:
    - a: numpy array of shape (m, d)
    - b: numpy array of shape (k, d)
    - k: number of closest rows to find
    - metric: the distance metric to use when making the distance matrix

    Returns:
    - indices: numpy array of shape (m, k) with the indices of the closest rows in `b`
    """
    distances = sp.spatial.distance.cdist(a, b, metric=metric)  # shape: (m, k)
    closest_indices = np.argsort(distances, axis=1)[:, :k]
    return closest_indices


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Calculating the Average of the _k_ Nearest Points""")
    return


@app.function
def average_closest_k_rows(a, b, k, metric='cosine'):
    """
    For each row in `a`, compute the average of the `k` closest rows in `b`.

    Parameters:
    - a: numpy array of shape (m, d)
    - b: numpy array of shape (n, d)
    - k: number of closest rows to average from `b`

    Returns:
    - avg_matrix: numpy array of shape (m, d), where each row is the average of `k` closest rows from `b`
    """
    closest_indices = find_closest_k_rows(a, b, k, metric=metric)
    m, d = a.shape
    avg_matrix = np.zeros((m, d))

    for i in range(m):
        nearest_rows = b[closest_indices[i]]  # shape: (n, d)
        avg_matrix[i] = np.mean(nearest_rows, axis=0)

    return avg_matrix


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: kNN State Assignment""")
    return


@app.function
def knn_assign_state_label(closest_indices, labels):
    """
    For each row in `a`, determines the state label

    Parameters:
    - a: numpy array of shape (m, d)
    - b: numpy array of shape (n, d)
    - k: number of closest rows to average from `b`

    Returns:
    - state_labels: a list of state labels assigned based on k-nearest neighbors
    """
    assigned_labels = []

    for i, idxs in enumerate(closest_indices):
        assigned_labels.append(majority_vote(labels[idxs]))
    return assigned_labels


@app.cell(hide_code=True)
def _():
    mo.md("""## Biological Matching""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Matching: Angelo Data""")
    return


@app.cell
def _():
    def load_and_process_mibi_data():
        # load the angelo data set for kNN mapping
        angelo_roi_spatial_summaries_df = pd.read_csv(
            os.path.join("data", "angelo", "angelo_data_rois_spatial_summaries_20251008.csv")
        )
    
        angelo_roi_spatial_summaries_df[angelo_roi_spatial_summaries_df.isna()] = 0
        #angelo_spatial_summaries_df = pd.read_csv("data/angelo/angelo_data_spatial_summaries_20250917.csv")
        angelo_roi_spatial_summaries_df.columns = angelo_roi_spatial_summaries_df.columns.str.replace("exCD8 T", "exhausted_T_cell")
        angelo_roi_spatial_summaries_df.columns = angelo_roi_spatial_summaries_df.columns.str.replace("effCD8 T", "effector_T_cell")
        angelo_roi_spatial_summaries_df.columns = angelo_roi_spatial_summaries_df.columns.str.replace("Tumor", "malignant_epithelial_cell")
        angelo_roi_spatial_summaries_df.columns = angelo_roi_spatial_summaries_df.columns.str.replace("group_degree_centrality", "degree_centrality")
        angelo_roi_spatial_summaries_df.columns = angelo_roi_spatial_summaries_df.columns.str.replace("group_closeness_centrality", "closeness_centrality")
        angelo_roi_spatial_summaries_df.columns = angelo_roi_spatial_summaries_df.columns.str.replace("group_clustering_coefficient", "average_clustering")
    
        selected_columns = angelo_roi_spatial_summaries_df.columns.str.contains('_T_cell|malignant_epithelial_cell')
    
        angelo_roi_spatial_summaries_df = angelo_roi_spatial_summaries_df.loc[:, selected_columns]
        angelo_roi_spatial_summaries_df = zscore_then_minmax_normalize(angelo_roi_spatial_summaries_df)
        return angelo_roi_spatial_summaries_df
    
    mibi_df = load_and_process_mibi_data()
    return (mibi_df,)


@app.cell
def _(mibi_df):
    def map_mibi_to_abm():
        common_columns = abm_window_info.columns.intersection(mibi_df.columns)

        closest_rows = find_closest_k_rows(
            mibi_df[common_columns].to_numpy(), 
            abm_window_info[common_columns].to_numpy(),
            k=10
        )
        assigned_labels = knn_assign_state_label(
            closest_indices=closest_rows,
            labels=abm_window_info['hierarchical_label'].values
        )
        return assigned_labels, common_columns
    
    knn_state_labels, common_columns = map_mibi_to_abm()
    return common_columns, knn_state_labels


@app.cell
def _(knn_state_labels):
    def _():
        with plt.style.context('forest_and_sky_academic'):
            pd.Series(knn_state_labels).value_counts().plot(kind='bar')
            plt.gca()
            plt.xlabel('State Label')
            plt.ylabel('Number of MIBI ROIs Assigned')
            plt.title('Mapping MIBI ROIs to ABM-derived TME States')
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(common_columns, mibi_df):
    # take the average of the closest k points in ABM space and project in 2d with umap
    def _():
        closest_k_avgs = average_closest_k_rows(
            mibi_df[common_columns].to_numpy(), 
            abm_window_info[common_columns].to_numpy(),
            k=10
        )
        print(closest_k_avgs)
        reducer = umap.UMAP(random_state=42)
        knearest_embedding = reducer.fit_transform(closest_k_avgs)
        fig, ax = plt.subplots()
        scat = ax.scatter(
            knearest_embedding[:, 0],
            knearest_embedding[:, 1],
            marker = '.',
            color = 'grey'
        )
        plt.show()
        plt.close()
        return closest_k_avgs
    _ = _()
    return


@app.cell
def _(knn_state_labels):
    def _():
        state_label_series = pd.Series(knn_state_labels, dtype='category').value_counts()
        # add +1 to the cluster label since clusters are 1-indexed
        df = pd.DataFrame({
            'state_label': state_label_series.index,
            'count': state_label_series.values
        })
        with plt.style.context("forest_and_sky_academic"):
            fig, ax = plt.subplots()
            sns.barplot(
                df,
                x='state_label',
                y='count',
                hue='state_label',
                palette=theme_palette,
                ax=ax 
            )
            ax.set_xlabel('State Label')
            ax.set_ylabel('Number of ROIs in State')
            ax.set_title('Distribution of TNBC ROIs Across ABM State Space\n(cosine distance)')
            plt.legend(title='State')
            plt.show()
            plt.close()


        # Load the anndata objects
        # filename for pickled ROI data object output from angelo_data_processing.py 
        angelo_data_fname = os.path.join(
            'data',
            'angelo',
            "angelo_anndata_rois.pkl"
        )
        with open(angelo_data_fname, "rb") as f:
            loaded_adatas = pickle.load(f)

        roi_mixing_score_classes = [adata.uns['mixing_score_class'] for adata in loaded_adatas]
        del roi_mixing_score_classes[1094]
        plot_df = pd.DataFrame({
            'state_label': pd.Series(knn_state_labels, dtype='category'),
            'mixing_score_class': roi_mixing_score_classes
        })
        #print(plot_df)
        plot_df['mixing_score_class'] = roi_mixing_score_classes
        with plt.style.context("forest_and_sky_academic"):
            combo_counts_data = plot_df.value_counts().plot.bar(
                title='Mixing Score - ABM State Association',
                color='grey'
            )
            plt.xlabel('Mixing Score Class & State Combination')
            plt.ylabel('Number of ROIs')
            ax = plt.gca()
            plt.xticks(rotation=45, ha='right')
            plt.show()
            plt.close()

    _()
    return


@app.cell
def _(knn_state_labels):
    def _():
        # Load the anndata objects
        angelo_data_fname = os.path.join(
            'data',
            'angelo',
            "angelo_anndata_rois.pkl"
        )
        with open(angelo_data_fname, "rb") as f:
            loaded_adatas = pickle.load(f)
        roi_mixing_score_classes = [adata.uns['mixing_score_class'] for adata in loaded_adatas]
        del roi_mixing_score_classes[1094]

        plot_df = pd.DataFrame({
            'state_label': pd.Series(knn_state_labels, dtype='category'),
            'mixing_score_class': roi_mixing_score_classes
        })

        # Create a pivot table for stacked bar chart
        stacked_data = plot_df.groupby(['state_label', 'mixing_score_class']).size().unstack(fill_value=0)

        # Define colors for mixing score classes
        mixing_colors = {
            'Mixed': plt.cm.tab10(0),
            'Compartmentalized': plt.cm.tab10(1), 
            'Cold': plt.cm.tab10(2)
        }

        # Reorder columns to match your legend order
        column_order = ['Mixed', 'Compartmentalized', 'Cold']
        stacked_data = stacked_data.reindex(columns=column_order, fill_value=0)

        with plt.style.context("forest_and_sky_academic"):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create stacked bar chart
            stacked_data.plot(
                kind='bar',
                stacked=True,
                ax=ax,
                color=[mixing_colors[col] for col in stacked_data.columns],
                width=0.7
            )

            ax.set_xlabel('State Label (Cluster)')
            ax.set_ylabel('Number of ROIs')
            ax.set_title('ROI Distribution by Cluster and Mixing Score Class')
            plt.xticks(rotation=0, ha='right')
            plt.legend(title='Mixing Score Class', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
