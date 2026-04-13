import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy as sp
    import scipy.spatial.distance as distance
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pickle
    from scipy.stats import entropy
    from collections import Counter
    from utils import ensure_directory, majority_vote
    from plotting_utilities import contour_scatter_plot


    # Style
    if "forest_and_sky_academic" in plt.style.available:
        plt.style.use("forest_and_sky_academic")
    else:
        plt.style.use("ggplot")

    # make a color palette
    theme_colors = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
    theme_palette = sns.color_palette(theme_colors)
    sns.palplot(theme_palette)
    plt.show()


@app.cell
def _():
    mo.md("""# Combined MIBI and Wang ROI Mapping to ABM State Space""")
    return


@app.cell
def _():
    # Configuration
    abm_features_file = "data/abm/processed/abm_windows_clustered_with_state_label_20251126.csv"
    abm_umap_file = "output/objects/abm/abm_full_features_pc_n647_umap_embedding.npy"

    wang_features_file = "data/wang/processed/wang_roi_poisson_spatial_summaries_normalized_common_features.csv"
    wang_meta_file = "data/wang/processed/wang_roi_poisson_spatial_summaries_normalized.csv"

    mibi_features_file = "data/angelo/processed/mibi_roi_poisson_spatial_summaries_normalized_common_features.csv"
    mibi_meta_file = "data/angelo/processed/mibi_roi_poisson_spatial_summaries_normalized.csv"

    wang_obj_file = "data/wang/processed/wang_roi_anndatas_list.pkl"
    mibi_obj_file = "data/angelo/processed/fully_annotated_anndata_objects.pkl"
    mibi_roi_obj_file = "data/angelo/processed/mibi_roi_poisson_anndata.pkl"

    output_dir = "output/figures/combined_mapping_figures"

    k_slider = mo.ui.slider(start=1, stop=50, value=1, label="k Neighbors")

    mo.vstack([
        mo.md("### Configuration"),
        k_slider
    ])
    return (
        abm_features_file,
        abm_umap_file,
        k_slider,
        mibi_features_file,
        mibi_meta_file,
        mibi_roi_obj_file,
        output_dir,
        wang_features_file,
        wang_meta_file,
        wang_obj_file,
    )


@app.cell
def _(output_dir):
    ensure_directory(output_dir)
    return


@app.cell
def _():
    len(['M0_macrophage', 'M0_macrophage_average_clustering',
           'M0_macrophage_closeness_centrality', 'M0_macrophage_degree_centrality',
           'M1_macrophage', 'M1_macrophage_average_clustering',
           'M1_macrophage_closeness_centrality', 'M1_macrophage_degree_centrality',
           'M2_macrophage', 'M2_macrophage_average_clustering',
           'M2_macrophage_closeness_centrality', 'M2_macrophage_degree_centrality',
           'effector_T_cell', 'effector_T_cell_average_clustering',
           'effector_T_cell_closeness_centrality',
           'effector_T_cell_degree_centrality', 'exhausted_T_cell',
           'exhausted_T_cell_average_clustering',
           'exhausted_T_cell_closeness_centrality',
           'exhausted_T_cell_degree_centrality', 'graph_density',
           'im_M0_macrophage_M0_macrophage', 'im_M0_macrophage_M1_macrophage',
           'im_M0_macrophage_M2_macrophage', 'im_M0_macrophage_effector_T_cell',
           'im_M0_macrophage_exhausted_T_cell',
           'im_M0_macrophage_malignant_epithelial_cell',
           'im_M1_macrophage_M0_macrophage', 'im_M1_macrophage_M1_macrophage',
           'im_M1_macrophage_M2_macrophage', 'im_M1_macrophage_effector_T_cell',
           'im_M1_macrophage_exhausted_T_cell',
           'im_M1_macrophage_malignant_epithelial_cell',
           'im_M2_macrophage_M0_macrophage', 'im_M2_macrophage_M1_macrophage',
           'im_M2_macrophage_M2_macrophage', 'im_M2_macrophage_effector_T_cell',
           'im_M2_macrophage_exhausted_T_cell',
           'im_M2_macrophage_malignant_epithelial_cell',
           'im_effector_T_cell_M0_macrophage', 'im_effector_T_cell_M1_macrophage',
           'im_effector_T_cell_M2_macrophage',
           'im_effector_T_cell_effector_T_cell',
           'im_effector_T_cell_exhausted_T_cell',
           'im_effector_T_cell_malignant_epithelial_cell',
           'im_exhausted_T_cell_M0_macrophage',
           'im_exhausted_T_cell_M1_macrophage',
           'im_exhausted_T_cell_M2_macrophage',
           'im_exhausted_T_cell_effector_T_cell',
           'im_exhausted_T_cell_exhausted_T_cell',
           'im_exhausted_T_cell_malignant_epithelial_cell',
           'im_malignant_epithelial_cell_M0_macrophage',
           'im_malignant_epithelial_cell_M1_macrophage',
           'im_malignant_epithelial_cell_M2_macrophage',
           'im_malignant_epithelial_cell_effector_T_cell',
           'im_malignant_epithelial_cell_exhausted_T_cell',
           'im_malignant_epithelial_cell_malignant_epithelial_cell',
           'malignant_epithelial_cell',
           'malignant_epithelial_cell_average_clustering',
           'malignant_epithelial_cell_closeness_centrality',
           'malignant_epithelial_cell_degree_centrality', 'moranI'])
    return


@app.cell
def _(df_wang_meta):
    df_wang_meta
    return


@app.cell
def _(
    abm_features_file,
    abm_umap_file,
    mibi_features_file,
    mibi_meta_file,
    wang_features_file,
    wang_meta_file,
    wang_obj_file,
):
    # Load Feature Data

    # 1. ABM
    df_abm = pd.read_csv(abm_features_file)
    # Clean ABM columns (remove 'avg_' prefix if present to match others)
    df_abm.columns = df_abm.columns.str.replace("avg_", "")

    abm_umap = np.load(abm_umap_file)

    # 2. Wang
    df_wang = pd.read_csv(wang_features_file)
    df_wang_meta = pd.read_csv(wang_meta_file)

    with open(wang_obj_file, "rb") as f:
        wang_adatas = pickle.load(f)

    # 3. MIBI
    df_mibi = pd.read_csv(mibi_features_file)
    df_mibi_meta = pd.read_csv(mibi_meta_file)

    print(f"ABM Shape: {df_abm.shape}")
    print(f"Wang Features: {df_wang.shape}, Meta: {df_wang_meta.shape}, Adatas: {len(wang_adatas)}")
    print(f"MIBI Features: {df_mibi.shape}, Meta: {df_mibi_meta.shape}")
    return abm_umap, df_abm, df_mibi, df_wang, df_wang_meta, wang_adatas


@app.cell
def _(mibi_roi_obj_file):
    def _():
        with open(mibi_roi_obj_file, 'rb') as f:
            mibi_objects = pickle.load(f)
        return mibi_objects
    mibi_adata_objects = _()
    mibi_roi_sample_ids = [a.obs['SampleID'].iloc[0] for a in mibi_adata_objects]
    return (mibi_roi_sample_ids,)


@app.cell
def _(df_abm, df_mibi, df_wang):
    # Align Columns
    # Find intersection of features
    common_features = df_abm.columns.intersection(df_wang.columns).intersection(df_mibi.columns)

    # Filter numeric features only (exclude metadata columns if any slipped in)
    # Usually spatial summaries are all numeric, but 'roi_index' etc might exist.
    # We should exclude ID columns.
    exclude_cols = ['roi_index', 'PatientID', 'BiopsyPhase', 'hierarchical_label', 'window_id', 'simulation_id', 'Time', 'time_index']
    final_features = [c for c in common_features if c not in exclude_cols]

    mo.md(f"**Common Features for Mapping:** {len(final_features)}")
    return common_features, final_features


@app.cell
def _(common_features):
    common_features.values
    return


@app.function
# KNN Mapping Function (Vectorized/Optimized)

def map_new_points_to_umap(new_data, ref_data, ref_umap, ref_state_label, k=1):
    """
    Maps new_data points to the embedding space of ref_data using average UMAP coordinates 
    of the k nearest neighbors in feature space.

    Args:
        new_data (np.array): (N_new, D) Features of new points
        ref_data (np.array): (N_ref, D) Features of reference points
        ref_umap (np.array): (N_ref, 2) UMAP coordinates of reference points
        k (int): Number of neighbors

    Returns:
        np.array: (N_new, 2) Projected UMAP coordinates
    """
    n_new = new_data.shape[0]
    projected_coords = np.zeros((n_new, 2))
    knn_indices_long = []
    state_labels = []

    # Process in chunks to manage memory if N_new is large
    chunk_size = 100
    for i in range(0, n_new, chunk_size):
        end = min(i + chunk_size, n_new)
        chunk = new_data[i:end]

        # Calculate distances (chunk vs ref_data)
        dists = distance.cdist(chunk, ref_data, metric='cosine')

        # Find k nearest indices
        knn_indices = np.argsort(dists, axis=1)[:, :k]
        knn_indices_unravel = [i[0] for i in knn_indices]
        knn_indices_long.append(knn_indices_unravel)

        # Average UMAP coords
        for j, idxs in enumerate(knn_indices):
            projected_coords[i + j] = np.mean(ref_umap[idxs], axis=0)

        # get the state label for the matched ABM point
        state_labels.append(ref_state_label[knn_indices_unravel].values)

    # Flatten using list comprehension
    state_labels_flat = [item for arr in state_labels for item in arr]

    return projected_coords, knn_indices, state_labels_flat


@app.cell
def _(abm_umap, df_abm, df_mibi, df_wang, final_features, k_slider):
    # Perform Mapping

    k_val = k_slider.value

    # Prepare Arrays
    X_ref = df_abm[final_features].values
    X_wang = df_wang[final_features].values
    X_mibi = df_mibi[final_features].values

    print(f"Mapping {len(X_wang)} Wang ROIs and {len(X_mibi)} MIBI ROIs using k={k_val}...")

    # Map
    wang_proj, wang_indices, wang_state_labels = map_new_points_to_umap(X_wang, X_ref, abm_umap, df_abm['hierarchical_label'], k=k_val)
    mibi_proj, mibi_indices, mibi_state_labels = map_new_points_to_umap(X_mibi, X_ref, abm_umap, df_abm['hierarchical_label'], k=k_val)

    # Combine for plotting
    combined_proj = np.vstack([wang_proj, mibi_proj])

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': combined_proj[:, 0],
        'y': combined_proj[:, 1],
        'Source': ['Wang et al'] * len(wang_proj) + ['Keren et al'] * len(mibi_proj)
    })
    return k_val, mibi_state_labels, plot_df, wang_proj, wang_state_labels


@app.cell
def _(wang_state_labels):
    np.array(wang_state_labels).shape
    return


@app.cell
def _(mibi_state_labels):
    np.array(mibi_state_labels).shape
    return


@app.cell
def _(df_wang_meta, mibi_roi_sample_ids, mibi_state_labels, wang_state_labels):
    combined_state_labels = np.concat([np.array(wang_state_labels), np.array(mibi_state_labels)])
    combined_ids = np.concat([df_wang_meta['PatientID'].values, mibi_roi_sample_ids])

    combined_meta = pd.DataFrame(
        {
            'patient_id':combined_ids,
            'state':combined_state_labels
        }
    )
    combined_meta.to_csv(
         os.path.join(
            'data', 
            'abm', 
            'processed',
            'abm_all_patient_rois_mapped_meta_data.csv'
        ),
        index=False
    )
    return (combined_meta,)


@app.cell
def _(plot_df):
    plot_df.to_csv(
        os.path.join(
            'data', 
            'abm', 
            'processed',
            'abm_all_patient_rois_mapped_umap_coordinate.csv'
        ),
        index=False
    )
    return


@app.cell
def _(combined_meta, plot_df):
    # combining the meta data and the umap coordinates
    # useful for IEEE paper?
    plot_with_meta = pd.concat([combined_meta, plot_df], axis=1)
    plot_with_meta.to_csv(
        os.path.join(
            'data', 
            'abm', 
            'processed',
            'abm_all_patient_rois_mapped_umap_coordinate_plus_metadata.csv'
        ),
        index=False
    )
    return


@app.cell
def _(abm_umap, df_abm, wang_proj):
    def _():
        merger_abm = pd.DataFrame({
            'x': abm_umap[:, 0],
            'y': abm_umap[:, 1],
            'state_label': df_abm.hierarchical_label
        })
        merger_wang = pd.DataFrame({
            'x': wang_proj[:, 0], 
            'y': wang_proj[:, 1]
        })
        merged = pd.merge(merger_wang, merger_abm, how='left', on=['x', 'y']).reset_index()
        return mo.ui.dataframe(merged)
    _()    
    return


@app.cell
def _(abm_umap, df_abm, k_val, output_dir, plot_df):
    # Visualization

    # Background Data (ABM)
    bg_df = pd.DataFrame({
        'x': abm_umap[:, 0],
        'y': abm_umap[:, 1],
        'State': df_abm['hierarchical_label'].values # Use labels for contour coloring
    })

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # We want a contour plot of the ABM states in the background.
    # Then overlay the combined points with black '+' markers.

    # Use contour_scatter_plot utility
    # It usually handles bg contour + fg scatter.
    # We want the fg scatter to have specific marker properties.

    # Prepare 'Group' col for scatter (we want single group 'Data')
    fg_df = plot_df.copy()
    fg_df['Group'] = 'Patient ROIs'

    # Use the function to plot contours and scatter
    # We configure the scatter to be black '+' markers via the params arguments

    contour_scatter_plot(
        bg_df,
        fg_df,
        contour_color_col='State',
        scatter_marker_col='Group', 
        scatter_marker_params={'markers': ['+']},
        scatter_cmap_params={'colors': ['#000000']}, # Hex for black
        marker_size=40,
        marker_alpha=0.6,
        ax=ax
    )

    ax.set_title(f"Combined MIBI & Wang ROIs Mapped to ABM State Space (k={k_val})")

    # Save
    save_path = f"{output_dir}/combined_mapping_k{k_val}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    mo.vstack([
        mo.md(f"### Mapped Visualization"),
        fig,
        mo.md(f"Saved to `{save_path}`")
    ])
    return bg_df, fg_df


@app.cell
def _(bg_df, k_val, output_dir, plot_df):
    # Visualization: Is there a difference between the IMC and MIBI in where they associate?
    def _():
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        # We want a contour plot of the ABM states in the background.
        # Then overlay the combined points with black '+' markers.

        # Use contour_scatter_plot utility
        # It usually handles bg contour + fg scatter.
        # We want the fg scatter to have specific marker properties.

        # Prepare 'Group' col for scatter (we want single group 'Data')
        fg_df = plot_df.copy()
        fg_df['Group'] = 'Patient ROIs'

        # Use the function to plot contours and scatter
        # We configure the scatter to be black '+' markers via the params arguments

        contour_scatter_plot(
            bg_df,
            fg_df,
            contour_color_col='State',
            scatter_marker_col='Source', 
            scatter_marker_params={'markers': ['+', 'o']},
            scatter_cmap_params={'colors': ['#000000', '#ff0000']}, # Hex for black
            marker_size=40,
            marker_alpha=0.8,
            ax=ax
        )

        ax.set_title(f"Combined MIBI & Wang ROIs Mapped to ABM State Space (k={k_val})")

        # Save
        save_path = f"{output_dir}/combined_mapping_k{k_val}_dataset_colored.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        save_path = f"{output_dir}/combined_mapping_k{k_val}_dataset_colored.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return mo.vstack([
            mo.md(f"### Mapped Visualization"),
            fig,
            mo.md(f"Saved to `{save_path}`")
        ])
    _()
    return


@app.cell
def _(bg_df):
    bg_df
    return


@app.cell
def _(fg_df):
    fg_df
    return


@app.cell
def _(bg_df, fg_df, k_val, output_dir):
    def _():
        bg_df2 = bg_df.copy()
        bg_df2['State'] = bg_df2['State']-1
        fig, ax = plt.subplots()

        # 1. Draw KDE and capture legend handles before they get replaced
        kde = sns.kdeplot(
            bg_df2, 
            x='x',
            y='y',
            hue='State',
            fill=False,
            palette=sns.color_palette(theme_colors[0:6]),
            alpha=0.3,
            ax=ax
        )
        kde_handles, kde_labels = ax.get_legend_handles_labels()

        # 2. Draw scatter (no legend yet)
        fg_df['Size'] = 3
        sns.scatterplot(
            fg_df, 
            x='x',
            y='y',
            hue='Source',
            palette='Set2',
            legend=False,  # prevent overwrite
            ax=ax,
            marker='.',
            size='Size',
            linewidth=0,
            edgecolor=None
        )

        # 3. Capture scatter handles separately
        scatter_handles, scatter_labels = [], []
        for artist, label in zip(ax.collections, ax.get_legend().texts if ax.get_legend() else []):
            # (Alternative: re-call get_legend_handles_labels() on a temp axis if scatter legend is needed)
            pass
        # Simpler approach: re-generate scatter handles using get_legend_handles_labels() on a fresh axis:
        _, scatter_labels = sns.scatterplot(
            fg_df, x='x', y='y', hue='Source', palette='Set2'
        ).get_legend_handles_labels()
        scatter_handles, _ = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        ax.collections[-1].set_sizes([10])
        ax.collections[-1].set_alpha(0.2)
        ax.collections[-1].set_edgecolor(None)

        # 4. Set axis labels
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        # 5. Build a merged legend (KDE first, scatter second)
        ax.legend(
            handles=kde_handles + scatter_handles,
            labels=kde_labels + scatter_labels,
            loc='upper right',
            title=""
        )

        plt.tight_layout()

        # 6. Save and display
        fig.savefig(
            os.path.join(
                f"{output_dir}/combined_mapping_k{k_val}_dataset_colored_in_theme.png"
            ), 
            dpi=300
        )
        plt.show()
        plt.close()
    _()
    return


@app.cell
def _(bg_df, fg_df, k_val, output_dir):
    def _():
        bg_df2 = bg_df.copy()
        bg_df2['State'] = bg_df2['State']-1
        fig, ax = plt.subplots()

        # 1. Draw KDE and capture legend handles before they get replaced
        kde = sns.kdeplot(
            bg_df2, 
            x='x',
            y='y',
            hue='State',
            fill=False,
            palette=sns.color_palette(theme_colors[0:6]),
            alpha=0.6,
            linewidth=1,
            ax=ax,
            zorder=1
        )
        kde_handles, kde_labels = ax.get_legend_handles_labels()

        # 2. Draw scatter (no legend yet)
        np.random.seed(123)
        fg_df['x_jit'] = fg_df['x']+(np.random.rand(len(fg_df)))/5
        fg_df['y_jit'] = fg_df['y']+(np.random.rand(len(fg_df)))/5
        sns.scatterplot(
            fg_df, 
            x='x_jit',
            y='y_jit',
            hue='Source',
            palette='Set2',
            legend=False,  # prevent overwrite
            ax=ax,
            marker='.',
            size='Size',
            linewidth=0,
            edgecolor=None,
            zorder=5
        )

        # 3. Capture scatter handles separately
        scatter_handles, scatter_labels = [], []
        for artist, label in zip(ax.collections, ax.get_legend().texts if ax.get_legend() else []):
            # (Alternative: re-call get_legend_handles_labels() on a temp axis if scatter legend is needed)
            pass
        # Simpler approach: re-generate scatter handles using get_legend_handles_labels() on a fresh axis:
        _, scatter_labels = sns.scatterplot(
            fg_df, x='x', y='y', hue='Source', palette='Set2'
        ).get_legend_handles_labels()
        scatter_handles, _ = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        ax.collections[-1].set_sizes([10])
        ax.collections[-1].set_alpha(0.8)
        ax.collections[-1].set_edgecolor(None)

        # 4. Set axis labels
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        # 5. Build a merged legend (KDE first, scatter second)
        ax.legend(
            handles=kde_handles + scatter_handles,
            labels=kde_labels + ['NeoTRIP', 'MIBI'], #scatter_labels,
            loc='upper right',
            title=""
        )

        plt.tight_layout()

        # 6. Save and display
        fig.savefig(
            os.path.join(
                f"{output_dir}/combined_mapping_k{k_val}_dataset_colored_in_theme_jittered.png"
            ), 
            dpi=300
        )
        fig.savefig(
            os.path.join(
                f"{output_dir}/combined_mapping_k{k_val}_dataset_colored_in_theme_jittered.svg"
            )
        )
        plt.show()
        plt.close()
    _()
    return


@app.cell(hide_code=True)
def _(bg_df):
    def _():
        with plt.style.context('forest_and_sky_academic'):
            fig, ax = plt.subplots()
            unique_state_labels = sorted(bg_df.State.unique())

            # Define color palette
            theme_palette = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', 
                             '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']

            # Plot filled contours for each cluster
            for i, state in enumerate(unique_state_labels):
                state_idx = bg_df.State == state
                points = bg_df[state_idx]

                # Only plot contours if we have enough points
                if len(points) > 10:
                    # Calculate point density using KDE
                    kde = sp.stats.gaussian_kde(points.T)

                    # Create grid for contour plotting
                    x_min, x_max = points[:, 0].min(), points[:, 0].max()
                    y_min, y_max = points[:, 1].min(), points[:, 1].max()
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = np.reshape(kde(positions).T, xx.shape)

                    # Get cluster color
                    cluster_color = theme_palette[i % len(theme_palette)]

                    # Convert hex color to RGB for alpha blending
                    from matplotlib.colors import to_rgba
                    rgb = to_rgba(cluster_color)

                    # Create colormap with varying alpha for this cluster
                    from matplotlib.colors import LinearSegmentedColormap
                    colors = [(rgb[0], rgb[1], rgb[2], 0.15),
                             (rgb[0], rgb[1], rgb[2], 0.7)]
                    n_bins = 100
                    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

                    # Plot contourf starting from second level (skip lowest density)
                    levels = np.linspace(density.min(), density.max(), 11)
                    ax.contourf(xx, yy, density, levels=levels[1:], cmap=cmap, antialiased=True)

            # Create custom legend with two sections
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            # State Label section (contour colors)
            state_handles = [Patch(facecolor=theme_palette[i % len(theme_palette)], 
                                  edgecolor='none', label=state) 
                            for i, state in enumerate(unique_state_labels)]

            # Combine with section titles
            all_handles = (state_handles + 
                           [Line2D([0], [0], color='none', label='')])

            # Create legend
            #legend = plt.legend(handles=all_handles, ncol=1, bbox_to_anchor=(1.02, 0.8),
            #                  frameon=False)

            # Add section titles manually
            #texts = legend.get_texts()
            #texts[0].set_text('State Label\n' + texts[0].get_text())
            #texts[0].set_weight('bold')

            ax.axis('off')
            plt.show()
            plt.close()
    _()
    return


@app.function
def knn_predict_state(new_data, ref_data, ref_labels, k=5):
    """
    Predicts state labels for new_data based on k-nearest neighbors in ref_data.
    """
    n_new = new_data.shape[0]
    predicted_labels = []

    # Process in chunks
    chunk_size = 100
    for i in range(0, n_new, chunk_size):
        end = min(i + chunk_size, n_new)
        chunk = new_data[i:end]

        dists = distance.cdist(chunk, ref_data, metric='cosine')
        knn_indices = np.argsort(dists, axis=1)[:, :k]

        for idxs in knn_indices:
            # majority_vote expects list/array of labels
            neighbor_labels = ref_labels[idxs]
            predicted_labels.append(majority_vote(neighbor_labels))

    return np.array(predicted_labels)


@app.cell
def _(
    df_abm,
    df_mibi,
    df_wang,
    df_wang_meta,
    final_features,
    k_val,
    output_dir,
    wang_adatas,
):
    # --- Stacked Bar Plot of ROI State Composition ---

    # 1. Prepare Data
    _X_ref = df_abm[final_features].values
    _y_ref = df_abm['hierarchical_label'].values

    _X_wang = df_wang[final_features].values
    _X_mibi = df_mibi[final_features].values

    # 2. Predict States
    print("Predicting states for Wang ROIs...")
    wang_states = knn_predict_state(_X_wang, _X_ref, _y_ref, k=k_val)

    print("Predicting states for MIBI ROIs...")
    mibi_states = knn_predict_state(_X_mibi, _X_ref, _y_ref, k=k_val)

    # 3. Combine with Metadata
    # Wang
    wang_res = df_wang_meta[['PatientID', 'roi_index']].copy()

    # Extract BiopsyPhase from adatas
    _phases_list = []
    for _idx in wang_res['roi_index']:
        _adata = wang_adatas[_idx]
        _phases_list.append(_adata.uns.get('biopsy_phase', 'Unknown'))

    wang_res['BiopsyPhase'] = _phases_list
    wang_res['AssignedState'] = wang_states
    wang_res['Dataset'] = 'Wang'

    wang_res.to_csv(f"data/wang/processed/wang_roi_abm_state_assignment.csv")

    # MIBI
    #mibi_res = df_mibi_meta[['PatientID']].copy() if 'PatientID' in df_mibi_meta.columns else pd.DataFrame({'PatientID': df_mibi_meta.index}) # Fallback
    #mibi_res['BiopsyPhase'] = 'Baseline'
    #mibi_res['AssignedState'] = mibi_states
    #mibi_res['Dataset'] = 'MIBI'

    # Combine
    #combined_res = pd.concat([wang_res, mibi_res], ignore_index=True)
    combined_res = wang_res.copy()

    # 4. Calculate Percentages
    # Group by Dataset, Patient, Phase, State
    counts = combined_res.groupby(['Dataset', 'PatientID', 'BiopsyPhase', 'AssignedState']).size().reset_index(name='Count')
    totals = combined_res.groupby(['Dataset', 'PatientID', 'BiopsyPhase']).size().reset_index(name='Total')

    merged = pd.merge(counts, totals, on=['Dataset', 'PatientID', 'BiopsyPhase'])
    merged['Percentage'] = (merged['Count'] / merged['Total']) * 100

    # 5. Plotting
    phases = ['Baseline', 'On-treatment', 'Post-treatment']

    unique_states = sorted(combined_res['AssignedState'].unique())
    palette = sns.color_palette(theme_colors, n_colors=len(unique_states))
    color_map = dict(zip(unique_states, palette))

    _fig, _axes = plt.subplots(3, 1, figsize=(15, 12), sharey=True)

    for i, phase in enumerate(phases):
        _ax = _axes[i]
        phase_data = merged[merged['BiopsyPhase'] == phase]

        if phase_data.empty:
            _ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            _ax.set_title(f"{phase} (N=0)")
            continue

        # Pivot
        pivot_df = phase_data.pivot(index='PatientID', columns='AssignedState', values='Percentage').fillna(0)

        # Sort
        first_state = unique_states[0]
        if first_state in pivot_df.columns:
            pivot_df = pivot_df.sort_values(by=first_state, ascending=False)

        # Plot
        pivot_df.plot(kind='bar', stacked=True, ax=_ax, color=[color_map.get(c, 'grey') for c in pivot_df.columns], width=0.8)

        _ax.set_title(f"{phase} (N={len(pivot_df)})")
        _ax.set_ylabel("Percentage of ROIs")
        _ax.set_xlabel("Patient")
        _ax.legend(title='ABM State', bbox_to_anchor=(1.05, 1), loc='upper left')

        # X-ticks
        if len(pivot_df) > 50:
            _ax.set_xticks([]) 
        else:
            _ax.tick_params(axis='x', rotation=90, labelsize=8)

    plt.tight_layout()

    _save_path = f"{output_dir}/roi_composition_stacked_bar.png"
    plt.savefig(_save_path, dpi=300, bbox_inches='tight')

    mo.vstack([
        mo.md("### ROI State Composition by Patient"),
        _fig,
        mo.md(f"Saved to `{_save_path}`")
    ])
    return (combined_res,)


app._unparsable_cell(
    r"""
    #def analyze_mixing(vector):
        \"\"\"
        Calculates Entropy and Gini Impurity for a categorical vector.
        \"\"\"
        if len(vector) == 0:
            return 0.0, 0.0

        # 1. Get proportions (probabilities)
        counts = Counter(vector)
        total = len(vector)
        probs = np.array([count / total for count in counts.values()])

        # 2. Shannon Entropy (using natural log, base e)
        # Higher value = More mixed
        shannon = entropy(probs, base=np.e)

        # 3. Gini Impurity (1 - sum of squared probabilities)
        # Higher value = More mixed
        gini = 1 - np.sum(probs**2)

        return shannon, gini

    # --- Examples ---
    vectors = {
        \"Consistent\": [\"A\", \"A\", \"A\", \"A\", \"A\", \"B\"], # Mostly A
        \"Mixed\":      [\"A\", \"B\", \"A\", \"B\", \"A\", \"B\"], # Even split
        \"Very Mixed\": [\"A\", \"B\", \"C\", \"D\", \"E\", \"F\"]  # Chaos
    }

    print(f\"{'Vector Type':<12} | {'Entropy':<8} | {'Gini':<8}\")
    print(\"-\" * 35)

    for name, vec in vectors.items():
        ent, gi = analyze_mixing(vec)
        print(f\"{name:<12} | {ent:.4f}   | {gi:.4f}\")
    """,
    name="_"
)


@app.cell
def _(combined_res):
    combined_res
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
