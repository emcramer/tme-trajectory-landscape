import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy as sp
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    import os
    import pickle
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist

    from statannotations.Annotator import Annotator
    from itertools import combinations
    from scipy.stats import mannwhitneyu
    import statsmodels.stats.multitest as smm

    # set the random seed
    np.random.seed(42)

    # theming for figures
    theme_colors = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
    theme_palette = sns.color_palette(theme_colors) # Initialization code that runs before all other cells
    plot_style = 'forest_and_sky_academic'
    sns.palplot(theme_palette)
    plt.show()
    plt.close()

    plt.rcParams['svg.fonttype'] = 'none'
    return (
        Annotator,
        BoundaryNorm,
        GridSpec,
        ListedColormap,
        StandardScaler,
        mannwhitneyu,
        mo,
        np,
        os,
        pd,
        pickle,
        plot_style,
        plt,
        smm,
        sns,
        theme_colors,
        theme_palette,
    )


@app.cell
def _(os, pd):
    # load the data frame that has the window information with the parameter values
    window_parameter_info = pd.read_csv(
        os.path.join(
            'data', 
            'abm', 
            'processed', 
            'window_info_with_sim_parameters.csv'
        )
    )

    # load the data frame that has the state labels for the windowed/time lagged embedding
    window_data_clustered = pd.read_csv(
        os.path.join(
            'data', 
            'abm', 
            'processed', 
            'abm_windows_clustered_with_state_label_20251126.csv'
        )
    )
    return window_data_clustered, window_parameter_info


@app.cell(hide_code=True)
def _():
    parameter_name_map = {
        "cell_definitions.cell_definition[@name='malignant_epithelial_cell'].phenotype.mechanics.cell_adhesion_affinities.cell_adhesion_affinity[@name='malignant_epithelial_cell']": "Malignant Epithelial Cell Adhesion (Self)",
        "cell_definitions.cell_definition[@name='M0_macrophage'].motility.options.advanced_chemotaxis.chemotactic_sensitivities.chemotactic_sensitivity[@substrate='apoptotic_debris']": "M0 Macrophage Chemotaxis Apoptotic Debris",
        "cell_definitions.cell_definition[@name='M0_macrophage'].motility.options.advanced_chemotaxis.chemotactic_sensitivities.chemotactic_sensitivity[@substrate='necrotic_debris']": "M0 Macrophage Chemotaxis Necrotic Debris",
        "cell_definitions.cell_definition[@name='M0_macrophage'].cell_transformations.transformation_rates.transformation_rate[@name='M1_macrophage']": "M0 Macrophage Transform to M1",
        "cell_definitions.cell_definition[@name='M0_macrophage'].cell_transformations.transformation_rates.transformation_rate[@name='M2_macrophage']": "M0 Macrophage Transform to M2",
        "cell_definitions.cell_definition[@name='M1_macrophage'].motility.options.advanced_chemotaxis.chemotactic_sensitivities.chemotactic_sensitivity[@substrate='apoptotic_debris']": "M1 Macrophage Chemotaxis Apoptotic Debris",
        "cell_definitions.cell_definition[@name='M1_macrophage'].motility.options.advanced_chemotaxis.chemotactic_sensitivities.chemotactic_sensitivity[@substrate='necrotic_debris']": "M1 Macrophage Chemotaxis Necrotic Debris",
        "cell_definitions.cell_definition[@name='M1_macrophage'].cell_interactions.other_dead_phagocytosis_rate": "M1 Macrophage Phagocytosis Other Dead",
        "cell_definitions.cell_definition[@name='M1_macrophage'].cell_interactions.apoptotic_phagocytosis_rate": "M1 Macrophage Phagocytosis Apoptotic",
        "cell_definitions.cell_definition[@name='M1_macrophage'].cell_interactions.necrotic_phagocytosis_rate": "M1 Macrophage Phagocytosis Necrotic",
        "cell_definitions.cell_definition[@name='M2_macrophage'].motility.options.advanced_chemotaxis.chemotactic_sensitivities.chemotactic_sensitivity[@substrate='apoptotic_debris']": "M2 Macrophage Chemotaxis Apoptotic Debris",
        "cell_definitions.cell_definition[@name='M2_macrophage'].motility.options.advanced_chemotaxis.chemotactic_sensitivities.chemotactic_sensitivity[@substrate='necrotic_debris']": "M2 Macrophage Chemotaxis Necrotic Debris",
        "cell_definitions.cell_definition[@name='M2_macrophage'].cell_interactions.other_dead_phagocytosis_rate": "M2 Macrophage Phagocytosis Other Dead",
        "cell_definitions.cell_definition[@name='M2_macrophage'].cell_interactions.apoptotic_phagocytosis_rate": "M2 Macrophage Phagocytosis Apoptotic",
        "cell_definitions.cell_definition[@name='M2_macrophage'].cell_interactions.necrotic_phagocytosis_rate": "M2 Macrophage Phagocytosis Necrotic",
        "cell_definitions.cell_definition[@name='effector_T_cell'].cell_interactions.attack_rates.attack_rate[@name='malignant_epithelial_cell']": "Effector T Cell Attack Malignant Epithelial",
        "cell_definitions.cell_definition[@name='effector_T_cell'].cell_transformations.transformation_rates.transformation_rate[@name='exhausted_T_cell']": "Effector T Cell Transform to Exhausted T Cell",
        "cell_definitions.cell_definition[@name='effector_T_cell'].attack_duration": "Effector T Cell Attack Duration",
        "cell_definitions.cell_definition[@name='effector_T_cell'].attack_damage_rate": "Effector T Cell Attack Damage Rate",
        "cell_definitions.cell_definition[@name='exhausted_T_cell'].secretion.substrate[@name='pro-inflammatory_factor'].secretion_rate": "Exhausted T Cell Secretion Rate Pro-Inflammatory Factor",
        "cell_definitions.cell_definition[@name='exhausted_T_cell'].secretion.substrate[@name='pro-inflammatory_factor'].secretion_target": "Exhausted T Cell Secretion Target Pro-Inflammatory Factor",
        "initial_conditions.cell_positions.filename": "Initial Cell Positions",
        "max_time": 'Maximum Simulation Time',
        "param_set_id": "Parameter Set ID"
    }

    parameter_name_map_inv = {v: k for k, v in parameter_name_map.items()}
    return (parameter_name_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Visualize Parameter Distributions Across Clusters

    To compare parameters across clusters, we can use a heatmap or a set of violin plots. A heatmap is good for seeing patterns across many parameters and clusters simultaneously. Violin plots are better for seeing the distribution of a single parameter across clusters.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Heatmap of Average Parameter Values Across Clusters""")
    return


@app.cell
def _(mo, window_data_clustered):
    mo.ui.dataframe(window_data_clustered)
    return


@app.cell
def _(window_parameter_info):
    parameters_df = window_parameter_info.drop(['start_time_step', 'end_time_step', 'window_index_in_sim', 'avg_minutes_of_window', 'initial_conditions.cell_positions.folder', 'save.SVG.enable', 'param_set_id'], axis=1).drop_duplicates()
    parameter_names = list(parameters_df.columns.values)
    return parameter_names, parameters_df


@app.cell
def _(parameter_names, parameters_df, pd, window_data_clustered):
    avg_params_per_cluster = pd.merge(window_data_clustered, parameters_df, on='sim_id', how='left')
    avg_params_per_cluster = avg_params_per_cluster[ ['hierarchical_label'] + parameter_names].groupby('hierarchical_label').mean()
    return (avg_params_per_cluster,)


@app.cell
def _(avg_params_per_cluster, mo):
    mo.ui.dataframe(avg_params_per_cluster)
    return


@app.cell
def _(StandardScaler, avg_params_per_cluster, np, pd):
    # For better visualization, we might want to scale the average parameter values
    # so that differences are more apparent. We can use z-score scaling or min-max scaling.

    # Also: need to handle potential NaNs if std is 0 for a parameter (e.g., if it's constant across clusters)

    # for min-max scaling
    minmax_scaled_avg_params = avg_params_per_cluster.apply(lambda x: (x - x.min()) / x.max(), axis=0).fillna(0).drop(['sim_id'], axis=1) 

    # z-score normalization
    standardized_avg_params = pd.DataFrame(
        StandardScaler().fit_transform(avg_params_per_cluster),
        columns=avg_params_per_cluster.columns
    )

    # log normalized
    logged_avg_params = avg_params_per_cluster.drop(['sim_id'], axis=1).apply(np.log10, axis=1) 
    return logged_avg_params, minmax_scaled_avg_params, standardized_avg_params


@app.cell
def _(mo, pd, window_data_clustered):
    # calculate the average time of a window within a cluster
    time_df = pd.DataFrame({
        'state_label': window_data_clustered['hierarchical_label'],
        'avg_window_minutes': (window_data_clustered['end_time_step'] + window_data_clustered['start_time_step']) / 2
    }).groupby('state_label').agg({'avg_window_minutes':['mean', 'median', 'max', 'min', 'std']})
    mo.ui.dataframe(time_df)
    return (time_df,)


@app.cell
def _(
    BoundaryNorm,
    GridSpec,
    ListedColormap,
    np,
    os,
    parameter_name_map,
    plot_style,
    plt,
    sns,
):
    # Function to create the parameter heatmap
    def plot_parameter_heatmap(
        data,
        cluster_labels,
        times,
        cluster_colors,
        time_cmap,
        param_cmap,
        normtype='Z-score Standardized',
        savename=None
    ):
        with plt.style.context(plot_style):

            n_states = data.shape[0]

            fig = plt.figure(figsize=(14, 11))
            gs = GridSpec(
                nrows=2,#3,
                ncols=1,
                height_ratios=[20, 1],#, 1],  # main heatmap + two annotation rows
                hspace=0.03
            )

            # --- Main heatmap ---
            ax_heatmap = fig.add_subplot(gs[0])
            sns.heatmap(
                data.T,
                cmap=param_cmap,
                annot=False,
                linewidths=0.5,
                cbar_kws={'label': f'Average Parameter Value ({normtype})', 'shrink':0.5},
                ax=ax_heatmap
            )

            ax_heatmap.set_title('Heatmap of Average Parameter Values per State')
            ax_heatmap.set_ylabel('Model Input Parameter')
            ax_heatmap.set_xlabel('')

            ax_heatmap.set_yticks(np.arange(data.shape[1]) + 0.5)
            ax_heatmap.set_yticklabels(
                [parameter_name_map[p] for p in data.columns],
                rotation=0
            )

            # Remove x tick labels (will be shown only once at bottom)
            ax_heatmap.set_xticklabels([])
            ax_heatmap.tick_params(axis='x', bottom=False)

            # --- Cluster label row (categorical) ---
            ax_cluster = fig.add_subplot(gs[1], sharex=ax_heatmap)

            cluster_array = np.array(cluster_labels)[None, :]

            n_clusters = len(cluster_colors)
            norm = BoundaryNorm(
                boundaries=np.arange(-0.5, n_clusters + 0.5, 1),
                ncolors=n_clusters
            )

            cluster_cmap = ListedColormap(cluster_colors)

            sns.heatmap(
                cluster_array,
                cmap=cluster_cmap,
                norm=norm,
                #cbar=False,
                #palette=cluster_cmap,
                ax=ax_cluster,
                cbar=False
            )

            ax_cluster.set_yticks([])
            ax_cluster.set_ylabel('Cluster', rotation=0, labelpad=30, va='center')
            ax_cluster.tick_params(axis='x', bottom=False, labelbottom=False)
            ax_cluster.set_xlim(ax_heatmap.get_xlim())

            # --- Time row (continuous) ---
            #ax_time = fig.add_subplot(gs[2], sharex=ax_heatmap)

            #time_array = np.array(times)
            #sns.heatmap(
            #    time_array.T,
            #    cmap=time_cmap,
            #    cbar=True,
            #    cbar_kws={'label': 'Time'},
            #    ax=ax_time
            #)

            #ax_time.set_yticks([])
            #ax_time.set_ylabel('Time', rotation=0, labelpad=30, va='center')

            # Final x-axis labeling
            #ax_time.set_xlabel('State Label')
            #ax_time.set_xticks(np.arange(n_states) + 0.5)
            #ax_time.set_xticklabels(range(1, n_states + 1))

            plt.tight_layout()

            if savename is not None:
                fig.savefig(
                    os.path.join(
                        'output', 
                        'figures',
                        'abm', 
                        'parameter_figures',
                        savename+'.png'
                    ), 
                    dpi=300
                )
                fig.savefig(
                    os.path.join(
                        'output', 
                        'figures',
                        'abm', 
                        'parameter_figures',
                        savename+'.svg'
                    )
                )

            plt.show()
            plt.close()

    def plot_parameter_heatmap3(
        data,
        cluster_labels,
        cluster_colors,
        time_cmap,
        param_cmap,
        normtype='Z-score Standardized',
        savename=None
    ):
        with plt.style.context(plot_style):

            n_states = data.shape[0]

            fig = plt.figure(figsize=(14, 11))
            gs = GridSpec(
                nrows=2,
                ncols=2,
                height_ratios=[20, 1],   # main heatmap + annotation row
                width_ratios=[20, 1],    # heatmap column + colorbar column
                hspace=0.01,
                wspace=0.05
            )

            # --- Main heatmap (no seaborn colorbar) ---
            ax_heatmap = fig.add_subplot(gs[0, 0])
            hm = sns.heatmap(
                data.T,
                cmap=param_cmap,
                annot=False,
                linewidths=0.5,
                cbar=False,  # turn off seaborn's built-in colorbar
                ax=ax_heatmap
            )

            ax_heatmap.set_title('Heatmap of Average Parameter Values per State')
            ax_heatmap.set_ylabel('Model Input Parameter')
            ax_heatmap.set_xlabel('')

            ax_heatmap.set_yticks(np.arange(data.shape[1]) + 0.5)
            ax_heatmap.set_yticklabels(
                [parameter_name_map[p] for p in data.columns],
                rotation=0
            )

            ax_heatmap.set_xticklabels([])
            ax_heatmap.tick_params(axis='x', bottom=False)

            # --- Standalone colorbar in its own axis ---
            ax_cbar = fig.add_subplot(gs[0, 1])
            cbar = fig.colorbar(
                hm.collections[0],
                cax=ax_cbar
            )
            cbar.set_label(f'Average Parameter Value ({normtype})')

            # --- Cluster annotation row (still spans both columns) ---
            ax_cluster = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)

            cluster_array = np.array(cluster_labels)[None, :]

            n_clusters = len(cluster_colors)
            norm = BoundaryNorm(
                boundaries=np.arange(-0.5, n_clusters + 0.5, 1),
                ncolors=n_clusters
            )
            cluster_cmap = ListedColormap(cluster_colors)

            sns.heatmap(
                cluster_array,
                cmap=cluster_cmap,
                norm=norm,
                ax=ax_cluster,
                cbar=False
            )

            ax_cluster.set_yticks([])
            ax_cluster.set_ylabel('TME State', rotation=0, labelpad=30, va='center')
            ax_cluster.tick_params(axis='x', bottom=False, labelbottom=False)
            ax_cluster.set_xlim(ax_heatmap.get_xlim())

            ax_cluster.set_xticklabels(range(1, n_states + 1))
            #ax_cluster.text(np.arange(1, n_states + 1), [0.5]*6, np.arange(1, n_states+1))

            for i in range(6):
                ax_cluster.text(
                    i + 0.5, 0.5,  # center of each cell
                    str(i + 1),
                    ha='center', va='center',
                    fontsize=12,
                    color='white'
                )

            plt.tight_layout()

            # --- Save outputs if requested ---
            if savename is not None:
                outdir = os.path.join('output','figures','abm','parameter_figures')
                os.makedirs(outdir, exist_ok=True)
                fig.savefig(os.path.join(outdir, savename+'.png'), dpi=300)
                fig.savefig(os.path.join(outdir, savename+'.svg'))

            plt.show()
            plt.close(fig)
    return plot_parameter_heatmap, plot_parameter_heatmap3


@app.cell
def _(ListedColormap, avg_params_per_cluster, np, theme_colors):
    cluster_cmap = ListedColormap(theme_colors)
    cluster_order = [i for i in avg_params_per_cluster.index.values-1]
    cluster_to_idx = {c: i for i, c in enumerate(cluster_order)}
    cluster_idx = np.array([cluster_to_idx[c] for c in cluster_order])
    cluster_array = cluster_idx
    return (cluster_array,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Heatmap Visualizations""")
    return


@app.cell
def _(
    cluster_array,
    minmax_scaled_avg_params,
    np,
    os,
    pickle,
    plot_parameter_heatmap3,
    theme_colors,
):
    plot_parameter_heatmap3(
        data=minmax_scaled_avg_params, 
        cluster_labels=cluster_array,
        cluster_colors=theme_colors[0:np.max(cluster_array)+1],
        time_cmap='viridis',
        param_cmap='coolwarm',
        normtype='Min-Max Normalized',
        savename='parameter_heatmap_minmax_normalized'
    )

    # save the data out for plotting in a full figure
    with open(
            os.path.join(
                'output',
                'figures',
                'paper-figures',
                'figure-5-panels',
                'data',
                'fig_5d_data.pkl'
            ),
            'wb'
        ) as f:
            panel_data = {
                'data': minmax_scaled_avg_params, 
                'cluster_labels': cluster_array,
                'cluster_colors':theme_colors[0:np.max(cluster_array)+1]
            }
            pickle.dump(panel_data, file=f)
            print(f'Panel data saved to {f}')
    return


@app.cell
def _(
    cluster_array,
    np,
    plot_parameter_heatmap,
    standardized_avg_params,
    theme_colors,
    time_df,
):
    plot_parameter_heatmap(
        data=standardized_avg_params.drop(['sim_id'], axis=1), 
        cluster_labels=cluster_array,
        times=time_df.filter(like='max').to_numpy(), # get the *max* time to show which states are terminal states
        cluster_colors=theme_colors[0:np.max(cluster_array)+1],
        time_cmap='viridis',
        param_cmap='coolwarm',
        normtype='Z-Score Standardized'
    )
    return


@app.cell
def _(
    cluster_array,
    logged_avg_params,
    np,
    plot_parameter_heatmap,
    theme_colors,
    time_df,
):
    plot_parameter_heatmap(
        data=logged_avg_params, 
        cluster_labels=cluster_array,
        times=time_df.filter(like='max').to_numpy(), # get the *max* time to show which states are terminal states
        cluster_colors=theme_colors[0:np.max(cluster_array)+1],
        time_cmap='viridis',
        param_cmap='coolwarm',
        normtype=r'$log_{10}$ Normalized'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This graphic relies heavily on proprer normalization to derive proper insight because the scale and units for each parameter are different. We want to avoid comparing two dramatically different quantities. The Latin Hypercube assumes a uniform distribution over all of the bins along the range of the parameter's values. This means that the parameters follow a uniform distribution, and using z-score standardization changes the shape of that distribution, potentially creating the impression that certain parameter-emergent behavior relationships should exist when that is not the case. $Log_{10}$ normalization compresses the range of each parameter, but it does not scale all of the parameters appropriately. I think that min-max normalization is the most appropriate approach here because it forces the values of each parameter into the same domain while negating the differences in their units (by making them essentially unit-less).""")
    return


@app.cell
def panel6c(
    os,
    parameter_name_map,
    parameter_names,
    parameters_df,
    pd,
    pickle,
    plt,
    sns,
    theme_palette,
    window_data_clustered,
):
    def _():
        df = pd.merge(window_data_clustered, parameters_df, on='sim_id', how='left')[['hierarchical_label'] + parameter_names]
        df = df.rename(columns=parameter_name_map)
        df = df.rename(columns={'hierarchical_label':'State'})

        with plt.style.context('forest_and_sky_academic'):
            fig, ax = plt.subplots()
            sns.violinplot(
                df,
                x='State', 
                y="Effector T Cell Transform to Exhausted T Cell",
                hue='State',
                palette=theme_palette
            )
            ax.set_xlabel('State')
            ax.set_ylabel(r'$CD8^+ T_{EFF} \rightarrow CD8^+ T_{EX}$ Rate')
            ax.set_title(r'Distribution of $CD8^+ T_{EFF} \rightarrow CD8^+ T_{EX}$'+ '\nTransformation Rate Parameter Value by TME State')

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'parameter_figures', 
                    'parameter_violin_plots',
                   f"{'Effector T Cell Transform to Exhausted T Cell'.replace(' ', '_').lower()}.png"
                ), 
                dpi=300
            )
            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'parameter_figures', 
                    'parameter_violin_plots',
                   "teff_to_tex_rate_violin.svg"
                )
            )

            plt.show()
            plt.close()

        # save the data out for plotting in a full figure
        with open(
                os.path.join(
                    'output',
                    'figures',
                    'paper-figures',
                    'figure-5-panels',
                    'data',
                    'fig_5e_data.pkl'
                ),
                'wb'
            ) as f2:
                panel_data2 = {
                    'df': df, 
                    'palette': theme_palette
                }
                pickle.dump(panel_data2, file=f2)
                print(f'Panel data saved to {f2}')
        return df
    _()
    return


@app.cell
def _(
    Annotator,
    os,
    parameter_name_map,
    parameter_names,
    parameters_df,
    pd,
    plt,
    sns,
    theme_palette,
    window_data_clustered,
):
    def _():
        df = pd.merge(window_data_clustered, parameters_df, on='sim_id', how='left')[['hierarchical_label'] + parameter_names]
        df = df.rename(columns=parameter_name_map)
        df = df.rename(columns={'hierarchical_label':'State'})

        with plt.style.context('forest_and_sky_academic'):
            fig, ax = plt.subplots()
            sns.violinplot(
                data=df,
                x='State', 
                y="Effector T Cell Transform to Exhausted T Cell",
                hue='State',
                palette=theme_palette,
                ax=ax
            )

            # Define pairwise comparisons for all states present
            states = sorted(df['State'].unique())
            pairs = [(1,2), (1,3),(1,4),(1,5),(1,6)]#list(combinations(states, 2))

            # Initialize Annotator
            annotator = Annotator(ax, pairs, data=df, x='State', y="Effector T Cell Transform to Exhausted T Cell")

            # Configure statistical test + correction
            annotator.configure(
                test='Mann-Whitney',
                comparisons_correction='BH',  # Benjamini–Hochberg FDR
                text_format='star',
                loc='outside',
                verbose=2
            )

            # Run annotation
            annotator.apply_and_annotate()

            ax.set_xlabel('State')
            ax.set_ylabel(r'$CD8 T_{EFF} \rightarrow CD8 T_{EX}$ Rate')

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'parameter_figures', 
                    'parameter_violin_plots',
                    f"{'Effector T Cell Transform to Exhausted T Cell'.replace(' ', '_').lower()}.png"
                ), 
                dpi=300
            )

            fig.savefig(
                os.path.join(
                    'output', 
                    'figures', 
                    'abm', 
                    'parameter_figures', 
                    'parameter_violin_plots',
                    f"{'Effector T Cell Transform to Exhausted T Cell'.replace(' ', '_').lower()}.svg"
                )
            )

            plt.show()
            plt.close()

        return df

    _()
    return


@app.cell
def _(
    mannwhitneyu,
    os,
    parameter_name_map,
    parameter_names,
    parameters_df,
    pd,
    plt,
    smm,
    sns,
    theme_palette,
    window_data_clustered,
):
    def plot_state1_summary_significance_footnote():
        df = pd.merge(
            window_data_clustered,
            parameters_df,
            on='sim_id',
            how='left'
        )[['hierarchical_label'] + parameter_names]

        df = df.rename(columns=parameter_name_map)
        df = df.rename(columns={'hierarchical_label': 'State'})

        # Ensure categorical consistency
        df['State'] = df['State'].astype(str)

        y_col = "Effector T Cell Transform to Exhausted T Cell"

        reference_state = "1"
        other_states = sorted(s for s in df['State'].unique() if s != reference_state)

        # --- statistics ---
        pvals = []
        for s in other_states:
            g1 = df[df['State'] == reference_state][y_col].dropna().values
            g2 = df[df['State'] == s][y_col].dropna().values

            if len(g1) >= 2 and len(g2) >= 2:
                _, p = mannwhitneyu(g1, g2, alternative='two-sided')
                pvals.append(p)
            else:
                pvals.append(1.0)

        # BH–FDR correction
        _, pvals_fdr, _, _ = smm.multipletests(pvals, method='fdr_bh')

        # Conservative summary significance
        summary_p = pvals_fdr.max()

        if summary_p < 1e-3:
            p_text = "p < 1×10⁻³"
        else:
            p_text = f"p < {summary_p:.3g}"

        footnote_text = (
            f"Comparisons between TME State 1 and all other states "
            f"were significant ({p_text}, Mann–Whitney U test with "
            f"Benjamini–Hochberg correction)."
        )

        # --- plotting ---
        with plt.style.context('forest_and_sky_academic'):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))

            sns.violinplot(
                data=df,
                x='State',
                y=y_col,
                hue='State',
                palette=theme_palette,
                ax=ax
            )

            ax.set_xlabel('TME State')
            ax.set_ylabel(r'$CD8 T_{EFF} \rightarrow CD8 T_{EX}$ Rate')

            # Footnote-style annotation outside axes
            fig.text(
                0.01,
                0.01,
                footnote_text,
                ha='left',
                va='bottom',
                fontsize=9
            )

            # Adjust layout to preserve footnote space
            fig.subplots_adjust(bottom=0.18)

            fig.savefig(
                os.path.join(
                    'output',
                    'figures',
                    'abm',
                    'parameter_figures',
                    'parameter_violin_plots',
                    f"{y_col.replace(' ', '_').lower()}_state1_summary_footnote.png"
                ),
                dpi=300,
                bbox_inches='tight'
            )

            plt.show()
            plt.close()

        return summary_p

    plot_state1_summary_significance_footnote()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
