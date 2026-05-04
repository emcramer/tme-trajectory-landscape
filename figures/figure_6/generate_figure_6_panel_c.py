import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle

from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# set the random seed
np.random.seed(42)

# theming for figures
theme_colors = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
theme_palette = sns.color_palette(theme_colors) # Initialization code that runs before all other cells
plot_style = 'forest_and_sky_academic' if 'forest_and_sky_academic' in plt.style.available else 'seaborn-whitegrid'

plt.rcParams['svg.fonttype'] = 'none'

# load the data frame that has the window information with the parameter values
window_parameter_info = pd.read_csv(
    os.path.join(
        'data', 
        'window_info_with_sim_parameters.csv'
    )
)

# load the data frame that has the state labels for the windowed/time lagged embedding
window_data_clustered = pd.read_csv(
    os.path.join(
        'data', 
        'abm_windows_clustered_with_state_label.csv'
    )
)

parameters_df = window_parameter_info.drop(['start_time_step', 'end_time_step', 'window_index_in_sim', 'avg_minutes_of_window', 'initial_conditions.cell_positions.folder', 'save.SVG.enable', 'param_set_id'], axis=1).drop_duplicates()
parameter_names = list(parameters_df.columns.values)

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

def gen_fig_6_panel_c():
    df = pd.merge(window_data_clustered, parameters_df, on='sim_id', how='left')[['hierarchical_label'] + parameter_names]
    df = df.rename(columns=parameter_name_map)
    df = df.rename(columns={'hierarchical_label':'State'})

    with plt.style.context(plot_style):
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

        y_col = "Effector T Cell Transform to Exhausted T Cell"
        states = sorted(df['State'].dropna().unique())
        pairs = list(combinations(states, 2))

        raw_pvals = []
        for a, b in pairs:
            va = df.loc[df['State'] == a, y_col].dropna().values
            vb = df.loc[df['State'] == b, y_col].dropna().values
            _, p = mannwhitneyu(va, vb, alternative='two-sided')
            raw_pvals.append(p)

        reject, adj_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='holm')

        all_sig = bool(reject.all())
        annotation = (
            f"All {len(pairs)} pairwise comparisons significant\n"
            f"(Mann–Whitney U, Holm-corrected p < 0.05)"
            if all_sig else
            f"{int(reject.sum())}/{len(pairs)} pairwise comparisons significant\n"
            f"(Mann–Whitney U, Holm-corrected p < 0.05)"
        )

        ax.text(
            0.98, 0.02, annotation,
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='lightgray'),
        )

        fig.savefig(
            os.path.join(
                'output', 
                "figure_6_panel_c.png"
            ), 
            dpi=300
        )
        fig.savefig(
            os.path.join(
                'output', 
               "figure_6_panel_c.svg"
            )
        )

        #plt.show()
        plt.close()

    # save the data out for plotting in a full figure
    with open(
            os.path.join(
                'output',
                'figure_6_panel_c_data.pkl'
            ),
            'wb'
        ) as f2:
            panel_data2 = {
                'df': df,
                'palette': theme_palette,
                'pairwise_stats': pd.DataFrame({
                    'state_a': [a for a, _ in pairs],
                    'state_b': [b for _, b in pairs],
                    'p_raw': raw_pvals,
                    'p_holm': adj_pvals,
                    'reject_at_0.05': reject,
                }),
                'all_significant': all_sig,
            }
            pickle.dump(panel_data2, file=f2)
            print(f'Panel data saved to {f2}')
    return df

if __name__ == "__main__":
    gen_fig_6_panel_c()