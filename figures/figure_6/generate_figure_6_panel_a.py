import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import os
import pickle

# settings for plotting style
plot_style = "forest_and_sky_academic"
plt.rcParams['svg.fonttype'] = 'none'

# make a color palette for seaborn
color_hex_codes = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
theme_palette = sns.color_palette(color_hex_codes)

plt.style.use('~/.matplotlib/stylelib/tme_research.mplstyle')

abm_window_data = pd.read_csv(
    os.path.join(
        "data",
        "abm_windows_clustered_with_state_label.csv"
    )
)

grouped = abm_window_data.groupby('sim_id')

trajectories_nested = []
for name, group in grouped:
    trajectories_nested.append(group['hierarchical_label'].values)

def plot_figure_6_panel_a():
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
            'fig_6_panel_a_data.pkl'
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

    # saveout
    fig.savefig(
        fname=os.path.join(
                "output", 
                "figure_6_panel_a.svg"
            )
    )
    fig.savefig(
        fname=os.path.join(
                "output", 
                "figure_6_panel_a.png"
            ),
        dpi=300
    )

    plt.close()
    return 

if __name__ == "__main__":
    plot_figure_6_panel_a()