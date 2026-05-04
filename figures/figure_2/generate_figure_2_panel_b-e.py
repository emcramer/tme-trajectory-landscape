import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import seaborn as sns

# settings for plotting style
plot_style = "forest_and_sky_academic" if "forest_and_sky_academic" in plt.style.available else "default"
plt.rcParams['svg.fonttype'] = 'none'

# make a color palette for seaborn
color_hex_codes = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
theme_palette = sns.color_palette(color_hex_codes)

abm_umap_embedding = pd.read_csv("data/abm_umap_embedding_state_labeled.csv")
abm_summaries_df_windowed = pd.read_csv("data/abm_window_info_and_averages.csv")

def enhance_legend_markers(ax=None, marker_size=10, alpha=1.0):
    """
    Increase the marker size and opacity (alpha) of legend markers.

    Parameters:
    - ax: Matplotlib Axes object. If None, uses current axes.
    - marker_size: New size for the legend markers.
    - alpha: Opacity (0.0 to 1.0) for the legend markers.
    """
    if ax is None:
        ax = plt.gca()
    
    legend = ax.get_legend()
    if legend is None:
        print("No legend found on the given axes.")
        return

    for handle in legend.legend_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(marker_size)
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(alpha)

def generate_figure_2_panel_b():
    plot_df = pd.DataFrame({
        'x':abm_umap_embedding['x'],
        'y':abm_umap_embedding['y'],
        'state_label':abm_umap_embedding['state_label']
    })
    # plot the result
    with plt.style.context(plot_style):
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
                'figure_2_panel_b.png'
            ), 
            dpi=300
        )
        fig.savefig(
            os.path.join(
                'output', 
                'figure_2_panel_b.svg'
            )
        )

        plt.close()

generate_figure_2_panel_b()

avg_time_per_timestep = (abm_summaries_df_windowed['start_time_step']+abm_summaries_df_windowed['end_time_step'])/2

def generate_figure_2_panel_c():
    with plt.style.context(plot_style):
        # plot the result
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot()

        scat = ax.scatter(
            abm_umap_embedding['x'],
            abm_umap_embedding['y'],
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
                'figure_2_panel_c.png'
            ), 
            dpi=300
        )
        fig.savefig(
            os.path.join(
                'output', 
                'figure_2_panel_c.svg'
            )
        )
        plt.close()

generate_figure_2_panel_c()

##########
# Figure 2, Panel D
##########

def generate_figure_2_panel_d():
    """
    Recreate the UMAP figure with simulation trajectories overlaid.

    The UMAP embedding CSV and the window-info CSV are row-aligned: row i in the
    embedding corresponds to row i in the info file. We use this alignment to pull
    out the (x, y) UMAP coordinates for each simulation's windows, ordered by their
    index within the simulation, and draw them as colored trajectories on top of
    the full grey point cloud.
    """

    umap_df = abm_umap_embedding.copy()
    info_df = abm_summaries_df_windowed.copy()

    # Sanity check: the two files must be row-aligned for the index trick to work.
    assert len(umap_df) == len(info_df), "UMAP and info CSVs must have the same number of rows"

    # Attach sim_id and ordering column from the info file to the UMAP coordinates.
    umap_df = umap_df.copy()
    umap_df["sim_id"] = info_df["sim_id"].values
    umap_df["window_index_in_sim"] = info_df["window_index_in_sim"].values

    # --- Simulations to highlight (matches the reference figure) ---
    sims_to_plot = [
        (44,  "#8a2be2"),  # purple
        (100, "#3ee0b8"),  # teal/mint
        (28,  "#e63946"),  # red
    ]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # Background: all timepoints as grey points.
    ax.scatter(
        umap_df["x"], umap_df["y"],
        s=4, c="#888888", alpha=0.5,
        linewidths=0, label="All Timepoints",
    )

    # Foreground: trajectory for each highlighted simulation, ordered in time.
    for sim_id, color in sims_to_plot:
        traj = (
            umap_df[umap_df["sim_id"] == sim_id]
            .sort_values("window_index_in_sim")
        )
        ax.plot(
            traj["x"], traj["y"],
            color=color, linewidth=2.0,
            solid_capstyle="round", solid_joinstyle="round",
            label=f"Sim {sim_id}",
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("output/figure_2_panel_d.png", dpi=300, bbox_inches="tight")
    plt.savefig("output/figure_2_panel_d.svg", bbox_inches="tight")
    plt.close()

generate_figure_2_panel_d()

##########
# Figure 2, Panel E
##########

def generate_figure_2_panel_e(cmap='inferno'):
    features = [
        'malignant_epithelial_cell',
        'effector_T_cell',
        'exhausted_T_cell',
        'malignant_epithelial_cell_degree_centrality',
    ]

    with plt.style.context(plot_style):
        fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4))

        for ax, feature in zip(axes, features):
            scat = ax.scatter(
                abm_umap_embedding['x'],
                abm_umap_embedding['y'],
                c=abm_summaries_df_windowed[feature],
                marker='.',
                alpha=0.6,
                s=5,
                cmap=cmap
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(feature)
            fig.colorbar(scat, ax=ax, label='Relative Population (%)')

        plt.tight_layout()

        fig.savefig(
            os.path.join('output', 'figure_2_panel_e.png'),
            dpi=300
        )
        fig.savefig(
            os.path.join('output', 'figure_2_panel_e.svg')
        )

        plt.close()

generate_figure_2_panel_e()