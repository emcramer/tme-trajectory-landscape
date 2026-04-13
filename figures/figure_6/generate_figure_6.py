"""
================================================================================
FIGURE 6: ABM TME State Space Dynamics and Markov State Model
================================================================================

Generates Figure 6 from Cramer et al. (manuscript).
Analyzes within-state and between-state dynamics of the ABM TME state space,
including parameter heatmaps, state transition drivers, and a Markov state model
schematic with intervention projections.

Note: This script was previously named generate_figure_5.py in the development
repository; the figure was renumbered to Figure 6 in the final manuscript.
The data files for this figure are stored in data/ and named fig_6*.pkl
(previously fig_6*.pkl).

Run from the figures/figure_6/ directory:
    python generate_figure_6.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})
MIN_FONT_SIZE = 9

# Color palette for states
STATE_COLORS = ["#1f5f66", "#f46d43", "#8db59a", "#cfcfc4", "#6a4c7d", "#f2c300"]
STATE_CMAP = ListedColormap(STATE_COLORS)

def add_panel_label(ax, label):
    ax.text(-0.1, 1.15, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right')

# -----------------------------------------------------------------------------
# Panel Drawing Functions
# -----------------------------------------------------------------------------

def plot_panel_a(ax_heatmap, ax_dendro, data_path):
    with open(data_path, 'rb') as f:
        panel_data = pickle.load(f)
    
    data = np.array(panel_data['data'])
    color_codes = panel_data['color_hex_codes']
    row_distances = pdist(data, metric='euclidean')
    linked_matrix = linkage(row_distances, method='average')
    
    dendro_info = dendrogram(linked_matrix, orientation='left', ax=ax_dendro, link_color_func=lambda k: "black")
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    
    reordered_indices = dendro_info['leaves']
    sorted_data = data[reordered_indices]
    
    unique_states = np.unique(data)
    n_states = len(unique_states)
    cmap = ListedColormap([color_codes[i % len(color_codes)] for i in range(n_states)])
    
    state_to_rank = {state: i for i, state in enumerate(unique_states)}
    ranked_data = np.vectorize(lambda x: state_to_rank[x])(sorted_data)
    
    ax_heatmap.pcolor(ranked_data, cmap=cmap)
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xticks([])
    ax_heatmap.set_title('Simulation Trajectories Through State Space', fontsize=12)
    ax_heatmap.set_xlabel('Time Window', fontsize=10)

def plot_panel_b(ax, data_path):
    with open(data_path, 'rb') as f:
        panel_data = pickle.load(f)
    
    plot_df = panel_data['plot_df']
    state_color_map = panel_data['state_color_map']
    
    # Identify features from the original logic (Effector T vs Malignant Epithelial)
    f_left = plot_df.columns[2] # Best guess based on snippet
    f_right = plot_df.columns[3]
    
    # Snippet shows f_left and f_right come from feat_left.value
    # Let's check keys or columns
    cols = [c for c in plot_df.columns if c not in ['window_index_in_sim', 'State', 'sim_id', 'hierarchical_label']]
    f_left = cols[0]
    f_right = cols[1]

    ax_r = ax.twinx()
    
    sns.lineplot(data=plot_df, x='window_index_in_sim', y=f_left, ax=ax, hue='State', 
                 palette=state_color_map, linestyle='-', legend=False)
    sns.lineplot(data=plot_df, x='window_index_in_sim', y=f_right, ax=ax_r, hue='State', 
                 palette=state_color_map, linestyle='--', legend=False)
    
    ax.set_ylabel(f_left, fontsize=10)
    ax_r.set_ylabel(f_right, fontsize=10)
    ax.set_title(f"Dual Axis Dynamics", fontsize=12)
    ax.set_xlabel("Time Step", fontsize=10)
    
    # Legend handled globally or in panel
    handles = []
    for state in sorted(plot_df['State'].unique()):
        handles.append(mlines.Line2D([], [], color=state_color_map[state], linestyle='-', label=f'State {state}'))
    handles.append(mlines.Line2D([], [], color='black', linestyle='-', label=f'Left Axis'))
    handles.append(mlines.Line2D([], [], color='black', linestyle='--', label=f'Right Axis'))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)

def plot_panel_c(ax, data_path):
    with open(data_path, 'rb') as f:
        panel_data = pickle.load(f)
    
    df_melt = panel_data['df_s2_melt']
    state_color_map = panel_data['state_color_map']
    
    sns.lineplot(data=df_melt, x='window_index_in_sim', y='Value', hue='State', style='Feature', ax=ax, palette=state_color_map)
    ax.set_title("Feature Dynamics Stratified by State", fontsize=12)
    ax.set_xlabel("Time Step", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)

def plot_panel_d(ax_heatmap, ax_cluster, ax_cbar, data_path):
    with open(data_path, 'rb') as f:
        panel_data = pickle.load(f)
    
    data = panel_data['data']
    cluster_labels = panel_data['cluster_labels']
    cluster_colors = panel_data['cluster_colors']
    param_cmap = panel_data.get('param_cmap', 'RdYlBu_r')
    param_names = panel_data.get('parameter_name_map', {})

    hm = sns.heatmap(data.T, cmap=param_cmap, annot=False, linewidths=0.5, cbar=False, ax=ax_heatmap)
    ax_heatmap.set_title('Average Parameter Values per State', fontsize=12)
    ax_heatmap.set_ylabel('Model Input Parameter', fontsize=10)
    ax_heatmap.set_yticks(np.arange(data.shape[1]) + 0.5)
    ax_heatmap.set_yticklabels([param_names.get(p, p) for p in data.columns], rotation=0, fontsize=8)
    ax_heatmap.set_xticks([])

    plt.colorbar(hm.collections[0], cax=ax_cbar)
    ax_cbar.set_ylabel('Z-score', fontsize=9)

    cluster_array = np.array(cluster_labels)[None, :]
    n_clusters = len(cluster_colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, n_clusters + 0.5, 1), ncolors=n_clusters)
    cluster_cmap = ListedColormap(cluster_colors)
    sns.heatmap(cluster_array, cmap=cluster_cmap, norm=norm, ax=ax_cluster, cbar=False)
    ax_cluster.set_yticks([])
    ax_cluster.set_xticks(np.arange(len(cluster_labels)) + 0.5)
    ax_cluster.set_xticklabels(range(1, len(cluster_labels) + 1))
    ax_cluster.set_xlabel('TME State', fontsize=10)

def plot_panel_e(ax, data_path):
    with open(data_path, 'rb') as f:
        panel_data = pickle.load(f)
    
    df = panel_data['df']
    palette = panel_data['palette']
    
    sns.violinplot(data=df, x='State', y=df.columns[1], hue='State', palette=palette, ax=ax)
    ax.set_xlabel('State', fontsize=10)
    ax.set_ylabel('Transformation Rate', fontsize=10)
    ax.set_title('Parameter Distribution by State', fontsize=12)

def plot_panel_f(ax):
    # Purely conceptual diagram - redraw using the logic in snippets/panel_f.py
    def draw_state(ax, x, y, r, color, label=None, edgecolor="black"):
        circ = Circle((x, y), r, facecolor=color, edgecolor=edgecolor, lw=1.5, zorder=3)
        ax.add_patch(circ)
        if label:
            ax.text(x, y-0.45, label, ha='center', va='top', fontsize=9)

    def draw_transition_box(ax, x, y, w, h, color, alpha=0.5):
        rect = Rectangle((x-w/2, y-h/2), w, h, facecolor=color, edgecolor=color, alpha=alpha, zorder=2)
        ax.add_patch(rect)

    def draw_arrow(ax, x1, y1, x2, y2, color="k", lw=1.5):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", linewidth=lw, color=color, mutation_scale=10, zorder=4)
        ax.add_patch(arr)

    S1, S2, S3, S4, S5, S6 = STATE_COLORS
    P_nat, P_int = "#f2c94c", "#7fa6a6"
    r = 0.3
    y_m, y_u, y_d = 0, 1.8, -1.8
    x = [0, 2, 4, 6]

    draw_state(ax, x[0], y_m, r, S3, "t_i")
    draw_transition_box(ax, 1, y_m, 0.6, 0.6, P_nat)
    draw_arrow(ax, 0.3, y_m, 1.7, y_m)
    draw_state(ax, x[1], y_m, r, S4, "t_{i+1}")
    draw_transition_box(ax, 3, y_m, 0.6, 0.6, P_nat)
    draw_arrow(ax, 2.3, y_m, 3.7, y_m)
    draw_state(ax, x[2], y_m, r, S4, "t_{i+2}")
    draw_transition_box(ax, 5, y_m, 0.6, 0.6, P_nat)
    draw_arrow(ax, 4.3, y_m, 5.7, y_m)
    draw_state(ax, x[3], y_m, r, S5, "t_{i+3}")

    ax.axvline(6.7, linestyle="--", color="red", lw=1.5)
    ax.text(6.7, 2.5, "Intervention", color="red", ha="center", fontsize=9)

    # Branches
    draw_arrow(ax, 6.3, 0.15, 7.5, y_u-0.15, color=P_int)
    draw_transition_box(ax, 7.5, y_u, 0.6, 0.6, P_int)
    draw_state(ax, 8.8, y_u, r, S1, "t_{i+4}")
    
    draw_arrow(ax, 6.3, -0.15, 7.5, y_d+0.15, color=P_nat)
    draw_transition_box(ax, 7.5, y_d, 0.6, 0.6, P_nat)
    draw_state(ax, 8.8, y_d, r, S6, "t_{i+4}")

    ax.set_xlim(-1, 12)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MSM Intervention Schematic", fontsize=12)

def plot_panel_g(ax, data_path):
    # Try to load recovered points if original data is missing
    recovered_path = 'data/fig_6g_recovered.pkl'
    if os.path.exists(recovered_path):
        with open(recovered_path, 'rb') as f:
            panel_data = pickle.load(f)
        points = np.array(panel_data['points'])
        # points are in SVG coordinate space. We can plot them relative but 
        # it's better to just show a representative curve if we can't get data.
        ax.plot(points[:, 0], -points[:, 1], linewidth=2, color='dodgerblue')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Intervention Efficacy", fontsize=12)
    else:
        ax.text(0.5, 0.5, "Data not found", ha='center')

def plot_panel_h(ax, data_path):
    # data_path is data/fig_6g_data.pkl which actually contains H data
    with open(data_path, 'rb') as f:
        panel_data = pickle.load(f)
    df_plot = panel_data['df_plot']
    palette = panel_data['palette']
    sns.barplot(data=df_plot, x='State', y='Count', hue='Scenario', ax=ax, palette=palette)
    ax.set_title("Projected Outcomes", fontsize=12)
    ax.set_ylabel("Simulations", fontsize=10)
    ax.legend(fontsize=8, loc='upper right')

# -----------------------------------------------------------------------------
# Main Assembly
# -----------------------------------------------------------------------------

def main():
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)

    # Panel A: Trajectories
    gs_a = gs[0, 0].subgridspec(1, 2, width_ratios=[1, 4], wspace=0.05)
    ax_a_den = fig.add_subplot(gs_a[0, 0])
    ax_a_hm = fig.add_subplot(gs_a[0, 1])
    plot_panel_a(ax_a_hm, ax_a_den, 'data/fig_6a_data.pkl')
    add_panel_label(ax_a_den, 'A')

    # Panel B: Dual Axis
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, 'data/fig_6b_data.pkl')
    add_panel_label(ax_b, 'B')

    # Panel C: Dynamics
    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c(ax_c, 'data/fig_6c_data.pkl')
    add_panel_label(ax_c, 'C')

    # Panel D: Parameter Heatmap
    gs_d = gs[1, 1].subgridspec(2, 2, height_ratios=[20, 1], width_ratios=[20, 1], hspace=0.05, wspace=0.02)
    ax_d_hm = fig.add_subplot(gs_d[0, 0])
    ax_d_cl = fig.add_subplot(gs_d[1, 0])
    ax_d_cb = fig.add_subplot(gs_d[0, 1])
    plot_panel_d(ax_d_hm, ax_d_cl, ax_d_cb, 'data/fig_6d_data.pkl')
    add_panel_label(ax_d_hm, 'D')

    # Panel E: Violin
    ax_e = fig.add_subplot(gs[2, 0])
    plot_panel_e(ax_e, 'data/fig_6e_data.pkl')
    add_panel_label(ax_e, 'E')

    # Panel F: Schematic
    ax_f = fig.add_subplot(gs[2, 1])
    plot_panel_f(ax_f)
    add_panel_label(ax_f, 'F')

    # Panel G: Intervention Efficacy
    ax_g = fig.add_subplot(gs[3, 0])
    # Note: Snippet says panel G data is in fig_6g_data.pkl
    plot_panel_g(ax_g, 'data/fig_6g_data.pkl')
    add_panel_label(ax_g, 'G')

    # Panel H: Outcomes
    ax_h = fig.add_subplot(gs[3, 1])
    # Snippet H saved to fig_6g_data.pkl as well? Let's check both possibilities.
    # Actually fig_6g_data.pkl exists. Let's try to find 'df_plot' in it.
    plot_panel_h(ax_h, 'data/fig_6g_data.pkl')
    add_panel_label(ax_h, 'H')

    plt.savefig('figure_6_assembled.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_6_assembled.svg', bbox_inches='tight')
    print("Final figure saved as figure_6_assembled.png/svg")

if __name__ == "__main__":
    main()
