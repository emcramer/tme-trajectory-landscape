import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os

# Style
if "forest_and_sky_academic" in plt.style.available:
    plt.style.use("forest_and_sky_academic")
else:
    plt.style.use("ggplot")

# make a color palette
theme_colors = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800', '#4a6d86', '#c28e9e', '#4f7942', '#d4a88c']
theme_palette = sns.color_palette(theme_colors)

def generate_figure_3_panel_c():
    bg_df = pd.read_csv("data/abm_umap_embedding_state_labeled.csv")
    bg_df['state_label'] = bg_df['state_label']-1

    fg_df = pd.read_csv("data/abm_all_patient_rois_mapped_umap_coordinate.csv")

    fig, ax = plt.subplots()

    # 1. Draw KDE and capture legend handles before they get replaced
    kde = sns.kdeplot(
        bg_df, 
        x='x',
        y='y',
        hue='state_label',
        fill=False,
        palette=sns.color_palette(theme_colors[0:6]),
        alpha=0.6,
        linewidth=1,
        ax=ax,
        zorder=1
    )
    # seaborn builds the kde legend directly (it doesn't label the contour
    # collections), so read handles/labels from the Legend object itself.
    kde_legend = ax.get_legend()
    assert kde_legend is not None, "kdeplot did not produce a legend"
    kde_handles = list(getattr(kde_legend, "legend_handles"))
    # state_label was decremented on load to drive the palette mapping; show 1–6 in the legend.
    kde_labels = [str(int(t.get_text()) + 1) for t in kde_legend.get_texts()]

    # 2. Draw scatter (no legend yet)
    np.random.seed(123)
    fg_df['x_jit'] = fg_df['x']+(np.random.rand(len(fg_df)))/5
    fg_df['y_jit'] = fg_df['y']+(np.random.rand(len(fg_df)))/5
    sources = sorted(fg_df['Source'].unique())
    source_palette = sns.color_palette('Set2', n_colors=len(sources))
    sns.scatterplot(
        fg_df,
        x='x_jit',
        y='y_jit',
        hue='Source',
        hue_order=sources,
        palette=source_palette,
        legend=False,
        ax=ax,
        marker='.',
        linewidth=0,
        edgecolor=None,
        zorder=5
    )

    ax.collections[-1].set_sizes([10])
    ax.collections[-1].set_alpha(0.8)
    ax.collections[-1].set_edgecolor(None)

    # Build proxy scatter handles for the merged legend (KDE legend already captured above).
    scatter_labels = sources
    scatter_handles = [
        Line2D([0], [0], marker='o', linestyle='', color=source_palette[i], markersize=8)
        for i in range(len(sources))
    ]

    # 4. Set axis labels
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    # 5. Build a merged legend (KDE first, scatter second)
    # Source legend (foreground data) — anchored to the upper right.
    source_legend = ax.legend(
        handles=scatter_handles,
        labels=scatter_labels,
        loc='upper right',
        title='Source',
    )
    ax.add_artist(source_legend)

    # State legend (background KDE) — placed beneath the source legend, then nudged
    # up by ~5 mm so it doesn't overlap features in the lower half of the plot.
    fig.canvas.draw()
    src_bbox = source_legend.get_window_extent().transformed(ax.transAxes.inverted())
    nudge_mm = 5
    nudge_axes_y = (nudge_mm / 25.4) * fig.dpi / ax.bbox.height
    ax.legend(
        handles=kde_handles,
        labels=kde_labels,
        loc='upper right',
        bbox_to_anchor=(src_bbox.xmax, src_bbox.ymin - 0.02 + nudge_axes_y),
        title='State',
    )

    plt.tight_layout()

    # 6. Save and display
    fig.savefig(
        os.path.join(
            f"output/figure_3_panel_c.png"
        ), 
        dpi=300
    )
    fig.savefig(
        os.path.join(
            f"output/figure_3_panel_c.svg"
        )
    )
    plt.close()

generate_figure_3_panel_c()