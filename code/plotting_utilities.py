import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy as sp
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.animation as animation
    from typing import Optional, Union, Any, Dict, List, Tuple
    import os
    import sys
    import colorsys
    import re
    import math

    USE_ADVANCED_PERCEPTUAL = False
    ADVANCED_NOTICE = ""

    try:
        # Try colour-science CIECAM02-UCS
        from colour import convert
        from colour.appearance import CAM02UCS_to_JMh, JMh_to_CAM02UCS
        USE_ADVANCED_PERCEPTUAL = True
        ADVANCED_NOTICE = "Using CIECAM02-UCS for perceptual differences."
    except Exception:
        try:
            # Fall back to colormath Lab
            from colormath.color_objects import sRGBColor, LabColor
            from colormath.color_conversions import convert_color
            USE_ADVANCED_PERCEPTUAL = True
            ADVANCED_NOTICE = "Using CIELab (colormath) for perceptual differences."
        except Exception:
            ADVANCED_NOTICE = (
                "Advanced perceptual libraries unavailable. "
                "Falling back to internal Lab approximation."
            )


@app.function
def contour_scatter_plot(
    background_data: pd.DataFrame,
    foreground_data: pd.DataFrame,
    contour_color_col: str,
    scatter_marker_col: str,
    ax: Optional[plt.Axes] = None,
    contour_cmap_params: Optional[Dict[str, Any]] = None,
    scatter_cmap_params: Optional[Dict[str, Any]] = None,
    scatter_marker_params: Optional[Dict[str, Any]] = None,
    min_contour_points: int = 10,
    n_contour_levels: int = 10,
    marker_size: int = 20,
    marker_alpha: float = 0.8,
    **scatter_kwargs
) -> Tuple[plt.Figure, plt.Axes]:

    # ------------------------------------------------------------------
    # 1. Setup Figure and Axes
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    contour_labels = sorted(background_data[contour_color_col].unique())
    scatter_labels = sorted(foreground_data[scatter_marker_col].unique())

    # ------------------------------------------------------------------
    # 2. Contour color handling (hex list supported)
    # ------------------------------------------------------------------
    if contour_cmap_params and "colors" in contour_cmap_params:
        contour_hex = contour_cmap_params["colors"]
        if len(contour_hex) < len(contour_labels):
            raise ValueError(
                "Number of contour colors must be >= number of contour labels."
            )
        contour_base_colors = [to_rgba(c, 1.0) for c in contour_hex]
    else:
        contour_base_colors = [
            to_rgba(plt.cm.tab10(i), 1.0)
            for i in range(len(contour_labels))
        ]

    alpha_min, alpha_max = (
        contour_cmap_params.get("alpha_range", (0.15, 0.7))
        if contour_cmap_params else (0.15, 0.7)
    )

    contourf_kwargs = {
        k: v for k, v in (contour_cmap_params or {}).items()
        if k not in {"colors", "alpha_range"}
    }

    # ------------------------------------------------------------------
    # 3. Scatter color handling (hex list + cycling + validation)
    # ------------------------------------------------------------------
    if scatter_cmap_params and "colors" in scatter_cmap_params:
        scatter_hex = scatter_cmap_params["colors"]
        if not scatter_hex:
            raise ValueError("scatter_cmap_params['colors'] must not be empty.")

        if len(scatter_hex) < len(scatter_labels):
            print(
                "Warning: Fewer scatter colors than labels; "
                "colors will be cycled in label-sorted order."
            )

        scatter_colors = [
            to_rgba(scatter_hex[i % len(scatter_hex)], 1.0)
            for i in range(len(scatter_labels))
        ]
    else:
        scatter_cmap = plt.cm.get_cmap(
            scatter_cmap_params.get("cmap", "tab10")
            if scatter_cmap_params else "tab10"
        )
        scatter_colors = [
            scatter_cmap(i % scatter_cmap.N)
            for i in range(len(scatter_labels))
        ]

    # ------------------------------------------------------------------
    # 4. Scatter marker handling
    # ------------------------------------------------------------------
    default_markers = ['o', 's', '^', 'd', 'p', '*', 'h', 'v', '<', '>']
    custom_markers = (
        scatter_marker_params.get("markers", default_markers)
        if scatter_marker_params else default_markers
    )

    # ------------------------------------------------------------------
    # 5. Plot Filled Contours
    # ------------------------------------------------------------------
    contour_handles = []

    for i, label in enumerate(contour_labels):
        label_idx = background_data[contour_color_col] == label
        points = background_data.loc[label_idx].iloc[:, :2].values

        if len(points) < min_contour_points:
            continue

        kde = sp.stats.gaussian_kde(points.T)

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        x_pad = 0.05 * (x_max - x_min)
        y_pad = 0.05 * (y_max - y_min)

        xx, yy = np.mgrid[
            (x_min - x_pad):(x_max + x_pad):100j,
            (y_min - y_pad):(y_max + y_pad):100j
        ]

        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        base_rgb = contour_base_colors[i]

        cmap = LinearSegmentedColormap.from_list(
            f"contour_{label}",
            [
                (*base_rgb[:3], alpha_min),
                (*base_rgb[:3], alpha_max),
            ],
            N=100
        )

        levels = np.linspace(density.min(), density.max(), n_contour_levels)

        ax.contourf(
            xx, yy, density,
            levels=levels[1:],
            cmap=cmap,
            antialiased=True,
            **contourf_kwargs
        )

        contour_handles.append(
            Patch(
                facecolor=base_rgb,
                edgecolor="none",
                alpha=alpha_max,
                label=str(label)
            )
        )

    # ------------------------------------------------------------------
    # 6. Plot Scatter Points
    # ------------------------------------------------------------------
    scatter_handles = []
    scatter_x = foreground_data.iloc[:, 0].values
    scatter_y = foreground_data.iloc[:, 1].values

    for i, label in enumerate(scatter_labels):
        idx = foreground_data[scatter_marker_col] == label
        marker = custom_markers[i % len(custom_markers)]
        color = scatter_colors[i]

        ax.scatter(
            scatter_x[idx],
            scatter_y[idx],
            marker=marker,
            color=color,
            s=marker_size,
            alpha=marker_alpha,
            label=str(label),
            **scatter_kwargs
        )

        scatter_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                markerfacecolor=color,
                color="w",
                markersize=10,
                label=str(label)
            )
        )

    # ------------------------------------------------------------------
    # 7. Legend
    # ------------------------------------------------------------------
    contour_title = Line2D([0], [0], color="none",
                           label=rf"$\bf{{{contour_color_col}}}$")
    scatter_title = Line2D([0], [0], color="none",
                           label=rf"$\bf{{{scatter_marker_col}}}$")
    spacer = Line2D([0], [0], color="none", label="")

    ax.legend(
        handles=(
            [contour_title] +
            contour_handles +
            [spacer, scatter_title] +
            scatter_handles
        ),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    # ------------------------------------------------------------------
    # 8. Labels
    # ------------------------------------------------------------------
    ax.set_xlabel(background_data.columns[0])
    ax.set_ylabel(background_data.columns[1])
    ax.set_title("Contour and Scatter Overlay Plot")

    return fig, ax


@app.cell
def _():
    # Ensure the function is defined in the script or imported
    # (Assuming the function definition above is available)

    def example():
        # --- Synthetic Data Generation ---
        np.random.seed(42)

        # Background Data (for contours)
        n_bg_points = 500
        bg_data = []
        states = ['Alpha', 'Beta', 'Gamma', 'Delta']
        for i, state in enumerate(states):
            mean_x = np.random.uniform(-5, 5)
            mean_y = np.random.uniform(-5, 5)
            cov = [[1.5, 0.5], [0.5, 1.0]]
            points = np.random.multivariate_normal([mean_x, mean_y], cov, size=100 + i * 50)
            df = pd.DataFrame(points, columns=['Dim_X', 'Dim_Y'])
            df['StateLabel'] = state
            bg_data.append(df)

        background_df = pd.concat(bg_data).reset_index(drop=True)

        # Foreground Data (for scatter)
        n_fg_points = 150
        fg_data = []
        scores = ['Low', 'Medium', 'High']
        # Select a subset of points for the foreground
        fg_indices = np.random.choice(background_df.index, n_fg_points, replace=False)
        foreground_df = background_df.loc[fg_indices].copy()

        # Randomly assign a 'Mixing Score Class' to foreground points
        foreground_df['MixingScoreClass'] = np.random.choice(scores, size=n_fg_points, p=[0.25, 0.5, 0.25])
        foreground_df = foreground_df[['Dim_X', 'Dim_Y', 'MixingScoreClass']].reset_index(drop=True)


        # --- Example 1: Basic Usage (New Figure and Axes) ---
        print("Running Example 1: Basic Usage...")
        fig1, ax1 = contour_scatter_plot(
            background_data=background_df,
            foreground_data=foreground_df,
            contour_color_col='StateLabel',
            scatter_marker_col='MixingScoreClass',
            # Optional: pass custom markers
            scatter_marker_params={'markers': ['X', 'D', 's']} 
        )
        fig1.suptitle("Example 1: Basic Plotting", fontsize=14)
        plt.show() # Uncomment to display the plot
        plt.close()

        # --- Example 2: Plotting onto an existing Axis ---
        print("Running Example 2: Plotting onto an existing Axis...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        fig2, ax2 = contour_scatter_plot(
            background_data=background_df,
            foreground_data=foreground_df,
            contour_color_col='StateLabel',
            scatter_marker_col='MixingScoreClass',
            ax=ax2, # Pass the existing axis
            # Override scatter color map
            scatter_cmap_params={'cmap': 'Set2'}
        )
        ax2.set_title("Example 2: Plotting onto Existing Axes with Custom Colormap", fontsize=12)
        ax2.axis('off') # Turn off the axis for a cleaner look
        # plt.show() # Uncomment to display the plot


        # Display all plots
        plt.show()
        plt.close()
        print("Example scripts finished running.")
    example()
    return


@app.function
def hex_to_rgb01(hex_color):
    """
    Convert hex string '#rrggbb' to RGB tuple in [0,1].
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


@app.function
def parse_rgb_string(s):
    """
    Parse 'rgb(r,g,b)' or 'r,g,b' where r,g,b may be 0–255 or 0–1 floats.
    """
    s = s.strip().lower()
    s = s.replace("rgb(", "").replace(")", "")
    parts = re.split(r"[, ]+", s)
    vals = [float(x) for x in parts if x != ""]
    if any(v > 1 for v in vals):  # assume 0–255
        vals = [v/255.0 for v in vals]
    return tuple(vals)


@app.function
def normalize_seed_color(seed_color):
    """
    Accept multiple formats:
        '#rrggbb'
        'rgb(r,g,b)'
        'r,g,b'
        (r,g,b) in [0,1] or [0,255]
        ('hsv', h, s, v)
        (h,s,v) if values look like HSV
    """
    if isinstance(seed_color, str):
        s = seed_color.strip()
        if s.startswith("#"):
            return hex_to_rgb01(s)
        elif "rgb" in s:
            return parse_rgb_string(s)
        else:
            # maybe "0.1, 0.4, 1.0"
            return parse_rgb_string(s)

    if isinstance(seed_color, tuple):
        if len(seed_color) == 4 and seed_color[0] == "hsv":
            _, h, ss, v = seed_color
            return colorsys.hsv_to_rgb(h, ss, v)
        if len(seed_color) == 3:
            r, g, b = seed_color
            # if they look like HSV (h in [0,1], s/v in [0,1])
            if 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1:
                # ambiguous: assume RGB unless explicitly flagged as HSV
                return (r, g, b)
            else:
                # assume 0–255 RGB
                return (r/255.0, g/255.0, b/255.0)

    raise ValueError("Unsupported seed_color format.")


@app.function
def rgb_to_lab_basic(rgb):
    """Internal fallback: approximate Lab conversion."""
    r,g,b = rgb

    # gamma expand
    def expand(c):
        return (c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4)
    rl, gl, bl = expand(r), expand(g), expand(b)

    # linear RGB → XYZ
    X = rl*0.4124 + gl*0.3576 + bl*0.1805
    Y = rl*0.2126 + gl*0.7152 + bl*0.0722
    Z = rl*0.0193 + gl*0.1192 + bl*0.9505

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    def f(t):
        return t**(1/3) if t>0.008856 else (7.787*t + 16/116)

    fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return (L,a,b)


@app.function
def perceptual_distance(rgb1, rgb2):
    """Compute perceptual distance using best available method."""
    if USE_ADVANCED_PERCEPTUAL:
        try:
            # If colour-science is installed, use its converters
            try:
                # convert RGB to CAM02-UCS J'a'b'
                # Normalising for colour-science: input RGB 0–1 as XYZ
                from colour import RGB_to_XYZ, XYZ_to_CAM02UCS
                XYZ1 = RGB_to_XYZ(rgb1)
                XYZ2 = RGB_to_XYZ(rgb2)
                Jab1 = XYZ_to_CAM02UCS(XYZ1)
                Jab2 = XYZ_to_CAM02UCS(XYZ2)
                return math.dist(Jab1, Jab2)
            except Exception:
                # fallback to colormath Lab
                c1 = sRGBColor(rgb1[0], rgb1[1], rgb1[2])
                c2 = sRGBColor(rgb2[0], rgb2[1], rgb2[2])
                lab1 = convert_color(c1, LabColor)
                lab2 = convert_color(c2, LabColor)
                return math.dist((lab1.lab_l,lab1.lab_a,lab1.lab_b),
                                 (lab2.lab_l,lab2.lab_a,lab2.lab_b))
        except Exception:
            pass

    # fallback internal approx Lab
    L1,a1,b1 = rgb_to_lab_basic(rgb1)
    L2,a2,b2 = rgb_to_lab_basic(rgb2)
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)


@app.function
def generate_opposed_palette(
    seed_color,
    n_colors,
    *,
    anchor_seed_hue=True,
    output_format="rgb",
    lightness_scale=1.0,
    saturation_scale=1.0,
    verbose=False
):
    """
    Generate a perceptually maximally-separated discrete color palette.

    Parameters
    ----------
    seed_color : various formats
    n_colors : int
    anchor_seed_hue : bool
        If True, first hue is exactly the seed hue.
    output_format : "rgb" or "hex"
    lightness_scale : float
        Multiply V (value) before conversion.
    saturation_scale : float
        Multiply S before conversion.
    verbose : bool
        Print dependency information.

    Returns
    -------
    list of RGB tuples or hex strings
    """

    if verbose:
        print(ADVANCED_NOTICE)

    # normalize input
    seed_rgb = normalize_seed_color(seed_color)
    seed_h, seed_s, seed_v = colorsys.rgb_to_hsv(*seed_rgb)

    # apply optional S/V scaling
    base_s = min(1.0, seed_s * saturation_scale)
    base_v = min(1.0, seed_v * lightness_scale)

    ###################################################################
    # Step 1: sample evenly around hue wheel
    ###################################################################
    if anchor_seed_hue:
        hues = [(seed_h + i/n_colors) % 1.0 for i in range(n_colors)]
    else:
        hues = [(i/n_colors) % 1.0 for i in range(n_colors)]

    rgb_candidates = [
        colorsys.hsv_to_rgb(h, base_s, base_v)
        for h in hues
    ]

    ###################################################################
    # Step 2: greedy farthest-neighbor traversal in perceptual space
    ###################################################################
    remaining = rgb_candidates.copy()

    # start at seed-most-similar
    start_idx = min(
        range(len(remaining)),
        key=lambda i: perceptual_distance(remaining[i], seed_rgb)
    )
    current = remaining.pop(start_idx)
    ordered = [current]

    while remaining:
        idx = max(
            range(len(remaining)),
            key=lambda i: perceptual_distance(current, remaining[i])
        )
        current = remaining.pop(idx)
        ordered.append(current)

    ###################################################################
    # Step 3: formatting output
    ###################################################################
    if output_format == "hex":
        def to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            )
        return [to_hex(c) for c in ordered]

    return ordered


@app.cell
def _():
    def colorpalette_example():
      palette = generate_opposed_palette("#1f77b4", 6)
      for i, c in enumerate(palette):
          plt.bar(i, 1, color=c)
      plt.show()
    colorpalette_example()
    return


@app.function
def save_animation_to_mp4(ani, fname="my_animation.mp4", **kwargs):
    opts = {
        'fps':20,
        'bitrate':1800,
    }
    opts.update(kwargs)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=opts['fps'], metadata=dict(artist='Me'), bitrate=opts['bitrate'])
    ani.save(fname, writer=writer)

@app.function
def stacked_bar_plot(
    df: pd.DataFrame,
    x_var: str,
    stack_var: str,
    normalize: bool = False,
    order_x: list[str] | None = None,
    order_stack: list[str] | None = None,
    missing: str = "drop",
    min_prop: float = 0.0,
    label_percent: bool = False,
    label_counts: bool = False,
    legend_pos: str = "right",
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Generate a stacked bar plot with enhancements and flexible legend/aesthetic control.

    Additional supported kwargs:
        legend_kwargs: dict of keyword arguments passed directly to ax.legend()
        (all other kwargs are passed to the DataFrame.plot() call)
    """

    # Protect against continuous x-variable misuse
    if pd.api.types.is_numeric_dtype(df[x_var]) and df[x_var].nunique() > 20:
        raise ValueError(f"x_var '{x_var}' appears continuous. Use a categorical/discrete variable.")

    # Handle missing values
    if missing == "drop":
        plot_df = df.dropna(subset=[x_var, stack_var])
    elif missing == "Unknown":
        plot_df = df.copy()
        plot_df[stack_var] = plot_df[stack_var].fillna("Unknown")
        plot_df[x_var] = plot_df[x_var].fillna("Unknown")
    else:
        raise ValueError("missing must be 'drop' or 'Unknown'")

    # Build contingency table (counts)
    table = pd.crosstab(plot_df[x_var], plot_df[stack_var])

    # Group rare categories by mean proportion threshold
    if min_prop > 0:
        prop_table = table.div(table.sum(axis=1), axis=0)
        low_cats = prop_table.mean(axis=0)[prop_table.mean(axis=0) < min_prop].index.tolist()
        if low_cats:
            table["Other"] = table[low_cats].sum(axis=1)
            table = table.drop(columns=low_cats)

    # Normalize bars if requested
    if normalize:
        table = table.div(table.sum(axis=1), axis=0)

    # Apply x-axis ordering
    if order_x is not None:
        table = table.reindex(order_x).dropna(axis=0, how="all")

    # Apply stacked category ordering
    if order_stack is not None:
        cols = [c for c in order_stack if c in table.columns]
        others = [c for c in table.columns if c not in cols]
        table = table[cols + others]

    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Extract legend kwargs if passed
    legend_kwargs = kwargs.pop("legend_kwargs", {})

    # Plot with forwarded aesthetics kwargs
    table.plot(kind="bar", stacked=True, ax=ax, **kwargs)

    # Correct y-axis ticks/labels
    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{v:.0%}" for v in ax.get_yticks()])
    else:
        max_height = table.sum(axis=1).max()
        ax.set_ylim(0, max_height * 1.05)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f"{int(v)}" for v in ax.get_yticks()])

    # Add segment annotations (counts and/or percentages)
    orig_counts = pd.crosstab(plot_df[x_var], plot_df[stack_var])
    for i, (_, row) in enumerate(orig_counts.reindex(table.index).iterrows()):
        cumulative = 0
        total = row.sum()
        for cat, count in row.items():
            if count > 0:
                height = table.iloc[i].get(cat, 0)
                midpoint = cumulative + height / 2

                if normalize and label_percent:
                    label = f"{height:.0%}"
                elif not normalize and label_counts:
                    label = f"{int(count)}"
                elif normalize and label_counts:
                    label = f"{int(count)}"
                elif not normalize and label_percent:
                    label = f"{count/total:.0%}"
                else:
                    label = None

                if label is not None:
                    ax.text(i, midpoint, label, ha="center", va="center", fontsize=9)

            cumulative += table.iloc[i].get(cat, 0)

    # Legend placement
    legend_pos = legend_pos.lower()
    if legend_pos == "right":
        ax.legend(title=stack_var, loc="center left", bbox_to_anchor=(1, 0.5), **legend_kwargs)
    elif legend_pos == "left":
        ax.legend(title=stack_var, loc="center right", bbox_to_anchor=(-0.2, 0.5), **legend_kwargs)
    elif legend_pos == "top":
        ax.legend(title=stack_var, loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=len(table.columns), **legend_kwargs)
    elif legend_pos == "bottom":
        ax.legend(title=stack_var, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(table.columns), **legend_kwargs)
    else:
        raise ValueError("legend_pos must be 'right', 'left', 'top', or 'bottom'")

    # Axis labels
    ax.set_xlabel(x_var)
    ax.set_ylabel("Proportion" if normalize else "Count")

    return fig, ax, table

@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
