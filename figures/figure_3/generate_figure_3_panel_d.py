import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import pickle
    import tifffile
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path
    import os
    from skimage.restoration import rolling_ball
    from scipy.ndimage import median_filter

    # Anchor paths to the repo root via __file__ so the notebook works from any CWD.
    base_dir = Path(__file__).resolve().parents[3]

    data_dir = base_dir / "data"
    wang_processed = data_dir / "wang" / "processed"
    wang_raw = data_dir / "wang" / "NTPublic" / "data" / "raw"
    images_dir = wang_raw / "images"
    panel_dir = wang_raw / "panel"
    # Save outputs alongside this script in <figure_3>/output/.
    output_dir = Path(__file__).resolve().parent / "output"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        images_dir,
        median_filter,
        mo,
        mpatches,
        np,
        output_dir,
        panel_dir,
        pd,
        pickle,
        plt,
        rolling_ball,
        tifffile,
        wang_processed,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Wang ROI Raw Data Visualization (Background Subtraction + Denoising + Scale Bar)
    """)
    return


@app.cell
def _(mo, panel_dir, pd, pickle, wang_processed):
    # Load Metadata
    state_assignment_path = wang_processed / "wang_roi_abm_state_assignment.csv"
    roi_list_path = wang_processed / "wang_roi_anndatas_list.pkl"
    metal_order_path = panel_dir / "metalReadOrder.csv"
    panel_map_path = panel_dir / "NeoTripFinalPanelToIMCTools1.csv"

    # Load State Assignments
    df_assignments = pd.read_csv(state_assignment_path)

    # Load Panel Info
    try:
        metal_order_df = pd.read_csv(metal_order_path, header=None)
        metal_list = metal_order_df[0].tolist()
    except Exception as e:
        metal_list = []
        print(f"Error loading metal order: {e}")

    # Load Target Mapping
    panel_df = pd.read_csv(panel_map_path)
    # Map Metal Tag -> Target
    metal_to_target = dict(zip(panel_df['Metal Tag'], panel_df['Target']))

    # Create Channel -> Target list
    channel_targets = []
    for metal in metal_list:
        protein_target = metal_to_target.get(metal, metal)
        channel_targets.append(protein_target)

    # Load ROI List (Heavy)
    with open(roi_list_path, 'rb') as f:
        roi_anndatas = pickle.load(f)

    mo.md(f"**Loaded Data:**\n- {len(df_assignments)} assigned ROIs\n- {len(roi_anndatas)} total ROIs loaded\n- {len(channel_targets)} channels identified")
    return channel_targets, df_assignments, roi_anndatas


@app.cell
def _(df_assignments, mo):
    # UI: State Selection
    states = sorted(df_assignments['AssignedState'].unique())
    state_selector = mo.ui.dropdown(
        options=[str(s) for s in states],
        value=str(states[0]) if states else None,
        label="Select TME State"
    )
    state_selector
    return (state_selector,)


@app.cell
def _(df_assignments, state_selector):
    # Filter ROIs by State
    selected_state = int(state_selector.value)
    filtered_df = df_assignments[df_assignments['AssignedState'] == selected_state]

    # UI: ROI Selection
    roi_options = {
        f"ROI {row.roi_index} (Pt: {row.PatientID}, {row.BiopsyPhase})": row.roi_index
        for _, row in filtered_df.iterrows()
    }
    return roi_options, selected_state


@app.cell
def _(mo, roi_options):
    roi_selector = mo.ui.dropdown(
        options=roi_options.keys(),
        label="Select ROI",
        value=list(roi_options.keys())[0] if roi_options else None
    )
    roi_selector
    return (roi_selector,)


@app.cell
def _(channel_targets, mo):
    # UI: Channel Selection
    target_options = {t: i for i, t in enumerate(channel_targets)}

    # Pre-select common markers (canonical color assignment lives in the
    # visualization cell's marker_color_map so the legend stays stable
    # regardless of selection order).
    preferred = ['CD8', 'panKeratin_AE3', 'DNA1', 'CD68']
    defaults = [m for m in preferred if m in target_options]
    if not defaults:
        defaults = list(target_options.keys())[:4]

    channel_selector = mo.ui.multiselect(
        options=target_options,
        value=defaults,
        label="Select Channels (Max 4 recommended)",
    )
    channel_selector
    return channel_selector, target_options


@app.cell
def _(mo):
    # Image Processing Settings
    radius_slider = mo.ui.slider(start=0, stop=100, step=5, value=30, label="Rolling Ball Radius (0=Off)")
    median_slider = mo.ui.slider(start=0, stop=5, step=1, value=0, label="Median Filter Size (0=Off)")

    settings_ui = mo.vstack([
        mo.md("**Processing Settings**"),
        radius_slider,
        median_slider
    ])
    settings_ui
    return median_slider, radius_slider


@app.cell
def _(images_dir, mo, roi_anndatas, roi_options, roi_selector, tifffile):
    # Load Image Data
    if roi_selector.value is None:
        mo.stop(True, "Please select an ROI.")

    selected_roi_idx = roi_options[roi_selector.value]

    # Get Metadata
    roi_data = roi_anndatas[selected_roi_idx]
    image_id = roi_data.uns.get('ImageID', 'Unknown')
    roi_box = roi_data.uns.get('roi_box', None)

    image_filename = f"{image_id}FullStack.tiff"
    image_path = images_dir / image_filename

    status_msg = f"**Selected:** ROI {selected_roi_idx}, Image: {image_id}, Box: {roi_box}"

    if not image_path.exists():
        mo.stop(True, f"Image file not found: {image_path}")

    # Read Image & Metadata
    pixels_per_micron = 1.0 # Default

    try:
        with tifffile.TiffFile(image_path) as tif:
            image_stack = tif.asarray()

            # Metadata extraction for resolution
            try:
                page = tif.pages[0]
                tags = page.tags

                # Check for XResolution and Unit
                if 'XResolution' in tags:
                    x_res = tags['XResolution'].value
                    # x_res is usually a tuple (numerator, denominator)
                    if isinstance(x_res, tuple):
                        res_val = x_res[0] / x_res[1]
                    else:
                        res_val = x_res

                    # Check unit
                    # 1 = No absolute unit, 2 = Inch, 3 = Centimeter
                    unit = tags['ResolutionUnit'].value if 'ResolutionUnit' in tags else 0

                    if unit == 3: # Centimeter
                        # res_val is pixels per cm
                        # pixels per micron = res_val / 10000
                        pixels_per_micron = res_val / 10000.0
                    elif unit == 2: # Inch
                        # res_val is pixels per inch (DPI)
                        # pixels per micron = res_val / 25400.0
                        pixels_per_micron = res_val / 25400.0
                    else:
                        # Often in scientific imaging, ResolutionUnit is not standard or is implicit.
                        # However, for IMC (Hyperion), it's typically 1 um/pixel.
                        # Let's check ImageDescription for 'mpp' (microns per pixel)
                        pass

                # Check ImageDescription for OME-XML or similar
                if 'ImageDescription' in tags:
                    desc = tags['ImageDescription'].value
                    if isinstance(desc, str) and 'PhysicalSizeX' in desc:
                        # Simple parse for OME-XML
                        import re
                        match = re.search(r'PhysicalSizeX="([\d\.]+)"', desc)
                        if match:
                            mpp = float(match.group(1)) # Microns per pixel
                            if mpp > 0:
                                pixels_per_micron = 1.0 / mpp

            except Exception as e_meta:
                print(f"Metadata read warning: {e_meta}")

    except Exception as e:
        mo.stop(True, f"Error reading image: {e}")

    # Fallback/Sanity Check: If resolution seems off (too high/low), reset to 1.0
    # IMC is usually 1 um/pixel.
    if pixels_per_micron < 0.01 or pixels_per_micron > 100:
        pixels_per_micron = 1.0
        status_msg += " (Resolution unavailable, assuming 1 µm/pixel)"
    else:
        status_msg += f" (Resolution: {pixels_per_micron:.2f} px/µm)"

    # Crop
    if roi_box:
        y1, x1, y2, x2 = roi_box
        if y2 > image_stack.shape[1] or x2 > image_stack.shape[2]:
             status_msg += " (Warning: ROI box might exceed image dimensions)"

        cropped_stack = image_stack[:, int(y1):int(y2), int(x1):int(x2)]
    else:
        cropped_stack = image_stack

    mo.md(status_msg)
    return cropped_stack, pixels_per_micron, selected_roi_idx


@app.cell
def _(
    channel_selector,
    cropped_stack,
    median_filter,
    median_slider,
    mo,
    mpatches,
    np,
    pixels_per_micron,
    plt,
    radius_slider,
    rolling_ball,
    target_options,
):
    # Composite-only visualization.
    selected_targets = channel_selector.value
    if not selected_targets:
        mo.stop(True, "Select at least one channel.")

    radius = radius_slider.value
    median_size = median_slider.value

    all_keys = list(target_options.keys())
    markers = [all_keys[t] for t in selected_targets]

    # Pin canonical colors for the common-marker defaults so the legend stays
    # stable regardless of selection order; other markers fall back to the
    # positional palette.
    marker_color_map = {
        "CD8": "red",
        "panKeratin_AE3": "green",
        "DNA1": "blue",
        "CD68": "yellow",
    }
    fallback_colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    h, w = cropped_stack.shape[1], cropped_stack.shape[2]
    composite = np.zeros((h, w, 3))
    legend_patches = []

    for i, (idx, target) in enumerate(zip(selected_targets, markers)):
        raw_img = cropped_stack[idx, :, :]

        if median_size > 0:
            raw_img = median_filter(raw_img, size=median_size)

        if radius > 0:
            try:
                bg = rolling_ball(raw_img, radius=radius)
                raw_img = raw_img - bg
                raw_img = np.clip(raw_img, 0, None)
            except Exception:
                pass

        p1, p99 = np.percentile(raw_img, (1, 99))
        if p99 > p1:
            norm_img = np.clip((raw_img - p1) / (p99 - p1), 0, 1)
        else:
            norm_img = np.zeros_like(raw_img)

        color_name = marker_color_map.get(target, fallback_colors[i % len(fallback_colors)])
        legend_patches.append(mpatches.Patch(color=color_name, label=target))

        if color_name == "red":
            composite[:, :, 0] += norm_img
        elif color_name == "green":
            composite[:, :, 1] += norm_img
        elif color_name == "blue":
            composite[:, :, 2] += norm_img
        elif color_name == "yellow":
            composite[:, :, 0] += norm_img
            composite[:, :, 1] += norm_img
        elif color_name == "cyan":
            composite[:, :, 1] += norm_img
            composite[:, :, 2] += norm_img
        elif color_name == "magenta":
            composite[:, :, 0] += norm_img
            composite[:, :, 2] += norm_img

    composite = np.clip(composite, 0, 1)

    aspect_ratio = w / h
    fig_height = 6
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax.imshow(composite)
    ax.axis("off")
    ax.legend(handles=legend_patches, loc='upper right', fontsize='small', framealpha=0.5)

    # Scale bar — choose a sensible micron length given the FOV width.
    bar_length_um = 100
    if w / pixels_per_micron < 200:
        bar_length_um = 50
    if w / pixels_per_micron < 50:
        bar_length_um = 10

    bar_length_px = bar_length_um * pixels_per_micron
    margin = w * 0.05
    x0 = w - margin - bar_length_px
    y0 = h - margin
    bar_height = h * 0.02

    ax.add_patch(mpatches.Rectangle(
        (x0, y0 - bar_height), bar_length_px, bar_height, color='white'
    ))
    ax.text(
        x0 + bar_length_px/2, y0 - bar_height - (h*0.01),
        f"{bar_length_um} µm",
        color='white', ha='center', va='bottom', fontsize=8, fontweight='bold',
    )

    plot_output = mo.as_html(fig)
    plot_output
    return (fig,)

@app.cell
def _(
    channel_targets,
    images_dir,
    median_filter,
    mpatches,
    np,
    output_dir,
    plt,
    roi_anndatas,
    rolling_ball,
    tifffile,
):
    # 2x3 grid of representative composite images, one per TME state.
    # States 1-3 top row, 4-6 bottom row. Each panel gets a 2 mm border
    # in its TME state theme color.
    # Body wrapped in `def _():` so its locals stay private (marimo enforces
    # unique top-level names across cells, and many of these (composite,
    # legend_patches, ax, i, ...) are also used by the visualization cell).
    def _():
        state_to_image = {
            "TME State 1": "NTImg0659",
            "TME State 2": "NTImg1042",
            "TME State 3": "NTImg0945",
            "TME State 4": "NTImg0640",
            "TME State 5": "NTImg1705",
            "TME State 6": "NTImg0914",
        }
        theme_colors = {
            "TME State 1": "#1a535c",
            "TME State 2": "#ee6c4d",
            "TME State 3": "#84a98c",
            "TME State 4": "#b8b8a8",
            "TME State 5": "#6b4f7b",
            "TME State 6": "#e6b800",
        }

        grid_marker_color_map = {
            "CD8": "red",
            "panKeratin_AE3": "green",
            "DNA1": "blue",
            "CD68": "yellow",
        }
        grid_radius = 30
        grid_median = 2

        image_id_to_roi_box = {}
        for adata in roi_anndatas:
            img_id = adata.uns.get("ImageID")
            roi_box = adata.uns.get("roi_box")
            if img_id and roi_box is not None and img_id not in image_id_to_roi_box:
                image_id_to_roi_box[img_id] = roi_box

        def build_composite(image_id):
            roi_box = image_id_to_roi_box.get(image_id)
            image_path = images_dir / f"{image_id}FullStack.tiff"
            with tifffile.TiffFile(image_path) as tif:
                image_stack = tif.asarray()
            if roi_box is not None:
                y1, x1, y2, x2 = roi_box
                cropped = image_stack[:, int(y1):int(y2), int(x1):int(x2)]
            else:
                cropped = image_stack
            h, w = cropped.shape[1], cropped.shape[2]
            composite = np.zeros((h, w, 3))
            for marker, color_name in grid_marker_color_map.items():
                if marker not in channel_targets:
                    continue
                idx = channel_targets.index(marker)
                raw_img = cropped[idx, :, :]
                if grid_median > 0:
                    raw_img = median_filter(raw_img, size=grid_median)
                if grid_radius > 0:
                    try:
                        bg = rolling_ball(raw_img, radius=grid_radius)
                        raw_img = np.clip(raw_img - bg, 0, None)
                    except Exception:
                        pass
                p1, p99 = np.percentile(raw_img, (1, 99))
                if p99 > p1:
                    norm = np.clip((raw_img - p1) / (p99 - p1), 0, 1)
                else:
                    norm = np.zeros_like(raw_img)
                if color_name == "red":
                    composite[:, :, 0] += norm
                elif color_name == "green":
                    composite[:, :, 1] += norm
                elif color_name == "blue":
                    composite[:, :, 2] += norm
                elif color_name == "yellow":
                    composite[:, :, 0] += norm
                    composite[:, :, 1] += norm
            return np.clip(composite, 0, 1)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        # 2 mm in points (1 inch = 25.4 mm = 72 pt).
        border_pt = 2 * 72 / 25.4

        for i, state in enumerate(state_to_image):
            row, col = divmod(i, 3)
            ax = axes[row][col]
            try:
                composite = build_composite(state_to_image[state])
                ax.imshow(composite)
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed: {e}", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(border_pt)
                spine.set_color(theme_colors[state])
                spine.set_visible(True)
            ax.set_title(state, color=theme_colors[state])

        legend_patches = [
            mpatches.Patch(color=color, label=marker)
            for marker, color in grid_marker_color_map.items()
            if marker in channel_targets
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(legend_patches),
            bbox_to_anchor=(0.5, -0.02),
            fontsize="small",
            framealpha=0.5,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "figure_3_panel_d.png"
        svg_path = output_dir / "figure_3_panel_d.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        print(f"Saved {png_path}")

    _()
    return


if __name__ == "__main__":
    app.run()
