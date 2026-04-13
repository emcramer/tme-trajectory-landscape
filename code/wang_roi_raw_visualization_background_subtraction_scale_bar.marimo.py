import marimo

__generated_with = "0.16.5"
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

    # Define paths
    base_dir = Path().resolve()
    # Handle case where we might be running relative to code/
    if base_dir.name == "code":
        base_dir = base_dir.parent

    data_dir = base_dir / "data"
    wang_processed = data_dir / "wang" / "processed"
    wang_raw = data_dir / "wang" / "NTPublic" / "data" / "raw"
    images_dir = wang_raw / "images"
    panel_dir = wang_raw / "panel"
    output_dir = base_dir / "output" / "figures" / "wang" / "roi_marker_visualizations"

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
    mo.md(r"""# Wang ROI Raw Data Visualization (Background Subtraction + Denoising + Scale Bar)""")
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

    # Pre-select common markers
    defaults = ['CD8', 'panKeratin_AE3', 'DNA1', 'CD68']
    for m in ['CD8', 'panKeratin_AE3', 'DNA1', 'CD68']:
        if m in target_options:
            defaults.append(m)
    if not defaults:
        defaults = list(target_options.keys())[:4]

    channel_selector = mo.ui.multiselect(
        options=target_options,
        value=defaults[:3],
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
    # Visualization Logic
    selected_targets = channel_selector.value
    if not selected_targets:
        mo.stop(True, "Select at least one channel.")

    radius = radius_slider.value
    median_size = median_slider.value

    # Get indices
    all_keys = list(target_options.keys())
    markers = [all_keys[t] for t in selected_targets]

    # Colors
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    # Create Figure
    num_channels = len(markers)
    fig, axes = plt.subplots(
        1, num_channels + 1, figsize=(4 * (num_channels + 1), 4), constrained_layout=True
    )
    if num_channels == 0:
        axes = [axes]

    # 1. Composite Image
    h, w = cropped_stack.shape[1], cropped_stack.shape[2]
    composite = np.zeros((h, w, 3))

    legend_patches = []
    legend_patches_composite = []

    for i, (idx, target) in enumerate(zip(selected_targets, markers)):
        raw_img = cropped_stack[idx, :, :]

        # 1. Median Filtering (Denoising)
        if median_size > 0:
             raw_img = median_filter(raw_img, size=median_size)

        # 2. Background Subtraction
        if radius > 0:
            try:
                bg = rolling_ball(raw_img, radius=radius)
                raw_img = raw_img - bg
                raw_img = np.clip(raw_img, 0, None)
            except Exception as e:
                pass

        # 3. Normalization
        p1, p99 = np.percentile(raw_img, (1, 99))
        if p99 > p1:
            norm_img = np.clip((raw_img - p1) / (p99 - p1), 0, 1)
        else:
            norm_img = np.zeros_like(raw_img)

        color_name = colors[i % len(colors)]

        legend_patches.append(mpatches.Patch(color=color_name, label=target))
        legend_patches_composite.append(mpatches.Patch(color=color_name, label=target))

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

        # Plot individual
        ax = axes[i + 1]
        ax.imshow(raw_img, cmap="gray")
        ax.set_title(f"{target} ({color_name})")
        ax.axis("off")

    composite = np.clip(composite, 0, 1)

    axes[0].imshow(composite)
    axes[0].set_title("Composite")
    axes[0].axis("off")
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize='small', framealpha=0.5)

    # Create Separate Composite Figure
    aspect_ratio = w / h
    fig_height = 6
    fig_width = fig_height * aspect_ratio
    fig_composite, ax_composite = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    ax_composite.imshow(composite)
    ax_composite.axis("off")
    ax_composite.legend(handles=legend_patches_composite, loc='upper right', fontsize='small', framealpha=0.5)

    # --- ADD SCALE BAR ---
    # Define scale bar length in microns
    bar_length_um = 100
    if w / pixels_per_micron < 200:
        bar_length_um = 50
    if w / pixels_per_micron < 50:
        bar_length_um = 10

    bar_length_px = bar_length_um * pixels_per_micron

    # Position: Bottom Right
    margin = w * 0.05
    x0 = w - margin - bar_length_px
    y0 = h - margin
    bar_height = h * 0.02

    # Draw Bar (Main Figure)
    rect = mpatches.Rectangle((x0, y0 - bar_height), bar_length_px, bar_height, color='white')
    axes[0].add_patch(rect)

    # Draw Label (Main Figure)
    axes[0].text(x0 + bar_length_px/2, y0 - bar_height - (h*0.01), f"{bar_length_um} µm", 
                 color='white', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Draw Bar (Composite Figure)
    rect_comp = mpatches.Rectangle((x0, y0 - bar_height), bar_length_px, bar_height, color='white')
    ax_composite.add_patch(rect_comp)

    # Draw Label (Composite Figure)
    ax_composite.text(x0 + bar_length_px/2, y0 - bar_height - (h*0.01), f"{bar_length_um} µm", 
                 color='white', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plot_output = mo.as_html(fig)
    plot_output
    return fig, fig_composite, markers


@app.cell
def _(
    fig,
    fig_composite,
    markers,
    median_slider,
    output_dir,
    radius_slider,
    selected_roi_idx,
    selected_state,
):
    def save():
        filename_base = (
            f"ROI_{selected_roi_idx}_State_{selected_state}_Markers_{'-'.join(markers)}_BGSub_{radius_slider.value}_Med_{median_slider.value}"
        )
        filename_base = "".join(
            c for c in filename_base if c.isalnum() or c in ("-", "_")
        ).strip()

        png_path = output_dir / f"{filename_base}.png"
        svg_path = output_dir / f"{filename_base}.svg"

        comp_png_path = output_dir / f"{filename_base}_composite.png"
        comp_svg_path = output_dir / f"{filename_base}_composite.svg"

        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)

        fig_composite.savefig(comp_png_path, dpi=300)
        fig_composite.savefig(comp_svg_path)

        return f"Saved output to: {png_path} and {comp_png_path}."

    save()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
