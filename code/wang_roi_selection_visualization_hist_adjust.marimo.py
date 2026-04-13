import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import pickle
    import tifffile
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path
    from skimage import exposure
    import os
    import re

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
    output_dir = base_dir / "output" / "figures" / "wang" / "roi_selection_maps"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        exposure,
        images_dir,
        mo,
        mpatches,
        np,
        output_dir,
        panel_dir,
        pd,
        pickle,
        plt,
        re,
        tifffile,
        wang_processed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Wang ROI Selection Visualization""")
    return


@app.cell(hide_code=True)
def _(mo, panel_dir, pd, pickle, wang_processed):
    # Load Metadata and Data
    state_assignment_path = wang_processed / "wang_roi_abm_state_assignment.csv"
    roi_list_path = wang_processed / "wang_roi_anndatas_list.pkl"
    metal_order_path = panel_dir / "metalReadOrder.csv"
    panel_map_path = panel_dir / "NeoTripFinalPanelToIMCTools1.csv"

    # Load State Assignments
    df_assignments = pd.read_csv(state_assignment_path)

    # Load ROI List (Heavy)
    with open(roi_list_path, 'rb') as f:
        roi_anndatas = pickle.load(f)

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

    mo.md(f"**Loaded Data:**\n- {len(df_assignments)} assigned ROIs\n- {len(roi_anndatas)} total ROIs loaded")
    return channel_targets, df_assignments, roi_anndatas


@app.cell(hide_code=True)
def _(df_assignments, roi_anndatas):
    # Process Data to link Images to ROIs

    # We need to extract ImageID and Box for each ROI in df_assignments
    # df_assignments has 'roi_index' which corresponds to roi_anndatas list index

    image_ids = []
    boxes = []
    biopsy_phases = []

    # Pre-fetch info from anndatas to avoid repeated lookups
    # Creating a dict for faster access might be good if indices are sparse, 
    # but here they should be consistent.

    for idx in df_assignments['roi_index']:
        if idx < len(roi_anndatas):
            adata = roi_anndatas[idx]
            image_ids.append(adata.uns.get('ImageID', 'Unknown'))
            boxes.append(adata.uns.get('roi_box', None))
            biopsy_phases.append(adata.uns.get('biopsy_phase', 'Unknown'))
        else:
            image_ids.append('Unknown')
            boxes.append(None)
            biopsy_phases.append('Unknown')

    df_linked = df_assignments.copy()
    df_linked['ImageID'] = image_ids
    df_linked['ROI_Box'] = boxes
    df_linked['BiopsyPhase'] = biopsy_phases

    # Filter out Unknowns
    df_linked = df_linked[df_linked['ImageID'] != 'Unknown']
    return (df_linked,)


@app.cell(hide_code=True)
def _(df_linked, mo):
    # UI: Phase Selection
    phases = sorted(df_linked['BiopsyPhase'].dropna().unique())
    phase_selector = mo.ui.dropdown(
        options=phases,
        value=phases[0] if phases else None,
        label="Select Biopsy Phase"
    )
    phase_selector
    return (phase_selector,)


@app.cell(hide_code=True)
def _(df_linked, mo, phase_selector):
    # UI: Patient/Image Selection

    if phase_selector.value is None:
        mo.stop(True, "Select a phase.")

    filtered_df = df_linked[df_linked['BiopsyPhase'] == phase_selector.value]

    # Group by ImageID (and include PatientID for display)
    # Some patients might have multiple images?
    unique_images = filtered_df[['PatientID', 'ImageID']].drop_duplicates()

    image_options = {
        f"{row.PatientID} - {row.ImageID}": row.ImageID
        for _, row in unique_images.iterrows()
    }

    image_selector = mo.ui.dropdown(
        options=image_options,
        value=list(image_options.keys())[0] if image_options else None,
        label="Select Field of View (Patient - ImageID)"
    )
    image_selector
    return (image_selector,)


@app.cell(hide_code=True)
def _(channel_targets, mo):
    # UI: Channel Selection for Background

    # Common structural markers
    defaults = ['DNA1', 'panKeratin_AE3', 'CD68', 'CD8']
    default_val = 'DNA1'
    for d in defaults:
        if d in channel_targets:
            default_val = d
            break

    channel_selector = mo.ui.dropdown(
        options=channel_targets,
        value=default_val,
        label="Select Background Channel",
    )
    channel_selector
    return (channel_selector,)


@app.cell(hide_code=True)
def _(mo):
    # UI: Visualization Settings
    line_width = mo.ui.slider(start=1, stop=10, step=1, value=2, label="Box Line Width")
    show_labels = mo.ui.checkbox(value=True, label="Show ROI Labels")
    use_clahe = mo.ui.checkbox(value=False, label="Local Hist Adjustment (CLAHE)")

    # Signal Intensity Controls
    gamma = mo.ui.slider(start=0.1, stop=2.5, step=0.1, value=1.0, label="Gamma Correction")
    gain = mo.ui.slider(start=1.0, stop=20.0, step=0.5, value=1.0, label="Signal Gain (Multiplier)")
    saturation_p = mo.ui.slider(start=80.0, stop=99.9, step=0.1, value=99.0, label="Saturation Percentile")

    settings_ui = mo.vstack([
        mo.md("**Plot Settings**"),
        line_width,
        show_labels,
        mo.md("**Signal Intensity**"),
        use_clahe,
        gamma,
        gain,
        saturation_p
    ])
    settings_ui
    return gain, gamma, line_width, saturation_p, show_labels, use_clahe


@app.cell(hide_code=True)
def _(
    channel_selector,
    channel_targets,
    df_linked,
    exposure,
    gain,
    gamma,
    image_selector,
    images_dir,
    line_width,
    mo,
    mpatches,
    np,
    phase_selector,
    plt,
    re,
    saturation_p,
    show_labels,
    tifffile,
    use_clahe,
):
    # Main Visualization Logic

    if not image_selector.value:
        mo.stop(True, "Please select an image.")

    selected_image_id = image_selector.value
    selected_phase = phase_selector.value
    selected_channel = channel_selector.value

    # Get associated ROIs
    rois_in_fov = df_linked[df_linked['ImageID'] == selected_image_id]

    # Load Image
    image_filename = f"{selected_image_id}FullStack.tiff"
    image_path = images_dir / image_filename

    if not image_path.exists():
        mo.stop(True, f"Image file not found: {image_path}")

    mo.md(f"Loading {image_filename}...")

    # Read Image
    pixels_per_micron = 1.0 # Default fallback

    try:
        with tifffile.TiffFile(image_path) as tif:
            image_stack = tif.asarray()
             # Metadata extraction (reusing logic from previous notebook)
            try:
                page = tif.pages[0]
                tags = page.tags
                if 'XResolution' in tags:
                    x_res = tags['XResolution'].value
                    if isinstance(x_res, tuple):
                        res_val = x_res[0] / x_res[1]
                    else:
                        res_val = x_res
                    unit = tags['ResolutionUnit'].value if 'ResolutionUnit' in tags else 0
                    if unit == 3: # cm
                        pixels_per_micron = res_val / 10000.0
                    elif unit == 2: # inch
                        pixels_per_micron = res_val / 25400.0
                if 'ImageDescription' in tags:
                    desc = tags['ImageDescription'].value
                    if isinstance(desc, str) and 'PhysicalSizeX' in desc:
                        match = re.search(r'PhysicalSizeX="([\d\.]+)"', desc)
                        if match:
                            mpp = float(match.group(1))
                            if mpp > 0:
                                pixels_per_micron = 1.0 / mpp
            except Exception:
                pass

    except Exception as e:
        mo.stop(True, f"Error reading image: {e}")

    # Find channel index
    try:
        channel_idx = channel_targets.index(selected_channel)
    except ValueError:
        channel_idx = 0 

    # Get raw channel data
    raw_img = image_stack[channel_idx, :, :]

    # Apply Gain (Multiplier) first
    proc_img = raw_img.astype(np.float32) * gain.value

    # Normalize for display
    if use_clahe.value:
        # Local histogram equalization (CLAHE)
        img_min, img_max = np.min(proc_img), np.max(proc_img)
        if img_max > img_min:
            img_rescaled = (proc_img - img_min) / (img_max - img_min)
            norm_img = exposure.equalize_adapthist(img_rescaled, clip_limit=0.03)
        else:
            norm_img = np.zeros_like(proc_img)
    else:
        # Standard Percentile Normalization
        p_low, p_high = np.percentile(proc_img, (1, saturation_p.value))
        if p_high > p_low:
            norm_img = np.clip((proc_img - p_low) / (p_high - p_low), 0, 1)
        else:
            norm_img = np.zeros_like(proc_img)

    # Apply Gamma Correction
    if gamma.value != 1.0:
        norm_img = np.power(norm_img, gamma.value)

    # Create Plot
    h, w = norm_img.shape
    fig, ax = plt.subplots(figsize=(10, 10 * (h/w)), constrained_layout=True)

    ax.imshow(norm_img, cmap='gray')
    ax.set_title(f"Patient: {rois_in_fov.iloc[0]['PatientID']} | Phase: {selected_phase} | Image: {selected_image_id}")
    ax.axis('off')

    # Draw ROI Boxes
    for _, row in rois_in_fov.iterrows():
        box = row['ROI_Box'] # y1, x1, y2, x2
        if box:
            y1, x1, y2, x2 = box
            width = x2 - x1
            height = y2 - y1

            # Create Rectangle
            rect = mpatches.Rectangle(
                (x1, y1), width, height,
                linewidth=line_width.value,
                edgecolor='yellow',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add Label
            if show_labels.value:
                ax.text(
                    x1 + width/2, y1 + height/2,
                    str(row['roi_index']),
                    color='yellow',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
                )

    # Scale Bar (Bottom Right)
    bar_length_um = 200 # Larger for full FOV
    bar_length_px = bar_length_um * pixels_per_micron

    margin = w * 0.02
    x0 = w - margin - bar_length_px
    y0 = h - margin
    bar_height = h * 0.005

    rect_bar = mpatches.Rectangle((x0, y0 - bar_height), bar_length_px, bar_height, color='white')
    ax.add_patch(rect_bar)
    ax.text(x0 + bar_length_px/2, y0 - bar_height - (h*0.005), f"{bar_length_um} µm", 
            color='white', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plot_output = mo.as_html(fig)
    plot_output
    return fig, selected_image_id, selected_phase


@app.cell(hide_code=True)
def _(fig, mo, output_dir, selected_image_id, selected_phase):
    def save():
        safe_id = "".join(c for c in selected_image_id if c.isalnum() or c in ("-", "_"))
        safe_phase = "".join(c for c in selected_phase if c.isalnum() or c in ("-", "_"))

        filename = f"FOV_Map_{safe_phase}_{safe_id}"
        png_path = output_dir / f"{filename}.png"
        svg_path = output_dir / f"{filename}.svg"

        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path)

        return f"Saved to {png_path}"

    save_btn = mo.ui.button(label="Save Plot", on_click=lambda _: save())
    save_btn
    return


if __name__ == "__main__":
    app.run()
