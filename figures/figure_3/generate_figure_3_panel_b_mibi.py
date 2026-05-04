import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import tifffile
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_pkl_path = data_dir / "mibi_roi_poisson_anndata.pkl"
    # Cell-type map lives at the repo root, not in this figure's local data/.
    repo_root = Path(__file__).resolve().parents[3]
    cell_types_csv = repo_root / "data" / "angelo" / "processed" / "mibi_all_cells.csv"
    return (
        cell_types_csv,
        data_dir,
        mo,
        mpatches,
        np,
        output_dir,
        pd,
        pickle,
        plt,
        roi_pkl_path,
        sns,
        tifffile,
    )


@app.cell
def _(mo, pickle, roi_pkl_path):
    # Load all ROIs and extract the available patient sample IDs.
    with open(roi_pkl_path, "rb") as f:
        all_rois = pickle.load(f)

    sample_ids = sorted(
        {
            str(r.obs["SampleID"].iloc[0])
            for r in all_rois
            if "SampleID" in r.obs
        },
        key=lambda s: int(s) if s.isdigit() else s,
    )

    sample_selector = mo.ui.dropdown(
        options=sample_ids,
        value="4",
        label="Select Patient Sample",
    )
    sample_selector
    return all_rois, sample_selector


@app.cell
def _(
    all_rois,
    cell_types_csv,
    data_dir,
    mo,
    mpatches,
    np,
    output_dir,
    pd,
    plt,
    sample_selector,
    sns,
    tifffile,
):
    selected = sample_selector.value
    if not selected:
        mo.stop(True, "Select a patient sample.")

    sample_rois = [
        r for r in all_rois
        if str(r.obs["SampleID"].iloc[0]) == str(selected)
    ]

    tiff_path = data_dir / f"p{selected}_labeledcellData.tiff"
    if not tiff_path.exists():
        mo.stop(True, f"TIFF not found: {tiff_path}")
    if not cell_types_csv.exists():
        mo.stop(True, f"Cell-type CSV not found: {cell_types_csv}")

    seg_mask = tifffile.imread(tiff_path)

    # Map (SampleID, Cell ID) -> cellType, then build an index lookup table.
    df_map = pd.read_csv(cell_types_csv, usecols=["SampleID", "Cell ID", "cellType"])
    df_sample = df_map[df_map["SampleID"].astype(str) == str(selected)]
    cell_type_map = dict(zip(df_sample["Cell ID"], df_sample["cellType"]))

    unique_types = sorted(set(cell_type_map.values()))
    palette = sns.color_palette("husl", len(unique_types))
    color_map = dict(zip(unique_types, palette))

    n_max = int(seg_mask.max())
    lookup = np.zeros((n_max + 1, 3), dtype=np.float32)  # background black
    for cid, ctype in cell_type_map.items():
        if cid <= n_max:
            lookup[cid] = color_map[ctype]
    rgb_img = lookup[seg_mask]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_img)
    ax.set_title(f"Sample {selected} with {len(sample_rois)} ROIs")
    ax.axis("off")

    for i, roi in enumerate(sample_rois):
        if "roi_box" in roi.uns:
            x1, y1, x2, y2 = roi.uns["roi_box"]
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="white", facecolor="none",
            ))
            ax.text(
                x1 + 24, y1 + 24, str(i),
                color="white", fontsize=12,
                ha="center", va="center", fontweight="bold",
            )

    legend_handles = [
        mpatches.Patch(color=color_map[t], label=t) for t in unique_types
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.,
    )

    plt.tight_layout()

    png_path = output_dir / "figure_3_panel_b_mibi.png"
    svg_path = output_dir / "figure_3_panel_b_mibi.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    return (fig,)


if __name__ == "__main__":
    app.run()
