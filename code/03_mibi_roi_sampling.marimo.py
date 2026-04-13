import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pickle
    import numpy as np
    import anndata
    import random
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    import warnings
    import math
    import os
    import tifffile
    import matplotlib

    warnings.filterwarnings("ignore", category=UserWarning, module="anndata")
    return (
        Path,
        PdfPages,
        math,
        matplotlib,
        mo,
        np,
        os,
        pd,
        pickle,
        plt,
        random,
        sns,
        tifffile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 1. Configuration""")
    return


@app.cell
def _(mo):
    sampling_algorithm = mo.ui.dropdown(
        options=["Rejection", "Quadtree", "Poisson"],
        value="Poisson",
        label="Sampling Algorithm",
    )
    return (sampling_algorithm,)


@app.cell
def _(mo, sampling_algorithm):
    input_file = mo.ui.text(
        "data/angelo/processed/fully_annotated_anndata_objects.pkl",
        label="Input AnnData Objects (.pkl)",
    )
    output_roi_file = mo.ui.text(
        f"data/angelo/processed/mibi_roi_{sampling_algorithm.value.lower()}_anndata.pkl",
        label="Output ROI Data (.pkl)",
    )
    output_report_file = mo.ui.text(
        f"data/angelo/processed/mibi_roi_{sampling_algorithm.value.lower()}_sampling_report.pdf",
        label="Output Report (.pdf)",
    )
    num_rois = mo.ui.number(start=1, stop=100, value=50, label="Number of ROIs per sample")
    roi_width = mo.ui.number(start=50, stop=2000, value=500, label="ROI Width (pixels)")
    roi_height = mo.ui.number(start=50, stop=2000, value=500, label="ROI Height (pixels)")
    max_iou = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.5, label="Max IoU")
    random_seed = mo.ui.number(42, label="Random Seed")

    config_ui = mo.vstack(
        [
            sampling_algorithm,
            input_file,
            output_roi_file,
            output_report_file,
            num_rois,
            roi_width,
            roi_height,
            max_iou,
            random_seed,
        ]
    )
    return (
        config_ui,
        input_file,
        max_iou,
        num_rois,
        output_report_file,
        output_roi_file,
        random_seed,
        roi_height,
        roi_width,
    )


@app.cell
def _(config_ui):
    config_ui
    return


@app.cell
def _(Path, math, np, pickle, random):
    # %% Helpers
    def calculate_iou(box1, box2):
        """Calculates Intersection over Union for two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def get_random_roi_box(max_x, max_y, width, height):
        if max_x < width or max_y < height:
            return None
        x1 = random.randint(0, int(max_x - width))
        y1 = random.randint(0, int(max_y - height))
        return (x1, y1, x1 + width, y1 + height)

    def _process_results(all_sampled_rois, all_boxes_by_sample):
        all_ious = []
        for sample_boxes in all_boxes_by_sample.values():
            for j in range(len(sample_boxes)):
                for k in range(j + 1, len(sample_boxes)):
                    all_ious.append(calculate_iou(sample_boxes[j], sample_boxes[k]))
        report_data = {
            "all_ious": all_ious,
            "cell_counts": [len(roi.obs) for roi in all_sampled_rois],
            "total_rois": len(all_sampled_rois),
        }
        return report_data

    def _get_base_data(input_path, random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        with open(input_path, "rb") as f:
            adata_list = pickle.load(f)
        return adata_list

    # %% Algorithm 1: Rejection Sampling
    def perform_sampling_rejection(
        input_path, num_rois, roi_width, roi_height, max_iou, random_seed
    ):
        adata_list = _get_base_data(input_path, random_seed)
        all_sampled_rois = []
        all_boxes_by_sample = {}
        for i, adata in enumerate(adata_list):
            sample_name = adata.uns.get("sample_name", f"Sample_{i}")
            coords = adata.obsm["spatial"]
            max_x, max_y = coords.max(axis=0)
            sampled_boxes = []
            max_attempts = num_rois * 100
            attempts = 0
            while len(sampled_boxes) < num_rois and attempts < max_attempts:
                attempts += 1
                box = get_random_roi_box(max_x, max_y, roi_width, roi_height)
                if box is None:
                    break
                if all(calculate_iou(box, b) < max_iou for b in sampled_boxes):
                    sampled_boxes.append(box)
            all_boxes_by_sample[sample_name] = sampled_boxes
            for box in sampled_boxes:
                x1, y1, x2, y2 = box
                roi_mask = (coords[:, 0] >= x1) & (coords[:, 0] < x2) & (coords[:, 1] >= y1) & (coords[:, 1] < y2)
                roi_adata = adata[roi_mask, :].copy()
                roi_adata.uns["roi_box"] = box
                all_sampled_rois.append(roi_adata)
        return all_sampled_rois, _process_results(all_sampled_rois, all_boxes_by_sample)

    # %% Algorithm 2: Quadtree-accelerated Rejection Sampling
    class Rectangle:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h
        def intersects(self, other):
            return not (other.x > self.x + self.w or other.x + other.w < self.x or other.y > self.y + self.h or other.y + other.h < self.y)

    class Quadtree:
        def __init__(self, boundary, capacity=4):
            self.boundary = boundary
            self.capacity = capacity
            self.points = []
            self.divided = False
        def subdivide(self):
            x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h
            self.northwest = Quadtree(Rectangle(x, y, w / 2, h / 2), self.capacity)
            self.northeast = Quadtree(Rectangle(x + w / 2, y, w / 2, h / 2), self.capacity)
            self.southwest = Quadtree(Rectangle(x, y + h / 2, w / 2, h / 2), self.capacity)
            self.southeast = Quadtree(Rectangle(x + w / 2, y + h / 2, w / 2, h / 2), self.capacity)
            self.divided = True
        def insert(self, point):
            if not self.boundary.intersects(point.uns["roi_rect"]): return False
            if len(self.points) < self.capacity:
                self.points.append(point)
                return True
            if not self.divided: self.subdivide()
            return (self.northeast.insert(point) or self.northwest.insert(point) or self.southeast.insert(point) or self.southwest.insert(point))
        def query(self, rect, found):
            if not self.boundary.intersects(rect): return
            for p in self.points:
                if rect.intersects(p.uns["roi_rect"]): found.append(p)
            if self.divided:
                self.northwest.query(rect, found)
                self.northeast.query(rect, found)
                self.southwest.query(rect, found)
                self.southeast.query(rect, found)

    def perform_sampling_quadtree(
        input_path, num_rois, roi_width, roi_height, max_iou, random_seed
    ):
        adata_list = _get_base_data(input_path, random_seed)
        all_sampled_rois, all_boxes_by_sample = [], {}
        for i, adata in enumerate(adata_list):
            sample_name = adata.uns.get("sample_name", f"Sample_{i}")
            coords = adata.obsm["spatial"]
            max_x, max_y = coords.max(axis=0)
            qtree = Quadtree(Rectangle(0, 0, max_x, max_y))
            sampled_boxes, max_attempts, attempts = [], num_rois * 100, 0
            while len(sampled_boxes) < num_rois and attempts < max_attempts:
                attempts += 1
                box = get_random_roi_box(max_x, max_y, roi_width, roi_height)
                if box is None: break
                rect = Rectangle(box[0], box[1], roi_width, roi_height)
                neighbors = []
                qtree.query(rect, neighbors)
                if all(calculate_iou(box, n.uns["roi_box"]) < max_iou for n in neighbors):
                    roi_mask = (coords[:, 0] >= box[0]) & (coords[:, 0] < box[2]) & (coords[:, 1] >= box[1]) & (coords[:, 1] < box[3])
                    roi_adata = adata[roi_mask, :].copy()
                    roi_adata.uns["roi_box"] = box
                    roi_adata.uns["roi_rect"] = rect
                    qtree.insert(roi_adata)
                    all_sampled_rois.append(roi_adata)
                    sampled_boxes.append(box)
            all_boxes_by_sample[sample_name] = sampled_boxes
        return all_sampled_rois, _process_results(all_sampled_rois, all_boxes_by_sample)

    # %% Algorithm 3: Poisson Disk Sampling
    def perform_sampling_poisson(
        input_path, num_rois, roi_width, roi_height, max_iou, random_seed
    ):
        adata_list = _get_base_data(input_path, random_seed)
        all_sampled_rois, all_boxes_by_sample = [], {}
        # Heuristic: min_dist between centers to likely satisfy IoU
        min_dist = (roi_width + roi_height) / 2 * (1 - max_iou) 

        for i, adata in enumerate(adata_list):
            sample_name = adata.uns.get("sample_name", f"Sample_{i}")
            coords = adata.obsm["spatial"]
            max_x, max_y = coords.max(axis=0)

            cell_size = min_dist / math.sqrt(2)
            grid_w, grid_h = int(math.ceil(max_x / cell_size)), int(math.ceil(max_y / cell_size))
            grid = [None] * (grid_w * grid_h)

            active_list, sampled_boxes = [], []

            # Initial sample
            box = get_random_roi_box(max_x, max_y, roi_width, roi_height)
            if box is None: continue

            active_list.append(box)
            sampled_boxes.append(box)
            gx, gy = int(box[0]/cell_size), int(box[1]/cell_size)
            grid[gx + gy * grid_w] = box

            while active_list and len(sampled_boxes) < num_rois:
                active_idx = random.randrange(len(active_list))
                active_box = active_list[active_idx]

                found_candidate = False
                for _ in range(30): # k candidates
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(min_dist, 2 * min_dist)

                    nx = active_box[0] + dist * math.cos(angle)
                    ny = active_box[1] + dist * math.sin(angle)

                    if not (0 <= nx < max_x - roi_width and 0 <= ny < max_y - roi_height):
                        continue

                    new_box = (nx, ny, nx + roi_width, ny + roi_height)

                    # Check neighbors
                    gx, gy = int(nx/cell_size), int(ny/cell_size)
                    is_valid = True
                    for i in range(max(0, gx - 2), min(grid_w, gx + 3)):
                        for j in range(max(0, gy - 2), min(grid_h, gy + 3)):
                            neighbor = grid[i + j * grid_w]
                            if neighbor and calculate_iou(new_box, neighbor) >= max_iou:
                                is_valid = False
                                break
                        if not is_valid: break

                    if is_valid:
                        active_list.append(new_box)
                        sampled_boxes.append(new_box)
                        grid[gx + gy * grid_w] = new_box
                        found_candidate = True
                        break # Found a candidate

                if not found_candidate:
                    active_list.pop(active_idx)

            all_boxes_by_sample[sample_name] = sampled_boxes
            for box in sampled_boxes:
                x1, y1, x2, y2 = box
                roi_mask = (coords[:, 0] >= x1) & (coords[:, 0] < x2) & (coords[:, 1] >= y1) & (coords[:, 1] < y2)
                roi_adata = adata[roi_mask, :].copy()
                roi_adata.uns["roi_box"] = box
                all_sampled_rois.append(roi_adata)

        return all_sampled_rois, _process_results(all_sampled_rois, all_boxes_by_sample)
    return (
        perform_sampling_poisson,
        perform_sampling_quadtree,
        perform_sampling_rejection,
    )


@app.cell
def _(Path, PdfPages, np, plt, sns):
    def generate_pdf_report(pdf_path, report_data, config_values):
        Path(pdf_path).parent.mkdir(exist_ok=True, parents=True)
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.clf()
            summary_text = (
                f"ROI Sampling Report\n\n"
                f"Configuration:\n"
                f"  - Algorithm: {config_values['sampling_algorithm']}\n"
                f"  - Input File: {config_values['input_file']}\n"
                f"  - Num ROIs per Sample: {config_values['num_rois']}\n"
                f"  - ROI Size: {config_values['roi_width']}x{config_values['roi_height']}\n"
                f"  - Max IoU: {config_values['max_iou']}\n\n"
                f"Results:\n"
                f"  - Total ROIs Generated: {report_data['total_rois']}\n"
                f"  - Average Cells per ROI: {np.mean(report_data['cell_counts']):.2f}\n"
                f"  - Average IoU: {np.mean(report_data['all_ious'] if report_data['all_ious'] else [0]):.4f}\n"
            )
            fig.text(0.1, 0.9, summary_text, va="top", ha="left", wrap=True, fontsize=10)
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(report_data["cell_counts"], kde=True, ax=ax)
            ax.set_title("Distribution of Cell Densities per ROI")
            ax.set_xlabel("Number of Cells")
            ax.set_ylabel("Frequency")
            pdf.savefig(fig)
            plt.close()

            if report_data["all_ious"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(report_data["all_ious"], kde=True, ax=ax)
                ax.set_title("Distribution of Pairwise IoU")
                ax.set_xlabel("IoU")
                ax.set_ylabel("Frequency")
                pdf.savefig(fig)
                plt.close()
    return (generate_pdf_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 2. ROI Sampling and Report Generation""")
    return


@app.cell
def _(
    Path,
    input_file,
    max_iou,
    mo,
    num_rois,
    output_roi_file,
    perform_sampling_poisson,
    perform_sampling_quadtree,
    perform_sampling_rejection,
    pickle,
    random_seed,
    roi_height,
    roi_width,
    sampling_algorithm,
):
    run_button = mo.ui.button(label="Run Sampling")

    SAMPLING_DISPATCH = {
        "Rejection": perform_sampling_rejection,
        "Quadtree": perform_sampling_quadtree,
        "Poisson": perform_sampling_poisson,
    }

    def run_all():
        algo_func = SAMPLING_DISPATCH[sampling_algorithm.value]

        sampled_rois, data = algo_func(
            input_file.value,
            num_rois.value,
            roi_width.value,
            roi_height.value,
            max_iou.value,
            random_seed.value,
        )

        Path(output_roi_file.value).parent.mkdir(exist_ok=True, parents=True)
        with open(output_roi_file.value, "wb") as f:
            pickle.dump(sampled_rois, f)

        return data, sampled_rois
    return (run_all,)


@app.cell
def _(run_all):
    report_data, rois = run_all()
    return report_data, rois


@app.cell
def _(pd, rois):
    sample_id_distribution = pd.Series([rois[i].obs['SampleID'].max() for i in range(len(rois))]).value_counts()
    return (sample_id_distribution,)


@app.cell
def _(plt, sample_id_distribution):
    def _():
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(sample_id_distribution)
        ax.set_xticks(list(set(sample_id_distribution)))
        plt.title('Distribution of the number of ROIs sampled from each FOV')
        plt.show()
        plt.close()
    _()
    return


@app.cell
def _(
    generate_pdf_report,
    input_file,
    max_iou,
    num_rois,
    output_report_file,
    report_data,
    roi_height,
    roi_width,
    sampling_algorithm,
):
    config_values = {
        "sampling_algorithm": sampling_algorithm.value,
        "input_file": input_file.value,
        "num_rois": num_rois.value,
        "roi_width": roi_width.value,
        "roi_height": roi_height.value,
        "max_iou": max_iou.value,
    }
    generate_pdf_report(output_report_file.value, report_data, config_values)
    return


@app.cell
def _(mo, np, plt, report_data, sns):
    mo.md("## 3. Sampling Results")
    if report_data is None:
        mo.md("Report will be displayed here after running the sampling.")
        summary_display = None
    else:
        avg_cells = np.mean(report_data["cell_counts"])
        avg_iou = np.mean(report_data["all_ious"]) if report_data["all_ious"] else 0

        summary = mo.md(
            f"""
            ### Metrics
            - **Total ROIs Generated:** {report_data['total_rois']}
            - **Average Cells per ROI:** {avg_cells:.2f}
            - **Average IoU:** {avg_iou:.4f}
            """
        )

        fig1, ax1 = plt.subplots()
        sns.histplot(report_data["cell_counts"], kde=True, ax=ax1)
        ax1.set_title("Distribution of Cell Densities per ROI")
        ax1.set_xlabel("Number of Cells")
        density_plot = mo.mpl.interactive(fig1)

        iou_plot = mo.md("No IoU distribution to show (no overlapping ROIs).")
        if report_data["all_ious"]:
            fig2, ax2 = plt.subplots()
            sns.histplot(report_data["all_ious"], kde=True, ax=ax2)
            ax2.set_title("Distribution of Pairwise IoU")
            ax2.set_xlabel("IoU")
            iou_plot = mo.mpl.interactive(fig2)

        # Note: Coverage map generation removed for brevity in this refactoring
        # but can be added back if needed.

        summary_display = mo.vstack(
            [
                summary,
                mo.md("### Cell Density Distribution"),
                density_plot,
                mo.md("### IoU Distribution"),
                iou_plot,
            ]
        )
    summary_display
    return


@app.cell
def _(mo):
    mo.md("""## 4. Visualization of Sampled ROIs""")
    return


@app.cell
def _(mo):
    # Button to trigger loading/refreshing available samples
    refresh_samples = mo.ui.button(label="Refresh Samples")
    return (refresh_samples,)


@app.cell
def _(mo, os, output_roi_file, pickle, refresh_samples):
    # This code block handles the sample selection logic
    # It depends on refresh_samples so it re-runs on click
    refresh_samples

    roi_file_path = output_roi_file.value
    available_samples = []

    # We check if the file exists to populate the dropdown
    # We assume the user has run the sampling or the file exists
    if os.path.exists(roi_file_path):
        try:
            with open(roi_file_path, "rb") as f:
                # We only need to peek at the data, but pickle loads all.
                # Since these are small ROIs, it should be fast.
                _loaded_rois = pickle.load(f)
            _ids = set()
            for _r in _loaded_rois:
                if "SampleID" in _r.obs:
                    _ids.add(_r.obs["SampleID"].iloc[0])
            available_samples = sorted(list(_ids), key=lambda x: int(x) if str(x).isdigit() else x)
        except Exception:
            pass

    sample_selector = mo.ui.dropdown(
        options=[str(s) for s in available_samples],
        label="Select Patient Sample",
    )
    return roi_file_path, sample_selector


@app.cell
def _(
    matplotlib,
    mo,
    np,
    os,
    pd,
    pickle,
    plt,
    roi_file_path,
    sample_selector,
    sns,
    tifffile,
):
    # Visualization Logic
    def visualize_sample(selected_sample_id):
        if not selected_sample_id:
            return mo.md("Please select a sample to visualize.")

        # 1. Load ROIs
        with open(roi_file_path, "rb") as f:
            all_rois = pickle.load(f)

        sample_rois = [
            r for r in all_rois
            if str(r.obs['SampleID'].iloc[0]) == str(selected_sample_id)
        ]

        if not sample_rois:
            return mo.md(f"No ROIs found for Sample {selected_sample_id}")

        # 2. Load Tiff Image
        tiff_path = f"data/angelo/TNBC_shareCellData/p{selected_sample_id}_labeledcellData.tiff"
        if not os.path.exists(tiff_path):
            return mo.md(f"Error: Image file not found at {tiff_path}")

        seg_mask = tifffile.imread(tiff_path)

        # 3. Load Mapping Data
        mapping_path = "data/angelo/processed/mibi_all_cells.csv"
        if not os.path.exists(mapping_path):
            return mo.md(f"Error: Mapping file not found at {mapping_path}")

        df_map = pd.read_csv(mapping_path, usecols=['SampleID', 'Cell ID', 'cellType'])
        df_sample = df_map[df_map['SampleID'].astype(str) == str(selected_sample_id)]

        cell_type_map = dict(zip(df_sample['Cell ID'], df_sample['cellType']))

        # 4. Create Colored Image
        unique_types = sorted(list(set(cell_type_map.values())))
        # Create a consistent palette
        palette = sns.color_palette("husl", len(unique_types))
        color_map = dict(zip(unique_types, palette))

        # Optimized coloring
        max_id = seg_mask.max()
        lookup = np.zeros((max_id + 1, 3), dtype=np.float32)

        # Background is black (0,0,0) by default
        for cid, ctype in cell_type_map.items():
            if cid <= max_id:
                lookup[cid] = color_map[ctype]

        rgb_img = lookup[seg_mask]

        # 5. Plot
        with plt.style.context('default'):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb_img)
            ax.set_title(f"Sample {selected_sample_id} with {len(sample_rois)} ROIs")
            ax.axis('off')

            # Add Boxes
            for i, roi in enumerate(sample_rois):
                if 'roi_box' in roi.uns:
                    x1, y1, x2, y2 = roi.uns['roi_box']
                    rect = matplotlib.patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.5, edgecolor='white', facecolor='none')
                    ax.add_patch(rect)
                    ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(i), color='white', fontsize=12, ha='center', va='center', fontweight='bold')    
            # Legend
            patches = [matplotlib.patches.Patch(color=color_map[t], label=t) for t in unique_types]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
        return mo.mpl.interactive(fig)

    viz_output = visualize_sample(sample_selector.value)
    return (viz_output,)


@app.cell
def _(mo, refresh_samples, sample_selector, viz_output):
    # Display UI
    mo.vstack(
        [
            refresh_samples,
            sample_selector,
            viz_output
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
