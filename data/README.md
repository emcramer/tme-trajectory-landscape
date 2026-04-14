# Data

The data files required to reproduce the manuscript figures are archived on
Zenodo at: **[https://doi.org/10.5281/zenodo.19226009](https://doi.org/10.5281/zenodo.19226009
> **Note:** This Zenodo link will be updated when the archive is published.

---

## Setup

Download the Zenodo archive and unpack it so that the directory structure
matches what is described below. From the repository root:

```bash
# Download and unpack (replace URL with actual Zenodo record URL)
wget https://zenodo.org/record/10.5281/zenodo.19226009/files/tme-trajectory-landscape-data.tar.gz
tar -xzf tme-trajectory-landscape-data.tar.gz
```

The resulting layout should be:
```
tme-trajectory-landscape/
├── data/
│   ├── abm/
│   │   ├── all_simulations_param_trajectory.pkl
│   │   ├── all_simulations_param_trajectory_20251022.pkl
│   │   └── processed/
│   │       ├── abm_windows_clustered_with_state_label_20260212.csv
│   │       ├── abm_leiden_cluster_means.csv
│   │       ├── abm_full_features_leiden_clusters_linkage_matrix.npy
│   │       ├── abm_full_features_leiden_clusters_distance_matrix.npy
│   │       ├── normed_scaled_features_local_time_20251123.csv
│   │       ├── abm_umap_embedding_state_labeled.csv
│   │       └── [other intermediate files]
│   ├── angelo/
│   │   ├── clinical_data.csv
│   │   ├── s2_clinical_data_table.csv
│   │   └── processed/
│   │       ├── fully_annotated_anndata_objects.pkl
│   │       ├── mibi_roi_poisson_anndata.pkl
│   │       ├── mibi_roi_poisson_spatial_summaries_normalized.csv
│   │       └── mibi_roi_poisson_spatial_summaries_normalized_common_features.csv
│   └── wang/
│       ├── processed/
│       │   ├── wang_roi_anndatas_list.pkl
│       │   ├── wang_roi_abm_state_assignment.csv
│       │   ├── wang_roi_poisson_spatial_summaries_normalized.csv
│       │   └── wang_roi_poisson_spatial_summaries_normalized_common_features.csv
│       └── NTPublic/              # Raw Wang et al. IMC images (see below)
│           ├── metalReadOrder.csv
│           └── NeoTripFinalPanelToIMCTools1.csv
└── output/
    └── objects/
        └── abm/
            ├── abm_full_features_pc_n647_umap_embedding.npy
            ├── abm_full_features_pca.npy
            └── [other model objects]
```

---

## Raw external datasets

Two external datasets are used in this study. They are **not** included in
the Zenodo archive and must be downloaded separately from their original sources.

### Angelo et al. (2018) MIBI TNBC cohort

- **Publication:** Keren et al., *Cell* 174(6), 2018.
  [doi:10.1016/j.cell.2018.08.039](https://doi.org/10.1016/j.cell.2018.08.039)
- **Data:** Available via the publication's supplementary materials or from
  the corresponding authors.
- **Expected location:** `data/angelo/TNBC_shareCellData/`

### Wang et al. NeoTRIP IMC dataset

- **Expected location:** `data/wang/NTPublic/`
- Contact the corresponding authors of the Wang et al. manuscript for access.

> The Figure 3D script (`wang_roi_raw_visualization_background_subtraction_scale_bar.marimo.py`)
> and the Figure 3B Wang sub-panel script require the raw TIFF images from the
> NTPublic directory. All other figure scripts work with the processed data
> files available on Zenodo.

---

## File descriptions

### ABM data (`data/abm/`)

| File | Description |
|------|-------------|
| `all_simulations_param_trajectory.pkl` | Per-simulation parameter values and TME state sequence for all 150 runs |
| `all_simulations_param_trajectory_20251022.pkl` | Earlier version; used by `01_abm_generate_embedding.py` |
| `processed/abm_windows_clustered_with_state_label_20260212.csv` | Final clustered time-step windows with TME state labels (111 MB) |
| `processed/abm_leiden_cluster_means.csv` | Mean feature values per Leiden cluster |
| `processed/abm_full_features_leiden_clusters_linkage_matrix.npy` | Hierarchical clustering linkage for the state clustermap |
| `processed/abm_full_features_leiden_clusters_distance_matrix.npy` | Distance matrix for Leiden clusters |
| `processed/normed_scaled_features_local_time_20251123.csv` | Yeo-Johnson + z-score normalized spatial features, recentered by time (116 MB) |
| `processed/abm_umap_embedding_state_labeled.csv` | UMAP 2D coordinates with state labels for each time-step window |
| `output/objects/abm/abm_full_features_pc_n647_umap_embedding.npy` | Pre-computed UMAP embedding (n=647 PCs) |

### MIBI / Angelo data (`data/angelo/`)

| File | Description |
|------|-------------|
| `clinical_data.csv` | Patient clinical data (survival, recurrence, stage, grade) |
| `processed/fully_annotated_anndata_objects.pkl` | Per-patient AnnData objects with cell type annotations |
| `processed/mibi_roi_poisson_anndata.pkl` | Poisson-sampled ROI AnnData objects |
| `processed/mibi_roi_poisson_spatial_summaries_normalized.csv` | Normalized spatial statistics for all MIBI ROIs |
| `processed/mibi_roi_poisson_spatial_summaries_normalized_common_features.csv` | Subset of features shared with Wang and ABM datasets |

### Wang / NeoTRIP data (`data/wang/`)

| File | Description |
|------|-------------|
| `processed/wang_roi_anndatas_list.pkl` | Per-ROI AnnData objects for Wang cohort |
| `processed/wang_roi_abm_state_assignment.csv` | Per-ROI ABM state assignments (kNN mapping) |
| `processed/wang_roi_poisson_spatial_summaries_normalized.csv` | Normalized spatial statistics for all Wang ROIs |
| `processed/wang_roi_poisson_spatial_summaries_normalized_common_features.csv` | Subset of features shared with MIBI and ABM datasets |
| `NTPublic/metalReadOrder.csv` | IMC panel channel order |
| `NTPublic/NeoTripFinalPanelToIMCTools1.csv` | IMC panel antibody-to-channel mapping |
