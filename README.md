# TME Trajectory Landscape

Code repository for:

> **Mapping the Trajectory Landscape of the Tumor Microenvironment in Triple
> Negative Breast Cancer Using Agent-Based Modeling**
> Eric Cramer et al. (manuscript in preparation)

This repository contains the analysis code and figure generation scripts
for the manuscript. Most processed inputs needed to regenerate the
figures are bundled in this repository under `figures/figure_<n>/data/`.
A handful of large supporting files (raw simulation outputs, raw IMC
TIFFs, full feature tables) are archived separately on
[Zenodo](https://doi.org/ZENODO_DOI) — see
[data/README.md](data/README.md) for download instructions.

---

## Repository structure

```text
tme-trajectory-landscape/
├── run                          # Top-level orchestrator (regenerates all figures)
├── environment.yml              # Conda environment
├── requirements.txt             # Pip-resolved dependency pin
│
├── code/                        # Analysis notebooks that produce upstream
│                                # processed data (kNN mapping, ROI sampling,
│                                # spatial summaries, etc.). Used during the
│                                # analysis pipeline; not invoked by `run`.
│
├── figures/
│   ├── FIGURE_GENERATION.md     # Panel-by-panel guide
│   ├── figure_2/                # ABM TME state space
│   │   ├── generate_figure_2.py            # orchestrator
│   │   ├── generate_figure_2_panel_a.py    # state clustermap
│   │   ├── generate_figure_2_panel_b-e.py  # UMAP scatters + trajectories
│   │   ├── generate_figure_2_panel_f.py    # Bokeh state snapshots
│   │   ├── data/                           # bundled inputs
│   │   └── output/                         # PNG/SVG outputs (gitignored)
│   ├── figure_3/                # Patient ROI mapping
│   │   ├── generate_figure_3.py
│   │   ├── generate_figure_3_panel_b_mibi.py
│   │   ├── generate_figure_3_panel_b_imc.py
│   │   ├── generate_figure_3_panel_c.py
│   │   ├── generate_figure_3_panel_d.py
│   │   ├── data/
│   │   └── output/
│   ├── figure_4/                # MIBI clinical outcomes
│   │   ├── generate_figure_4.py            # all panels A–F + Cox PH table
│   │   ├── panel_c_tme_recurrence_analysis.py     # standalone, extra stats
│   │   ├── panel_d_e_tme_survival_analysis.py     # standalone, extra stats
│   │   ├── tme_style.py / tme_research_1.mplstyle # local style helpers
│   │   ├── data/
│   │   └── outputs/
│   ├── figure_5/                # NeoTRIP (Wang) clinical outcomes
│   │   ├── generate_figure_5.py
│   │   ├── data/
│   │   └── outputs/
│   └── figure_6/                # Markov state model + interventions
│       ├── generate_figure_6.py
│       ├── generate_figure_6_panel_a.py
│       ├── generate_figure_6_panel_b.py
│       ├── generate_figure_6_panel_c.py
│       ├── generate_figure_6_panel_e_and_f.py
│       ├── figure_6_panel_d_msm_schematic.svg     # Inkscape schematic
│       ├── data/
│       └── output/
│
├── data/                        # Repo-level data tree for raw inputs
│   └── README.md                # Zenodo download / setup instructions
│
└── outputs/                     # Aggregated figure outputs from `run`
                                 # (gitignored)
```

Each `figures/figure_<n>/` directory is self-contained: its panel scripts
read from the local `data/` subfolder and write to the local `output/`
(or `outputs/`) subfolder. Panels that depend on raw IMC TIFFs or raw
PhysiCell outputs reach into the repo-level `data/` tree — those
external dependencies are noted in
[figures/FIGURE_GENERATION.md](figures/FIGURE_GENERATION.md).

---

## Quick start

### 1. Install dependencies

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate tme-trajectory-landscape
```

Pip:
```bash
pip install -r requirements.txt
```

### 2. Download supplementary data (optional)

The bundled `figures/figure_<n>/data/` directories contain everything
needed to regenerate the panels of Figures 2, 4, 5, and 6 and most
panels of Figure 3. Two Figure 3 panels (B-IMC and D) and Figure 2
Panel F additionally need raw PhysiCell simulation outputs and raw IMC
TIFFs that live in the repo-level `data/` tree — download those from
the Zenodo archive when you need them.

See [data/README.md](data/README.md) for download details and the
expected directory layout.

### 3. Regenerate figures

The `run` script orchestrates every figure that has bundled data:

```bash
bash run
```

It calls each `figures/figure_<n>/generate_figure_<n>.py` orchestrator
in turn, then copies the per-figure outputs into `outputs/figure_<n>/`
at the repo root. Per-figure orchestrators can also be run individually:

```bash
cd figures/figure_4 && python generate_figure_4.py
cd figures/figure_5 && python generate_figure_5.py
cd figures/figure_6 && python generate_figure_6.py
cd figures/figure_2 && python generate_figure_2.py
cd figures/figure_3 && python generate_figure_3.py
```

The final multi-panel figures in the manuscript were assembled in
Inkscape from the SVG outputs. See
[figures/FIGURE_GENERATION.md](figures/FIGURE_GENERATION.md) for the
panel-by-panel layout and any panels that are pure schematics
(BioRender / Inkscape only).

---

## Overview of the study

Triple negative breast cancer (TNBC) is characterized by a dynamic tumor
microenvironment (TME) whose composition and spatial organization
influence therapeutic response. Existing patient data provide only a
single static snapshot of TME state, limiting our ability to understand
TME dynamics and predict treatment outcomes.

We developed a PhysiCell agent-based model (ABM) of the TNBC TME
featuring malignant epithelial cells, CD8+ T cells (effector and
exhausted states), and macrophages (M0, M1, M2 polarizations). By
systematically exploring the ABM parameter space (Latin Hypercube
sampling, 150 simulations), we computed spatial statistics at each time
step and embedded the resulting trajectories in a reduced-dimensional
state space using PCA and UMAP.

Hierarchical clustering of the state space identified 6 discrete TME
states and their transition dynamics. We then mapped patient ROIs from
two independent TNBC cohorts — the MIBI dataset (Angelo et al., *Cell*
2018) and the NeoTRIP IMC dataset (Wang et al.) — onto this
ABM-derived state space, enabling inference of TME dynamics from
static tissue images and associations with clinical outcomes including
recurrence and treatment response.

---

## External datasets

This work uses two publicly available patient datasets:

- **Angelo et al. (2018)** MIBI TNBC cohort: [doi:10.1016/j.cell.2018.08.039](https://doi.org/10.1016/j.cell.2018.08.039)
- **Wang et al.** NeoTRIP IMC dataset: [citation TBD]

---

## Citation

If you use this code, please cite:

> Cramer et al. (manuscript in preparation). *Mapping the Trajectory Landscape
> of the Tumor Microenvironment in Triple Negative Breast Cancer Using
> Agent-Based Modeling.*

---

## License

[MIT License](LICENSE)
