#!/usr/bin/env python3
"""
================================================================================
FIGURE 4: TME State Classification and Clinical Outcomes in TNBC
================================================================================

This figure integrates:
  A) ROI State Heterogeneity Heatmap - showing intra-patient variation
  B) Stacked Barplot - ROI distribution by Mixing Score class (Keren et al.)
  C) TME State Proportions vs Recurrence - violin/box plots with statistics
  D) Combined Cox PH Forest Plot - TME state group HRs + Mixing Score HRs
  E) Kaplan-Meier Survival Analysis - DFS by TME State Groups (former Panel D)
  F) Kaplan-Meier Survival Analysis - DFS by Mixing Score Class (new)

The model comparison table (formerly Panel F) is saved separately as
'coxph_comparison_table.png' and printed to console with full formatting.

The figure legend is placed as a single row at the top of the figure.

Uses CoxPHFitter(penalizer=0.01) with binary cutoff indicators for
TME state groups. Survival data sourced from clinical_data.csv
(Overall Survival endpoint, 39 patients).

Run from the figures/figure_4/ directory:
    python generate_figure_4.py

Author: Eric Cramer
Date: February 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, to_hex
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency
from scipy.cluster.hierarchy import linkage, dendrogram
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from pathlib import Path
import warnings
import sys

from tme_style import (
    TME_COLORS, TME_COLORS_LIST, MIXING_COLORS, OUTCOME_COLORS,
    apply_tme_style, get_tme_cmap, add_significance_stars, format_pvalue,
    get_tme_legend_handles, get_mixing_legend_handles, style_spines, add_panel_label
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_PATH = Path('tme_research_1.mplstyle')

SURVIVAL_DATA_PATH = DATA_DIR / 'clinical_data.csv'
PROPORTIONS_DATA_PATH = DATA_DIR / 'mibi_roi_tme_proportions_clinical.csv'
ROI_SPATIAL_PATH = DATA_DIR / 'mibi_roi_spatial_summaries_state_labeled.csv'
MIXING_SCORE_DATA_PATH = DATA_DIR / 'mibi_patient_mixing_score_labels.csv'

FIGURE_SIZE = (10, 17)
DPI = 300

# =============================================================================
# FONT SIZE CONTROLS
# =============================================================================

FONT_MAIN_TITLE = 14
FONT_PANEL_TITLE = 12
FONT_AXIS_LABEL = 11
FONT_TICK_LABEL = 10
FONT_TICK_LABEL_SMALL = 8
FONT_ANNOTATION = 9
FONT_LEGEND = 10         # Top-row legend font (tight single row)
FONT_TABLE = 10
FONT_TABLE_HEADER = 10
FONT_RUG_LABEL = 9

# =============================================================================

STATE1_CUTPOINT = 1.0
STATE2_CUTPOINT = 0.111
STATE6_CUTPOINT = 0.333

TME_GROUP_COLORS = {
    'State_1_High': TME_COLORS[1],
    'State_2_High': TME_COLORS[2],
    'State_6_High': TME_COLORS[6],
    'Other': '#8B7355'
}


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_merge_data():
    """
    Load survival data, TME proportions, mixing scores, and ROI spatial data.
    Returns (df_clinical, df_roi, df_merged).
    """
    print("Loading data...")

    df_survival = pd.read_csv(SURVIVAL_DATA_PATH)
    df_survival = df_survival.rename(columns={'Survival': 'Event', 'Survival_time': 'Time'})
    print(f"  Survival data: {len(df_survival)} patients (OS endpoint)")

    df_props = pd.read_csv(PROPORTIONS_DATA_PATH)
    prop_cols = ['SampleID', 'Majority_State',
                 'State_1_Prop', 'State_2_Prop', 'State_3_Prop',
                 'State_5_Prop', 'State_6_Prop',
                 'TIL_score', 'STAGE', 'GRADE', 'AGE_AT_DX']
    prop_cols = [c for c in prop_cols if c in df_props.columns]
    print(f"  TME proportions data: {len(df_props)} patients")

    df_clinical = df_survival.merge(df_props[prop_cols], on='SampleID', how='left')
    print(f"  After merging survival + proportions: {len(df_clinical)} patients")

    df_mixing = pd.read_csv(MIXING_SCORE_DATA_PATH)
    print(f"  Mixing Score data: {len(df_mixing)} patients")

    df_clinical = df_clinical.merge(
        df_mixing[['PatientID', 'mixing_class_label']],
        left_on='SampleID',
        right_on='PatientID',
        how='left'
    )
    df_clinical = df_clinical.rename(columns={'mixing_class_label': 'mixing_score_class'})

    df_roi = pd.read_csv(ROI_SPATIAL_PATH)
    print(f"  ROI spatial data: {len(df_roi)} ROIs")

    merge_cols = ['SampleID', 'mixing_score_class', 'Event', 'Time']
    optional_cols = ['TIL_score', 'STAGE', 'GRADE', 'AGE_AT_DX']
    merge_cols += [c for c in optional_cols if c in df_clinical.columns]

    df_merged = df_roi.merge(
        df_clinical[merge_cols],
        left_on='PatientID',
        right_on='SampleID',
        how='left'
    )
    print(f"  Merged data: {len(df_merged)} ROIs with clinical annotations")

    n_matched = df_merged['mixing_score_class'].notna().sum()
    print(f"  Successfully matched: {n_matched} ROIs ({n_matched/len(df_merged)*100:.1f}%)")

    return df_clinical, df_roi, df_merged


def assign_tme_groups(df):
    """Assign patients to TME groups based on state proportions."""
    groups = []
    for idx, row in df.iterrows():
        s1 = row.get('State_1_Prop', 0) or 0
        s2 = row.get('State_2_Prop', 0) or 0
        s6 = row.get('State_6_Prop', 0) or 0

        if s6 >= STATE6_CUTPOINT:
            groups.append('State_6_High')
        elif s1 >= STATE1_CUTPOINT:
            groups.append('State_1_High')
        elif s2 >= STATE2_CUTPOINT:
            groups.append('State_2_High')
        else:
            groups.append('Other')
    return groups


def sort_linkage_by_mean(Z, column_means):
    """Sort linkage matrix so lower mean values are on the left."""
    n_samples = len(column_means)
    n_nodes = 2 * n_samples - 1
    node_means = np.zeros(n_nodes)
    node_counts = np.zeros(n_nodes)

    node_means[:n_samples] = column_means
    node_counts[:n_samples] = 1

    Z_sorted = Z.copy()

    for i in range(len(Z)):
        cluster_idx = n_samples + i
        c1 = int(Z_sorted[i, 0])
        c2 = int(Z_sorted[i, 1])

        count1 = node_counts[c1]
        count2 = node_counts[c2]
        mean1 = node_means[c1]
        mean2 = node_means[c2]

        new_count = count1 + count2
        new_mean = (mean1 * count1 + mean2 * count2) / new_count

        node_counts[cluster_idx] = new_count
        node_means[cluster_idx] = new_mean

        if mean1 > mean2:
            Z_sorted[i, 0] = c2
            Z_sorted[i, 1] = c1

    return Z_sorted


# =============================================================================
# COX MODEL FITTING (shared by Panel D and table output)
# =============================================================================

def fit_cox_models(df_clinical):
    """
    Fit Cox PH models for TME state groups and Mixing Score class.

    Uses a common patient dataset (patients with all variables present) so
    that model metrics are directly comparable.

    Returns:
        tme_forest  : list of dicts with HR/CI/p for TME model covariates
        mix_forest  : list of dicts with HR/CI/p for Mixing Score model covariates
        models_stats: dict keyed by model name with c-index, AIC, BIC, LRT p
        n_common    : number of patients used in all models
    """
    df = df_clinical.copy()
    df['TME_Group'] = assign_tme_groups(df)

    df['is_State_2_High'] = (df['TME_Group'] == 'State_2_High').astype(int)
    df['is_State_6_High'] = (df['TME_Group'] == 'State_6_High').astype(int)

    df['mix_Cold'] = df['mixing_score_class'].map(
        lambda x: 1 if x == 'Cold' else (0 if pd.notna(x) else np.nan))
    df['mix_Mixed'] = df['mixing_score_class'].map(
        lambda x: 1 if x == 'Mixed' else (0 if pd.notna(x) else np.nan))

    all_cols = ['Time', 'Event', 'is_State_2_High', 'is_State_6_High',
                'mix_Cold', 'mix_Mixed']
    df_common = df[all_cols].dropna()
    n_common = len(df_common)
    print(f"  Cox models: using {n_common} patients with complete data")

    def model_stats(cph, n, k):
        ll = cph.log_likelihood_
        return {
            'c_idx': cph.concordance_index_,
            'll': ll,
            'aic': 2 * k - 2 * ll,
            'bic': k * np.log(n) - 2 * ll,
            'df': k,
            'n': n,
            'lrt_p': cph.log_likelihood_ratio_test().p_value
        }

    models_stats = {}
    tme_forest = []
    mix_forest = []

    # Model 1: TME State Groups
    try:
        data1 = df_common[['Time', 'Event', 'is_State_2_High', 'is_State_6_High']]
        cph1 = CoxPHFitter(penalizer=0.01)
        cph1.fit(data1, duration_col='Time', event_col='Event')
        models_stats['TME States'] = model_stats(cph1, n_common, 2)

        for var, state_num, display_name in [
            ('is_State_2_High', 2, 'State 2 High'),
            ('is_State_6_High', 6, 'State 6 High'),
        ]:
            tme_forest.append({
                'name': display_name,
                'hr': cph1.summary.loc[var, 'exp(coef)'],
                'hr_lower': cph1.summary.loc[var, 'exp(coef) lower 95%'],
                'hr_upper': cph1.summary.loc[var, 'exp(coef) upper 95%'],
                'p': cph1.summary.loc[var, 'p'],
                'state': state_num,
                'model': 'TME States',
            })
    except Exception as e:
        print(f"  TME model failed: {e}")

    # Model 2: Mixing Score Class
    try:
        data2 = df_common[['Time', 'Event', 'mix_Mixed', 'mix_Cold']]
        cph2 = CoxPHFitter(penalizer=0.01)
        cph2.fit(data2, duration_col='Time', event_col='Event')
        models_stats['Mixing Score'] = model_stats(cph2, n_common, 2)

        for var, mix_class, display_name in [
            ('mix_Mixed', 'Mixed', 'Mixed'),
            ('mix_Cold', 'Cold', 'Cold'),
        ]:
            mix_forest.append({
                'name': display_name,
                'hr': cph2.summary.loc[var, 'exp(coef)'],
                'hr_lower': cph2.summary.loc[var, 'exp(coef) lower 95%'],
                'hr_upper': cph2.summary.loc[var, 'exp(coef) upper 95%'],
                'p': cph2.summary.loc[var, 'p'],
                'mixing_class': mix_class,
                'model': 'Mixing Score',
            })
    except Exception as e:
        print(f"  Mixing Score model failed: {e}")

    # Model 3: Combined (for table only)
    try:
        data3 = df_common[all_cols]
        cph3 = CoxPHFitter(penalizer=0.01)
        cph3.fit(data3, duration_col='Time', event_col='Event')
        models_stats['Combined'] = model_stats(cph3, n_common, 4)
    except Exception as e:
        print(f"  Combined model failed: {e}")

    return tme_forest, mix_forest, models_stats, n_common


# =============================================================================
# MODEL COMPARISON TABLE: console + separate PNG
# =============================================================================

def print_model_comparison(models_stats):
    """Print model comparison results to console with aligned tabular formatting."""

    models_list = [m for m in ['TME States', 'Mixing Score', 'Combined']
                   if m in models_stats]

    metrics = [
        ('N',               'n'),
        ('C-index',         'c_idx'),
        ('Log-likelihood',  'll'),
        ('AIC',             'aic'),
        ('BIC',             'bic'),
        ('LRT p-value',     'lrt_p'),
    ]

    def fmt(key, val):
        if key == 'n':      return str(int(val))
        if key == 'c_idx':  return f"{val:.3f}"
        if key == 'll':     return f"{val:.2f}"
        if key in ('aic', 'bic'): return f"{val:.2f}"
        if key == 'lrt_p':
            return f"{val:.4f}" if val >= 0.0001 else "<0.0001"
        return str(val)

    col0_w = max(len(m[0]) for m in metrics) + 2
    col_w  = max(max(len(m) for m in models_list), 12) + 2

    sep = "=" * (col0_w + col_w * len(models_list) + 4)

    print("\n" + sep)
    print("  COX PROPORTIONAL HAZARDS MODEL COMPARISON")
    print(sep)
    header = "  " + f"{'Metric':<{col0_w}}" + "".join(f"{m:>{col_w}}" for m in models_list)
    print(header)
    print("  " + "-" * (col0_w + col_w * len(models_list)))

    for metric_label, metric_key in metrics:
        row = "  " + f"{metric_label:<{col0_w}}"
        for model in models_list:
            val = models_stats[model].get(metric_key, 'N/A')
            row += f"{fmt(metric_key, val):>{col_w}}"
        print(row)

    print(sep)

    if 'TME States' in models_stats and 'Mixing Score' in models_stats:
        c_diff = models_stats['TME States']['c_idx'] - models_stats['Mixing Score']['c_idx']
        print(f"\n  ΔC-index (TME States − Mixing Score) = {c_diff:+.3f}")

    print()


def save_coxph_table(models_stats, output_dir):
    """
    Save model comparison table as a standalone PNG and print to console.

    Args:
        models_stats: dict from fit_cox_models()
        output_dir  : Path to output directory
    """
    # Print to console
    print_model_comparison(models_stats)

    display_models = [m for m in ['TME States', 'Mixing Score'] if m in models_stats]
    metric_labels = ['N', 'C-index', 'AIC', 'BIC', 'LRT vs Null']

    def fmt_cell(metric, model):
        r = models_stats[model]
        if metric == 'N':           return str(int(r['n']))
        if metric == 'C-index':     return f"{r['c_idx']:.3f}"
        if metric == 'AIC':         return f"{r['aic']:.2f}"
        if metric == 'BIC':         return f"{r['bic']:.2f}"
        if metric == 'LRT vs Null':
            p = r['lrt_p']
            return f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
        return ''

    table_data = [[fmt_cell(m, mdl) for mdl in display_models] for m in metric_labels]

    # Determine best for each metric
    best_cidx = max(display_models, key=lambda x: models_stats[x]['c_idx'])
    best_aic  = min(display_models, key=lambda x: models_stats[x]['aic'])
    best_bic  = min(display_models, key=lambda x: models_stats[x]['bic'])

    fig_t, ax_t = plt.subplots(figsize=(6, 3.2))
    ax_t.axis('off')

    tbl = ax_t.table(
        cellText=table_data,
        rowLabels=metric_labels,
        colLabels=display_models,
        cellLoc='center',
        rowLoc='right',
        loc='center',
        colWidths=[0.40] * len(display_models)
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    # Header row styling
    for col_idx in range(len(display_models)):
        tbl[(0, col_idx)].set_facecolor('#4a4a4a')
        tbl[(0, col_idx)].set_text_props(color='white', fontweight='bold',
                                          fontsize=11)

    # Row label styling
    for row_idx in range(len(metric_labels)):
        tbl[(row_idx + 1, -1)].set_text_props(fontweight='bold', fontsize=11)

    # Highlight best cells
    for row_idx, metric in enumerate(metric_labels):
        for col_idx, model in enumerate(display_models):
            cell_row = row_idx + 1
            if metric == 'C-index' and model == best_cidx:
                tbl[(cell_row, col_idx)].set_facecolor('#d4edda')
            elif metric == 'AIC' and model == best_aic:
                tbl[(cell_row, col_idx)].set_facecolor('#d4edda')
            elif metric == 'BIC' and model == best_bic:
                tbl[(cell_row, col_idx)].set_facecolor('#d4edda')
            elif metric == 'LRT vs Null' and models_stats[model]['lrt_p'] < 0.05:
                tbl[(cell_row, col_idx)].set_facecolor('#d4edda')

    if 'TME States' in models_stats and 'Mixing Score' in models_stats:
        c_diff = (models_stats['TME States']['c_idx']
                  - models_stats['Mixing Score']['c_idx'])
        ax_t.text(0.5, 0.04,
                  f"ΔC-index (TME States − Mixing Score) = {c_diff:+.3f}",
                  transform=ax_t.transAxes, ha='center',
                  fontsize=10, style='italic')

    ax_t.set_title('Cox PH Model Comparison: TME States vs Mixing Score',
                   fontsize=13, fontweight='bold', pad=12)

    table_path = output_dir / 'coxph_comparison_table.png'
    fig_t.savefig(table_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig_t)
    print(f"  Saved comparison table: {table_path}")

    return table_path


# =============================================================================
# PANEL A: ROI STATE HETEROGENEITY HEATMAP
# =============================================================================

def create_panel_A(fig, gs_panel, df_roi, df_clinical):
    """Panel A: ROI State Heterogeneity Heatmap."""
    df_roi_local = df_roi.copy()
    df_roi_local['ROI_Index'] = df_roi_local.groupby('PatientID').cumcount() + 1

    heatmap_data = df_roi_local.pivot(
        index='ROI_Index', columns='PatientID', values='state_label'
    )

    patient_mixing = df_clinical.set_index('SampleID')['mixing_score_class']

    try:
        sorted_cols = sorted(heatmap_data.columns, key=lambda x: int(x))
        heatmap_data = heatmap_data[sorted_cols]
    except Exception:
        pass

    heatmap_filled = heatmap_data.fillna(0)
    Z = linkage(heatmap_filled.T, method='ward')
    col_means = heatmap_filled.mean(axis=0).values
    Z_sorted = sort_linkage_by_mean(Z, col_means)

    dendro_res = dendrogram(Z_sorted, no_plot=True)
    order = dendro_res['leaves']

    heatmap_ordered = heatmap_data.iloc[:, order]
    patient_ids_ordered = heatmap_ordered.columns
    mixing_labels_ordered = [patient_mixing.get(pid, 'Unknown')
                              for pid in patient_ids_ordered]

    gs_A = gs_panel.subgridspec(
        4, 1,
        height_ratios=[0.12, 1, 0.1, 0.1],
        width_ratios=[1],
        hspace=0.02,
        wspace=0.02
    )

    ax_dendro  = fig.add_subplot(gs_A[0, 0])
    ax_heatmap = fig.add_subplot(gs_A[1, 0])
    ax_rug     = fig.add_subplot(gs_A[2, 0], sharex=ax_heatmap)
    ax_ffov    = fig.add_subplot(gs_A[3, 0], sharex=ax_heatmap)

    # Dendrogram
    dendrogram(Z_sorted, ax=ax_dendro, color_threshold=0,
               above_threshold_color='black')
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # Heatmap
    state_cmap = ListedColormap(TME_COLORS_LIST)
    n_rows, n_patients = heatmap_ordered.shape

    ax_heatmap.imshow(
        heatmap_ordered.values,
        aspect='auto',
        cmap=state_cmap,
        vmin=0.5, vmax=6.5,
        interpolation='nearest',
        extent=[0, n_patients, n_rows - 0.5, -0.5]
    )

    ax_heatmap.set_ylabel('ROI Index', fontsize=FONT_AXIS_LABEL)
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks(range(len(heatmap_ordered.index)))
    ax_heatmap.set_yticklabels(heatmap_ordered.index,
                                fontsize=FONT_TICK_LABEL_SMALL)
    ax_heatmap.grid(which='major', alpha=0)
    ax_heatmap.set_xticks(np.arange(0, n_patients + 1, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, heatmap_ordered.shape[0], 1),
                           minor=True)
    ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax_heatmap.tick_params(which='minor', bottom=False, left=False)

    # Mixing Score rug
    n_patients = len(patient_ids_ordered)
    for i, label in enumerate(mixing_labels_ordered):
        color = MIXING_COLORS.get(label, 'gray')
        ax_rug.add_patch(Rectangle((i, 0), 1, 1, facecolor=color,
                                    edgecolor='none'))
    ax_rug.set_xlim(0, n_patients)
    ax_rug.set_ylim(0, 1)
    ax_rug.set_yticks([])
    ax_rug.set_ylabel('Mixing\nScore', fontsize=FONT_RUG_LABEL,
                       rotation=0, ha='right', va='center')

    # FFOV label rug
    ffov_labels = heatmap_ordered.mode().iloc[0].values
    print(ffov_labels)
    print([TME_COLORS.get(label, 'gray') for label in ffov_labels])
    for i, label in enumerate(ffov_labels):
        color = TME_COLORS.get(label, 'gray')
        ax_ffov.add_patch(Rectangle((i, 0), 1, 1, facecolor=color,
                                     edgecolor='none'))
    ax_ffov.set_xlim(0, n_patients)
    ax_ffov.set_ylim(0, 1)
    ax_ffov.set_yticks([])
    ax_ffov.set_ylabel('TME State\nLabel', fontsize=FONT_RUG_LABEL,
                        rotation=0, ha='right', va='center')

    ax_ffov.set_xticks(np.arange(n_patients) + 0.5)
    ax_ffov.set_xticklabels(patient_ids_ordered, fontsize=6, rotation=90)
    ax_ffov.set_xlabel('Patient ID', fontsize=FONT_TICK_LABEL)

    ax_heatmap.set_title('A. ROI State Heterogeneity Across Patients',
                          fontsize=FONT_PANEL_TITLE, fontweight='bold',
                          pad=15, loc='left', y=1.1)

    state_handles = [Patch(facecolor=TME_COLORS[i], edgecolor='black',
                            linewidth=0.5, label=f'State {i}')
                     for i in range(1, 7)]
    mixing_handles = [Patch(facecolor=MIXING_COLORS[k], edgecolor='black',
                             linewidth=0.5, label=k)
                      for k in ['Compartmentalized', 'Mixed', 'Cold']]

    return state_handles, mixing_handles


# =============================================================================
# PANEL B: STACKED BAR PLOT
# =============================================================================

def create_panel_B(ax, df_merged):
    """Panel B: ROI State Distribution by Mixing Score Class."""
    mix_classes = ['Compartmentalized', 'Mixed', 'Cold']
    all_states = [1, 2, 3, 4, 5, 6]

    proportions = {}
    counts = {}
    totals = {}

    for mix_class in mix_classes:
        subset = df_merged[df_merged['mixing_score_class'] == mix_class]
        state_counts = subset['state_label'].value_counts()
        total = len(subset)
        totals[mix_class] = total
        proportions[mix_class] = {s: state_counts.get(s, 0) / total
                                   if total > 0 else 0 for s in all_states}
        counts[mix_class] = {s: state_counts.get(s, 0) for s in all_states}

    contingency = pd.DataFrame(counts).T
    contingency_nonzero = contingency.loc[:, (contingency != 0).any(axis=0)]
    if contingency_nonzero.shape[1] > 1:
        chi2_stat, p_overall, dof, expected = chi2_contingency(contingency_nonzero)
    else:
        chi2_stat, p_overall = 0, 1.0

    x = np.arange(len(mix_classes))
    width = 0.6
    bottom = np.zeros(len(mix_classes))

    for state in all_states:
        heights = [proportions[mc].get(state, 0) for mc in mix_classes]
        ax.bar(x, heights, width, bottom=bottom, label=f'State {state}',
               color=TME_COLORS[state], edgecolor='white', linewidth=0.5)
        bottom += heights

    for i, mc in enumerate(mix_classes):
        ax.text(i, 1.02, f'n={totals[mc]}', ha='center', fontsize=FONT_ANNOTATION)

    ax.set_xticks(x)
    ax.set_xticklabels(mix_classes, fontsize=FONT_TICK_LABEL,
                        rotation=20, ha='right')
    ax.set_ylabel('Proportion of ROIs', fontsize=FONT_AXIS_LABEL)
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.5, 2.5)

    sig_str = add_significance_stars(p_overall)
    ax.text(0.5, 0.95,
            f'χ² = {chi2_stat:.1f}, p = {p_overall:.2e} {sig_str}',
            transform=ax.transAxes, ha='center', fontsize=FONT_ANNOTATION,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='gray'))

    style_spines(ax)
    ax.set_title('B. ROI State Distribution \nby Mixing Score Class',
                  fontsize=FONT_PANEL_TITLE, fontweight='bold', loc='left')


# =============================================================================
# PANEL C: TME STATE PROPORTIONS VS RECURRENCE
# =============================================================================

def create_panel_C(fig, gs_panel, df_clinical):
    """Panel C: TME State Proportions and Recurrence."""
    gs_C = gs_panel.subgridspec(1, 2, wspace=0.35)
    ax_s1 = fig.add_subplot(gs_C[0, 0])
    ax_s6 = fig.add_subplot(gs_C[0, 1])

    df = df_clinical.copy()
    df['Event_Label'] = df['Event'].map({0: 'No Recurrence', 1: 'Recurrence'})

    for ax, state_col, state_num, title_prefix in [
        (ax_s1, 'State_1_Prop', 1, 'C. TME State Proportions and Recurrence'),
        (ax_s6, 'State_6_Prop', 6, ''),
    ]:
        if state_col not in df.columns:
            continue

        parts = ax.violinplot(
            [df[df['Event'] == 0][state_col].dropna(),
             df[df['Event'] == 1][state_col].dropna()],
            positions=[0, 1],
            showmeans=False, showmedians=False, showextrema=False
        )
        for pc in parts['bodies']:
            pc.set_facecolor(TME_COLORS[state_num])
            pc.set_alpha(0.3)

        bp = ax.boxplot(
            [df[df['Event'] == 0][state_col].dropna(),
             df[df['Event'] == 1][state_col].dropna()],
            positions=[0, 1], widths=0.2, patch_artist=True, showfliers=True
        )
        for patch in bp['boxes']:
            patch.set_facecolor(TME_COLORS[state_num])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        no_event = df[df['Event'] == 0][state_col].dropna()
        event    = df[df['Event'] == 1][state_col].dropna()
        _, pval  = mannwhitneyu(no_event, event, alternative='two-sided')

        y_max = df[state_col].max()
        stars = add_significance_stars(pval)
        ax.plot([0, 0, 1, 1], [y_max+0.05, y_max+0.08, y_max+0.08, y_max+0.05],
                'k-', linewidth=1)
        ax.text(0.5, y_max + 0.10, f'{stars}\np={pval:.3f}',
                ha='center', fontsize=FONT_ANNOTATION, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f'No Recurrence\n(n={len(no_event)})',
             f'Recurrence\n(n={len(event)})'],
            fontsize=FONT_TICK_LABEL
        )
        ax.set_ylabel(f'State {state_num} Proportion', fontsize=FONT_AXIS_LABEL)
        ax.set_ylim(-0.05, y_max + 0.20)
        style_spines(ax)

        if title_prefix:
            ax.set_title(title_prefix, fontsize=FONT_PANEL_TITLE,
                          fontweight='bold', loc='left')


# =============================================================================
# PANEL D: COMBINED COX PH FOREST PLOT (TME States + Mixing Score)
# =============================================================================

def create_panel_D(ax, tme_forest, mix_forest, models_stats):
    """
    Panel D: Combined Cox PH Forest Plot

    Shows hazard ratios from two separate Cox models on the same axes:
      - Upper section: TME State Groups (reference = State 1 High)
      - Lower section: Mixing Score Class (reference = Compartmentalized)

    X-axis limits are set to just outside the outermost 95% CI bounds.
    """
    if not tme_forest and not mix_forest:
        ax.text(0.5, 0.5, 'Model fitting failed', transform=ax.transAxes,
                ha='center', va='center')
        return

    # Collect all CI bounds for tight axis scaling
    all_entries = tme_forest + mix_forest
    all_lowers = [r['hr_lower'] for r in all_entries]
    all_uppers = [r['hr_upper'] for r in all_entries]

    x_min_data = min(all_lowers)
    x_max_data = max(all_uppers)
    x_range = x_max_data - x_min_data
    padding = x_range * 0.06  # 6% padding

    # Ensure HR=1 reference is always in view
    x_min = min(x_min_data - padding, 0.85)
    x_max = x_max_data + padding

    ax.set_xlim(x_min, x_max)

    # ---- Y positions ----
    # TME section: y = 3 (State 2 High), y = 2 (State 6 High)
    # Gap / separator at y = 1
    # Mixing section: y = 0 (Mixed), y = -1 (Cold)
    tme_y = [3, 2]
    mix_y = [0, -1]

    # x in axes fraction, y in data coords — keeps annotations right-aligned
    # regardless of the computed data x-limits
    ann_transform = ax.get_yaxis_transform()
    ann_x = 0.28   # left edge of annotation text (axes fraction)
    ann_fs = 8.5   # annotation font size

    # ---- Plot TME model estimates (squares) ----
    for res, y in zip(tme_forest, tme_y):
        color = TME_COLORS[res['state']]
        ax.plot([res['hr_lower'], res['hr_upper']], [y, y],
                color=color, linewidth=2.5, solid_capstyle='butt', zorder=4)
        ax.scatter([res['hr']], [y], color=color, s=80, zorder=5,
                   edgecolor='black', linewidth=0.8, marker='s')
        # Combined HR [CI], p-value on one line below the marker
        p_str = f"{res['p']:.3f}" if res['p'] >= 0.001 else "<0.001"
        ax.text(ann_x, y - 0.15,
                f"HR={res['hr']:.2f} [{res['hr_lower']:.2f}\u2013{res['hr_upper']:.2f}],  p={p_str}",
                transform=ann_transform,
                ha='left', va='top', fontsize=ann_fs, color='#222222')

    # ---- Plot Mixing Score model estimates (diamonds) ----
    for res, y in zip(mix_forest, mix_y):
        color = MIXING_COLORS.get(res['mixing_class'], '#666666')
        ax.plot([res['hr_lower'], res['hr_upper']], [y, y],
                color=color, linewidth=2.5, solid_capstyle='butt', zorder=4)
        ax.scatter([res['hr']], [y], color=color, s=80, zorder=5,
                   edgecolor='black', linewidth=0.8, marker='D')
        p_str = f"{res['p']:.3f}" if res['p'] >= 0.001 else "<0.001"
        ax.text(ann_x, y - 0.15,
                f"HR={res['hr']:.2f} [{res['hr_lower']:.2f}\u2013{res['hr_upper']:.2f}],  p={p_str}",
                transform=ann_transform,
                ha='left', va='top', fontsize=ann_fs, color='#222222')

    # ---- Reference line at HR = 1 ----
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.8, zorder=2)

    # ---- Section separator ----
    ax.axhline(y=1.0, color='#cccccc', linestyle='-', linewidth=1.0, zorder=1)

    # ---- Section header text (inside plot at left edge) ----
    x_text = x_min + (x_max - x_min) * 0.1
    ax.text(x_text, 3.80, 'TME State Groups',
            fontsize=FONT_ANNOTATION - 0.5, fontweight='bold',
            style='italic', va='bottom', color='#333333')
    ax.text(x_text, 3.52, '(ref: State 1 High)',
            fontsize=FONT_ANNOTATION - 1.5, va='bottom', color='#666666')
    ax.text(x_text, 0.75, 'Mixing Score Class',
            fontsize=FONT_ANNOTATION - 0.5, fontweight='bold',
            style='italic', va='bottom', color='#333333')
    ax.text(x_text, 0.47, '(ref: Compartmentalized)',
            fontsize=FONT_ANNOTATION - 1.5, va='bottom', color='#666666')

    # ---- Y-axis tick labels ----
    all_y      = tme_y + mix_y
    all_labels = [r['name'] for r in tme_forest] + [r['name'] for r in mix_forest]
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=FONT_TICK_LABEL)
    ax.set_ylim(-2.0, 4.5)

    # ---- Model stats box ----
    stats_lines = []
    if 'TME States' in models_stats:
        stats_lines.append(f"\u25a0 TME C: {models_stats['TME States']['c_idx']:.3f}")
    if 'Mixing Score' in models_stats:
        stats_lines.append(f"\u25c6 Mix C: {models_stats['Mixing Score']['c_idx']:.3f}")
    if stats_lines:
        ax.text(0.98, 0.02, '\n'.join(stats_lines),
                transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=FONT_AXIS_LABEL)
    style_spines(ax)
    ax.set_title('D. Cox PH Forest Plot:\nTME States & Mixing Score',
                  fontsize=FONT_PANEL_TITLE, fontweight='bold', loc='left')


# =============================================================================
# PANEL E: KAPLAN-MEIER BY TME STATE GROUPS  (former Panel D)
# =============================================================================

def create_panel_E(ax, df_clinical):
    """
    Panel E: Kaplan-Meier Survival Curves by TME State Groups.
    (Moved from Panel D in v4.)
    """
    df = df_clinical.copy()
    df['TME_Group'] = assign_tme_groups(df)

    kmf = KaplanMeierFitter()
    group_order = ['State_1_High', 'State_2_High', 'State_6_High']

    ANNOTATION_LABELS = {
        'State_1_High': 'Effector-Dominant',
        'State_2_High': 'Undifferentiated',
        'State_6_High': 'Immune Excluded',
    }

    curve_endpoints = {}

    for group in group_order:
        mask = df['TME_Group'] == group
        if mask.sum() >= 2:
            kmf.fit(
                df.loc[mask, 'Time'],
                df.loc[mask, 'Event'],
                label=f"{group.replace('_', ' ')} (n={mask.sum()})"
            )
            kmf.plot_survival_function(
                ax=ax, ci_show=True,
                color=TME_GROUP_COLORS[group], linewidth=2.5
            )
            # Record last time point and survival probability for annotation
            timeline = kmf.timeline
            sf = kmf.survival_function_.values.flatten()
            curve_endpoints[group] = (timeline[-1], sf[-1])

    # Add text annotations near the end of each curve, slightly above
    for group, (t_end, sf_end) in curve_endpoints.items():
        ax.text(
            t_end, sf_end + 0.04,
            ANNOTATION_LABELS[group],
            #color=TME_GROUP_COLORS[group],
            color='black',
            fontsize=FONT_ANNOTATION - 1,
            fontweight='bold',
            ha='right',
            va='bottom',
        )

    groups_present = [g for g in group_order
                      if (df['TME_Group'] == g).sum() >= 2]
    df_for_test = df[df['TME_Group'].isin(groups_present)]

    if len(groups_present) >= 2:
        lr_result = multivariate_logrank_test(
            df_for_test['Time'], df_for_test['TME_Group'], df_for_test['Event']
        )
        p_logrank = lr_result.p_value
    else:
        p_logrank = 1.0

    ax.set_xlabel('Time (days)', fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel('Disease-Free Survival\nProbability', fontsize=FONT_AXIS_LABEL)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', fontsize=FONT_ANNOTATION, frameon=True)

    sig_str = add_significance_stars(p_logrank)
    ax.text(0.95, 0.95, f'Log-rank p = {p_logrank:.4f} {sig_str}',
            transform=ax.transAxes, fontsize=FONT_TICK_LABEL,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='gray'))

    style_spines(ax)
    ax.set_title('E. Kaplan-Meier: DFS by\nTME State Groups',
                  fontsize=FONT_PANEL_TITLE, fontweight='bold', loc='left')


# =============================================================================
# PANEL F: KAPLAN-MEIER BY MIXING SCORE CLASS  (replaces old table panel)
# =============================================================================

def create_panel_F(ax, df_clinical):
    """
    Panel F: Kaplan-Meier Survival Curves by Mixing Score Class.
    (Replaces the Model Comparison Table from v4 Panel F.)
    """
    df = df_clinical.dropna(subset=['mixing_score_class', 'Time', 'Event']).copy()

    kmf = KaplanMeierFitter()
    mix_order = ['Compartmentalized', 'Mixed', 'Cold']

    for mix_class in mix_order:
        mask = df['mixing_score_class'] == mix_class
        if mask.sum() >= 2:
            kmf.fit(
                df.loc[mask, 'Time'],
                df.loc[mask, 'Event'],
                label=f"{mix_class} (n={mask.sum()})"
            )
            kmf.plot_survival_function(
                ax=ax, ci_show=True,
                color=MIXING_COLORS.get(mix_class, '#888888'),
                linewidth=2.5
            )

    groups_present = [g for g in mix_order
                      if (df['mixing_score_class'] == g).sum() >= 2]
    df_test = df[df['mixing_score_class'].isin(groups_present)]

    if len(groups_present) >= 2:
        lr_result = multivariate_logrank_test(
            df_test['Time'], df_test['mixing_score_class'], df_test['Event']
        )
        p_logrank = lr_result.p_value
    else:
        p_logrank = 1.0

    ax.set_xlabel('Time (days)', fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel('Disease-Free Survival\nProbability', fontsize=FONT_AXIS_LABEL)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', fontsize=FONT_ANNOTATION, frameon=True)

    sig_str = add_significance_stars(p_logrank)
    ax.text(0.95, 0.95, f'Log-rank p = {p_logrank:.4f} {sig_str}',
            transform=ax.transAxes, fontsize=FONT_TICK_LABEL,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor='gray'))

    style_spines(ax)
    ax.set_title('F. Kaplan-Meier: DFS by\nMixing Score Class',
                  fontsize=FONT_PANEL_TITLE, fontweight='bold', loc='left')


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================

def create_main_figure():
    """
    Generate Figure 4 (v5) with layout:

        Row 0 (thin):  Single-row figure legend spanning full width
        Row 1:         Panel A (heatmap, cols 0-1) + Panel B (stacked bar, col 2)
        Row 2:         Panel C (violin plots, cols 0-1) + Panel D (forest plot, col 2)
        Row 3:         Panel E (KM-TME, left half) + Panel F (KM-mixing, right half)

    Model comparison table is saved separately as 'coxph_comparison_table.png'
    and printed to console.
    """
    print("\n" + "=" * 70)
    print("GENERATING FIGURE 4 v6: TME State Classification and Clinical Outcomes")
    print("=" * 70)

    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
        print(f"Applied style from: {STYLE_PATH}")
    else:
        apply_tme_style()
        print("Applied fallback TME style")

    df_clinical, df_roi, df_merged = load_and_merge_data()

    # Fit all Cox models once; results feed Panel D and the saved table
    print("\nFitting Cox PH models...")
    tme_forest, mix_forest, models_stats, n_common = fit_cox_models(df_clinical)

    # Save comparison table as separate PNG + print to console
    save_coxph_table(models_stats, OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # Figure layout: 4 rows × 3 cols
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=FIGURE_SIZE)

    gs_outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.02, 1],
        hspace=0.01,
        left=0.08, right=0.95, top=0.96, bottom=0.05
    )

    gs_main = gs_outer[1].subgridspec(
        4, 3,
        height_ratios=[0.01, 1.4, 1.0, 1.0],
        width_ratios=[1.2, 1, 1],
        hspace=0.35,
        wspace=0.30,
        #left=0.08, right=0.95, top=0.96, bottom=0.05
    )

    # =========================================================================
    # ROW 0: Single-row figure legend
    # =========================================================================
    ax_legend = fig.add_subplot(gs_outer[0, :])
    ax_legend.axis('off')

    state_handles = [
        Patch(facecolor=TME_COLORS[i], edgecolor='black',
              linewidth=0.5, label=f'State {i}')
        for i in range(1, 7)
    ]
    mixing_handles = [
        Patch(facecolor=MIXING_COLORS[k], edgecolor='black',
              linewidth=0.5, label=k)
        for k in ['Compartmentalized', 'Mixed', 'Cold']
    ]
    # Invisible spacer separates TME states from Mixing Score entries
    spacer = Patch(facecolor='none', edgecolor='none', label=' ')
    all_handles = state_handles + [spacer] + mixing_handles

    ax_legend.legend(
        handles=all_handles,
        loc='center',
        ncol=len(all_handles),   # all items in one row
        fontsize=FONT_LEGEND,
        frameon=True,
        fancybox=False,
        edgecolor='#cccccc',
        handlelength=0.9,
        handleheight=0.8,
        handletextpad=0.4,
        columnspacing=0.6,
        borderpad=0.5,
    )

    # =========================================================================
    # ROW 1: Panel A (heatmap) + Panel B (stacked bar)
    # =========================================================================
    print("\nCreating Panel A: ROI State Heterogeneity...")
    gs_A = gs_main[1, 0:2]
    create_panel_A(fig, gs_A, df_roi, df_clinical)

    print("Creating Panel B: State Distribution by Mixing Class...")
    ax_B = fig.add_subplot(gs_main[1, 2])
    create_panel_B(ax_B, df_merged)

    # =========================================================================
    # ROW 2: Panel C (violin) + Panel D (combined forest plot)
    # =========================================================================
    print("Creating Panel C: TME State Proportions vs Recurrence...")
    gs_C = gs_main[2, 0:2]
    create_panel_C(fig, gs_C, df_clinical)

    print("Creating Panel D: Combined Cox PH Forest Plot...")
    ax_D = fig.add_subplot(gs_main[2, 2])
    create_panel_D(ax_D, tme_forest, mix_forest, models_stats)

    # =========================================================================
    # ROW 3: Panel E (KM-TME) + Panel F (KM-mixing) — equal width, side by side
    # =========================================================================
    gs_bottom = gs_main[3, :].subgridspec(1, 2, wspace=0.35)

    print("Creating Panel E: Kaplan-Meier by TME State Groups...")
    ax_E = fig.add_subplot(gs_bottom[0, 0])
    create_panel_E(ax_E, df_clinical)

    print("Creating Panel F: Kaplan-Meier by Mixing Score Class...")
    ax_F = fig.add_subplot(gs_bottom[0, 1])
    create_panel_F(ax_F, df_clinical)

    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle(
        'Figure 4: TME State Classification and Clinical Outcomes in TNBC',
        fontsize=FONT_MAIN_TITLE,
        fontweight='bold',
        y=0.99
    )

    # Reduce whitespace around the figure
    gs_outer.tight_layout(fig)

    return fig, df_merged


def save_outputs(fig, df_merged):
    """Save figure (PNG + SVG) and merged dataset."""
    print("\nSaving outputs...")

    base_name = 'figure_4_mibi_tme_clinical_outcomes_v6'

    png_path = OUTPUT_DIR / f'{base_name}.png'
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {png_path}")

    svg_path = OUTPUT_DIR / f'{base_name}.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {svg_path}")

    csv_path = OUTPUT_DIR / 'mibi_roi_merged_clinical.csv'
    df_merged.to_csv(csv_path, index=False)
    print(f"  Saved merged data: {csv_path}")

    return [png_path, svg_path, csv_path]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    fig, df_merged = create_main_figure()
    saved_files = save_outputs(fig, df_merged)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("FIGURE 4 v6 GENERATION COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    return saved_files


if __name__ == "__main__":
    main()
