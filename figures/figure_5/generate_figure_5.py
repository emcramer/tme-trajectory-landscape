#!/usr/bin/env python3
"""
================================================================================
FIGURE 5: TME State Dynamics in NeoTRIP Trial
================================================================================

Generates Figure 5 from Cramer et al. (manuscript).
Maps patient ROIs from the NeoTRIP (Wang et al.) IMC dataset onto the ABM TME
state space and analyzes clinical associations with TME state assignments.

Note: This script was previously named create_figure6_v12.py in the development
repository; the figure was renumbered to Figure 5 in the final manuscript.

Author: Eric Cramer
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact, kruskal
import warnings
import os

warnings.filterwarnings('ignore')


def tex_escape(s):
    """Escape special LaTeX characters in a string."""
    s = str(s)
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    s = s.replace('_', r'\_')
    return s


# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False  # disabled: requires a full LaTeX install
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# ============================================================================
# COLOR SCHEMES
# ============================================================================
STATE_COLORS = {
    1: '#1a535c', 2: '#ee6c4d', 3: '#84a98c',
    4: '#b8b8a8', 5: '#6b4f7b', 6: '#e6b800',
}
STATE_NAMES_SHORT = {1: 'S1', 2: 'S2', 3: 'S3', 4: 'S4', 5: 'S5', 6: 'S6'}
RESPONSE_COLORS = {'pCR': '#2e7d32', 'RD': '#c62828'}
ARM_COLORS = {'C': '#6b7280', 'C&I': '#3b82f6'}
PHASE_ORDER = ['Baseline', 'On-treatment', 'Post-treatment']


# ============================================================================
# DATA LOADING
# ============================================================================
def load_all_data():
    roi_data = pd.read_csv('data/wang_roi_abm_state_assignment.csv')
    clinical_data = pd.read_csv('data/wang_full_fov_clinical_tme_state_labeled.csv')
    spatial_data = pd.read_csv('data/wang_full_fov_spatial_summaries_normalized.csv')
    umap_data = pd.read_csv('data/abm_umap_embedding_state_labeled.csv')
    spatial_merged = spatial_data.merge(
        clinical_data[['ImageID', 'Assigned_State']].drop_duplicates(),
        on='ImageID', how='inner'
    )
    return roi_data, clinical_data, spatial_merged, umap_data


# ============================================================================
# PANEL A: Within-Patient TME State Heterogeneity
# ============================================================================
def create_panel_a(ax, roi_data, clinical_data):
    phases = ['Baseline', 'On-treatment', 'Post-treatment']
    states = [1, 2, 3, 4, 5, 6]

    patient_dominant_states = {}
    for phase in phases:
        phase_data = clinical_data[clinical_data['biopsy_phase'] == phase]
        dominant = phase_data.groupby('PatientID')['Assigned_State'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        patient_dominant_states[phase] = dominant

    patient_phase_compositions = {}
    for phase in phases:
        phase_data = roi_data[roi_data['BiopsyPhase'] == phase]
        patient_states = phase_data.groupby(['PatientID', 'AssignedState']).size().unstack(fill_value=0)
        patient_states = patient_states.reindex(columns=states, fill_value=0)
        row_sums = patient_states.sum(axis=1)
        patient_props = patient_states.div(row_sums, axis=0)
        patient_phase_compositions[phase] = {'proportions': patient_props, 'counts': row_sums}

    # Enlarged layout: bars fill more of the axes
    phase_width = 0.29
    phase_starts = [0.04, 0.36, 0.68]

    for ph_idx, phase in enumerate(phases):
        if phase not in patient_phase_compositions:
            continue
        props = patient_phase_compositions[phase]['proportions']
        counts = patient_phase_compositions[phase]['counts']
        if len(props) == 0:
            continue

        sort_key = props.apply(lambda row: tuple(-row[s] for s in states), axis=1)
        sorted_patients = sort_key.sort_values().index.tolist()
        n_patients = len(sorted_patients)
        bar_width = phase_width / max(n_patients, 1)
        x_start = phase_starts[ph_idx]
        bar_bottom = 0.12
        bar_height_total = 0.76

        for p_idx, patient_id in enumerate(sorted_patients):
            x_pos = x_start + p_idx * bar_width
            y_bottom = bar_bottom
            for state in states:
                prop = props.loc[patient_id, state] if patient_id in props.index else 0
                if prop > 0:
                    height = prop * bar_height_total
                    rect = Rectangle((x_pos, y_bottom), bar_width * 0.9, height,
                        facecolor=STATE_COLORS[state], edgecolor='white',
                        linewidth=0.1, transform=ax.transAxes)
                    ax.add_patch(rect)
                    y_bottom += height

        # Rug plot
        rug_height = 0.04
        rug_bottom = 0.07
        for p_idx, patient_id in enumerate(sorted_patients):
            x_pos = x_start + p_idx * bar_width
            dom_state = int(patient_dominant_states[phase].get(patient_id, 1))
            rect = Rectangle((x_pos, rug_bottom), bar_width * 0.9, rug_height,
                facecolor=STATE_COLORS[dom_state], edgecolor='white',
                linewidth=0.1, transform=ax.transAxes)
            ax.add_patch(rect)

        if ph_idx == 0:
            ax.text(0.02, rug_bottom + rug_height/2, 'Patient\nState', fontsize=9,
                   ha='right', va='center', transform=ax.transAxes, style='italic')

        ax.text(x_start + phase_width/2, 0.94, phase, fontsize=12, fontweight='bold',
               ha='center', transform=ax.transAxes)
        ax.text(x_start + phase_width/2, 0.90, f'n={n_patients} patients',
               fontsize=10, ha='center', transform=ax.transAxes, color='gray')

    ax.text(0.005, 0.50, 'TME State\nProportion', fontsize=11, ha='center', va='center',
           rotation=90, transform=ax.transAxes)

    legend_x = 0.10
    for state in states:
        x_pos = legend_x + (state-1) * 0.12
        rect = Rectangle((x_pos, 0.005), 0.035, 0.03,
                         facecolor=STATE_COLORS[state], edgecolor='white', linewidth=0.5,
                         transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x_pos + 0.04, 0.02, f'S{state}', fontsize=10,
               ha='left', va='center', transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(r'\textbf{A. Within-Patient TME State Heterogeneity}', fontsize=13,
                fontweight='bold', loc='left', pad=5)


# ============================================================================
# PANEL B: Biological Profiles Heatmap
# ============================================================================
def create_panel_b_heatmap(ax, spatial_data, cax=None):
    features = {
        'effector_T_cell': r'Effector CD8$^+$ T cells',
        'exhausted_T_cell': r'Exhausted CD8$^+$ T cells',
        'malignant_epithelial_cell': 'Tumor cells',
        'other': 'Stromal/Other cells',
        'effector_T_cell_degree_centrality': r'CD8$^+$ T cell network connectivity',
        'im_effector_T_cell_malignant_epithelial_cell': 'Immune-Tumor interaction',
    }
    states = [1, 2, 3, 4, 5, 6]

    matrix = []
    feature_labels = []
    pvals = []
    for feat, label in features.items():
        if feat in spatial_data.columns:
            row = []
            groups = []
            for state in states:
                state_data = spatial_data[spatial_data['Assigned_State'] == state][feat]
                row.append(state_data.mean())
                groups.append(state_data.values)
            matrix.append(row)
            feature_labels.append(label)
            groups_clean = [g[~np.isnan(g)] for g in groups if len(g) > 0]
            if len(groups_clean) >= 2:
                try:
                    _, p = kruskal(*groups_clean)
                    pvals.append(p)
                except Exception:
                    pvals.append(1.0)
            else:
                pvals.append(1.0)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels([f'S{s}' for s in states], fontsize=11, fontweight='bold')

    # Build ytick labels with significance underneath
    sig_strings = []
    for p in pvals:
        sig_strings.append('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
    combined_labels = []
    for label, sig in zip(feature_labels, sig_strings):
        if sig:
            combined_labels.append(f'{label}\n({sig})')
        else:
            combined_labels.append(label)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(combined_labels, fontsize=10)

    if cax is not None:
        cbar = plt.colorbar(im, cax=cax)
    else:
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.06)
    #cbar.set_label('Z-score', fontsize=11)
    cbar.ax.set_title("Z-score", fontsize=11, pad=10, ha='left', va='bottom')
    cbar.ax.tick_params(labelsize=9)

    for j, state in enumerate(states):
        ax.add_patch(Rectangle((j-0.5, -0.7), 1, 0.4,
                               facecolor=STATE_COLORS[state],
                               edgecolor='white', linewidth=0.5, clip_on=False))

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(r'\textbf{B. TME State Biological Profiles}', fontsize=13,
                fontweight='bold', loc='left', pad=12)


# ============================================================================
# PANEL D: Patient-Level Balloon Plot
# ============================================================================
def create_panel_c(fig, gs_b, clinical_data):
    phases = ['Baseline', 'On-treatment', 'Post-treatment']
    states = [1, 2, 3, 4, 5, 6]
    clinical_order = ['pCR\nC', 'pCR\nC&I', 'RD\nC', 'RD\nC&I']

    patient_summary = clinical_data.groupby(
        ['PatientID', 'biopsy_phase', 'pCR', 'Arm']
    ).agg({
        'Assigned_State': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    patient_summary['clinical_group'] = patient_summary['pCR'] + '\n' + patient_summary['Arm']

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-3.5, vcenter=0, vmax=3.5)
    p_values = {}

    max_count = 0
    for phase in phases:
        phase_df = patient_summary[patient_summary['biopsy_phase'] == phase]
        if len(phase_df) > 0:
            contingency = pd.crosstab(phase_df['Assigned_State'], phase_df['clinical_group'])
            if len(contingency) > 0:
                max_count = max(max_count, contingency.values.max())

    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_b,
                                                 width_ratios=[1, 1, 1, 0.08],
                                                 wspace=0.20)
    axes = [fig.add_subplot(gs_inner[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs_inner[0, 3])

    for ph_idx, (phase, ax) in enumerate(zip(phases, axes)):
        phase_df = patient_summary[patient_summary['biopsy_phase'] == phase]
        if len(phase_df) == 0:
            continue

        contingency = pd.crosstab(phase_df['Assigned_State'], phase_df['clinical_group'])
        cols_present = [c for c in clinical_order if c in contingency.columns]
        contingency = contingency.reindex(columns=cols_present, fill_value=0)
        rowsums = contingency.sum(axis=1)
        valid_states = [int(i[0]+1) for i in np.argwhere(rowsums > 5)]
        contingency = contingency.reindex(index=valid_states, fill_value=0)

        _, p_val, _, expected = chi2_contingency(contingency)
        p_values[phase] = p_val

        row_totals = contingency.sum(axis=1)
        col_totals = contingency.sum(axis=0)
        grand_total = contingency.values.sum()

        if grand_total > 0:
            expected = np.outer(row_totals, col_totals) / grand_total
            with np.errstate(divide='ignore', invalid='ignore'):
                residuals = np.where(expected > 0,
                                    (contingency.values - expected) / np.sqrt(expected), 0)
        else:
            residuals = np.zeros_like(contingency.values, dtype=float)

        for i, state in enumerate(states):
            for j, group in enumerate(cols_present):
                count = contingency.loc[state, group] if state in contingency.index else 0
                resid = residuals[i, j] if i < len(residuals) else 0
                size = (count / max_count) * 800 if max_count > 0 else 0
                if size > 0:
                    color = cmap(norm(resid))
                    ax.scatter(j, 5-i, s=size, c=[color], edgecolors='black', linewidths=0.5)
                    ax.text(j, 5-i, str(int(count)), fontsize=9, ha='center', va='center',
                           fontweight='bold', color='white' if abs(resid) > 1.5 else 'black')
                else:
                    ax.scatter(j, 5-i, s=20, c='white', edgecolors='gray', linewidths=0.3, alpha=0.5)

        ax.set_xlim(-0.6, len(cols_present)-0.4)
        ax.set_ylim(-0.6, 5.6)
        ax.set_xticks(range(len(cols_present)))
        ax.set_xticklabels([tex_escape(c) for c in cols_present], fontsize=10)
        ax.set_yticks(range(6))
        ax.set_yticklabels([f'S{s}' for s in reversed(states)], fontsize=10)
        ax.set_aspect('equal')

        p_val = p_values.get(phase, 1.0) * 3
        sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ' (NS)'
        if ph_idx == 0:
            ax.set_title(r'\textbf{D. TME State Enrichment by Clinical Group (Patient-Level)}'
                         + f'\n{phase}\nn={int(grand_total)}, p={p_val:.3f}{sig}',
                        fontsize=13, loc='left')
        else:
            ax.set_title(f'{phase}\nn={int(grand_total)}, p={p_val:.3f}{sig}', fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ph_idx == 0:
            ax.set_ylabel('TME State', fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Std. Residual', fontsize=10)
    cbar.ax.tick_params(labelsize=9)


# ============================================================================
# PANEL E: State 1 Prevalence Over Treatment
# ============================================================================
def create_panel_d(ax, clinical_data):
    patient_states = clinical_data.groupby(['PatientID', 'biopsy_phase', 'pCR', 'Arm']).agg({
        'Assigned_State': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    pivot = patient_states.pivot_table(
        index=['PatientID', 'pCR', 'Arm'], columns='biopsy_phase',
        values='Assigned_State', aggfunc='first'
    ).reset_index()
    complete = pivot.dropna(subset=['Baseline', 'On-treatment', 'Post-treatment']).copy()

    complete['s1_baseline'] = (complete['Baseline'] == 1).astype(int)
    complete['s1_ontreatment'] = (complete['On-treatment'] == 1).astype(int)
    complete['s1_posttreatment'] = (complete['Post-treatment'] == 1).astype(int)

    phases = ['Baseline', 'On-treatment', 'Post-treatment']
    phase_labels = ['BL', 'OT', 'PT']
    x_positions = [0, 1, 2]

    for arm in ['C', 'C&I']:
        arm_data = complete[complete['Arm'] == arm]
        n_total = len(arm_data)
        prevalences = []
        for phase in phases:
            col = f's1_{phase.lower().replace("-", "")}'
            prevalences.append(arm_data[col].sum() / n_total * 100)

        line_style = '-' if arm == 'C&I' else '--'
        marker = 'o' if arm == 'C&I' else 's'
        ax.plot(x_positions, prevalences, line_style, marker=marker, markersize=10,
                color=ARM_COLORS[arm], linewidth=2.5, label=tex_escape(f'{arm} (n={n_total})'),
                markeredgecolor='white', markeredgewidth=1)

        for _, (x, y, phase) in enumerate(zip(x_positions, prevalences, phases)):
            if (arm == 'C&I') and (phase == 'Baseline'):
                yoffset, xoffset = -13, -10
            elif (arm == 'C') and (phase == 'Baseline'):
                yoffset, xoffset = 10, -10
            else:
                yoffset = 7 if arm == 'C&I' else -18
                xoffset = 0
            ax.annotate(f'{y:.0f}\\%', (x, y), textcoords='offset points',
                        xytext=(xoffset, yoffset), ha='center', fontsize=11, fontweight='bold',
                        color=ARM_COLORS[arm])

    c_pt = complete[complete['Arm'] == 'C']['s1_posttreatment'].mean() * 100
    ci_pt = complete[complete['Arm'] == 'C&I']['s1_posttreatment'].mean() * 100
    diff = ci_pt - c_pt
    ax.plot([2.1, 2.1], [c_pt, ci_pt], 'k-', linewidth=1.5)
    ax.plot([2.06, 2.14], [c_pt, c_pt], 'k-', linewidth=1.5)
    ax.plot([2.06, 2.14], [ci_pt, ci_pt], 'k-', linewidth=1.5)
    ax.text(2.18, (ci_pt + c_pt)/2, f'+{diff:.0f}\\%', fontsize=12, fontweight='bold',
            va='center', color='#1e40af')

    c_data = complete[complete['Arm'] == 'C']
    ci_data = complete[complete['Arm'] == 'C&I']
    table = [[ci_data['s1_posttreatment'].sum(), len(ci_data) - ci_data['s1_posttreatment'].sum()],
             [c_data['s1_posttreatment'].sum(), len(c_data) - c_data['s1_posttreatment'].sum()]]
    _, p_val = fisher_exact(table)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(phase_labels, fontsize=12)
    ax.set_ylabel(r'Patients in State 1 (\%)', fontsize=12)
    ax.set_ylim(45, 95)
    ax.set_xlim(-0.3, 2.5)
    ax.text(0.02, 0.8, f'PT: p={p_val:.3f}', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#dbeafe', edgecolor='#3b82f6', alpha=0.9))
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title(r'\textbf{E. ICB Maintains State 1 Prevalence}', fontsize=13,
                fontweight='bold', loc='left')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)


# ============================================================================
# PANEL C: Spatial Organization (State 1 vs State 6)
# ============================================================================
def create_panel_e_spatial(ax, spatial_data):
    features = {
        'effector_T_cell_degree_centrality': 'T cell\nConnectivity',
        'effector_T_cell_average_clustering': 'T cell\nClustering',
        'im_effector_T_cell_malignant_epithelial_cell': 'Immune-Tumor\nInteraction',
    }
    x = np.arange(len(features))
    width = 0.35

    s1_data = spatial_data[spatial_data['Assigned_State'] == 1]
    s6_data = spatial_data[spatial_data['Assigned_State'] == 6]
    s1_vals = [s1_data[f].mean() for f in features.keys()]
    s6_vals = [s6_data[f].mean() for f in features.keys()]

    ax.bar(x - width/2, s1_vals, width, color=STATE_COLORS[1],
           label=f'State 1 (n={len(s1_data)})', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, s6_vals, width, color=STATE_COLORS[6],
           label=f'State 6 (n={len(s6_data)})', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(features.values(), fontsize=10)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1.6, 1.2)

    for i, feat in enumerate(features.keys()):
        s1 = s1_data[feat].dropna()
        s6 = s6_data[feat].dropna()
        _, p = mannwhitneyu(s1, s6)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_max = max(s1_vals[i], s6_vals[i]) + 0.15
        ax.text(i, y_max + 0.1, sig, ha='center', fontsize=12, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_title(r'\textbf{C. Spatial Organization: S1 vs S6}', fontsize=13,
                fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================================
# PANEL F: Exemplar Patient Trajectories (2x2 grid)
# ============================================================================
def create_panel_f_trajectories(fig, gs_t, abm_data, clinical_data):
    exemplars = [
        ('NT023', 'pCR', 'C&I', 'Stable Immune Hot'),
        ('NT011', 'pCR', 'C', 'Recovery'),
        ('NT056', 'pCR', 'C', 'Paradox (Resolved)'),
        ('NT001', 'RD', 'C&I', 'Non-Responder'),
    ]

    gs_inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_t,
                                                 wspace=0.08, hspace=0.18)
    axes = []

    for idx, (patient_id, response, _, _) in enumerate(exemplars):
        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(gs_inner[row, col])
        axes.append(ax)

        ax.scatter(abm_data['x'], abm_data['y'],
                  c=[STATE_COLORS.get(int(s), 'gray') for s in abm_data['state_label']],
                  s=2, alpha=0.3)

        patient_clinical = clinical_data[clinical_data['PatientID'] == patient_id]
        phase_markers = {'Baseline': 'o', 'On-treatment': '^', 'Post-treatment': 's'}
        phase_order = ['Baseline', 'On-treatment', 'Post-treatment']

        trajectory_points = []
        for phase in phase_order:
            phase_data = patient_clinical[patient_clinical['biopsy_phase'] == phase]
            if len(phase_data) > 0:
                xp = phase_data['umap_x'].mean()
                yp = phase_data['umap_y'].mean()
                state = int(phase_data['Assigned_State'].mode()[0])
                trajectory_points.append((xp, yp, state, phase))

        if len(trajectory_points) > 1:
            xs = [p[0] for p in trajectory_points]
            ys = [p[1] for p in trajectory_points]
            ax.plot(xs, ys, 'k-', linewidth=3, alpha=0.7, zorder=5)
            ax.plot(xs, ys, '-', c='white', linewidth=1, alpha=1.0, zorder=7)

        for xp, yp, state, phase in trajectory_points:
            marker = phase_markers[phase]
            ax.scatter([xp], [yp], c=[STATE_COLORS[state]], s=120, marker=marker,
                      edgecolors='black', linewidths=1.5, zorder=10)

        if len(trajectory_points) > 0:
            for i, (xp, yp, state, phase) in enumerate(trajectory_points):
                if phase.lower() == 'baseline':
                    label_text = 'BL'
                elif 'on' in phase.lower():
                    label_text = 'OT'
                else:
                    label_text = 'PT'
                offset = (5, 5) if i % 2 == 0 else (-15, -15)
                ax.annotate(label_text, (xp, yp), xytext=offset, textcoords='offset points',
                           fontsize=9, fontweight='bold')

        resp_color = RESPONSE_COLORS[response]
        ax.set_title(f'{patient_id} ({response})', fontsize=12, fontweight='bold',
                    color=resp_color, pad=3)

        ax.set_xlim(abm_data['x'].min() - 1, abm_data['x'].max() + 1)
        ax.set_ylim(abm_data['y'].min() - 1, abm_data['y'].max() + 1)
        ax.set_aspect('equal')
        ax.axis('off')

    # Phase legend in bottom-left panel
    axes[2].text(0.05, 0.05, r'$\bullet$BL $\blacktriangle$OT $\blacksquare$PT', fontsize=12,
                transform=axes[2].transAxes, ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Panel title above top-left
    axes[0].text(0.0, 1.12, r'\textbf{F. Exemplar Patient Trajectories}', fontsize=13,
                fontweight='bold', transform=axes[0].transAxes)
    return axes


# ============================================================================
# FIGURE 6 LEGEND
# ============================================================================
def create_f6_legend(fig, legend_offset: float = 0.05):
    row_height = 0.016
    swatch_w = 0.018
    swatch_h = 0.012

    y = legend_offset
    fig.text(0.08, y + 0.003, 'Arm:', fontsize=10, fontweight='bold', va='center')
    x_cursor = 0.125
    for arm, color in ARM_COLORS.items():
        rect = Rectangle((x_cursor, y), swatch_w, swatch_h,
                         facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.8,
                         transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(x_cursor + swatch_w + 0.005, y + swatch_h / 2, tex_escape(arm), fontsize=9,
                ha='left', va='center')
        x_cursor += 0.065

    y += row_height
    fig.text(0.08, y + 0.003, 'Response:', fontsize=10, fontweight='bold', va='center')
    x_cursor = 0.165
    for resp, color in RESPONSE_COLORS.items():
        rect = Rectangle((x_cursor, y), swatch_w, swatch_h,
                         facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.8,
                         transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(x_cursor + swatch_w + 0.005, y + swatch_h / 2, resp, fontsize=9,
                ha='left', va='center')
        x_cursor += 0.065

    y += row_height
    fig.text(0.08, y + 0.003, 'TME States:', fontsize=10, fontweight='bold', va='center')
    x_cursor = 0.175
    for state in [1, 2, 3, 4, 5, 6]:
        rect = Rectangle((x_cursor, y), swatch_w, swatch_h,
                         facecolor=STATE_COLORS[state], edgecolor='white', linewidth=0.5,
                         transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(x_cursor + swatch_w + 0.005, y + swatch_h / 2, f'S{state}', fontsize=9,
                ha='left', va='center')
        x_cursor += 0.05


# ============================================================================
# MAIN FIGURE ASSEMBLY
# ============================================================================
def create_figure_6():
    print("=" * 60)
    print("Creating Figure 5")
    print("=" * 60)

    print("\nLoading data...")
    roi_data, clinical_data, spatial_data, umap_data = load_all_data()
    print(f"  Loaded {len(roi_data)} ROIs, {len(clinical_data)} samples, {len(spatial_data)} spatial records")

    fig = plt.figure(figsize=(12, 19))

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[0.9, 1.1, 0.85, 1.6],
                           width_ratios=[1, 1, 1],
                           left=0.02, right=0.98, top=0.975, bottom=0.055,
                           hspace=0.25, 
                           #hspace=0.15,
                           wspace=0.30)

    # --- Panel A (full width) ---
    ax_a = fig.add_subplot(gs[0, :])
    print("Creating Panel A...")
    create_panel_a(ax_a, roi_data, clinical_data)

    # --- Panel B (2/3 width) with manual position for label alignment ---
    ax_b_placeholder = fig.add_subplot(gs[1, 0:2])
    gs_pos = ax_b_placeholder.get_position()
    fig.delaxes(ax_b_placeholder)

    b_shift = 0.12  # slightly larger shift for bigger ytick labels
    b_left = gs_pos.x0 + b_shift
    b_width = gs_pos.width - b_shift - 0.03
    cbar_left = b_left + b_width + 0.008
    cbar_width = 0.012

    ax_b = fig.add_axes([b_left, gs_pos.y0, b_width, gs_pos.height])
    cax_b = fig.add_axes([cbar_left, gs_pos.y0 + gs_pos.height * 0.15,
                          cbar_width, gs_pos.height * 0.7])

    print("Creating Panel B...")
    create_panel_b_heatmap(ax_b, spatial_data, cax=cax_b)

    # --- Panel C (1/3 width) ---
    ax_c = fig.add_subplot(gs[1, 2])
    print("Creating Panel C...")
    create_panel_e_spatial(ax_c, spatial_data)

    # --- Panel D (2/3 width) ---
    print("Creating Panel D...")
    create_panel_c(fig, gs[2, 0:2], clinical_data)

    # --- Panel E (1/3 width) ---
    ax_e = fig.add_subplot(gs[2, 2])
    print("Creating Panel E...")
    create_panel_d(ax_e, clinical_data)

    # --- Panel F (full width, 2x2) ---
    print("Creating Panel F...")
    create_panel_f_trajectories(fig, gs[3, :], umap_data, clinical_data)

    # --- Legend ---
    create_f6_legend(fig, 0.01)

    # --- Save ---
    print("\nSaving figures...")
    os.makedirs('outputs', exist_ok=True)
    for fmt, dpi_val in [('png', 300), ('svg', None), ('pdf', None)]:
        fname = f'outputs/figure_5.{fmt}'
        print(f"  Saving {fname}...")
        save_kwargs = dict(bbox_inches='tight', facecolor='white')
        if dpi_val:
            save_kwargs['dpi'] = dpi_val
        plt.savefig(fname, **save_kwargs)

    plt.close()
    print("\n" + "=" * 60)
    print("Figure 5 complete.")
    print("=" * 60)


if __name__ == '__main__':
    create_figure_6()
