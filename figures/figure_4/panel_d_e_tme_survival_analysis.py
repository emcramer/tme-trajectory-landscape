#!/usr/bin/env python3
"""
TME State Proportions and Disease-Free Survival Analysis
=========================================================

This script performs survival analysis examining how TME state proportions
relate to disease-free survival in TNBC patients.

Stratification groups:
1. High TME State 1 proportion
2. High TME State 6 proportion  
3. High TME State 2 proportion
4. Low amounts of States 1, 2, and 6 (i.e., predominantly States 3/5)

Analyses:
- Kaplan-Meier survival curves with log-rank tests
- Optimal cutpoint determination using maximally selected rank statistics
- Median cutpoint analysis for comparison
- Cox proportional hazards modeling

Hypotheses:
- Higher State 1 proportion → longer disease-free survival (favorable)
- Higher State 6 proportion → shorter disease-free survival (unfavorable)
- Other states → intermediate prognosis

Author: Eric Cramer
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Survival analysis packages
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times

# =============================================================================
# Configuration and Style Settings
# =============================================================================

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 100

# Define colors consistent with TME state palette
# State 1: favorable (blue/teal)
# State 2: intermediate (green)
# State 6: unfavorable/cold (gold/amber)
# Other: mixed/unclear (gray/purple)
TME_COLORS = {
    'State_1_High': '#2E86AB',   # Blue - favorable
    'State_2_High': '#55a868',   # Green - intermediate
    'State_6_High': '#D4A017',   # Gold - unfavorable (cold)
    'Other': '#8B7355'           # Brown - other states dominant
}

# Output paths
OUTPUT_DIR = 'outputs'

# =============================================================================
# Helper Functions
# =============================================================================

def find_optimal_cutpoint(data, time_col, event_col, marker_col, min_group_size=5):
    """
    Find optimal cutpoint for a continuous marker using maximally selected
    rank statistics (similar to survminer's surv_cutpoint in R).
    
    This finds the cutpoint that maximizes the log-rank test statistic,
    effectively finding the split that best separates survival curves.
    
    Parameters:
    -----------
    data : DataFrame
        Data containing survival info and marker
    time_col : str
        Column name for survival time
    event_col : str
        Column name for event indicator
    marker_col : str
        Column name for continuous marker to find cutpoint
    min_group_size : int
        Minimum number of patients in each group
        
    Returns:
    --------
    dict with optimal_cutpoint, test_statistic, p_value
    """
    marker_values = data[marker_col].values
    times = data[time_col].values
    events = data[event_col].values
    
    # Get unique sorted values as potential cutpoints
    unique_vals = np.unique(marker_values)
    
    # We need at least min_group_size in each group
    # So we test cutpoints that leave enough samples on each side
    best_stat = -np.inf
    best_cutpoint = np.median(marker_values)
    best_pvalue = 1.0
    
    results = []
    
    for cutpoint in unique_vals:
        high_mask = marker_values >= cutpoint
        low_mask = ~high_mask
        
        n_high = high_mask.sum()
        n_low = low_mask.sum()
        
        # Skip if groups are too small
        if n_high < min_group_size or n_low < min_group_size:
            continue
        
        # Perform log-rank test
        try:
            result = logrank_test(
                times[high_mask], times[low_mask],
                events[high_mask], events[low_mask]
            )
            
            results.append({
                'cutpoint': cutpoint,
                'test_stat': result.test_statistic,
                'p_value': result.p_value,
                'n_high': n_high,
                'n_low': n_low
            })
            
            if result.test_statistic > best_stat:
                best_stat = result.test_statistic
                best_cutpoint = cutpoint
                best_pvalue = result.p_value
                
        except Exception as e:
            continue
    
    return {
        'optimal_cutpoint': best_cutpoint,
        'test_statistic': best_stat,
        'p_value': best_pvalue,
        'all_results': pd.DataFrame(results) if results else None
    }


def assign_tme_groups(df, state1_cut, state2_cut, state6_cut, method='priority'):
    """
    Assign patients to TME groups based on state proportion cutpoints.
    
    Priority assignment (if multiple states are "high"):
    1. If State 6 >= cutpoint → State_6_High (unfavorable takes precedence for clinical relevance)
    2. Elif State 1 >= cutpoint → State_1_High  
    3. Elif State 2 >= cutpoint → State_2_High
    4. Else → Other (predominantly States 3/5)
    
    This priority scheme ensures we capture the prognostically important groups.
    """
    groups = []
    
    for idx, row in df.iterrows():
        s1 = row['State_1_Prop']
        s2 = row['State_2_Prop']
        s6 = row['State_6_Prop']
        
        # Priority-based assignment
        # We prioritize State 6 (unfavorable) to ensure we identify high-risk patients
        if s6 >= state6_cut:
            groups.append('State_6_High')
        elif s1 >= state1_cut:
            groups.append('State_1_High')
        elif s2 >= state2_cut:
            groups.append('State_2_High')
        else:
            groups.append('Other')
    
    return groups


def plot_km_curves(df, group_col, time_col, event_col, ax, title, colors_dict,
                   show_ci=True, show_censored=True):
    """
    Plot Kaplan-Meier survival curves for multiple groups.
    
    Parameters:
    -----------
    df : DataFrame
        Data with survival info and group assignments
    group_col : str
        Column containing group labels
    time_col : str
        Column with survival times
    event_col : str
        Column with event indicators
    ax : matplotlib axis
        Axis to plot on
    title : str
        Plot title
    colors_dict : dict
        Mapping of group names to colors
    """
    kmf = KaplanMeierFitter()
    
    groups = df[group_col].unique()
    
    for group in groups:
        mask = df[group_col] == group
        group_data = df[mask]
        
        if len(group_data) < 2:
            continue
            
        kmf.fit(
            group_data[time_col],
            group_data[event_col],
            label=f"{group} (n={len(group_data)})"
        )
        
        color = colors_dict.get(group, '#888888')
        
        kmf.plot_survival_function(
            ax=ax,
            ci_show=show_ci,
            color=color,
            linewidth=2.5
        )
        
        if show_censored:
            # Mark censored observations
            censored = group_data[group_data[event_col] == 0]
            if len(censored) > 0:
                # Get survival probability at censoring times
                for t in censored[time_col]:
                    try:
                        surv_prob = kmf.survival_function_at_times(t).values[0]
                        ax.plot(t, surv_prob, '|', color=color, markersize=8, markeredgewidth=2)
                    except:
                        pass
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Disease-Free Survival Probability', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    
    return ax


# =============================================================================
# Load and Prepare Data
# =============================================================================

print("=" * 80)
print("TME STATE PROPORTIONS AND DISEASE-FREE SURVIVAL ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv('data/mibi_roi_tme_proportions_clinical.csv')

# The survival time column appears to be 'Time' based on earlier inspection
# Event column is 'Event' (1 = recurrence/event, 0 = censored)
time_col = 'Time'
event_col = 'Event'

print(f"\n--- Dataset Overview ---")
print(f"Total patients: {len(df)}")
print(f"Events (recurrences): {df[event_col].sum()}")
print(f"Censored: {(df[event_col] == 0).sum()}")
print(f"Median follow-up time: {df[time_col].median():.0f} days ({df[time_col].median()/365.25:.1f} years)")
print(f"Follow-up range: {df[time_col].min():.0f} - {df[time_col].max():.0f} days")

print(f"\n--- TME State Proportion Distributions ---")
for state in ['State_1_Prop', 'State_2_Prop', 'State_3_Prop', 'State_5_Prop', 'State_6_Prop']:
    print(f"  {state}: median={df[state].median():.3f}, mean={df[state].mean():.3f}, "
          f"range=[{df[state].min():.3f}, {df[state].max():.3f}]")

# =============================================================================
# PART 1: Find Optimal Cutpoints
# =============================================================================

print("\n" + "=" * 80)
print("PART 1: OPTIMAL CUTPOINT DETERMINATION")
print("=" * 80)
print("\nUsing maximally selected rank statistics to find cutpoints that")
print("best separate survival curves for each TME state proportion.\n")

# Find optimal cutpoints for each state
optimal_cuts = {}

for state_col in ['State_1_Prop', 'State_2_Prop', 'State_6_Prop']:
    result = find_optimal_cutpoint(df, time_col, event_col, state_col, min_group_size=5)
    optimal_cuts[state_col] = result
    
    print(f"--- {state_col} ---")
    print(f"  Optimal cutpoint: {result['optimal_cutpoint']:.4f}")
    print(f"  Log-rank χ² statistic: {result['test_statistic']:.3f}")
    print(f"  P-value: {result['p_value']:.4f}")
    
    # Count patients above/below cutpoint
    n_high = (df[state_col] >= result['optimal_cutpoint']).sum()
    n_low = (df[state_col] < result['optimal_cutpoint']).sum()
    print(f"  Patients ≥ cutpoint: {n_high}")
    print(f"  Patients < cutpoint: {n_low}\n")

# Also compute median cutpoints for comparison
median_cuts = {
    'State_1_Prop': df['State_1_Prop'].median(),
    'State_2_Prop': df['State_2_Prop'].median(),
    'State_6_Prop': df['State_6_Prop'].median()
}

print("--- Median Cutpoints (for comparison) ---")
for state_col, med in median_cuts.items():
    print(f"  {state_col}: {med:.4f}")

# =============================================================================
# PART 2: Assign TME Groups
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: TME GROUP ASSIGNMENT")
print("=" * 80)

# Assign groups using OPTIMAL cutpoints
df['TME_Group_Optimal'] = assign_tme_groups(
    df,
    state1_cut=optimal_cuts['State_1_Prop']['optimal_cutpoint'],
    state2_cut=optimal_cuts['State_2_Prop']['optimal_cutpoint'],
    state6_cut=optimal_cuts['State_6_Prop']['optimal_cutpoint']
)

# Assign groups using MEDIAN cutpoints
df['TME_Group_Median'] = assign_tme_groups(
    df,
    state1_cut=median_cuts['State_1_Prop'],
    state2_cut=median_cuts['State_2_Prop'],
    state6_cut=median_cuts['State_6_Prop']
)

print("\n--- Group Distribution (Optimal Cutpoints) ---")
opt_counts = df['TME_Group_Optimal'].value_counts()
for group in ['State_1_High', 'State_2_High', 'State_6_High', 'Other']:
    if group in opt_counts.index:
        n = opt_counts[group]
        events = df[df['TME_Group_Optimal'] == group][event_col].sum()
        print(f"  {group}: n={n} ({n/len(df)*100:.1f}%), events={events}")

print("\n--- Group Distribution (Median Cutpoints) ---")
med_counts = df['TME_Group_Median'].value_counts()
for group in ['State_1_High', 'State_2_High', 'State_6_High', 'Other']:
    if group in med_counts.index:
        n = med_counts[group]
        events = df[df['TME_Group_Median'] == group][event_col].sum()
        print(f"  {group}: n={n} ({n/len(df)*100:.1f}%), events={events}")

# =============================================================================
# PART 3: Kaplan-Meier Analysis
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: KAPLAN-MEIER SURVIVAL ANALYSIS")
print("=" * 80)

# --- 3A: KM Analysis with Optimal Cutpoints ---
print("\n--- Kaplan-Meier Analysis (Optimal Cutpoints) ---\n")

# Calculate median survival for each group
kmf = KaplanMeierFitter()
print("Median Disease-Free Survival by Group (Optimal Cutpoints):")

median_survivals_opt = {}
for group in ['State_1_High', 'State_2_High', 'State_6_High', 'Other']:
    mask = df['TME_Group_Optimal'] == group
    if mask.sum() >= 2:
        kmf.fit(df.loc[mask, time_col], df.loc[mask, event_col])
        median_surv = kmf.median_survival_time_
        ci = median_survival_times(kmf.confidence_interval_)
        median_survivals_opt[group] = median_surv
        
        if np.isinf(median_surv):
            print(f"  {group}: Median not reached (>{df[time_col].max():.0f} days)")
        else:
            print(f"  {group}: {median_surv:.0f} days ({median_surv/365.25:.1f} years)")

# Multivariate log-rank test
groups_present = df['TME_Group_Optimal'].unique()
if len(groups_present) > 1:
    result_opt = multivariate_logrank_test(
        df[time_col], df['TME_Group_Optimal'], df[event_col]
    )
    print(f"\nMultivariate Log-Rank Test (Optimal):")
    print(f"  Test statistic: {result_opt.test_statistic:.3f}")
    print(f"  P-value: {result_opt.p_value:.4f}")
    
    if result_opt.p_value < 0.05:
        print(f"  ✓ SIGNIFICANT difference between TME groups")
    else:
        print(f"  ✗ No significant difference between TME groups")

# --- 3B: KM Analysis with Median Cutpoints ---
print("\n--- Kaplan-Meier Analysis (Median Cutpoints) ---\n")

print("Median Disease-Free Survival by Group (Median Cutpoints):")
median_survivals_med = {}
for group in ['State_1_High', 'State_2_High', 'State_6_High', 'Other']:
    mask = df['TME_Group_Median'] == group
    if mask.sum() >= 2:
        kmf.fit(df.loc[mask, time_col], df.loc[mask, event_col])
        median_surv = kmf.median_survival_time_
        median_survivals_med[group] = median_surv
        
        if np.isinf(median_surv):
            print(f"  {group}: Median not reached (>{df[time_col].max():.0f} days)")
        else:
            print(f"  {group}: {median_surv:.0f} days ({median_surv/365.25:.1f} years)")

groups_present = df['TME_Group_Median'].unique()
if len(groups_present) > 1:
    result_med = multivariate_logrank_test(
        df[time_col], df['TME_Group_Median'], df[event_col]
    )
    print(f"\nMultivariate Log-Rank Test (Median):")
    print(f"  Test statistic: {result_med.test_statistic:.3f}")
    print(f"  P-value: {result_med.p_value:.4f}")
    
    if result_med.p_value < 0.05:
        print(f"  ✓ SIGNIFICANT difference between TME groups")
    else:
        print(f"  ✗ No significant difference between TME groups")

# --- Pairwise comparisons ---
print("\n--- Pairwise Log-Rank Tests (Optimal Cutpoints) ---")
groups_to_compare = [g for g in ['State_1_High', 'State_6_High', 'State_2_High', 'Other'] 
                     if g in df['TME_Group_Optimal'].unique()]

pairwise_results = []
for i, g1 in enumerate(groups_to_compare):
    for g2 in groups_to_compare[i+1:]:
        mask1 = df['TME_Group_Optimal'] == g1
        mask2 = df['TME_Group_Optimal'] == g2
        
        if mask1.sum() >= 2 and mask2.sum() >= 2:
            result = logrank_test(
                df.loc[mask1, time_col], df.loc[mask2, time_col],
                df.loc[mask1, event_col], df.loc[mask2, event_col]
            )
            pairwise_results.append({
                'Comparison': f"{g1} vs {g2}",
                'Chi-square': result.test_statistic,
                'P-value': result.p_value
            })
            sig = '*' if result.p_value < 0.05 else ''
            print(f"  {g1} vs {g2}: χ²={result.test_statistic:.3f}, p={result.p_value:.4f} {sig}")

# =============================================================================
# PART 4: Cox Proportional Hazards Model
# =============================================================================

print("\n" + "=" * 80)
print("PART 4: COX PROPORTIONAL HAZARDS MODEL")
print("=" * 80)

# --- 4A: Cox PH with TME Groups (Categorical) ---
print("\n--- Cox PH Model: TME Groups (Optimal Cutpoints) ---")
print("Reference group: State_1_High (hypothesized favorable prognosis)\n")

# Create dummy variables with State_1_High as reference
cox_df_opt = df[[time_col, event_col, 'TME_Group_Optimal']].copy()
cox_df_opt = pd.get_dummies(cox_df_opt, columns=['TME_Group_Optimal'], drop_first=False)

# Manually set State_1_High as reference by dropping it
dummy_cols_opt = [c for c in cox_df_opt.columns if c.startswith('TME_Group_Optimal_')]
ref_col = 'TME_Group_Optimal_State_1_High'

if ref_col in dummy_cols_opt:
    dummy_cols_opt.remove(ref_col)
    
    # Fit Cox model
    cph_opt = CoxPHFitter()
    cph_opt.fit(cox_df_opt[[time_col, event_col] + dummy_cols_opt], 
                duration_col=time_col, event_col=event_col)
    
    print("Cox PH Model Results (Reference: State_1_High):")
    print("-" * 70)
    
    cox_summary_opt = cph_opt.summary
    for var in dummy_cols_opt:
        group_name = var.replace('TME_Group_Optimal_', '')
        hr = cox_summary_opt.loc[var, 'exp(coef)']
        hr_lower = cox_summary_opt.loc[var, 'exp(coef) lower 95%']
        hr_upper = cox_summary_opt.loc[var, 'exp(coef) upper 95%']
        pval = cox_summary_opt.loc[var, 'p']
        
        sig = '*' if pval < 0.05 else ''
        print(f"  {group_name}:")
        print(f"    HR = {hr:.3f} (95% CI: {hr_lower:.3f} - {hr_upper:.3f})")
        print(f"    P-value = {pval:.4f} {sig}")
        print()
    
    print(f"Model Concordance (C-index): {cph_opt.concordance_index_:.3f}")
    print(f"Log-likelihood ratio test p-value: {cph_opt.log_likelihood_ratio_test().p_value:.4f}")
else:
    print("  Note: State_1_High group not present in data")

# --- 4B: Cox PH with Median Cutpoints ---
print("\n--- Cox PH Model: TME Groups (Median Cutpoints) ---")
print("Reference group: State_1_High\n")

cox_df_med = df[[time_col, event_col, 'TME_Group_Median']].copy()
cox_df_med = pd.get_dummies(cox_df_med, columns=['TME_Group_Median'], drop_first=False)

dummy_cols_med = [c for c in cox_df_med.columns if c.startswith('TME_Group_Median_')]
ref_col_med = 'TME_Group_Median_State_1_High'

if ref_col_med in dummy_cols_med:
    dummy_cols_med.remove(ref_col_med)
    
    cph_med = CoxPHFitter()
    cph_med.fit(cox_df_med[[time_col, event_col] + dummy_cols_med],
                duration_col=time_col, event_col=event_col)
    
    print("Cox PH Model Results (Reference: State_1_High):")
    print("-" * 70)
    
    cox_summary_med = cph_med.summary
    for var in dummy_cols_med:
        group_name = var.replace('TME_Group_Median_', '')
        hr = cox_summary_med.loc[var, 'exp(coef)']
        hr_lower = cox_summary_med.loc[var, 'exp(coef) lower 95%']
        hr_upper = cox_summary_med.loc[var, 'exp(coef) upper 95%']
        pval = cox_summary_med.loc[var, 'p']
        
        sig = '*' if pval < 0.05 else ''
        print(f"  {group_name}:")
        print(f"    HR = {hr:.3f} (95% CI: {hr_lower:.3f} - {hr_upper:.3f})")
        print(f"    P-value = {pval:.4f} {sig}")
        print()
    
    print(f"Model Concordance (C-index): {cph_med.concordance_index_:.3f}")
    print(f"Log-likelihood ratio test p-value: {cph_med.log_likelihood_ratio_test().p_value:.4f}")

# --- 4C: Cox PH with Continuous Proportions ---
print("\n--- Cox PH Model: Continuous State Proportions ---\n")

cox_df_cont = df[[time_col, event_col, 'State_1_Prop', 'State_2_Prop', 'State_6_Prop']].copy()

cph_cont = CoxPHFitter()
cph_cont.fit(cox_df_cont, duration_col=time_col, event_col=event_col)

print("Cox PH Model with Continuous Proportions:")
print("-" * 70)
print(f"{'Variable':<20} {'HR':>10} {'95% CI':>20} {'P-value':>12}")
print("-" * 70)

for var in ['State_1_Prop', 'State_2_Prop', 'State_6_Prop']:
    hr = cph_cont.summary.loc[var, 'exp(coef)']
    hr_lower = cph_cont.summary.loc[var, 'exp(coef) lower 95%']
    hr_upper = cph_cont.summary.loc[var, 'exp(coef) upper 95%']
    pval = cph_cont.summary.loc[var, 'p']
    sig = '*' if pval < 0.05 else ''
    
    ci_str = f"({hr_lower:.3f} - {hr_upper:.3f})"
    print(f"{var:<20} {hr:>10.3f} {ci_str:>20} {pval:>10.4f} {sig}")

print("-" * 70)
print(f"\nModel Concordance (C-index): {cph_cont.concordance_index_:.3f}")
print(f"Log-likelihood ratio test p-value: {cph_cont.log_likelihood_ratio_test().p_value:.4f}")

print("\nInterpretation:")
print("  HR < 1: Higher proportion associated with LOWER hazard (better survival)")
print("  HR > 1: Higher proportion associated with HIGHER hazard (worse survival)")

# =============================================================================
# PART 5: Generate Figures
# =============================================================================

print("\n" + "=" * 80)
print("PART 5: GENERATING FIGURES")
print("=" * 80)

# --- Figure 1: Main KM Curves (Optimal and Median Cutpoints) ---
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Optimal cutpoints
ax1 = axes[0]
plot_km_curves(df, 'TME_Group_Optimal', time_col, event_col, ax1,
               'A. Optimal Cutpoints', TME_COLORS)

# Add log-rank p-value
if len(df['TME_Group_Optimal'].unique()) > 1:
    result = multivariate_logrank_test(df[time_col], df['TME_Group_Optimal'], df[event_col])
    p_text = f"Log-rank p = {result.p_value:.4f}" if result.p_value >= 0.0001 else "Log-rank p < 0.0001"
    ax1.text(0.95, 0.95, p_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: Median cutpoints
ax2 = axes[1]
plot_km_curves(df, 'TME_Group_Median', time_col, event_col, ax2,
               'B. Median Cutpoints', TME_COLORS)

if len(df['TME_Group_Median'].unique()) > 1:
    result = multivariate_logrank_test(df[time_col], df['TME_Group_Median'], df[event_col])
    p_text = f"Log-rank p = {result.p_value:.4f}" if result.p_value >= 0.0001 else "Log-rank p < 0.0001"
    ax2.text(0.95, 0.95, p_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Disease-Free Survival by TME State Group', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save Figure 1
fig1_path_png = f"{OUTPUT_DIR}/tme_km_survival_curves.png"
fig1_path_svg = f"{OUTPUT_DIR}/tme_km_survival_curves.svg"
fig1.savefig(fig1_path_png, dpi=300, bbox_inches='tight', facecolor='white')
fig1.savefig(fig1_path_svg, format='svg', bbox_inches='tight', facecolor='white')
print(f"\nFigure 1 saved:")
print(f"  PNG: {fig1_path_png}")
print(f"  SVG: {fig1_path_svg}")
plt.close(fig1)

# --- Figure 2: Cox PH Forest Plot ---
fig2, ax = plt.subplots(figsize=(10, 6))

# Prepare data for forest plot (using optimal cutpoints model)
if ref_col in [c for c in cox_df_opt.columns if c.startswith('TME_Group_Optimal_')]:
    forest_data = []
    
    # Add reference group
    forest_data.append({
        'Group': 'State_1_High (Reference)',
        'HR': 1.0,
        'HR_lower': 1.0,
        'HR_upper': 1.0,
        'p_value': np.nan
    })
    
    # Add other groups
    for var in dummy_cols_opt:
        group_name = var.replace('TME_Group_Optimal_', '')
        forest_data.append({
            'Group': group_name,
            'HR': cox_summary_opt.loc[var, 'exp(coef)'],
            'HR_lower': cox_summary_opt.loc[var, 'exp(coef) lower 95%'],
            'HR_upper': cox_summary_opt.loc[var, 'exp(coef) upper 95%'],
            'p_value': cox_summary_opt.loc[var, 'p']
        })
    
    forest_df = pd.DataFrame(forest_data)
    
    # Plot forest plot
    y_positions = range(len(forest_df))
    
    for i, row in forest_df.iterrows():
        color = TME_COLORS.get(row['Group'].replace(' (Reference)', ''), '#888888')
        
        # Plot confidence interval
        ax.plot([row['HR_lower'], row['HR_upper']], [i, i], 
                color=color, linewidth=2, solid_capstyle='round')
        
        # Plot point estimate
        ax.plot(row['HR'], i, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    # Reference line at HR=1
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(forest_df['Group'])
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax.set_title('Cox Proportional Hazards Model: TME Groups (Optimal Cutpoints)\n'
                 'Reference: State_1_High', fontsize=13, fontweight='bold', loc='left')
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    ax.set_xlim(0.1, 20)
    
    # Add HR values as text
    for i, row in forest_df.iterrows():
        if pd.notna(row['p_value']):
            hr_text = f"HR={row['HR']:.2f} ({row['HR_lower']:.2f}-{row['HR_upper']:.2f})"
            p_text = f"p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else "p<0.001"
            ax.text(12, i, f"{hr_text}\n{p_text}", va='center', fontsize=9)
        else:
            ax.text(12, i, "Reference", va='center', fontsize=9, fontstyle='italic')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save Figure 2
    fig2_path_png = f"{OUTPUT_DIR}/tme_cox_forest_plot.png"
    fig2_path_svg = f"{OUTPUT_DIR}/tme_cox_forest_plot.svg"
    fig2.savefig(fig2_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(fig2_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"\nFigure 2 saved:")
    print(f"  PNG: {fig2_path_png}")
    print(f"  SVG: {fig2_path_svg}")
    
plt.close(fig2)

# --- Figure 3: Individual State KM Curves (High vs Low) ---
fig3, axes = plt.subplots(1, 3, figsize=(15, 5))

states_to_plot = [
    ('State_1_Prop', 'State 1', TME_COLORS['State_1_High']),
    ('State_2_Prop', 'State 2', TME_COLORS['State_2_High']),
    ('State_6_Prop', 'State 6', TME_COLORS['State_6_High'])
]

for idx, (state_col, state_name, color) in enumerate(states_to_plot):
    ax = axes[idx]
    
    # Use optimal cutpoint
    cut = optimal_cuts[state_col]['optimal_cutpoint']
    
    high_mask = df[state_col] >= cut
    low_mask = ~high_mask
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    if high_mask.sum() >= 2:
        kmf_high.fit(df.loc[high_mask, time_col], df.loc[high_mask, event_col],
                     label=f'High {state_name} (n={high_mask.sum()})')
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.5)
    
    if low_mask.sum() >= 2:
        kmf_low.fit(df.loc[low_mask, time_col], df.loc[low_mask, event_col],
                    label=f'Low {state_name} (n={low_mask.sum()})')
        kmf_low.plot_survival_function(ax=ax, ci_show=True, color='gray', linewidth=2.5, linestyle='--')
    
    # Log-rank test
    if high_mask.sum() >= 2 and low_mask.sum() >= 2:
        result = logrank_test(
            df.loc[high_mask, time_col], df.loc[low_mask, time_col],
            df.loc[high_mask, event_col], df.loc[low_mask, event_col]
        )
        p_text = f"Log-rank p = {result.p_value:.4f}" if result.p_value >= 0.0001 else "p < 0.0001"
        ax.text(0.95, 0.95, p_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Disease-Free Survival', fontsize=11)
    ax.set_title(f'{chr(65+idx)}. {state_name} (cutpoint={cut:.2f})', fontweight='bold', loc='left')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.legend(loc='lower left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Disease-Free Survival by Individual TME State Proportions (Optimal Cutpoints)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save Figure 3
fig3_path_png = f"{OUTPUT_DIR}/tme_individual_state_km.png"
fig3_path_svg = f"{OUTPUT_DIR}/tme_individual_state_km.svg"
fig3.savefig(fig3_path_png, dpi=300, bbox_inches='tight', facecolor='white')
fig3.savefig(fig3_path_svg, format='svg', bbox_inches='tight', facecolor='white')
print(f"\nFigure 3 saved:")
print(f"  PNG: {fig3_path_png}")
print(f"  SVG: {fig3_path_svg}")
plt.close(fig3)

# --- Figure 4: Comprehensive Summary Figure ---
fig4 = plt.figure(figsize=(16, 12))

# Create grid
gs = fig4.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Panel A: KM curves (optimal)
ax_a = fig4.add_subplot(gs[0, :2])
plot_km_curves(df, 'TME_Group_Optimal', time_col, event_col, ax_a,
               'A. Disease-Free Survival by TME Group (Optimal Cutpoints)', TME_COLORS)
if len(df['TME_Group_Optimal'].unique()) > 1:
    result = multivariate_logrank_test(df[time_col], df['TME_Group_Optimal'], df[event_col])
    p_text = f"Log-rank p = {result.p_value:.4f}"
    ax_a.text(0.98, 0.98, p_text, transform=ax_a.transAxes, fontsize=10,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: Group distribution pie chart
ax_b = fig4.add_subplot(gs[0, 2])
group_counts = df['TME_Group_Optimal'].value_counts()
colors_pie = [TME_COLORS.get(g, '#888888') for g in group_counts.index]
wedges, texts, autotexts = ax_b.pie(group_counts.values, labels=group_counts.index,
                                     autopct='%1.1f%%', colors=colors_pie,
                                     explode=[0.02]*len(group_counts))
ax_b.set_title('B. TME Group Distribution', fontweight='bold', loc='left')

# Panel C-E: Individual state KM curves
for idx, (state_col, state_name, color) in enumerate(states_to_plot):
    ax = fig4.add_subplot(gs[1, idx])
    
    cut = optimal_cuts[state_col]['optimal_cutpoint']
    high_mask = df[state_col] >= cut
    low_mask = ~high_mask
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    if high_mask.sum() >= 2:
        kmf_high.fit(df.loc[high_mask, time_col], df.loc[high_mask, event_col],
                     label=f'High (n={high_mask.sum()})')
        kmf_high.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)
    
    if low_mask.sum() >= 2:
        kmf_low.fit(df.loc[low_mask, time_col], df.loc[low_mask, event_col],
                    label=f'Low (n={low_mask.sum()})')
        kmf_low.plot_survival_function(ax=ax, ci_show=True, color='gray', linewidth=2, linestyle='--')
    
    if high_mask.sum() >= 2 and low_mask.sum() >= 2:
        result = logrank_test(
            df.loc[high_mask, time_col], df.loc[low_mask, time_col],
            df.loc[high_mask, event_col], df.loc[low_mask, event_col]
        )
        ax.text(0.95, 0.95, f"p={result.p_value:.3f}", transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Time (days)', fontsize=10)
    ax.set_ylabel('DFS Probability', fontsize=10)
    ax.set_title(f'{chr(67+idx)}. {state_name} (cut={cut:.2f})', fontweight='bold', loc='left', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Panel F: Forest plot
ax_f = fig4.add_subplot(gs[2, :2])

if ref_col in [c for c in cox_df_opt.columns if c.startswith('TME_Group_Optimal_')]:
    y_positions = range(len(forest_df))
    
    for i, row in forest_df.iterrows():
        color = TME_COLORS.get(row['Group'].replace(' (Reference)', ''), '#888888')
        ax_f.plot([row['HR_lower'], row['HR_upper']], [i, i], color=color, linewidth=2)
        ax_f.plot(row['HR'], i, 'o', color=color, markersize=8, markeredgecolor='white')
    
    ax_f.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_f.set_yticks(y_positions)
    ax_f.set_yticklabels(forest_df['Group'])
    ax_f.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
    ax_f.set_title('F. Cox Proportional Hazards Model', fontweight='bold', loc='left')
    ax_f.set_xscale('log')
    ax_f.set_xlim(0.1, 15)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)

# Panel G: Summary statistics table
ax_g = fig4.add_subplot(gs[2, 2])
ax_g.axis('off')

summary_text = "G. Summary Statistics\n" + "-"*30 + "\n\n"
summary_text += f"Total patients: {len(df)}\n"
summary_text += f"Events: {df[event_col].sum()}\n"
summary_text += f"Median follow-up: {df[time_col].median()/365.25:.1f} years\n\n"
summary_text += "Optimal Cutpoints:\n"
summary_text += f"  State 1: {optimal_cuts['State_1_Prop']['optimal_cutpoint']:.3f}\n"
summary_text += f"  State 2: {optimal_cuts['State_2_Prop']['optimal_cutpoint']:.3f}\n"
summary_text += f"  State 6: {optimal_cuts['State_6_Prop']['optimal_cutpoint']:.3f}\n\n"
summary_text += f"Cox model C-index: {cph_opt.concordance_index_:.3f}"

ax_g.text(0.1, 0.9, summary_text, transform=ax_g.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

plt.suptitle('TME State Proportions and Disease-Free Survival in TNBC',
             fontsize=15, fontweight='bold', y=0.98)

# Save Figure 4
fig4_path_png = f"{OUTPUT_DIR}/tme_survival_summary.png"
fig4_path_svg = f"{OUTPUT_DIR}/tme_survival_summary.svg"
fig4.savefig(fig4_path_png, dpi=300, bbox_inches='tight', facecolor='white')
fig4.savefig(fig4_path_svg, format='svg', bbox_inches='tight', facecolor='white')
print(f"\nFigure 4 (Summary) saved:")
print(f"  PNG: {fig4_path_png}")
print(f"  SVG: {fig4_path_svg}")
plt.close(fig4)

# =============================================================================
# PART 6: Hypothesis Evaluation Summary
# =============================================================================

print("\n" + "=" * 80)
print("PART 6: HYPOTHESIS EVALUATION SUMMARY")
print("=" * 80)

print("\n" + "─" * 80)
print("HYPOTHESIS 1: Higher State 1 proportion → Longer disease-free survival")
print("─" * 80)

# Get State 1 results
s1_opt_cut = optimal_cuts['State_1_Prop']['optimal_cutpoint']
s1_high_mask = df['State_1_Prop'] >= s1_opt_cut
s1_low_mask = ~s1_high_mask

if s1_high_mask.sum() >= 2 and s1_low_mask.sum() >= 2:
    kmf_s1_high = KaplanMeierFitter()
    kmf_s1_low = KaplanMeierFitter()
    kmf_s1_high.fit(df.loc[s1_high_mask, time_col], df.loc[s1_high_mask, event_col])
    kmf_s1_low.fit(df.loc[s1_low_mask, time_col], df.loc[s1_low_mask, event_col])
    
    med_high = kmf_s1_high.median_survival_time_
    med_low = kmf_s1_low.median_survival_time_
    
    s1_result = logrank_test(
        df.loc[s1_high_mask, time_col], df.loc[s1_low_mask, time_col],
        df.loc[s1_high_mask, event_col], df.loc[s1_low_mask, event_col]
    )
    
    print(f"\n  Optimal cutpoint: {s1_opt_cut:.3f}")
    print(f"  High State 1 (n={s1_high_mask.sum()}): Median DFS = {'Not reached' if np.isinf(med_high) else f'{med_high:.0f} days'}")
    print(f"  Low State 1 (n={s1_low_mask.sum()}): Median DFS = {'Not reached' if np.isinf(med_low) else f'{med_low:.0f} days'}")
    print(f"  Log-rank test: χ² = {s1_result.test_statistic:.3f}, p = {s1_result.p_value:.4f}")
    
    # Cox model result
    s1_hr = cph_cont.summary.loc['State_1_Prop', 'exp(coef)']
    s1_p = cph_cont.summary.loc['State_1_Prop', 'p']
    print(f"  Cox HR (continuous): {s1_hr:.3f}, p = {s1_p:.4f}")
    
    if s1_hr < 1:
        print("\n  ✓ DIRECTION CONSISTENT: Higher State 1 associated with LOWER hazard")
        if s1_result.p_value < 0.05:
            print("  ✓ STATISTICALLY SIGNIFICANT by log-rank test")
        else:
            print("  ⚠ Not statistically significant (p > 0.05)")
    else:
        print("\n  ✗ DIRECTION INCONSISTENT: Higher State 1 associated with higher hazard")

print("\n" + "─" * 80)
print("HYPOTHESIS 2: Higher State 6 proportion → Shorter disease-free survival")
print("─" * 80)

s6_opt_cut = optimal_cuts['State_6_Prop']['optimal_cutpoint']
s6_high_mask = df['State_6_Prop'] >= s6_opt_cut
s6_low_mask = ~s6_high_mask

if s6_high_mask.sum() >= 2 and s6_low_mask.sum() >= 2:
    kmf_s6_high = KaplanMeierFitter()
    kmf_s6_low = KaplanMeierFitter()
    kmf_s6_high.fit(df.loc[s6_high_mask, time_col], df.loc[s6_high_mask, event_col])
    kmf_s6_low.fit(df.loc[s6_low_mask, time_col], df.loc[s6_low_mask, event_col])
    
    med_high = kmf_s6_high.median_survival_time_
    med_low = kmf_s6_low.median_survival_time_
    
    s6_result = logrank_test(
        df.loc[s6_high_mask, time_col], df.loc[s6_low_mask, time_col],
        df.loc[s6_high_mask, event_col], df.loc[s6_low_mask, event_col]
    )
    
    print(f"\n  Optimal cutpoint: {s6_opt_cut:.3f}")
    print(f"  High State 6 (n={s6_high_mask.sum()}): Median DFS = {'Not reached' if np.isinf(med_high) else f'{med_high:.0f} days'}")
    print(f"  Low State 6 (n={s6_low_mask.sum()}): Median DFS = {'Not reached' if np.isinf(med_low) else f'{med_low:.0f} days'}")
    print(f"  Log-rank test: χ² = {s6_result.test_statistic:.3f}, p = {s6_result.p_value:.4f}")
    
    s6_hr = cph_cont.summary.loc['State_6_Prop', 'exp(coef)']
    s6_p = cph_cont.summary.loc['State_6_Prop', 'p']
    print(f"  Cox HR (continuous): {s6_hr:.3f}, p = {s6_p:.4f}")
    
    if s6_hr > 1:
        print("\n  ✓ DIRECTION CONSISTENT: Higher State 6 associated with HIGHER hazard")
        if s6_result.p_value < 0.05:
            print("  ✓ STATISTICALLY SIGNIFICANT by log-rank test")
        else:
            print("  ⚠ Not statistically significant (p > 0.05)")
    else:
        print("\n  ✗ DIRECTION INCONSISTENT: Higher State 6 associated with lower hazard")

print("\n" + "─" * 80)
print("OVERALL ASSESSMENT")
print("─" * 80)

# Determine assessment strings
s1_assessment = "Trend supports hypothesis" if s1_hr < 1 else "Does not support hypothesis"
s6_assessment = "Trend supports hypothesis" if s6_hr > 1 else "Does not support hypothesis"
c_index = cph_opt.concordance_index_
disc_ability = 'good' if c_index > 0.65 else ('moderate' if c_index > 0.55 else 'limited')

print(f"""
The analysis examined how TME state proportions relate to disease-free survival
using both Kaplan-Meier analysis and Cox proportional hazards modeling.

Key findings:
1. TME State 1 (hypothesized favorable): {s1_assessment}
2. TME State 6 (hypothesized unfavorable): {s6_assessment}
3. TME State 2 (hypothesized intermediate): See figure for details

Methodological considerations:
- Sample size (n={len(df)}) limits statistical power
- Optimal cutpoints may be optimistic due to within-sample selection
- Median cutpoints provide more conservative estimates
- Cox model C-index = {c_index:.3f} indicates {disc_ability} discriminative ability
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. {fig1_path_png}")
print(f"  2. {fig1_path_svg}")
print(f"  3. {fig2_path_png}")
print(f"  4. {fig2_path_svg}")
print(f"  5. {fig3_path_png}")
print(f"  6. {fig3_path_svg}")
print(f"  7. {fig4_path_png}")
print(f"  8. {fig4_path_svg}")
