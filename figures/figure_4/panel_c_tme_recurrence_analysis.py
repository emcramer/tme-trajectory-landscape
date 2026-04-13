#!/usr/bin/env python3
"""
TME State Proportions and Recurrence Analysis
==============================================

This script tests the hypotheses:
1. Higher proportion of TME State 1 ROIs → less likely to have recurrence
2. Higher proportion of TME State 6 ROIs → more likely to have recurrence

Statistical approaches:
- Mann-Whitney U test: Compare state proportions between recurrence vs. non-recurrence groups
- Logistic regression: Model recurrence probability as a function of state proportions
- Point-biserial correlation: Assess linear relationship between continuous proportions and binary outcome

Author: Analysis for Eric's TME research
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# =============================================================================
# Load and Prepare Data
# =============================================================================

print("=" * 70)
print("TME STATE PROPORTIONS AND RECURRENCE ANALYSIS")
print("=" * 70)

# Load data
df = pd.read_csv('data/mibi_roi_tme_proportions_clinical.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Total patients: {len(df)}")

# Identify key variables
print("\n--- Key Variables ---")
print(f"State 1 Proportion range: {df['State_1_Prop'].min():.3f} - {df['State_1_Prop'].max():.3f}")
print(f"State 6 Proportion range: {df['State_6_Prop'].min():.3f} - {df['State_6_Prop'].max():.3f}")
print(f"\nRecurrence events (Event=1): {df['Event'].sum()}")
print(f"No recurrence (Event=0): {(df['Event'] == 0).sum()}")

# Separate groups
recurrence = df[df['Event'] == 1]
no_recurrence = df[df['Event'] == 0]

print(f"\nPatients with recurrence: n = {len(recurrence)}")
print(f"Patients without recurrence: n = {len(no_recurrence)}")

# =============================================================================
# Statistical Test 1: Mann-Whitney U Test
# =============================================================================
# This non-parametric test compares the distributions of state proportions
# between patients who had recurrence vs. those who did not

print("\n" + "=" * 70)
print("STATISTICAL TEST 1: Mann-Whitney U Test")
print("=" * 70)
print("\nComparing TME state proportions between recurrence groups")
print("(Non-parametric test appropriate for non-normal distributions)")

# State 1 analysis
state1_recur = recurrence['State_1_Prop'].values
state1_no_recur = no_recurrence['State_1_Prop'].values

u_stat_1, p_value_1 = mannwhitneyu(state1_no_recur, state1_recur, alternative='greater')
# alternative='greater' tests if no-recurrence group has HIGHER State 1 proportions

print(f"\n--- State 1 Proportion ---")
print(f"Hypothesis: Higher State 1 → LESS likely to have recurrence")
print(f"  No recurrence group: median = {np.median(state1_no_recur):.3f}, mean = {np.mean(state1_no_recur):.3f}")
print(f"  Recurrence group:    median = {np.median(state1_recur):.3f}, mean = {np.mean(state1_recur):.3f}")
print(f"  Mann-Whitney U statistic: {u_stat_1}")
print(f"  P-value (one-tailed, no_recur > recur): {p_value_1:.4f}")

if p_value_1 < 0.05:
    print(f"  ✓ SIGNIFICANT: No-recurrence patients have significantly higher State 1 proportions")
else:
    print(f"  ✗ Not significant at α = 0.05")

# State 6 analysis
state6_recur = recurrence['State_6_Prop'].values
state6_no_recur = no_recurrence['State_6_Prop'].values

u_stat_6, p_value_6 = mannwhitneyu(state6_recur, state6_no_recur, alternative='greater')
# alternative='greater' tests if recurrence group has HIGHER State 6 proportions

print(f"\n--- State 6 Proportion ---")
print(f"Hypothesis: Higher State 6 → MORE likely to have recurrence")
print(f"  No recurrence group: median = {np.median(state6_no_recur):.3f}, mean = {np.mean(state6_no_recur):.3f}")
print(f"  Recurrence group:    median = {np.median(state6_recur):.3f}, mean = {np.mean(state6_recur):.3f}")
print(f"  Mann-Whitney U statistic: {u_stat_6}")
print(f"  P-value (one-tailed, recur > no_recur): {p_value_6:.4f}")

if p_value_6 < 0.05:
    print(f"  ✓ SIGNIFICANT: Recurrence patients have significantly higher State 6 proportions")
else:
    print(f"  ✗ Not significant at α = 0.05")

# =============================================================================
# Statistical Test 2: Logistic Regression
# =============================================================================
# Model the probability of recurrence as a function of state proportions

print("\n" + "=" * 70)
print("STATISTICAL TEST 2: Logistic Regression")
print("=" * 70)
print("\nModeling P(Recurrence) ~ State Proportions")

# Prepare data for logistic regression
df_clean = df[['Event', 'State_1_Prop', 'State_6_Prop']].dropna()

# Model 1: State 1 only
print("\n--- Model 1: State 1 Proportion Only ---")
X1 = sm.add_constant(df_clean['State_1_Prop'])
y = df_clean['Event']
model1 = sm.Logit(y, X1).fit(disp=0)

print(f"  Coefficient (β): {model1.params['State_1_Prop']:.4f}")
print(f"  Odds Ratio: {np.exp(model1.params['State_1_Prop']):.4f}")
print(f"  95% CI for OR: ({np.exp(model1.conf_int().loc['State_1_Prop', 0]):.4f}, {np.exp(model1.conf_int().loc['State_1_Prop', 1]):.4f})")
print(f"  P-value: {model1.pvalues['State_1_Prop']:.4f}")

if model1.params['State_1_Prop'] < 0 and model1.pvalues['State_1_Prop'] < 0.05:
    print(f"  ✓ SIGNIFICANT: Higher State 1 proportion associated with LOWER recurrence odds")
elif model1.pvalues['State_1_Prop'] < 0.05:
    print(f"  ✓ SIGNIFICANT: Effect in unexpected direction")
else:
    print(f"  ✗ Not significant at α = 0.05")

# Model 2: State 6 only
print("\n--- Model 2: State 6 Proportion Only ---")
X6 = sm.add_constant(df_clean['State_6_Prop'])
model6 = sm.Logit(y, X6).fit(disp=0)

print(f"  Coefficient (β): {model6.params['State_6_Prop']:.4f}")
print(f"  Odds Ratio: {np.exp(model6.params['State_6_Prop']):.4f}")
print(f"  95% CI for OR: ({np.exp(model6.conf_int().loc['State_6_Prop', 0]):.4f}, {np.exp(model6.conf_int().loc['State_6_Prop', 1]):.4f})")
print(f"  P-value: {model6.pvalues['State_6_Prop']:.4f}")

if model6.params['State_6_Prop'] > 0 and model6.pvalues['State_6_Prop'] < 0.05:
    print(f"  ✓ SIGNIFICANT: Higher State 6 proportion associated with HIGHER recurrence odds")
elif model6.pvalues['State_6_Prop'] < 0.05:
    print(f"  ✓ SIGNIFICANT: Effect in unexpected direction")
else:
    print(f"  ✗ Not significant at α = 0.05")

# Model 3: Both states together
print("\n--- Model 3: Both States Combined ---")
X_both = sm.add_constant(df_clean[['State_1_Prop', 'State_6_Prop']])
model_both = sm.Logit(y, X_both).fit(disp=0)

print(f"\n  State 1 Prop:")
print(f"    Coefficient (β): {model_both.params['State_1_Prop']:.4f}")
print(f"    Odds Ratio: {np.exp(model_both.params['State_1_Prop']):.4f}")
print(f"    P-value: {model_both.pvalues['State_1_Prop']:.4f}")

print(f"\n  State 6 Prop:")
print(f"    Coefficient (β): {model_both.params['State_6_Prop']:.4f}")
print(f"    Odds Ratio: {np.exp(model_both.params['State_6_Prop']):.4f}")
print(f"    P-value: {model_both.pvalues['State_6_Prop']:.4f}")

print(f"\n  Model AIC: {model_both.aic:.2f}")
print(f"  Model Pseudo R²: {model_both.prsquared:.4f}")

# =============================================================================
# Statistical Test 3: Correlation Analysis
# =============================================================================

print("\n" + "=" * 70)
print("STATISTICAL TEST 3: Correlation Analysis")
print("=" * 70)

# Point-biserial correlation (Pearson between continuous and binary)
r_state1, p_corr1 = pearsonr(df_clean['State_1_Prop'], df_clean['Event'])
r_state6, p_corr6 = pearsonr(df_clean['State_6_Prop'], df_clean['Event'])

print("\nPoint-biserial correlation with recurrence event:")
print(f"\n  State 1 Proportion:")
print(f"    Correlation (r): {r_state1:.4f}")
print(f"    P-value: {p_corr1:.4f}")
print(f"    Interpretation: {'Negative' if r_state1 < 0 else 'Positive'} correlation " +
      f"({'supports' if r_state1 < 0 else 'opposes'} hypothesis)")

print(f"\n  State 6 Proportion:")
print(f"    Correlation (r): {r_state6:.4f}")
print(f"    P-value: {p_corr6:.4f}")
print(f"    Interpretation: {'Positive' if r_state6 > 0 else 'Negative'} correlation " +
      f"({'supports' if r_state6 > 0 else 'opposes'} hypothesis)")

# =============================================================================
# Create Visualization
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

# Define colors consistent with Eric's TME_COLORS palette
# State 1 is typically more "favorable" (teal), State 6 more "unfavorable" (gold/orange)
color_state1 = '#1f77b4'  # Blue
color_state6 = '#d4a017'  # Gold/amber
color_recur = '#c44e52'   # Red for recurrence
color_no_recur = '#55a868'  # Green for no recurrence

fig = plt.figure(figsize=(12, 10))

# Panel A: Boxplot comparison - State 1
ax1 = fig.add_subplot(2, 2, 1)
bp1 = ax1.boxplot([state1_no_recur, state1_recur], 
                   positions=[0, 1], widths=0.6,
                   patch_artist=True)
bp1['boxes'][0].set_facecolor(color_no_recur)
bp1['boxes'][0].set_alpha(0.7)
bp1['boxes'][1].set_facecolor(color_recur)
bp1['boxes'][1].set_alpha(0.7)

# Add individual points with jitter
np.random.seed(42)
jitter1 = np.random.uniform(-0.15, 0.15, len(state1_no_recur))
jitter2 = np.random.uniform(-0.15, 0.15, len(state1_recur))
ax1.scatter(0 + jitter1, state1_no_recur, alpha=0.6, s=50, c=color_no_recur, edgecolors='white', linewidth=0.5, zorder=3)
ax1.scatter(1 + jitter2, state1_recur, alpha=0.6, s=50, c=color_recur, edgecolors='white', linewidth=0.5, zorder=3)

ax1.set_xticks([0, 1])
ax1.set_xticklabels(['No Recurrence\n(n={})'.format(len(state1_no_recur)), 
                     'Recurrence\n(n={})'.format(len(state1_recur))])
ax1.set_ylabel('State 1 Proportion')
ax1.set_title('A. TME State 1 Proportion by Recurrence Status', fontweight='bold', loc='left')

# Add significance annotation
y_max = max(state1_no_recur.max(), state1_recur.max()) + 0.05
ax1.plot([0, 1], [y_max, y_max], 'k-', lw=1)
sig_text = f'p = {p_value_1:.3f}' if p_value_1 >= 0.001 else 'p < 0.001'
if p_value_1 < 0.05:
    sig_text += ' *'
ax1.text(0.5, y_max + 0.02, sig_text, ha='center', va='bottom', fontsize=10)

ax1.set_ylim(-0.05, 1.15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Boxplot comparison - State 6
ax2 = fig.add_subplot(2, 2, 2)
bp2 = ax2.boxplot([state6_no_recur, state6_recur], 
                   positions=[0, 1], widths=0.6,
                   patch_artist=True)
bp2['boxes'][0].set_facecolor(color_no_recur)
bp2['boxes'][0].set_alpha(0.7)
bp2['boxes'][1].set_facecolor(color_recur)
bp2['boxes'][1].set_alpha(0.7)

# Add individual points with jitter
jitter3 = np.random.uniform(-0.15, 0.15, len(state6_no_recur))
jitter4 = np.random.uniform(-0.15, 0.15, len(state6_recur))
ax2.scatter(0 + jitter3, state6_no_recur, alpha=0.6, s=50, c=color_no_recur, edgecolors='white', linewidth=0.5, zorder=3)
ax2.scatter(1 + jitter4, state6_recur, alpha=0.6, s=50, c=color_recur, edgecolors='white', linewidth=0.5, zorder=3)

ax2.set_xticks([0, 1])
ax2.set_xticklabels(['No Recurrence\n(n={})'.format(len(state6_no_recur)), 
                     'Recurrence\n(n={})'.format(len(state6_recur))])
ax2.set_ylabel('State 6 Proportion')
ax2.set_title('B. TME State 6 Proportion by Recurrence Status', fontweight='bold', loc='left')

# Add significance annotation
y_max = max(state6_no_recur.max(), state6_recur.max()) + 0.05
ax2.plot([0, 1], [y_max, y_max], 'k-', lw=1)
sig_text = f'p = {p_value_6:.3f}' if p_value_6 >= 0.001 else 'p < 0.001'
if p_value_6 < 0.05:
    sig_text += ' *'
ax2.text(0.5, y_max + 0.02, sig_text, ha='center', va='bottom', fontsize=10)

ax2.set_ylim(-0.05, 1.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel C: Scatter plot - State 1 vs recurrence with logistic curve
ax3 = fig.add_subplot(2, 2, 3)

# Plot data points with jitter on y-axis for visibility
y_jitter = df_clean['Event'] + np.random.uniform(-0.05, 0.05, len(df_clean))
colors_scatter = [color_recur if e == 1 else color_no_recur for e in df_clean['Event']]
ax3.scatter(df_clean['State_1_Prop'], y_jitter, alpha=0.7, s=60, c=colors_scatter, edgecolors='white', linewidth=0.5)

# Plot logistic regression curve
x_range = np.linspace(0, 1, 100)
X_pred = sm.add_constant(x_range)
y_pred = model1.predict(X_pred)
ax3.plot(x_range, y_pred, color=color_state1, lw=2.5, label='Logistic fit')

ax3.set_xlabel('State 1 Proportion')
ax3.set_ylabel('Recurrence Probability')
ax3.set_yticks([0, 0.5, 1])
ax3.set_yticklabels(['0 (No)', '0.5', '1 (Yes)'])
ax3.set_xlim(-0.05, 1.05)
ax3.set_ylim(-0.15, 1.15)

# Add annotation with model results
text_str = f'OR = {np.exp(model1.params["State_1_Prop"]):.2f}\np = {model1.pvalues["State_1_Prop"]:.3f}'
ax3.text(0.95, 0.95, text_str, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax3.set_title('C. Logistic Regression: State 1 → Recurrence', fontweight='bold', loc='left')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Panel D: Scatter plot - State 6 vs recurrence with logistic curve
ax4 = fig.add_subplot(2, 2, 4)

# Plot data points
ax4.scatter(df_clean['State_6_Prop'], y_jitter, alpha=0.7, s=60, c=colors_scatter, edgecolors='white', linewidth=0.5)

# Plot logistic regression curve
X_pred6 = sm.add_constant(x_range)
y_pred6 = model6.predict(X_pred6)
ax4.plot(x_range, y_pred6, color=color_state6, lw=2.5, label='Logistic fit')

ax4.set_xlabel('State 6 Proportion')
ax4.set_ylabel('Recurrence Probability')
ax4.set_yticks([0, 0.5, 1])
ax4.set_yticklabels(['0 (No)', '0.5', '1 (Yes)'])
ax4.set_xlim(-0.05, 1.05)
ax4.set_ylim(-0.15, 1.15)

# Add annotation with model results
text_str = f'OR = {np.exp(model6.params["State_6_Prop"]):.2f}\np = {model6.pvalues["State_6_Prop"]:.3f}'
ax4.text(0.95, 0.95, text_str, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax4.set_title('D. Logistic Regression: State 6 → Recurrence', fontweight='bold', loc='left')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_no_recur, alpha=0.7, label='No Recurrence'),
                   Patch(facecolor=color_recur, alpha=0.7, label='Recurrence')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
           bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=11)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.suptitle('TME State Proportions and Recurrence in TNBC', fontsize=14, fontweight='bold', y=0.98)

# Save figure
output_path = 'outputs/tme_state_recurrence_analysis'
plt.savefig(output_path+'.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path+'.svg', bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to: {output_path}")

plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

print("\n📊 HYPOTHESIS 1: Higher State 1 proportion → Less likely to have recurrence")
print("-" * 60)
if np.mean(state1_no_recur) > np.mean(state1_recur):
    print(f"  Direction: CONSISTENT with hypothesis")
    print(f"    Mean State 1 in no-recurrence group: {np.mean(state1_no_recur):.3f}")
    print(f"    Mean State 1 in recurrence group: {np.mean(state1_recur):.3f}")
else:
    print(f"  Direction: INCONSISTENT with hypothesis")
    
print(f"  Mann-Whitney U test: p = {p_value_1:.4f} {'(significant)' if p_value_1 < 0.05 else '(not significant)'}")
print(f"  Logistic regression OR: {np.exp(model1.params['State_1_Prop']):.3f}, p = {model1.pvalues['State_1_Prop']:.4f}")
print(f"  Correlation with recurrence: r = {r_state1:.3f}, p = {p_corr1:.4f}")

print("\n📊 HYPOTHESIS 2: Higher State 6 proportion → More likely to have recurrence")
print("-" * 60)
if np.mean(state6_recur) > np.mean(state6_no_recur):
    print(f"  Direction: CONSISTENT with hypothesis")
    print(f"    Mean State 6 in recurrence group: {np.mean(state6_recur):.3f}")
    print(f"    Mean State 6 in no-recurrence group: {np.mean(state6_no_recur):.3f}")
else:
    print(f"  Direction: INCONSISTENT with hypothesis")

print(f"  Mann-Whitney U test: p = {p_value_6:.4f} {'(significant)' if p_value_6 < 0.05 else '(not significant)'}")
print(f"  Logistic regression OR: {np.exp(model6.params['State_6_Prop']):.3f}, p = {model6.pvalues['State_6_Prop']:.4f}")
print(f"  Correlation with recurrence: r = {r_state6:.3f}, p = {p_corr6:.4f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Interpret results
alpha = 0.05
state1_supported = (p_value_1 < alpha) and (np.mean(state1_no_recur) > np.mean(state1_recur))
state6_supported = (p_value_6 < alpha) and (np.mean(state6_recur) > np.mean(state6_no_recur))

if state1_supported:
    print("\n✓ State 1 hypothesis SUPPORTED: Patients with higher State 1 proportions")
    print("  have significantly fewer recurrence events. This suggests that TME State 1")
    print("  may represent a favorable microenvironment configuration.")
else:
    if np.mean(state1_no_recur) > np.mean(state1_recur):
        print("\n⚠ State 1 hypothesis shows expected TREND but is NOT statistically significant.")
        print("  The sample size may be insufficient to detect the effect, or the")
        print("  effect size may be small.")
    else:
        print("\n✗ State 1 hypothesis NOT SUPPORTED: No evidence of protective effect.")

if state6_supported:
    print("\n✓ State 6 hypothesis SUPPORTED: Patients with higher State 6 proportions")
    print("  have significantly more recurrence events. This suggests that TME State 6")
    print("  may represent an unfavorable/immunologically 'cold' microenvironment.")
else:
    if np.mean(state6_recur) > np.mean(state6_no_recur):
        print("\n⚠ State 6 hypothesis shows expected TREND but is NOT statistically significant.")
        print("  The sample size may be insufficient to detect the effect, or the")
        print("  effect size may be small.")
    else:
        print("\n✗ State 6 hypothesis NOT SUPPORTED: No evidence of adverse effect.")

print("\n" + "=" * 70)
print("METHODOLOGICAL NOTES")
print("=" * 70)
print("""
1. Mann-Whitney U test: Non-parametric comparison appropriate for proportions
   that may not be normally distributed.

2. Logistic regression: Models probability of recurrence as function of
   continuous state proportions. OR < 1 indicates protective effect,
   OR > 1 indicates risk factor.

3. Sample size consideration: n = {} patients total, with {} recurrence events.
   This limits statistical power for detecting moderate effect sizes.

4. Multiple testing: Two primary hypotheses tested. Consider Bonferroni
   correction (α = 0.025) for conservative interpretation.
""".format(len(df), df['Event'].sum()))
