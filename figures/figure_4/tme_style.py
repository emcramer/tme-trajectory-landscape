"""
================================================================================
TME Research Style Module
================================================================================
Companion module for tme_research.mplstyle

Provides:
  - Color palettes for TME states, mixing scores, and outcomes
  - Helper functions for consistent figure styling
  - Convenience functions for common plot elements

Usage:
    import matplotlib.pyplot as plt
    from tme_style import *
    
    # Apply the style
    apply_tme_style()
    
    # Access colors
    color = TME_COLORS[1]  # Teal for State 1
    
    # Use palettes
    for state, color in TME_COLORS.items():
        plt.bar(state, value, color=color)

Author: Eric
Date: January 2025
================================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# =============================================================================
# COLOR PALETTES
# =============================================================================

# TME State Colors (States 1-6)
# These represent the six ABM-derived tumor microenvironment states
TME_COLORS = {
    1: '#1a535c',  # Teal - Immune-rich/Compartmentalized
    2: '#ee6c4d',  # Coral - Intermediate
    3: '#84a98c',  # Sage - Transitional
    4: '#b8b8a8',  # Gray - Neutral
    5: '#6b4f7b',  # Purple - Pre-exclusion
    6: '#e6b800',  # Gold - Cold/Immune-excluded
}

# TME State Colors as a list (for colormaps)
TME_COLORS_LIST = ['#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800']

# Mixing Score Class Colors (Expert classification from Keren et al.)
MIXING_COLORS = {
    'Compartmentalized': '#1f77b4',  # Blue (tab10[0])
    'Mixed': '#ff7f0e',              # Orange (tab10[1])
    'Cold': '#2ca02c',               # Green (tab10[2])
}

# Clinical Outcome Colors
OUTCOME_COLORS = {
    'No Event': '#27ae60',    # Green - Good outcome
    'Event': '#e74c3c',       # Red - Poor outcome
    'Censored': '#95a5a6',    # Gray - Censored
}

# Survival Curve Colors
SURVIVAL_COLORS = {
    'High': '#2c3e50',        # Dark blue-gray
    'Low': '#bdc3c7',         # Light gray
    'Treatment': '#27ae60',   # Green
    'Control': '#e74c3c',     # Red
}

# General categorical palette (extended)
CATEGORICAL_COLORS = [
    '#1a535c', '#ee6c4d', '#84a98c', '#b8b8a8', '#6b4f7b', '#e6b800',
    '#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c',
]

# Sequential palette for heatmaps (warm)
SEQUENTIAL_WARM = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', 
                   '#fc8d59', '#ef6548', '#d7301f', '#990000']

# Diverging palette for correlation heatmaps
DIVERGING_COLORS = ['#1a535c', '#4d8a8c', '#80c1bc', '#b3d9d4', 
                    '#ffffff', '#f4c2c2', '#e57373', '#c62828']


# =============================================================================
# STYLE APPLICATION
# =============================================================================

def apply_tme_style(style_path=None):
    """
    Apply the TME research matplotlib style.
    
    Args:
        style_path: Path to the .mplstyle file. If None, looks in common locations.
    
    Returns:
        bool: True if style was applied successfully
    """
    if style_path is None:
        # Try common locations
        possible_paths = [
            Path('tme_research.mplstyle'),
            Path.home() / '.config/matplotlib/stylelib/tme_research.mplstyle',
            Path('/home/claude/tme_research.mplstyle'),
            Path.cwd() / 'tme_research.mplstyle',
        ]
        
        for path in possible_paths:
            if path.exists():
                style_path = path
                break
    
    if style_path and Path(style_path).exists():
        plt.style.use(str(style_path))
        print(f"Applied TME research style from: {style_path}")
        return True
    else:
        # Apply key settings manually as fallback
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '0.3',
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'axes.labelsize': 10,
            'font.size': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'grid.alpha': 0.7,
            'grid.linewidth': 0.5,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=TME_COLORS_LIST)
        print("Applied TME research style (fallback settings)")
        return True


def get_tme_cmap(n_states=6):
    """
    Get a discrete colormap for TME states.
    
    Args:
        n_states: Number of states (default 6)
    
    Returns:
        matplotlib.colors.ListedColormap
    """
    colors = TME_COLORS_LIST[:n_states]
    return mcolors.ListedColormap(colors)


def get_tme_norm(n_states=6):
    """
    Get a BoundaryNorm for discrete TME state coloring.
    
    Args:
        n_states: Number of states (default 6)
    
    Returns:
        matplotlib.colors.BoundaryNorm
    """
    boundaries = np.arange(0.5, n_states + 1.5, 1)
    return mcolors.BoundaryNorm(boundaries, n_states)


# =============================================================================
# LEGEND HELPERS
# =============================================================================

def get_tme_legend_handles(states=None):
    """
    Get legend handles for TME states.
    
    Args:
        states: List of states to include (default: all 1-6)
    
    Returns:
        list: List of Patch objects for legend
    """
    if states is None:
        states = range(1, 7)
    
    return [Patch(facecolor=TME_COLORS[s], edgecolor='black', 
                  linewidth=0.5, label=f'State {s}') for s in states]


def get_mixing_legend_handles():
    """
    Get legend handles for mixing score classes.
    
    Returns:
        list: List of Patch objects for legend
    """
    return [Patch(facecolor=MIXING_COLORS[k], edgecolor='black',
                  linewidth=0.5, label=k) 
            for k in ['Compartmentalized', 'Mixed', 'Cold']]


def get_outcome_legend_handles():
    """
    Get legend handles for clinical outcomes.
    
    Returns:
        list: List of Patch objects for legend
    """
    return [Patch(facecolor=OUTCOME_COLORS[k], edgecolor='black',
                  linewidth=0.5, label=k) 
            for k in ['No Event', 'Event']]


# =============================================================================
# ANNOTATION HELPERS
# =============================================================================

def add_significance_stars(p_value):
    """
    Convert p-value to significance stars.
    
    Args:
        p_value: P-value from statistical test
    
    Returns:
        str: Significance annotation ('***', '**', '*', or 'ns')
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def add_significance_bracket(ax, x1, x2, y, p_value, height=0.02, fontsize=10):
    """
    Add a significance bracket between two groups.
    
    Args:
        ax: Matplotlib axes object
        x1, x2: X positions of the two groups
        y: Y position for the bracket
        p_value: P-value for annotation
        height: Height of the bracket arms
        fontsize: Font size for annotation
    """
    stars = add_significance_stars(p_value)
    
    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], 
            'k-', linewidth=1.5)
    
    # Add text
    ax.text((x1 + x2) / 2, y + height * 1.5, stars,
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


def format_pvalue(p_value, style='scientific'):
    """
    Format p-value for display.
    
    Args:
        p_value: P-value to format
        style: 'scientific' or 'decimal'
    
    Returns:
        str: Formatted p-value string
    """
    if style == 'scientific':
        if p_value < 0.001:
            return f'p = {p_value:.2e}'
        else:
            return f'p = {p_value:.3f}'
    else:
        if p_value < 0.001:
            return 'p < 0.001'
        else:
            return f'p = {p_value:.3f}'


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def save_figure(fig, base_path, formats=['png', 'pdf', 'svg'], dpi=300):
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure object
        base_path: Base path without extension (e.g., 'output/figure_1')
        formats: List of formats to save
        dpi: DPI for raster formats
    
    Returns:
        list: List of saved file paths
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for fmt in formats:
        path = base_path.with_suffix(f'.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        saved_paths.append(path)
        print(f"  ✓ Saved: {path}")
    
    return saved_paths


def add_panel_label(ax, label, x=-0.1, y=1.1, fontsize=14, fontweight='bold'):
    """
    Add a panel label (A, B, C, etc.) to an axes.
    
    Args:
        ax: Matplotlib axes object
        label: Panel label string (e.g., 'A', 'B')
        x, y: Position in axes coordinates
        fontsize: Font size for label
        fontweight: Font weight for label
    """
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, va='top', ha='right')


def style_spines(ax, visible=['bottom', 'left'], linewidth=0.8):
    """
    Style axis spines consistently.
    
    Args:
        ax: Matplotlib axes object
        visible: List of spines to keep visible
        linewidth: Line width for visible spines
    """
    for spine in ['top', 'right', 'bottom', 'left']:
        if spine in visible:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(linewidth)
        else:
            ax.spines[spine].set_visible(False)


# =============================================================================
# COLORBAR HELPERS
# =============================================================================

def add_discrete_colorbar(fig, ax, cmap, n_levels, label='', 
                          orientation='vertical', shrink=0.8):
    """
    Add a discrete colorbar for categorical data.
    
    Args:
        fig: Matplotlib figure object
        ax: Axes object or mappable
        cmap: Colormap (ListedColormap or name)
        n_levels: Number of discrete levels
        label: Colorbar label
        orientation: 'vertical' or 'horizontal'
        shrink: Shrink factor
    
    Returns:
        Colorbar object
    """
    import matplotlib.cm as cm
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap, n_levels)
    
    norm = mcolors.BoundaryNorm(np.arange(0.5, n_levels + 1.5), n_levels)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation=orientation, shrink=shrink)
    cbar.set_label(label)
    cbar.set_ticks(range(1, n_levels + 1))
    
    return cbar


# =============================================================================
# QUICK SETUP FUNCTION
# =============================================================================

def setup_figure(nrows=1, ncols=1, figsize=None, style=True):
    """
    Quick setup for a new figure with TME style.
    
    Args:
        nrows, ncols: Subplot grid dimensions
        figsize: Figure size (auto-calculated if None)
        style: Whether to apply TME style
    
    Returns:
        tuple: (fig, axes)
    """
    if style:
        apply_tme_style()
    
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    return fig, axes


# =============================================================================
# STYLE INFO
# =============================================================================

def print_palette_info():
    """Print information about available color palettes."""
    print("=" * 60)
    print("TME Research Color Palettes")
    print("=" * 60)
    
    print("\nTME State Colors:")
    for state, color in TME_COLORS.items():
        print(f"  State {state}: {color}")
    
    print("\nMixing Score Colors:")
    for label, color in MIXING_COLORS.items():
        print(f"  {label}: {color}")
    
    print("\nOutcome Colors:")
    for label, color in OUTCOME_COLORS.items():
        print(f"  {label}: {color}")
    
    print("\nSurvival Colors:")
    for label, color in SURVIVAL_COLORS.items():
        print(f"  {label}: {color}")
    
    print("=" * 60)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print_palette_info()
    
    # Create a demo figure
    apply_tme_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Demo 1: Bar plot with TME colors
    ax = axes[0, 0]
    states = list(TME_COLORS.keys())
    values = [0.6, 0.3, 0.05, 0.02, 0.01, 0.02]
    ax.bar(states, values, color=[TME_COLORS[s] for s in states], edgecolor='black')
    ax.set_xlabel('TME State')
    ax.set_ylabel('Proportion')
    ax.set_title('TME State Distribution')
    add_panel_label(ax, 'A')
    
    # Demo 2: Line plot with default cycle
    ax = axes[0, 1]
    x = np.linspace(0, 10, 100)
    for i in range(4):
        ax.plot(x, np.sin(x + i), label=f'Group {i+1}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series')
    ax.legend()
    add_panel_label(ax, 'B')
    
    # Demo 3: Scatter with outcome colors
    ax = axes[1, 0]
    np.random.seed(42)
    for outcome, color in OUTCOME_COLORS.items():
        if outcome != 'Censored':
            x = np.random.randn(20)
            y = np.random.randn(20)
            ax.scatter(x, y, c=color, label=outcome, s=50, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Outcome Distribution')
    ax.legend()
    add_panel_label(ax, 'C')
    
    # Demo 4: Heatmap-style
    ax = axes[1, 1]
    data = np.random.rand(6, 3)
    im = ax.imshow(data, cmap=get_tme_cmap(), aspect='auto')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Compart.', 'Mixed', 'Cold'])
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'State {i+1}' for i in range(6)])
    ax.set_title('Confusion Matrix Demo')
    fig.colorbar(im, ax=ax, shrink=0.8)
    add_panel_label(ax, 'D')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/tme_style_demo.png', dpi=150)
    print("\nDemo figure saved to /mnt/user-data/outputs/tme_style_demo.png")
