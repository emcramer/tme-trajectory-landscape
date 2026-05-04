import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from matplotlib.patches import Rectangle as _Rect

# =============================================================================
# FONT SIZE CONTROLS (Updated: +2 points)
# =============================================================================
# Adjust these variables to control font sizes across all panels.

FONT_MAIN_TITLE = 16        # was 14
FONT_PANEL_TITLE = 14       # was 12
FONT_AXIS_LABEL = 13        # was 11
FONT_TICK_LABEL = 12        # was 10
FONT_TICK_LABEL_SMALL = 10  # was 8
FONT_ANNOTATION = 11        # was 9
FONT_LEGEND = 14            # was 12
FONT_TABLE = 12             # was 10
FONT_TABLE_HEADER = 12      # was 10
FONT_RUG_LABEL = 11         # was 9

# ============================================================================
# PUBLICATION-QUALITY SETTINGS
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = FONT_AXIS_LABEL
plt.rcParams['axes.titlesize'] = FONT_PANEL_TITLE
plt.rcParams['xtick.labelsize'] = FONT_TICK_LABEL
plt.rcParams['ytick.labelsize'] = FONT_TICK_LABEL
plt.rcParams['legend.fontsize'] = FONT_LEGEND
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False 
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# Figure configuration
FIGURE_SIZE = (10, 16)  # Width x Height
DPI = 300

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

df = pd.read_csv("data/panel_b_data.csv")

# Get selected feature values
x_col_sd = 'malignant_epithelial_cell'
y_col_sd = 'effector_T_cell'

# Grid definition (identical to biologically-constrained cell)
grid_res_sd = 25
x_min_sd, x_max_sd = df[x_col_sd].min(), df[x_col_sd].max()
y_min_sd, y_max_sd = df[y_col_sd].min(), df[y_col_sd].max()
x_range_sd = np.linspace(x_min_sd, x_max_sd, grid_res_sd)
y_range_sd = np.linspace(max(0, y_min_sd), y_max_sd, grid_res_sd)
X_sd, Y_sd = np.meshgrid(x_range_sd, y_range_sd)

# Maximum length for any single sub-arrow segment (in data units).
# Arrows longer than this are split into N = ceil(L / max_segment_len)
# equal sub-arrows chained end-to-end along the original direction.
max_segment_len = (y_max_sd - max(0, y_min_sd)) / grid_res_sd * 0.5

with plt.style.context('default'):
    fig_sd, ax_sd = plt.subplots(figsize=(12, 11))
    ax_sd.set_position([0.1, 0.12, 0.75, 0.7])

    for state_val in [1, 6]:
        state_df = df[df['State'] == state_val]
        if state_df.empty:
            continue

        # 1. Occupancy Density
        counts, xeb, yeb = np.histogram2d(
            state_df[x_col_sd], state_df[y_col_sd],
            bins=[x_range_sd, y_range_sd]
        )
        cmap_sd = LinearSegmentedColormap.from_list(
            f'state_{state_val}_sd',
            [(1, 1, 1, 0), TME_COLORS[state_val]]
        )
        counts_masked = np.ma.masked_where(counts == 0, counts)
        bin_width_y = y_range_sd[1] - y_range_sd[0]
        yeb_shifted = np.where(yeb < 0, yeb + bin_width_y, yeb)
        im = ax_sd.pcolormesh(xeb, yeb_shifted, counts_masked.T, cmap=cmap_sd, alpha=0.4, shading='auto')

        cax = fig_sd.add_axes([0.15 + (state_val == 6) * 0.4, 0.90, 0.3, 0.02])
        plt.colorbar(im, cax=cax, label=f'State {state_val} Time Step Occupancy', orientation='horizontal')

        # 2. Vector Field
        flow_data_sd = []
        for sim_id, sim_group in state_df.groupby('sim_id'):
            sim_group = sim_group.sort_values('window_index_in_sim')
            x_vals = sim_group[x_col_sd].values
            y_vals = sim_group[y_col_sd].values
            dx = np.diff(x_vals)
            dy = np.diff(y_vals)
            x_mid = x_vals[:-1]
            y_mid = y_vals[:-1]
            for i in range(len(dx)):
                flow_data_sd.append([x_mid[i], y_mid[i], dx[i], dy[i]])

        if flow_data_sd:
            flow_df_sd = pd.DataFrame(flow_data_sd, columns=['x', 'y', 'dx', 'dy'])
            U_sd = np.zeros_like(X_sd)
            V_sd = np.zeros_like(Y_sd)
            Count_sd = np.zeros_like(X_sd)

            xb_sd = np.digitize(flow_df_sd['x'], x_range_sd) - 1
            yb_sd = np.digitize(flow_df_sd['y'], y_range_sd) - 1

            for i in range(len(flow_df_sd)):
                xb = xb_sd[i]
                yb = yb_sd[i]
                if 0 <= xb < grid_res_sd and 0 <= yb < grid_res_sd:
                    U_sd[yb, xb] += flow_df_sd['dx'].iloc[i]
                    V_sd[yb, xb] += flow_df_sd['dy'].iloc[i]
                    Count_sd[yb, xb] += 1

            mask_sd = Count_sd > 0
            U_sd[mask_sd] /= Count_sd[mask_sd]
            V_sd[mask_sd] /= Count_sd[mask_sd]

            # Biological floor constraint (same as biologically-constrained cell)
            needs_clip = (Y_sd > 0) & (V_sd < 0) & ((Y_sd + V_sd) < 0)
            t = np.where(needs_clip, Y_sd / (-V_sd), 1.0)
            U_sd = U_sd * t
            V_sd = V_sd * t

            # Subdivision: break each grid-cell arrow into N equal
            # sub-arrows chained end-to-end along the original direction.
            # N = ceil(L / max_segment_len) so each segment is at most
            # max_segment_len long. Sub-arrows starting below y=0 are
            # dropped; the clip path below handles any residual overshoot.
            sub_x, sub_y, sub_u, sub_v = [], [], [], []
            for i in range(U_sd.shape[0]):
                for j in range(U_sd.shape[1]):
                    u, v = U_sd[i, j], V_sd[i, j]
                    ox, oy = X_sd[i, j], Y_sd[i, j]
                    L = np.hypot(u, v)
                    if L == 0:
                        continue
                    N = int(np.ceil(L / max_segment_len))
                    su, sv = u / N, v / N
                    for k in range(N):
                        sx = ox + k * su
                        sy = oy + k * sv
                        if sy >= 0:
                            sub_x.append(sx)
                            sub_y.append(sy)
                            sub_u.append(su)
                            sub_v.append(sv)

            q_sd = ax_sd.quiver(
                np.array(sub_x),
                np.array(sub_y),
                np.array(sub_u),
                np.array(sub_v),
                color=TME_COLORS[state_val],
                alpha=0.8,
                units='xy',
                width=0.015,
                headwidth=4,
                headlength=5,
                label=f'State {state_val} Flow'
            )
            _clip_sd = _Rect(
                (x_min_sd - 1, 0),
                x_max_sd - x_min_sd + 2,
                y_max_sd + 1,
                transform=ax_sd.transData,
                visible=False
            )
            ax_sd.add_patch(_clip_sd)
            q_sd.set_clip_path(_clip_sd)

        # 3. Mark Origin
        min_time = state_df['end_time_step'].min()
        origin_data = state_df[state_df['end_time_step'] == min_time]
        origin_x = origin_data[x_col_sd].mean()
        origin_y = origin_data[y_col_sd].mean()

        ax_sd.scatter(
            origin_x,
            origin_y,
            color=TME_COLORS[origin_data['State'].iloc[0]],
            marker='X',
            s=150,
            edgecolors='black',
            linewidth=1,
            zorder=10,
            label=f'State {state_val} Origin'
        )

    ax_sd.set_xlabel(f"{x_col_sd.replace('_', ' ').title()}")
    ax_sd.set_ylabel(f"{y_col_sd.replace('_', ' ').title()}")
    ax_sd.set_xlim([82.8, 91.7])
    ax_sd.set_ylim([0.0, 5.25])
    ax_sd.legend(loc='upper right')
    ax_sd.grid(True, linestyle='--', alpha=0.3)

    #plt.show()

    output_dir = os.path.join(
        'output'
    )
    os.makedirs(output_dir, exist_ok=True)
    fig_sd.savefig(
        os.path.join(
            output_dir,
            'figure_6_panel_b.png'
        ),
        dpi=300,
        bbox_inches='tight',
        transparent=True
    )
    fig_sd.savefig(
        os.path.join(
            output_dir,
            'figure_6_panel_b.svg'
        ),
        bbox_inches='tight'
    )
    plt.close()