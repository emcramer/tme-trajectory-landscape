import marimo

__generated_with = "0.16.5"
app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.md(
        r"""
    # ABM State 1 & 6 Phase Portraits
    This notebook generates interactive phase portraits for key cell
    populations in the ABM simulations, focusing on State 1 (Immune-rich)
    and State 6 (Cold/Immune-excluded).

    Select two features from the dropdowns below to visualize their average
    dynamical relationship over time within each state.
    """
    )
    return


@app.cell
def _():
    import os
    import sys
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Ensure the code directory is in the path for project utilities
    sys.path.append(os.path.join(os.getcwd(), 'code'))
    from tme_style import apply_tme_style, TME_COLORS

    # Apply standard project styling
    #apply_tme_style()
    return LinearSegmentedColormap, TME_COLORS, os, pd, plt


@app.cell
def _(plt):
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
    return


@app.cell
def _(os, pd):
    # Load the data
    input_path = 'data/abm/processed/state_1_state_6_within_state_dynamics.csv'
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
    else:
        df = pd.DataFrame() # Empty dataframe if file not found

    # Get numeric columns for selection, excluding identifiers
    feature_options = sorted([
        col for col in df.select_dtypes(include='number').columns 
        if col not in ['sim_id', 'start_time_step', 'end_time_step', 'window_index_in_sim', 'state_label', 'State']
    ])
    return df, feature_options


@app.cell
def _(feature_options):
    # UI elements for feature selection
    x_feature = mo.ui.dropdown(
        options=feature_options, 
        value='malignant_epithelial_cell', 
        label="X-Axis Feature:"
    )
    y_feature = mo.ui.dropdown(
        options=feature_options, 
        value='effector_T_cell', 
        label="Y-Axis Feature:"
    )

    # Display UI elements
    mo.vstack([x_feature, y_feature], justify='start')
    return x_feature, y_feature


@app.cell
def _(TME_COLORS, df, plt, x_feature, y_feature):
    # Display a message if the dataframe is empty
    if df.empty:
        mo.output.replace(
            mo.md("## Data not found!\n\nPlease ensure the data file exists at `data/abm/processed/state_1_state_6_within_state_dynamics.csv`.")
        )
        mo.stop()

    with plt.style.context('default'):

        # Get selected feature values from the UI elements
        x_col = x_feature.value
        y_col = y_feature.value

        # Calculate average trajectories for each state
        # We group by State and time (end_time_step), then calculate the mean
        avg_trajectories = df.groupby(['State', 'end_time_step'])[[x_col, y_col]].mean().reset_index()

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectories for each state
        for state, group in avg_trajectories.groupby('State'):
            # Sort by time to ensure correct plotting order
            group = group.sort_values('end_time_step')

            # Plot the trajectory line
            ax.plot(group[x_col], group[y_col], color=TME_COLORS[state], label=f'State {state} Avg. Trajectory', linewidth=2)

            # Plot scatter points for each time step
            ax.scatter(group[x_col], group[y_col], color=TME_COLORS[state], s=30, zorder=3)

            # Add arrows to show direction
            # We take pairs of consecutive points to draw an arrow between them
            for i in range(len(group) - 1):
                p1 = group.iloc[i]
                p2 = group.iloc[i+1]
                dx = p2[x_col] - p1[x_col]
                dy = p2[y_col] - p1[y_col]

                # Quiver is used to draw arrows
                # We place the arrow at the midpoint of the segment
                ax.quiver(
                    p1[x_col] + dx/2, p1[y_col] + dy/2, dx, dy,
                    color=TME_COLORS[state], 
                    #color='black',
                    angles='xy', scale_units='xy', scale=1, 
                    width=0.005, headwidth=4, headlength=6, zorder=4
                )

            ax.scatter(group[x_col].values[0], group[y_col].values[0], color='red', marker='x', s=100, label='Origin (Start)', zorder=5)

        # Formatting
        ax.set_xlabel(f"{x_col.replace('_', ' ').title()}")
        ax.set_ylabel(f"{y_col.replace('_', ' ').title()}")
        ax.set_title(f"Average Phase Portrait: {y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
        ax.legend(title="TME State")
        ax.grid(True, linestyle='--', alpha=0.3)

        # Return figure to be displayed by Marimo
        plt.tight_layout()

        plt.show()
        plt.close()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Origin-Normalized Phase Portraits
    To better compare the *relative* dynamics between states, this section
    normalizes the trajectories such that they both begin at (0, 0).
    This removes differences in the initial abundance of cells and highlights
    how the relationship between features changes over time.
    """
    )
    return


@app.cell
def _(TME_COLORS, df, plt, x_feature, y_feature):
    # Get selected feature values
    x_col_norm = x_feature.value
    y_col_norm = y_feature.value

    # Calculate average trajectories
    avg_traj_norm = df.groupby(['State', 'end_time_step'])[[x_col_norm, y_col_norm]].mean().reset_index()

    with plt.style.context('default'):
        fig_norm, ax_norm = plt.subplots(figsize=(10, 8))

        for state_norm, group_norm in avg_traj_norm.groupby('State'):
            # Sort by time
            group_norm = group_norm.sort_values('end_time_step')

            # Center at zero
            x0_norm = group_norm[x_col_norm].iloc[0]
            y0_norm = group_norm[y_col_norm].iloc[0]

            x_adj = group_norm[x_col_norm] #- x0_norm
            y_adj = group_norm[y_col_norm] #- y0_norm

            # Plot adjusted trajectory
            ax_norm.plot(x_adj, y_adj, color=TME_COLORS[state_norm], 
                         label=f'State {state_norm} Normalized', linewidth=2)
            ax_norm.scatter(x_adj, y_adj, color=TME_COLORS[state_norm], s=30, zorder=3)

            # Add arrows
            for j in range(len(group_norm) - 1):
                dx_norm = x_adj.iloc[j+1] - x_adj.iloc[j]
                dy_norm = y_adj.iloc[j+1] - y_adj.iloc[j]

                ax_norm.quiver(
                    x_adj.iloc[j] + dx_norm/2, y_adj.iloc[j] + dy_norm/2, dx_norm, dy_norm,
                    color=TME_COLORS[state_norm], 
                    angles='xy', scale_units='xy', scale=1, 
                    width=0.005, headwidth=4, headlength=6
                )

        # Origin marker
        #ax_norm.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        #ax_norm.axvline(0, color='black', linewidth=0.5, alpha=0.5)
        #ax_norm.scatter(0, 0, color='red', marker='x', s=100, label='Origin (Start)', zorder=5)

        ax_norm.set_xlabel(f"Delta {x_col_norm.replace('_', ' ').title()}")
        ax_norm.set_ylabel(f"Delta {y_col_norm.replace('_', ' ').title()}")
        ax_norm.set_title(f"Origin-Normalized Phase Portrait\n{y_col_norm.replace('_', ' ').title()} vs {x_col_norm.replace('_', ' ').title()}")
        ax_norm.legend(title="TME State")
        ax_norm.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        plt.show()
        plt.close()
    return


@app.cell
def _():
    mo.md(
        r"""
    # System Vector Field
    This section visualizes the underlying dynamical "flow" of the system by
    aggregating the displacements $(dx, dy)$ across all simulations into a
    vector field. This helps identify attractors (where vectors converge)
    and the general direction of the system's evolution in the phase space.
    """
    )
    return


@app.cell
def _(df, pd, plt, x_feature, y_feature):
    import numpy as np

    def _():
        # Get selected feature values
        x_col_v = x_feature.value
        y_col_v = y_feature.value

        # Calculate displacements for all individual simulations
        # We want to see the flow everywhere trajectories have been
        flow_data = []
        for sim_id, sim_group in df.groupby('sim_id'):
            sim_group = sim_group.sort_values('window_index_in_sim')
            x_vals = sim_group[x_col_v].values
            y_vals = sim_group[y_col_v].values

            dx = np.diff(x_vals)
            dy = np.diff(y_vals)

            # Midpoints for the vector origin
            x_mid = x_vals[:-1]
            y_mid = y_vals[:-1]

            for i in range(len(dx)):
                flow_data.append([x_mid[i], y_mid[i], dx[i], dy[i]])

        flow_df = pd.DataFrame(flow_data, columns=['x', 'y', 'dx', 'dy'])

        # Define a grid for the vector field
        grid_res = 20
        x_range = np.linspace(df[x_col_v].min(), df[x_col_v].max(), grid_res)
        y_range = np.linspace(df[y_col_v].min(), df[y_col_v].max(), grid_res)
        X, Y = np.meshgrid(x_range, y_range)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        Count = np.zeros_like(X)

        # Bin the displacements into the grid
        # This is a simple way to estimate the average velocity at each point
        x_bins = np.digitize(flow_df['x'], x_range) - 1
        y_bins = np.digitize(flow_df['y'], y_range) - 1

        for i in range(len(flow_df)):
            xb = x_bins[i]
            yb = y_bins[i]
            if 0 <= xb < grid_res and 0 <= yb < grid_res:
                U[yb, xb] += flow_df['dx'].iloc[i]
                V[yb, xb] += flow_df['dy'].iloc[i]
                Count[yb, xb] += 1

        # Average the vectors
        mask = Count > 0
        U[mask] /= Count[mask]
        V[mask] /= Count[mask]

        with plt.style.context('default'):
            fig_v, ax_v = plt.subplots(figsize=(10, 8))

            # Plot the vector field
            # We use a color scale based on the magnitude of the velocity
            M = np.hypot(U, V)
            q = ax_v.quiver(X, Y, U, V, M, cmap='viridis', units='xy', scale=1, 
                            width=0.015, headwidth=3, headlength=5, alpha=0.8)

            fig_v.colorbar(q, label='Velocity Magnitude (Normalized Displacement)')

            # Add a streamplot for better visualization of the flow manifold
            # Streamplot requires a regular grid with no NaNs, so we use it where we have data
            if Count.sum() > 0:
                ax_v.streamplot(
                    x_range, 
                    y_range, 
                    U, 
                    V, 
                    color='gray', 
                    #alpha=0.3, 
                    linewidth=1, 
                    density=1.5
                )

            ax_v.set_xlabel(f"{x_col_v.replace('_', ' ').title()}")
            ax_v.set_ylabel(f"{y_col_v.replace('_', ' ').title()}")
            ax_v.set_title(f"System Vector Field and Flow: {y_col_v.replace('_', ' ').title()} vs {x_col_v.replace('_', ' ').title()}")
            ax_v.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return (np,)


@app.cell
def _():
    mo.md(
        r"""
    # State-Specific Vector Fields and Occupancy
    This section further breaks down the system dynamics by plotting the
    flow and occupancy for States 1 and 6 separately. This reveals if
    different states are driven by distinct underlying manifolds or if they
    occupy different regions of a shared global attractor.

    The background colors represent the relative density (occupancy) of
    each state in the phase space.
    """
    )
    return


@app.cell
def _(TME_COLORS, df, np, pd, plt, x_feature, y_feature):
    def _():
        with plt.style.context('default'):
            # Get selected feature values
            x_col_s = x_feature.value
            y_col_s = y_feature.value

            states_to_plot = [1, 6]
            fig_s, axes_s = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

            # Grid definition for consistent comparison
            grid_res_s = 20
            x_min, x_max = df[x_col_s].min(), df[x_col_s].max()
            y_min, y_max = df[y_col_s].min(), df[y_col_s].max()
            x_range_s = np.linspace(x_min, x_max, grid_res_s)
            y_range_s = np.linspace(y_min, y_max, grid_res_s)
            X_s, Y_s = np.meshgrid(x_range_s, y_range_s)

            for idx, state_val in enumerate(states_to_plot):
                ax_s = axes_s[idx]
                state_df = df[df['State'] == state_val]

                # 1. Background Density (Occupancy)
                if not state_df.empty:
                    # We use a 2D histogram as a proxy for density
                    counts, xeb, yeb = np.histogram2d(
                        state_df[x_col_s], state_df[y_col_s], 
                        bins=[x_range_s, y_range_s]
                    )
                    # Plot occupancy as a contour or pcolormesh
                    # We use the state's primary color but as a light gradient
                    from matplotlib.colors import LinearSegmentedColormap
                    cmap_s = LinearSegmentedColormap.from_list(
                        f'state_{state_val}_cmap', 
                        ['#ffffff', TME_COLORS[state_val]]
                    )
                    ax_s.pcolormesh(xeb, yeb, counts.T, cmap=cmap_s, alpha=0.3, shading='auto')

                # 2. State-Specific Vector Field
                flow_data_s = []
                for sim_id, sim_group in state_df.groupby('sim_id'):
                    sim_group = sim_group.sort_values('window_index_in_sim')
                    x_vals = sim_group[x_col_s].values
                    y_vals = sim_group[y_col_s].values
                    dx = np.diff(x_vals)
                    dy = np.diff(y_vals)
                    x_mid = x_vals[:-1]
                    y_mid = y_vals[:-1]
                    for i in range(len(dx)):
                        flow_data_s.append([x_mid[i], y_mid[i], dx[i], dy[i]])

                if flow_data_s:
                    flow_df_s = pd.DataFrame(flow_data_s, columns=['x', 'y', 'dx', 'dy'])
                    U_s = np.zeros_like(X_s)
                    V_s = np.zeros_like(Y_s)
                    Count_s = np.zeros_like(X_s)

                    xb_s = np.digitize(flow_df_s['x'], x_range_s) - 1
                    yb_s = np.digitize(flow_df_s['y'], y_range_s) - 1

                    for i in range(len(flow_df_s)):
                        xb = xb_s[i]
                        yb = yb_s[i]
                        if 0 <= xb < grid_res_s and 0 <= yb < grid_res_s:
                            U_s[yb, xb] += flow_df_s['dx'].iloc[i]
                            V_s[yb, xb] += flow_df_s['dy'].iloc[i]
                            Count_s[yb, xb] += 1

                    mask_s = Count_s > 0
                    U_s[mask_s] /= Count_s[mask_s]
                    V_s[mask_s] /= Count_s[mask_s]

                    # Plot the state-specific quiver
                    M_s = np.hypot(U_s, V_s)
                    ax_s.quiver(X_s, Y_s, U_s, V_s, color=TME_COLORS[state_val], 
                                alpha=0.8, width=0.005)

                    # Add streamplot for that state
                    ax_s.streamplot(
                        x_range_s, 
                        y_range_s, 
                        U_s, 
                        V_s, 
                        color=TME_COLORS[state_val], 
                        #alpha=0.4, 
                        linewidth=1
                    )

                ax_s.set_title(f"State {state_val} Flow & Occupancy")
                ax_s.set_xlabel(f"{x_col_s.replace('_', ' ').title()}")
                if idx == 0:
                    ax_s.set_ylabel(f"{y_col_s.replace('_', ' ').title()}")
                ax_s.grid(True, linestyle='--', alpha=0.2)

            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Superimposed State Dynamics
    This final section superimposes the vector fields and occupancy densities
    for both State 1 and State 6 on a single plot. This provides a direct
    comparison of their dynamical landscapes.

    The red 'X' marks the average initial condition (timepoint 0) for each state.
    """
    )
    return


@app.cell
def _(
    LinearSegmentedColormap,
    TME_COLORS,
    df,
    np,
    os,
    pd,
    plt,
    x_feature,
    y_feature,
):
    def _():

        # Get selected feature values
        x_col_c = x_feature.value
        y_col_c = y_feature.value

        # Grid definition
        grid_res_c = 25
        x_min, x_max = df[x_col_c].min(), df[x_col_c].max()
        y_min, y_max = df[y_col_c].min(), df[y_col_c].max()
        x_range_c = np.linspace(x_min, x_max, grid_res_c)
        y_range_c = np.linspace(y_min, y_max, grid_res_c)
        X_c, Y_c = np.meshgrid(x_range_c, y_range_c)

        with plt.style.context('default'):
            fig_c, ax_c = plt.subplots(figsize=(12, 11)) # Increased height slightly

            # Position for the main plot
            ax_c.set_position([0.1, 0.12, 0.75, 0.7]) # Left, Bottom, Width, Height

            for state_val in [1, 6]:
                state_df = df[df['State'] == state_val]
                if state_df.empty:
                    continue

                # 1. Occupancy Density
                counts, xeb, yeb = np.histogram2d(
                    state_df[x_col_c], state_df[y_col_c], 
                    bins=[x_range_c, y_range_c]
                )

                # Custom colormap for this state
                cmap_c = LinearSegmentedColormap.from_list(
                    f'state_{state_val}_combined', 
                    [(1,1,1,0), TME_COLORS[state_val]] # Transparent to state color
                )

                # Mask zero counts so they are transparent (removed)
                counts_masked = np.ma.masked_where(counts == 0, counts)
                im = ax_c.pcolormesh(xeb, yeb, counts_masked.T, cmap=cmap_c, alpha=0.4, shading='auto')

                # Colorbar at the bottom
                #cax = fig_c.add_axes([0.15 + (state_val == 6) * 0.4, 0.08, 0.3, 0.02])

                # Colorbar at the top
                # TODO: modify the line below to move the colorbar axes above the vector field plot area but below the plot title
                cax = fig_c.add_axes([0.15 + (state_val == 6) * 0.4, 0.90, 0.3, 0.02])

                plt.colorbar(im, cax=cax, label=f'State {state_val} Time Step Occupancy', orientation='horizontal')

                # 2. Vector Field
                flow_data_c = []
                for sim_id, sim_group in state_df.groupby('sim_id'):
                    sim_group = sim_group.sort_values('window_index_in_sim')
                    x_vals = sim_group[x_col_c].values
                    y_vals = sim_group[y_col_c].values

                    dx = np.diff(x_vals)
                    dy = np.diff(y_vals)
                    x_mid = x_vals[:-1]
                    y_mid = y_vals[:-1]
                    for i in range(len(dx)):
                        flow_data_c.append([x_mid[i], y_mid[i], dx[i], dy[i]])

                if flow_data_c:
                    flow_df_c = pd.DataFrame(flow_data_c, columns=['x', 'y', 'dx', 'dy'])
                    U_c = np.zeros_like(X_c)
                    V_c = np.zeros_like(Y_c)
                    Count_c = np.zeros_like(X_c)

                    xb_c = np.digitize(flow_df_c['x'], x_range_c) - 1
                    yb_c = np.digitize(flow_df_c['y'], y_range_c) - 1

                    for i in range(len(flow_df_c)):
                        xb = xb_c[i]
                        yb = yb_c[i]
                        if 0 <= xb < grid_res_c and 0 <= yb < grid_res_c:
                            U_c[yb, xb] += flow_df_c['dx'].iloc[i]
                            V_c[yb, xb] += flow_df_c['dy'].iloc[i]
                            Count_c[yb, xb] += 1

                    mask_c = Count_c > 0
                    U_c[mask_c] /= Count_c[mask_c]
                    V_c[mask_c] /= Count_c[mask_c]

                    # Adjust arrow sizes to be smaller and proportional to magnitude
                    M_c = np.hypot(U_c, V_c)
                    ax_c.quiver(
                        X_c, 
                        Y_c, 
                        U_c, 
                        V_c, 
                        color=TME_COLORS[state_val], 
                        alpha=0.8, 
                        #width=0.002, 
                        #scale=0.5, 
                        units='xy', 
                        headwidth=3, 
                        headlength=4, 
                        label=f'State {state_val} Flow'
                    )

                # 3. Mark Origin using the first time point
                min_time = state_df['end_time_step'].min()
                origin_data = state_df[state_df['end_time_step'] == min_time]
                origin_x = origin_data[x_col_c].mean()
                origin_y = origin_data[y_col_c].mean()

                ax_c.scatter(
                    origin_x, 
                    origin_y, 
                    color= TME_COLORS[origin_data['State'].iloc[0]], #'red',
                    marker='X', 
                    s=150, 
                    edgecolors='black', 
                    linewidth=1, 
                    zorder=10, 
                    label=f'State {state_val} Origin'
                )

            ax_c.set_xlabel(f"{x_col_c.replace('_', ' ').title()}")
            ax_c.set_ylabel(f"{y_col_c.replace('_', ' ').title()}")

            # adjust axes to capture full field with buffer room
            ax_c.set_xlim([82.5, 92])
            ax_c.set_ylim([-0.5, 5.5])

            #fig_c.suptitle(
            #    f"Vector Fields & TME State Occupancy: States 1 & 6\n{y_col_c.replace('_', ' ').title()} vs {x_col_c.replace('_', ' ').title()}",
             #   y=0.96,
             #   fontsize=FONT_MAIN_TITLE
            #)

            # Place legend to the right
            ax_c.legend(loc='upper right')#, bbox_to_anchor=(1.05, 1))
            ax_c.grid(True, linestyle='--', alpha=0.3)

            plt.show()

            output_dir = os.path.join(
                'output', 
                'figures',
                'abm',
                'phase_portraits'
            )
            os.makedirs(output_dir, exist_ok=True)
            fig_c.savefig(
                os.path.join(
                    output_dir,
                    'tme_state_occupancy_s1_s6_vector_fields.png'
                ),
                dpi=300,
                bbox_inches = 'tight',
                transparent=True
            )
            fig_c.savefig(
                os.path.join(
                    output_dir,
                    'tme_state_occupancy_s1_s6_vector_fields.svg'
                ),
                bbox_inches = 'tight'
            )
            plt.close()
    _()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Biologically-Constrained Superimposed State Dynamics
    This plot repeats the superimposed vector field above, but enforces the
    biological constraint that cell populations cannot be negative. At grid
    rows where the y-axis floor is zero (or below), any downward-pointing
    velocity components are zeroed out. This prevents arrows from crossing
    the y=0 boundary, giving a more accurate picture of the attractor
    geometry near the extinction boundary.
    """
    )
    return


@app.cell
def _(
    LinearSegmentedColormap,
    TME_COLORS,
    df,
    np,
    os,
    pd,
    plt,
    x_feature,
    y_feature,
):
    def _():

        # Get selected feature values
        x_col_bc = x_feature.value
        y_col_bc = y_feature.value

        # Grid definition
        grid_res_bc = 25
        x_min_bc, x_max_bc = df[x_col_bc].min(), df[x_col_bc].max()
        y_min_bc, y_max_bc = df[y_col_bc].min(), df[y_col_bc].max()
        x_range_bc = np.linspace(x_min_bc, x_max_bc, grid_res_bc)
        # Floor the y range at 0: the effector T cell population cannot be
        # negative, so no histogram bin should start below 0. Any overshoot
        # in the raw data (artefact of the time-delay embedding window) is
        # absorbed into the first bin above 0.
        y_range_bc = np.linspace(max(0, y_min_bc), y_max_bc, grid_res_bc)
        X_bc, Y_bc = np.meshgrid(x_range_bc, y_range_bc)

        with plt.style.context('default'):
            fig_bc, ax_bc = plt.subplots(figsize=(12, 11))
            ax_bc.set_position([0.1, 0.12, 0.75, 0.7])

            for state_val in [1, 6]:
                state_df = df[df['State'] == state_val]
                if state_df.empty:
                    continue

                # 1. Occupancy Density
                counts, xeb, yeb = np.histogram2d(
                    state_df[x_col_bc], state_df[y_col_bc],
                    bins=[x_range_bc, y_range_bc]
                )
                cmap_bc = LinearSegmentedColormap.from_list(
                    f'state_{state_val}_bc',
                    [(1, 1, 1, 0), TME_COLORS[state_val]]
                )
                counts_masked = np.ma.masked_where(counts == 0, counts)
                # Shift any bin edges below y=0 upward by one bin width so
                # that bins which straddle the biological floor are displaced
                # above it rather than clipped or collapsed.
                bin_width_y = y_range_bc[1] - y_range_bc[0]
                yeb_shifted = np.where(yeb < 0, yeb + bin_width_y, yeb)
                im = ax_bc.pcolormesh(xeb, yeb_shifted, counts_masked.T, cmap=cmap_bc, alpha=0.4, shading='auto')

                cax = fig_bc.add_axes([0.15 + (state_val == 6) * 0.4, 0.90, 0.3, 0.02])
                plt.colorbar(im, cax=cax, label=f'State {state_val} Time Step Occupancy', orientation='horizontal')

                # 2. Vector Field
                flow_data_bc = []
                for sim_id, sim_group in state_df.groupby('sim_id'):
                    sim_group = sim_group.sort_values('window_index_in_sim')
                    x_vals = sim_group[x_col_bc].values
                    y_vals = sim_group[y_col_bc].values
                    dx = np.diff(x_vals)
                    dy = np.diff(y_vals)
                    x_mid = x_vals[:-1]
                    y_mid = y_vals[:-1]
                    for i in range(len(dx)):
                        flow_data_bc.append([x_mid[i], y_mid[i], dx[i], dy[i]])

                if flow_data_bc:
                    flow_df_bc = pd.DataFrame(flow_data_bc, columns=['x', 'y', 'dx', 'dy'])
                    U_bc = np.zeros_like(X_bc)
                    V_bc = np.zeros_like(Y_bc)
                    Count_bc = np.zeros_like(X_bc)

                    xb_bc = np.digitize(flow_df_bc['x'], x_range_bc) - 1
                    yb_bc = np.digitize(flow_df_bc['y'], y_range_bc) - 1

                    for i in range(len(flow_df_bc)):
                        xb = xb_bc[i]
                        yb = yb_bc[i]
                        if 0 <= xb < grid_res_bc and 0 <= yb < grid_res_bc:
                            U_bc[yb, xb] += flow_df_bc['dx'].iloc[i]
                            V_bc[yb, xb] += flow_df_bc['dy'].iloc[i]
                            Count_bc[yb, xb] += 1

                    mask_bc = Count_bc > 0
                    U_bc[mask_bc] /= Count_bc[mask_bc]
                    V_bc[mask_bc] /= Count_bc[mask_bc]

                    # Biological floor constraint: shorten arrows whose tips would
                    # cross y=0, without altering their angle.
                    #
                    # Geometry: treat the arrow as the hypotenuse of a right
                    # triangle whose vertical leg runs from the tail (height Y
                    # above y=0) down to y=0, and whose hypotenuse lies along the
                    # arrow's direction. The required scale factor is:
                    #
                    #   t = Y / |V|   (vertical distance to floor / vertical component)
                    #
                    # where Y > 0 (tail above floor) and |V| > 0 (pointing down).
                    # t is therefore in (0, 1). Multiplying both U and V by t
                    # shortens the hypotenuse proportionally so the tip lands
                    # exactly on y=0 while the angle is unchanged.
                    #
                    # Guard: only clip when the tail is strictly above y=0 (Y > 0).
                    # Grid points at or below y=0 are excluded so t stays positive
                    # and no arrow direction is reversed.
                    #
                    # Note: the need for this correction arises from the time-delay
                    # embedding window used to compute displacements. The cell
                    # population never actually goes negative — deceleration events
                    # occur faster than the window can resolve, so the averaged
                    # instantaneous velocity appears more negative than it truly is.
                    needs_clip = (Y_bc > 0) & (V_bc < 0) & ((Y_bc + V_bc) < 0)
                    t = np.where(needs_clip, Y_bc / (-V_bc), 1.0)
                    U_bc = U_bc * t
                    V_bc = V_bc * t

                    from matplotlib.patches import Rectangle as _Rect
                    q_bc = ax_bc.quiver(
                        X_bc,
                        Y_bc,
                        U_bc,
                        V_bc,
                        color=TME_COLORS[state_val],
                        alpha=0.8,
                        units='xy',
                        headwidth=3,
                        headlength=4,
                        label=f'State {state_val} Flow'
                    )
                    # Clip the rendered quiver (including arrowheads) at y=0.
                    # Even after scaling arrow vectors so their tips land at y=0,
                    # matplotlib draws the arrowhead beyond the tip, causing visual
                    # overshoot. set_clip_path restricts rendering of every part of
                    # the artist — shaft and head — to the region y >= 0.
                    _clip = _Rect(
                        (x_min_bc - 1, 0),
                        x_max_bc - x_min_bc + 2,
                        y_max_bc + 1,
                        transform=ax_bc.transData,
                        visible=False
                    )
                    ax_bc.add_patch(_clip)
                    q_bc.set_clip_path(_clip)

                # 3. Mark Origin
                min_time = state_df['end_time_step'].min()
                origin_data = state_df[state_df['end_time_step'] == min_time]
                origin_x = origin_data[x_col_bc].mean()
                origin_y = origin_data[y_col_bc].mean()

                ax_bc.scatter(
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

            ax_bc.set_xlabel(f"{x_col_bc.replace('_', ' ').title()}")
            ax_bc.set_ylabel(f"{y_col_bc.replace('_', ' ').title()}")
            ax_bc.set_xlim([82.8, 91.7])
            ax_bc.set_ylim([0.0, 5.25])
            ax_bc.legend(loc='upper right')
            ax_bc.grid(True, linestyle='--', alpha=0.3)

            plt.show()

            output_dir = os.path.join(
                'output',
                'figures',
                'abm',
                'phase_portraits'
            )
            os.makedirs(output_dir, exist_ok=True)
            fig_bc.savefig(
                os.path.join(
                    output_dir,
                    'tme_state_occupancy_s1_s6_vector_fields_bioconstrained.png'
                ),
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            fig_bc.savefig(
                os.path.join(
                    output_dir,
                    'tme_state_occupancy_s1_s6_vector_fields_bioconstrained.svg'
                ),
                bbox_inches='tight'
            )
            plt.close()
    _()
    return


@app.cell
def _():
    mo.md(
        r"""
    # Subdivided Arrow Vector Field
    Same biologically-constrained vector field as above, but long arrows are
    broken into chains of shorter sub-arrows placed end-to-end along the
    original direction. Each sub-arrow has the same angle as the full arrow;
    only the segment length is reduced. Sub-arrows whose starting point falls
    below y=0 are dropped, and the y=0 clip path still catches any residual
    arrowhead overshoot.
    """
    )
    return


@app.cell
def panel6b(
    LinearSegmentedColormap,
    TME_COLORS,
    df,
    np,
    os,
    pd,
    plt,
    x_feature,
    y_feature,
):
    def _():

        # Get selected feature values
        x_col_sd = x_feature.value
        y_col_sd = y_feature.value

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

                    from matplotlib.patches import Rectangle as _Rect
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

            plt.show()

            output_dir = os.path.join(
                'output',
                'figures',
                'abm',
                'phase_portraits'
            )
            os.makedirs(output_dir, exist_ok=True)
            fig_sd.savefig(
                os.path.join(
                    output_dir,
                    'tme_state_occupancy_s1_s6_vector_fields_subdivided.png'
                ),
                dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            fig_sd.savefig(
                os.path.join(
                    output_dir,
                    'tme_state_occupancy_s1_s6_vector_fields_subdivided.svg'
                ),
                bbox_inches='tight'
            )
            plt.close()
    _()
    return


if __name__ == "__main__":
    app.run()
