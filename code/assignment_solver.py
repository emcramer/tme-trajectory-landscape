import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cellsimport marimo as mo
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA 
    import umap.umap_ as umap 
    import os


@app.cell(hide_code=True)
def _():
    mo.md("""# Linear Assignment Functions for Solving, Visualization, and Animation""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""## Functions""")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Linear Assignment Solver""")
    return


@app.function
def solve_assignment(data_a: np.ndarray, data_b: np.ndarray, **cdist_kwargs) -> tuple[list[tuple], str]:
    """
    Solves the linear assignment problem to match each observation in data_a 
    to a unique, nearest neighbor in data_b using a specified distance metric.

    Args:
        data_a (np.ndarray): The first dataset (n x m), observations to be matched.
        data_b (np.ndarray): The second dataset (p x m, where p >= n), potential matches.
        **cdist_kwargs: Arbitrary keyword arguments passed directly to scipy.spatial.distance.cdist.

    Returns:
        tuple[list[tuple], str]: A tuple containing:
            1. list[tuple]: Optimal assignments: (index_A, index_B, point_A, point_B, cost).
            2. str: The name of the distance metric used.
    """
    n, m_a = data_a.shape
    p, m_b = data_b.shape

    # 1. Validation and Setup
    if m_a != m_b:
        raise ValueError("The number of features (m) must be the same for both datasets.")

    metric_used = cdist_kwargs.get('metric', 'cosine')
    cdist_kwargs['metric'] = metric_used 

    # 2. Calculate the Cost Matrix
    cost_matrix = cdist(data_a, data_b, **cdist_kwargs)

    # 3. Solve the Linear Assignment Problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Compile the results
    assignments = []

    for i in range(len(row_ind)):
        idx_a = row_ind[i]
        idx_b = col_ind[i]
        cost = cost_matrix[idx_a, idx_b]

        assignment_tuple = (
            idx_a,                            # Index 0: Index in data set A
            idx_b,                            # Index 1: Index in data set B
            data_a[idx_a].tolist(),           # Index 2: Point/vector in A
            data_b[idx_b].tolist(),           # Index 3: Point/vector in B
            cost                              # Index 4: The cost/distance of this specific match
        )
        assignments.append(assignment_tuple)

    return assignments, metric_used


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Plotting LAP Assignments""")
    return


@app.function
def plot_assignments(data_a: np.ndarray, data_b: np.ndarray, assignments: list[tuple], metric_name: str, 
                     title: str = "Optimal Linear Assignment", dim_reduction: str = 'pca'):
    """
    Visualizes the assignment problem results. If data dimensions > 2, it first
    applies dimensionality reduction. Defaults to PCA for high-dimensional data.

    Args:
        data_a (np.ndarray): Dataset A.
        data_b (np.ndarray): Dataset B.
        assignments (list[tuple]): The optimal assignments list.
        metric_name (str): The name of the distance metric used (e.g., 'cosine').
        title (str): The title for the plot.
        dim_reduction (str): Method to use ('pca' or 'umap'). Default 'pca'.
    """
    M = data_a.shape[1]
    plot_data_a = data_a
    plot_data_b = data_b
    reduction_method = ""

    if M > 2:
        # 1. Combine data for consistent dimensionality reduction
        # We need the assignment lines to connect the *reduced* points correctly.
        combined_data = np.vstack([data_a, data_b])

        # 2. Apply Dimensionality Reduction
        if dim_reduction == 'pca':
            try:
                # Target 2 dimensions for plotting
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(combined_data)
                reduction_method = "PCA (2 Components)"
            except Exception as e:
                print(f"PCA error: {e}. Cannot plot high-dimensional data.")
                return

        elif dim_reduction == 'umap':
            # UMAP requires the umap-learn library, which may not be available.
            try:
                # reducer = umap.UMAP(n_components=2, random_state=42)
                # reduced_data = reducer.fit_transform(combined_data)
                # reduction_method = "UMAP"
                print("UMAP is not fully implemented in this environment due to library dependencies. Falling back to PCA.")
                dim_reduction = 'pca' # Fallback for display purposes
                return # Skip plotting for now to avoid errors if UMAP is truly unavailable
            except NameError:
                print("UMAP library (umap-learn) not found. Cannot plot high-dimensional data with UMAP.")
                return

        else:
            print(f"Unsupported dimensionality reduction method: {dim_reduction}. Skipping visualization.")
            return

        # 3. Separate the reduced data
        plot_data_a = reduced_data[:data_a.shape[0]]
        plot_data_b = reduced_data[data_a.shape[0]:]

    else:
        # Data is already 2D
        reduction_method = "Raw 2D Data"

    # --- Plotting ---
    a_x, a_y = plot_data_a[:, 0], plot_data_a[:, 1]
    b_x, b_y = plot_data_b[:, 0], plot_data_b[:, 1]

    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Data Set A
    plt.scatter(a_x, a_y, color='skyblue', s=150, edgecolors='black', label='Data A (n)', zorder=3)

    # Plot Data Set B
    plt.scatter(b_x, b_y, color='salmon', s=100, alpha=0.6, label='Data B (p > n)', zorder=2)

    total_cost = 0

    # Draw assignments using the REDUCED coordinates
    for idx_a, idx_b, _, _, cost in assignments: 
        p_a = plot_data_a[idx_a]
        p_b = plot_data_b[idx_b]

        plt.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], 'k--', alpha=0.5, linewidth=1.5, zorder=1)

        # Annotations
        plt.annotate(f'A{idx_a}', (p_a[0], p_a[1]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9, fontweight='bold')
        plt.annotate(f'B{idx_b}', (p_b[0], p_b[1]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9, color='darkred')

        total_cost += cost

    plt.title(f"{title} (Projection: {reduction_method})\nTotal Cost ({metric_name} metric): {total_cost:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel(f"{'PC1' if M > 2 else 'Feature 1'}")
    plt.ylabel(f"{'PC2' if M > 2 else 'Feature 2'}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


@app.cell
def _():
    mo.md("""### Function: Overlapping Tweening Animation""")
    return


@app.function
def create_assignment_overlapping_animation(data_a: np.ndarray, data_b: np.ndarray, assignments: list[tuple], 
                                            metric_name: str, frames: int = 100, interval_ms: int = 50, 
                                            apply_dim_reduction: bool = True) -> animation.FuncAnimation | None:
    """
    Creates a tweening animation showing points in A moving to their assigned points in B
    and returns the FuncAnimation object for interactive display.

    Dimensionality reduction (PCA) is applied only if apply_dim_reduction is True AND 
    the data has more than 2 dimensions.

    Args:
        data_a (np.ndarray): Dataset A.
        data_b (np.ndarray): Dataset B.
        assignments (list[tuple]): The optimal assignments list.
        metric_name (str): The name of the distance metric used.
        frames (int): Total number of frames for the animation.
        interval_ms (int): Delay between frames in milliseconds.
        apply_dim_reduction (bool): If True, apply PCA if M > 2. Defaults to True.

    Returns:
        FuncAnimation | None: The Matplotlib animation object, or None if plotting fails.
    """
    M = data_a.shape[1]
    plot_data_a = data_a
    plot_data_b = data_b
    reduction_method = "Raw 2D Data"

    # --- 1. Conditional Dimensionality Reduction Setup ---
    if apply_dim_reduction and M > 2:
        combined_data = np.vstack([data_a, data_b])
        try:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(combined_data)
            reduction_method = "PCA (2 Components)"

            plot_data_a = reduced_data[:data_a.shape[0]]
            plot_data_b = reduced_data[data_a.shape[0]:]
        except Exception as e:
            print(f"PCA error during animation setup: {e}. Skipping animation.")
            return None
    elif M > 2 and not apply_dim_reduction:
        print("Warning: Cannot animate in raw feature space as M > 2 and dim reduction is disabled. Skipping animation.")
        return None

    # --- 2. Animation Data Preparation ---
    start_positions = []
    end_positions = []
    total_cost = 0

    for idx_a, idx_b, _, _, cost in assignments:
        # Start position is the point in A (or reduced A)
        start_positions.append(plot_data_a[idx_a])
        # End position is the target point in B (or reduced B)
        end_positions.append(plot_data_b[idx_b])
        total_cost += cost

    start_positions = np.array(start_positions)
    end_positions = np.array(end_positions)

    # --- 3. Matplotlib Figure Setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Data B (Fixed points)
    ax.scatter(plot_data_b[:, 0], plot_data_b[:, 1], color='salmon', s=100, alpha=0.6, label='Data B (Target)', zorder=2)
    for i, (x, y) in enumerate(plot_data_b):
        ax.annotate(f'B{i}', (x, y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9, color='darkred')

    # Initialize Data A points (will be updated)
    scat = ax.scatter(start_positions[:, 0], start_positions[:, 1], color='skyblue', s=150, edgecolors='black', label='Data A (Moving)', zorder=3)

    # Initialize the lines connecting A to B
    lines = [ax.plot([], [], 'k--', alpha=0.2, linewidth=0.5, zorder=1)[0] for _ in range(len(assignments))]

    title_projection = reduction_method if apply_dim_reduction else "Raw Data"
    ax.set_title(f"Assignment Overlapping Animation (Projection: {title_projection})\nTotal Cost ({metric_name} metric): {total_cost:.4f}", fontsize=14, fontweight='bold')

    x_label = 'PC1' if M > 2 and apply_dim_reduction else 'Feature 1'
    y_label = 'PC2' if M > 2 and apply_dim_reduction else 'Feature 2'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set limits based on all points
    all_x = np.concatenate([start_positions[:, 0], plot_data_b[:, 0]])
    all_y = np.concatenate([start_positions[:, 1], plot_data_b[:, 1]])
    ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
    ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)

    # --- 4. The Animation Update Function ---
    def update(frame):
        t = frame / frames
        new_positions = start_positions + (end_positions - start_positions) * t
        scat.set_offsets(new_positions)

        for i in range(len(assignments)):
            target_x, target_y = end_positions[i]
            current_x, current_y = new_positions[i]
            lines[i].set_data([current_x, target_x], [current_y, target_y])

        ax.set_title(f"Assignment Overlapping Animation (Projection: {title_projection}) - Frame {frame}/{frames}\nTotal Cost ({metric_name} metric): {total_cost:.4f}", fontsize=14, fontweight='bold')

        return [scat] + lines

    # --- 5. Create and Return the Animation ---
    ani = animation.FuncAnimation(fig, update, frames=frames + 1, interval=interval_ms, blit=False)
    return ani


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Side By Side Tweening Animation""")
    return


@app.function
def create_assignment_side_by_side_animation(data_a: np.ndarray, data_b: np.ndarray, assignments: list[tuple], 
                                            metric_name: str, frames: int = 100, interval_ms: int = 50, 
                                            apply_dim_reduction: bool = True, 
                                            initial_x_offset: float = 15.0, 
                                            show_ghost_trail: bool = True,
                                            a_color: str = 'skyblue', a_size: int = 180, a_marker: str = 'o',
                                            b_color: str = 'salmon', b_size: int = 120, b_marker: str = 'o',
                                            a_label: str = 'Data A (Moving)',       # NEW: Label for Data A in legend
                                            b_label: str = 'Data B (Target)',       # NEW: Label for Data B in legend
                                            plot_style: str | None = None) -> animation.FuncAnimation | None: # NEW: Matplotlib style context
    """
    Creates a tweening animation where Data A starts offset from Data B and moves
    to its assigned targets in B. Returns the FuncAnimation object.

    Args:
        data_a (np.ndarray): Dataset A.
        data_b (np.ndarray): Dataset B.
        assignments (list[tuple]): The optimal assignments list.
        metric_name (str): The name of the distance metric used.
        frames (int): Total number of frames for the animation.
        interval_ms (int): Delay between frames in milliseconds.
        apply_dim_reduction (bool): If True, apply PCA if M > 2. Defaults to True.
        initial_x_offset (float): The horizontal distance to separate Data A's starting points from Data B.
        show_ghost_trail (bool): If True, leaves a faint 'ghost' marker at A's initial position.
        a_color (str): Color of Data A points.
        a_size (int): Size of Data A points.
        a_marker (str): Marker style of Data A points.
        b_color (str): Color of Data B points.
        b_size (int): Size of Data B points.
        b_marker (str): Marker style of Data B points.
        a_label (str): Label for the Data A points in the plot legend.
        b_label (str): Label for the Data B points in the plot legend.
        plot_style (str | None): Matplotlib style context to use (e.g., 'ggplot', 'dark_background').

    Returns:
        FuncAnimation | None: The Matplotlib animation object, or None if plotting fails.
    """
    M = data_a.shape[1]
    plot_data_a = data_a
    plot_data_b = data_b
    reduction_method = "Raw 2D Data"

    # --- 1. Conditional Dimensionality Reduction Setup ---
    if apply_dim_reduction and M > 2:
        combined_data = np.vstack([data_a, data_b])
        try:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(combined_data)
            reduction_method = "PCA (2 Components)"

            plot_data_a = reduced_data[:data_a.shape[0]]
            plot_data_b = reduced_data[data_a.shape[0]:]
        except Exception as e:
            print(f"PCA error during animation setup: {e}. Skipping animation.")
            return None
    elif M > 2 and not apply_dim_reduction:
        print("Warning: Cannot animate in raw feature space as M > 2 and dim reduction is disabled. Skipping animation.")
        return None

    # --- 2. Animation Data Preparation ---
    start_positions_raw = []
    end_positions = []
    total_cost = 0

    for idx_a, idx_b, _, _, cost in assignments:
        start_positions_raw.append(plot_data_a[idx_a])
        end_positions.append(plot_data_b[idx_b])
        total_cost += cost

    start_positions_raw = np.array(start_positions_raw)
    end_positions = np.array(end_positions)

    # Apply X-offset for the side-by-side start
    offset_vector = np.array([-initial_x_offset, 0])
    start_positions_offset = start_positions_raw + offset_vector

    # Store initial positions for the ghost trail (these never change)
    ghost_positions = start_positions_offset.copy()

    # --- 3. Matplotlib Figure Setup ---
    # Apply custom style context, defaulting to 'seaborn-v0_8-whitegrid' if None is provided
    style_to_use = plot_style if plot_style is not None else 'seaborn-v0_8-whitegrid'

    with plt.style.context(style_to_use):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot Data B (Fixed Target) - Uses b_label
        ax.scatter(end_positions[:, 0], end_positions[:, 1], 
                   color=b_color, s=b_size, alpha=0.7, label=b_label, marker=b_marker, zorder=2)
        for i, (x, y) in enumerate(end_positions):
            ax.annotate(f'B{assignments[i][1]}', (x, y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9, color='darkred')

        # Plot Ghost/Shadow Impressions (Static) - Uses a_label for the ghost trail
        if show_ghost_trail:
            ax.scatter(ghost_positions[:, 0], ghost_positions[:, 1], 
                       color='gray', s=a_size * 0.5, alpha=0.2, label=f'{a_label} (Origin)', marker=a_marker, zorder=1)
            for i, (x, y) in enumerate(ghost_positions):
                ax.annotate(f'A{assignments[i][0]}', (x, y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9, color='gray', alpha=0.5)

        # Initialize Data A points (will be updated) - Uses a_label
        scat = ax.scatter(start_positions_offset[:, 0], start_positions_offset[:, 1], 
                          color=a_color, s=a_size, edgecolors='black', label=a_label, marker=a_marker, zorder=3)

        # Initialize the lines connecting A to B
        lines = [ax.plot([], [], 'k--', alpha=0.4, linewidth=1.5, zorder=1)[0] for _ in range(len(assignments))]

        title_projection = reduction_method if apply_dim_reduction else "Raw Data"
        ax.set_title(f"Side-by-Side Assignment Animation (Projection: {title_projection})\nTotal Cost ({metric_name} metric): {total_cost:.4f}", fontsize=14, fontweight='bold')

        x_label = 'PC1' if M > 2 and apply_dim_reduction else 'Feature 1'
        y_label = 'PC2' if M > 2 and apply_dim_reduction else 'Feature 2'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set limits based on all points (A start and B)
        all_x = np.concatenate([start_positions_offset[:, 0], end_positions[:, 0]])
        all_y = np.concatenate([start_positions_offset[:, 1], end_positions[:, 1]])
        ax.set_xlim(all_x.min() - 1.0, all_x.max() + 1.0)
        ax.set_ylim(all_y.min() - 1.0, all_y.max() + 1.0)

        # --- 4. The Animation Update Function ---
        def update(frame):
            t = frame / frames

            # Linear Interpolation: Moves from the OFFSET start position to the B target position
            new_positions = start_positions_offset + (end_positions - start_positions_offset) * t

            scat.set_offsets(new_positions)

            # Update the lines
            for i in range(len(assignments)):
                target_x, target_y = end_positions[i]
                current_x, current_y = new_positions[i]
                lines[i].set_data([current_x, target_x], [current_y, target_y])

            ax.set_title(f"Side-by-Side Assignment Animation (Projection: {title_projection}) - Frame {frame}/{frames}\nTotal Cost ({metric_name} metric): {total_cost:.4f}", fontsize=14, fontweight='bold')

            return [scat] + lines

        # --- 5. Create and Return the Animation ---
        ani = animation.FuncAnimation(fig, update, frames=frames + 1, interval=interval_ms, blit=False)
        return ani


@app.cell(hide_code=True)
def _():
    mo.md("""### Function: Saving Animations to .mp4 Video Files""")
    return


@app.function
def save_animation_to_mp4(ani, fname="my_animation.mp4", **kwargs):
    opts = {
        'fps':20,
        'bitrate':1800,
    }
    opts.update(kwargs)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=opts['fps'], metadata=dict(artist='Me'), bitrate=opts['bitrate'])
    ani.save(fname, writer=writer)


@app.cell(hide_code=True)
def _():
    mo.md("""## Examples""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Example 1: Simple Linear Assignment Solving in 2D

    Solving linear assignment problem when matching _m_ observations in data set _A_ to _n_ observations in dataset _B_ where _m_ < _n_.
    """
    )
    return


@app.cell
def _():
    def example_1():
        # Set a seed for reproducibility
        np.random.seed(42)

        # Define parameters: N=5 (A rows), P=10 (B rows), M=2 (features for easy plotting)
        N = 5  
        P = 10 
        M = 2 

        # Generate dummy 2D data
        data_a_2d = np.random.uniform(low=0, high=10, size=(N, M))
        data_b_2d = np.random.uniform(low=0, high=10, size=(P, M))

        # Engineer a cluster to demonstrate the unique assignment constraint:
        data_b_2d[5] = data_a_2d[1] + np.array([0.5, 0.5])
        data_b_2d[6] = data_a_2d[2] + np.array([0.6, 0.6])

        print(f"--- Running Optimal Assignment Solver ---")
        print(f"Data A ({data_a_2d.shape[0]} rows), Data B ({data_b_2d.shape[0]} rows)")

        # 1. Solve using default 'cosine' distance
        assignments_cosine, metric_cos = solve_assignment(data_a_2d, data_b_2d, metric='cosine')
        print(f"\nCompleted {len(assignments_cosine)} assignments using '{metric_cos}' metric.")

        # 2. Solve using 'euclidean' distance to demonstrate flexibility
        assignments_euclidean, metric_euc = solve_assignment(data_a_2d, data_b_2d, metric='euclidean')
        print(f"Completed {len(assignments_euclidean)} assignments using '{metric_euc}' metric.")


        # 3. Visualize the results (Cosine Distance)
        # FIX: Pass the metric_cos variable to the plot function
        plot_assignments(
            data_a_2d, 
            data_b_2d, 
            assignments_cosine, 
            metric_name=metric_cos, 
            title="Optimal Assignment (Cosine Distance)"
        )
        plt.show()
        plt.close()
    example_1()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Example 2: Linear Assignment with High Dimensional Data""")
    return


@app.cell
def _():
    def example_2():
        # --- Example Usage with 30-Dimensional Data ---

        # Set a seed for reproducibility
        np.random.seed(42)

        # Define parameters: N=10 (A rows), P=20 (B rows), M=30 (features)
        N = 10 
        P = 20 
        M = 30 

        # Generate dummy 30-dimensional data (high-D)
        data_a_hd = np.random.normal(loc=5, scale=2, size=(N, M))
        data_b_hd = np.random.normal(loc=5, scale=2, size=(P, M))

        # Engineer a cluster for one specific match to test assignment quality
        # Make A[0] very similar to B[0]
        data_b_hd[0] = data_a_hd[0] * 1.001 + np.random.normal(0, 0.01, M)

        print(f"--- Running Optimal Assignment Solver (M={M} Dimensions) ---")
        print(f"Data A ({data_a_hd.shape[0]} rows), Data B ({data_b_hd.shape[0]} rows)")

        # 1. Solve using default 'cosine' distance
        assignments_cosine, metric_cos = solve_assignment(data_a_hd, data_b_hd, metric='cosine')
        print(f"\nCompleted {len(assignments_cosine)} assignments using '{metric_cos}' metric.")


        # 2. Visualize the high-dimensional results using the default PCA reduction
        # The plot function will automatically detect M > 2 and apply PCA.
        plot_assignments(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine, 
            metric_name=metric_cos, 
            dim_reduction='pca', # Explicitly set PCA (default for >2D)
            title="Optimal Assignment (Cosine) - High Dimensional"
        )

        # Optional: Run and visualize a different metric (e.g., Euclidean)
        assignments_euclidean, metric_euc = solve_assignment(data_a_hd, data_b_hd, metric='euclidean')
        print(f"\nCompleted {len(assignments_euclidean)} assignments using '{metric_euc}' metric.")

    example_2()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Example 3: Animation Demonstration with Overlapping Tween using High Dimensional Data""")
    return


@app.cell
def _():
    def example_3():
        # --- Example Usage with 30-Dimensional Data ---

        # Set a seed for reproducibility
        np.random.seed(42)

        # Define parameters: N=10 (A rows), P=20 (B rows), M=30 (features)
        N = 10 
        P = 20 
        M = 30 

        # Generate dummy 30-dimensional data (high-D)
        data_a_hd = np.random.normal(loc=5, scale=2, size=(N, M))
        data_b_hd = np.random.normal(loc=5, scale=2, size=(P, M))

        # Engineer a cluster for one specific match to test assignment quality
        data_b_hd[0] = data_a_hd[0] * 1.001 + np.random.normal(0, 0.01, M)

        print(f"--- Running Optimal Assignment Solver (M={M} Dimensions) ---")
        print(f"Data A ({data_a_hd.shape[0]} rows), Data B ({data_b_hd.shape[0]} rows)")

        # Solve using default 'cosine' distance
        assignments_cosine, metric_cos = solve_assignment(data_a_hd, data_b_hd, metric='cosine')
        print(f"\nCompleted {len(assignments_cosine)} assignments using '{metric_cos}' metric.")


        # --- Run and Create Animation Object ---
        print(f"\n--- Creating Animation Object ---")

        # The animation object is now returned, which your execution environment 
        # may automatically display inline.
        ani = create_assignment_overlapping_animation(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine, 
            metric_name=metric_cos, 
            frames=50,       
            interval_ms=40   
        )

        if ani:
            # In some environments, simply returning the object will display it.
            print("Animation object created. If not displayed automatically, try saving it manually in an environment with the required movie writers.")
            mo.Html(ani.to_html5_video())
        else:
            print("Could not create animation object.")

        # Display the static plot as a fallback if the animation doesn't render.
        plot_assignments(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine, 
            metric_name=metric_cos, 
            dim_reduction='pca', 
            title="Optimal Assignment (Cosine) - High Dimensional"
        )

        return ani

    ani = example_3()
    return (ani,)


@app.cell
def _(ani):
    mo.Html(ani.to_html5_video())
    return


@app.cell(hide_code=True)
def _():
    mo.md("""### Example 4: Side-by-side Tween Animation Demonstration using High Dimensional Data""")
    return


@app.cell
def _():
    def example_4():
        # --- Example Usage ---

        np.random.seed(1337)

        # --- High-Dimensional Example (30D) ---
        N_hd = 12 
        P_hd = 25 
        M_hd = 30 
        # Generate data clustered around 0 for A, and around 10 for B to make the movement dramatic
        data_a_hd = np.random.normal(loc=0, scale=1.5, size=(N_hd, M_hd))
        data_b_hd = np.random.normal(loc=10, scale=1.5, size=(P_hd, M_hd))

        print(f"--- Running Optimal Assignment Solver (M={M_hd} Dimensions) ---")
        assignments_cosine_hd, metric_cos_hd = solve_assignment(data_a_hd, data_b_hd, metric='cosine')
        print(f"Completed {len(assignments_cosine_hd)} assignments using '{metric_cos_hd}' metric.")


        # 1. Animate High-Dimensional Data (PCA applied and side-by-side)
        print(f"\n--- Creating Side-by-Side Animation (PCA + Ghost Trail) ---")
        animation_side_by_side = create_assignment_side_by_side_animation(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine_hd, 
            metric_name=metric_cos_hd, 
            apply_dim_reduction=True, # Apply PCA
            initial_x_offset=5.0,    # Large offset for clear separation
            show_ghost_trail=True     # Show the initial ghost positions
        )
        if animation_side_by_side:
            plt.show() 

        # 2. Static Plot High-Dimensional Data (Fallback)
        plot_assignments(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine_hd, 
            metric_name=metric_cos_hd, 
            dim_reduction='pca', 
            title="Optimal Assignment (Cosine) - High Dimensional"
        )

        return animation_side_by_side

    ani_ss = example_4()
    return (ani_ss,)


@app.cell
def _(ani_ss):
    mo.Html(ani_ss.to_html5_video())
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    ### Example 5: Demonstration using Pre-Computed UMAP Coordinates for High Dimensional Data Sets

    Also include demonstration of side-by-side tween animation with custom aesthetics.
    """
    )
    return


@app.cell
def _():
    # testing the animation using pre-computed umap coordinates
    def example_5():
        # --- Example Usage ---

        np.random.seed(1337)

        # --- High-Dimensional Example (30D) ---
        N_hd = 100 
        P_hd = 1000
        M_hd = 30 
        # Generate data clustered around 0 for A, and around 10 for B to make the movement dramatic
        data_a_hd = np.random.normal(loc=0, scale=1.5, size=(N_hd, M_hd))
        data_b_hd = np.random.normal(loc=10, scale=1.5, size=(P_hd, M_hd))

        data_a_reducer = umap.UMAP(random_state=42)
        data_a_embedding = data_a_reducer.fit_transform(data_a_hd)

        data_b_reducer = umap.UMAP(random_state=42)
        data_b_embedding = data_b_reducer.fit_transform(data_b_hd)

        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(data_a_embedding[:, 0], data_a_embedding[:, 1], color=plt.cm.tab10(1))
        ax[0].set_title('Data A UMAP')
        ax[1].scatter(data_b_embedding[:, 0], data_b_embedding[:, 1], color=plt.cm.tab10(2))
        ax[1].set_title('Data B UMAP')
        plt.tight_layout()
        plt.show()
        plt.close()

        print(f"--- Running Optimal Assignment Solver (M={M_hd} Dimensions) ---")
        assignments_cosine_hd, metric_cos_hd = solve_assignment(data_a_hd, data_b_hd, metric='cosine')
        print(f"Completed {len(assignments_cosine_hd)} assignments using '{metric_cos_hd}' metric.")

        # 1. Animate High-Dimensional Data (PCA applied and side-by-side)
        print(f"\n--- Creating Side-by-Side Animation (PCA + Ghost Trail) ---")
        animation_side_by_side = create_assignment_side_by_side_animation(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine_hd, 
            metric_name=metric_cos_hd, 
            apply_dim_reduction=True,
            initial_x_offset=18.0, 
            show_ghost_trail=True,

            # --- CUSTOM STYLING ARGUMENTS ---
            a_color='C2',             # Data A color changed to the Matplotlib 'C2' (green)
            a_size=200,               
            a_marker='v',             
            b_color='C3',             # Data B color changed to the Matplotlib 'C3' (red)
            b_size=150,               
            b_marker='s',             
            a_label='Moving Samples', # NEW: Custom label for A
            b_label='Reference Nodes',# NEW: Custom label for B
            plot_style='ggplot'       # NEW: Use the 'ggplot' style context
            # -----------------------------------
        )
        if animation_side_by_side:
            plt.show() 

        # 2. Static Plot High-Dimensional Data (Fallback)
        plot_assignments(
            data_a_hd, 
            data_b_hd, 
            assignments_cosine_hd, 
            metric_name=metric_cos_hd, 
            dim_reduction='pca', 
            title="Optimal Assignment (Cosine) - High Dimensional"
        )

        return animation_side_by_side
    ex5_ani = example_5()
    return (ex5_ani,)


@app.cell
def _(ex5_ani):
    mo.Html(ex5_ani.to_html5_video())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
