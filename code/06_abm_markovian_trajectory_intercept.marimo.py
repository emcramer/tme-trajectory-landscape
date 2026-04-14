import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import os
    import pickle

    # Set plot style
    sns.set_context("talk")
    sns.set_style("whitegrid")

    plt.rcParams['svg.fonttype'] = 'none'
    return mo, np, os, pd, pickle, plt, sns


@app.cell
def _(mo):
    mo.md(
        """
    # ABM Markovian Trajectory Intercept Analysis

    This notebook explores the potential to redirect ABM simulation trajectories by "switching" their underlying dynamics to those of a different parameter set.

    Using a **Markov State Model (MSM)** surrogate, we ask:
    *If we intervene at time $t$ and change the system parameters (dynamics), what is the probability of ending up in a desired state?*
    """
    )
    return


@app.cell
def _(pd):
    # Load Data
    data_dir = "data/abm/processed"
    state_labels_file = f"{data_dir}/abm_windows_clustered_with_state_label_20251126.csv"
    params_file = f"{data_dir}/window_info_with_sim_parameters.csv"

    # Read state labels
    df_states = pd.read_csv(state_labels_file, usecols=['sim_id', 'start_time_step', 'window_index_in_sim', 'hierarchical_label'])

    # Read parameters
    df_params = pd.read_csv(params_file)

    # Merge
    df = pd.merge(df_states, df_params, on=['sim_id', 'start_time_step', 'window_index_in_sim'])

    # Sort
    df = df.sort_values(by=['sim_id', 'start_time_step'])

    # Create next state column
    df['next_state'] = df.groupby('sim_id')['hierarchical_label'].shift(-1)

    # Ensure states are integers
    if df['hierarchical_label'].dtype == float:
        df['hierarchical_label'] = df['hierarchical_label'].fillna(-1).astype(int)
    if df['next_state'].dtype == float:
        df['next_state'] = df['next_state'].fillna(-1).astype(int)

    # Get all unique states and max time
    unique_states = sorted([s for s in df['hierarchical_label'].unique() if s != -1])
    n_states = len(unique_states)
    max_time_idx = df['window_index_in_sim'].max()

    # Helper for parameter names
    param_cols = [c for c in df_params.columns if c.startswith('cell_definitions') or c.startswith('param_')]
    def clean_param_name(name):
        return name.split('.')[-1]
    param_options = {clean_param_name(c): c for c in param_cols}
    return (
        clean_param_name,
        df,
        max_time_idx,
        n_states,
        param_options,
        unique_states,
    )


@app.cell
def _(mo, param_options):
    mo.md("### 1. Define Intervention Dynamics")

    intervention_param_select = mo.ui.dropdown(
        options=param_options,
        value=list(param_options.keys())[0],
        label="Intervention Parameter"
    )

    intervention_range = mo.ui.range_slider(
        start=0, stop=100, step=1, value=[0, 33],
        label="Percentile Range (Intervention)"
    )

    mo.md(f"""
    Select the parameter regime that represents the **Target Dynamics** (e.g., "Good Outcome" parameters).
    The transition matrices calculated from these simulations will be used as the "new rules" for the intercepted trajectories.

    {intervention_param_select}
    {intervention_range}
    """)
    return intervention_param_select, intervention_range


@app.cell
def _(
    clean_param_name,
    df,
    intervention_param_select,
    intervention_range,
    max_time_idx,
    mo,
    n_states,
    np,
    unique_states,
):
    # Calculate Intervention Matrices

    i_col = intervention_param_select.value
    i_low_pct, i_high_pct = intervention_range.value
    i_low_val = df[i_col].quantile(i_low_pct / 100.0)
    i_high_val = df[i_col].quantile(i_high_pct / 100.0)

    df_intervention = df[
        (df[i_col] >= i_low_val) & (df[i_col] <= i_high_val)
    ].dropna(subset=['next_state'])

    i_sim_count = df_intervention['sim_id'].nunique()

    # Pre-calculate transition matrices for all t
    # Shape: (T, S, S)
    intervention_matrices = np.zeros((max_time_idx + 1, n_states, n_states))

    state_to_idx = {s: i for i, s in enumerate(unique_states)}

    for _t in range(max_time_idx + 1):
        _sub_df = df_intervention[df_intervention['window_index_in_sim'] == _t]
        _counts = np.zeros((n_states, n_states))
        for _, _row in _sub_df.iterrows():
            if _row['next_state'] != -1:
                _s_from = _row['hierarchical_label']
                _s_to = _row['next_state']
                if _s_from in state_to_idx and _s_to in state_to_idx:
                    _counts[state_to_idx[_s_from], state_to_idx[_s_to]] += 1

        # Normalize
        _row_sums = _counts.sum(axis=1, keepdims=True)
        # Handle zeros safely
        _probs = np.divide(_counts, _row_sums, out=np.zeros_like(_counts), where=_row_sums!=0)

        # If a row is all zeros (state not observed at this t), make it identity or uniform? 
        # Identity is safer (stuck) than uniform (random jump). 
        # Or we can fallback to global average for that state. 
        # For this demo, let's use Identity for unobserved states to avoid artifacts.
        for _i in range(n_states):
            if _row_sums[_i] == 0:
                _probs[_i, _i] = 1.0

        intervention_matrices[_t] = _probs

    mo.md(f"**Intervention Dynamics defined using {i_sim_count} simulations.** (`{clean_param_name(i_col)}`: {i_low_val:.5f} - {i_high_val:.5f})")
    return intervention_matrices, state_to_idx


@app.cell
def _(mo, unique_states):
    mo.md("### 2. Select Target Simulations to Analyze")

    target_state_multiselect = mo.ui.multiselect(
        options=[str(s) for s in unique_states],
        label="Select Undesirable Final States (Simulations ending here will be analyzed)"
    )

    desired_state_multiselect = mo.ui.multiselect(
        options=[str(s) for s in unique_states],
        label="Select Desired Final States (Goal of intervention)"
    )

    mo.md(f"""
    We will find all simulations that naturally ended in the **Undesirable States**. 
    Then, we will simulate "what if" we applied the Intervention Dynamics at different time points.
    We measure success by the probability of landing in the **Desired States**.

    {target_state_multiselect}
    {desired_state_multiselect}
    """)
    return desired_state_multiselect, target_state_multiselect


@app.cell
def _(desired_state_multiselect, df, mo, target_state_multiselect):
    # Identify Target Simulations

    target_end_states = [int(s) for s in target_state_multiselect.value]
    desired_end_states = [int(s) for s in desired_state_multiselect.value]

    # Find sim_ids that end in target_end_states
    # Get the last state of every simulation
    last_states = df.sort_values('window_index_in_sim').groupby('sim_id').last()['hierarchical_label']

    target_sim_ids = last_states[last_states.isin(target_end_states)].index.tolist()

    # Filter the main dataframe to just these simulations
    df_target_sims = df[df['sim_id'].isin(target_sim_ids)].copy()

    mo.md(f"Found **{len(target_sim_ids)}** simulations ending in states {target_end_states}.")
    return desired_end_states, df_target_sims


@app.cell
def _(
    desired_end_states,
    df_target_sims,
    intervention_matrices,
    max_time_idx,
    mo,
    n_states,
    np,
    state_to_idx,
):
    # 3. Perform Virtual Intervention Analysis
    # We want to calculate: For each intervention time t_int, what is the prob of ending in Desired Set?

    # We can do this efficiently. 
    # For a simulation S, at t_int, it is in state s_observed.
    # We initialize a probability vector v with 1.0 at s_observed.
    # We propagate v forward from t_int to T using intervention_matrices.
    # We sum probabilities of desired states.

    # To save time, we can pre-compute the "Probability of reaching Desired Set from State S at time T_int"
    # Let R[t, s] = Probability of ending in Desired Set given we are in state s at time t, following Intervention Dynamics.
    # We can compute R backwards from T to 0.

    # Initialize R matrix: (T+1, N_states)
    R = np.zeros((max_time_idx + 1, n_states))

    # At T_final (technically max_time_idx), R[T, s] = 1 if s in Desired else 0
    # Actually, the matrices are transition FROM t TO t+1. 
    # The last transition is at max_time_idx - 1 going to max_time_idx.

    desired_indices = [state_to_idx[s] for s in desired_end_states]

    # Boundary condition at the very end
    R[max_time_idx, :] = 0.0
    for d_idx in desired_indices:
        R[max_time_idx, d_idx] = 1.0

    # Backward pass
    # R[t] = Transition[t] * R[t+1]
    # Matrix mult: (N, N) * (N, 1) -> (N, 1)
    for _t in range(max_time_idx - 1, -1, -1):
        # Trans[t] is P(s_t -> s_{t+1})
        # Value(s_t) = sum_s_{t+1} P(s_t -> s_{t+1}) * Value(s_{t+1})
        R[_t] = intervention_matrices[_t] @ R[_t+1]


    # Now map this back to our observed trajectories
    # For each simulation, for each time t, lookup R[t, observed_state_at_t]

    results = []

    # Group by time to average across simulations
    # We want average "Success Probability" at time t

    avg_success_prob = np.zeros(max_time_idx + 1)
    counts_at_t = np.zeros(max_time_idx + 1)

    # Iterate through observed data
    for _, _row in df_target_sims.iterrows():
        _t = _row['window_index_in_sim']
        _s = _row['hierarchical_label']
        if _s in state_to_idx:
            _s_idx = state_to_idx[_s]
            _prob = R[_t, _s_idx]

            avg_success_prob[_t] += _prob
            counts_at_t[_t] += 1

    # Normalize
    avg_success_prob = np.divide(avg_success_prob, counts_at_t, out=np.zeros_like(avg_success_prob), where=counts_at_t!=0)

    mo.md("Analysis Complete. Probability map computed via Backward Induction.")
    return (avg_success_prob,)


@app.cell
def _(
    avg_success_prob,
    clean_param_name,
    intervention_param_select,
    intervention_range,
    max_time_idx,
    mo,
    np,
    os,
    pickle,
    plt,
):
    # Visualization
    t_axis = np.arange(max_time_idx + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_axis, avg_success_prob, linewidth=3, color='dodgerblue')
    ax.set_xlabel("Time of Intervention (Window Index)")
    ax.set_ylabel("Prob. of Reaching Desired State")
    ax.set_title("Intervention Efficacy vs. Time (Markov Model)")
    ax.set_ylim(0, 1.05)

    # Add a smooth trend line or threshold
    ax.axhline(0.5, linestyle='--', color='gray', alpha=0.5, label="50% Probability")

    ax.legend()

    # Save Figure
    output_dir = "output/figures/abm/markov_interception"
    os.makedirs(output_dir, exist_ok=True)

    # Construct filename
    p_name = clean_param_name(intervention_param_select.value)
    p_min, p_max = intervention_range.value
    fname = f"intercept_trajectory_{p_name}_{p_min}_{p_max}.png"
    fig.savefig(f"{output_dir}/{fname}", dpi=300, bbox_inches='tight')

    # saving to svg with smaller file size -> raster the data and keep text vector
    # Rasterize all non-text elements automatically
    for artist in ax.get_children():
        if not isinstance(artist, plt.Text):
            try:
                artist.set_rasterized(True)
            except AttributeError:
                pass  # Some artists don't support rasterization

    fname_svg = f"intercept_trajectory_{p_name}_{p_min}_{p_max}.svg"
    fig.savefig(f"{output_dir}/{fname_svg}", bbox_inches='tight')

    # save the data out for plotting in the full manuscript figure
    with open(
            os.path.join(
                'output',
                'figures',
                'paper-figures',
                'figure-5-panels',
                'data',
                'fig_5g_data.pkl'
            ),
            'wb'
        ) as f:
            panel_data = {
                't_axis': t_axis, 
                'avg_success_prob': avg_success_prob
            }
            pickle.dump(panel_data, file=f)
            print(f'Panel data saved to {f}')

    # Display results
    mo.md(f"""


    ### Trajectory Intercept Results





    The plot below shows the average probability that the selected "Bad Outcome" simulations would have ended in a "Desired State" **IF** their parameters were switched to the Intervention Set at time $t$.





    *   **High Probability:** Intervention at this time is effective.


    *   **Drop-off:** The "Point of No Return" is where this curve sharply declines.





    {mo.as_html(fig)}





    *Figure saved to `{output_dir}/{fname}`*


    """)
    return (output_dir,)


@app.cell
def _(max_time_idx, mo):
    mo.md("### 4. Detailed Outcome Analysis: Before vs. After Intervention")

    intervention_time_slider = mo.ui.slider(
        start=0, stop=max_time_idx, step=1, value=int(max_time_idx/2),
        label="Simulate Intervention at Time Step:"
    )

    mo.md(f"""
    Select a time point to simulate the intervention. 
    The chart below will compare the **Actual Final States** (where the simulations really ended) vs. the **Predicted Final States** (where they would end if parameters were switched at this time).

    {intervention_time_slider}
    """)
    return (intervention_time_slider,)


@app.cell
def _(
    clean_param_name,
    df_target_sims,
    intervention_matrices,
    intervention_param_select,
    intervention_range,
    intervention_time_slider,
    max_time_idx,
    mo,
    n_states,
    np,
    os,
    output_dir,
    pd,
    pickle,
    plt,
    sns,
    state_to_idx,
    unique_states,
):
    # Calculate Expected Final State Counts

    t_int = intervention_time_slider.value

    # 1. Get distribution of states at t_int for the target simulations
    # We only care about simulations that have data at t_int
    sims_at_t = df_target_sims[df_target_sims['window_index_in_sim'] == t_int]

    # Initial state vector (counts)
    current_counts = np.zeros(n_states)
    for _, _row in sims_at_t.iterrows():
        _s = _row['hierarchical_label']
        if _s in state_to_idx:
            current_counts[state_to_idx[_s]] += 1

    # 2. Propagate forward using Intervention Matrices
    # V_next = V_curr * M
    propagated_counts = current_counts.copy()

    for _t in range(t_int, max_time_idx):
        propagated_counts = propagated_counts @ intervention_matrices[_t]

    # 3. Get Actual Final States for these specific simulations
    # (To ensure fair comparison, we only look at sims that existed at t_int)
    valid_sim_ids = sims_at_t['sim_id'].unique()

    # Get the last record for each of these sims
    actual_outcomes = df_target_sims[df_target_sims['sim_id'].isin(valid_sim_ids)].sort_values('window_index_in_sim').groupby('sim_id').last()

    actual_counts = np.zeros(n_states)
    for _s in actual_outcomes['hierarchical_label']:
        if _s in state_to_idx:
            actual_counts[state_to_idx[_s]] += 1

    # 4. Prepare Plotting Data
    plot_data = []

    # Normalize to percentages for easier reading? Or keep counts? 
    # Counts are better to show magnitude of population.

    for _i, _s in enumerate(unique_states):
        # Actual
        plot_data.append({
            'State': str(_s),
            'Count': actual_counts[_i],
            'Scenario': 'Actual Outcome (No Intervention)'
        })
        # Predicted
        plot_data.append({
            'State': str(_s),
            'Count': propagated_counts[_i],
            'Scenario': f'Predicted w/ Intervention at t={t_int}'
        })

    df_plot = pd.DataFrame(plot_data)

    # Plot
    fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_plot, x='State', y='Count', hue='Scenario', ax=ax_bar, palette=['#e74c3c', '#2ecc71'])

    ax_bar.set_title(f"Projected Outcomes: Intervention at t={t_int}")
    ax_bar.set_ylabel("Number of Simulations (Expected)")
    ax_bar.set_xlabel("Final System State")
    plt.tight_layout()

    # Save
    p_name_bar = clean_param_name(intervention_param_select.value)
    p_min_bar, p_max_bar = intervention_range.value
    fname_bar = f"outcome_comparison_{p_name_bar}_{p_min_bar}_{p_max_bar}_t{t_int}.png"
    fig_bar.savefig(f"{output_dir}/{fname_bar}", dpi=300, bbox_inches='tight')

    fname_bar_svg = f"outcome_comparison_{p_name_bar}_{p_min_bar}_{p_max_bar}_t{t_int}.svg"
    fig_bar.savefig(f"{output_dir}/{fname_bar_svg}", bbox_inches='tight')

    # save the data out for plotting in the full manuscript figure
    with open(
            os.path.join(
                'output',
                'figures',
                'paper-figures',
                'figure-5-panels',
                'data',
                'fig_5g_data.pkl'
            ),
            'wb'
        ) as f2:
            panel_data2 = {
                'df_plot': df_plot, 
                'palette': ['#e74c3c', '#2ecc71']
            }
            pickle.dump(panel_data2, file=f2)
            print(f'Panel data saved to {f2}')

    mo.md(f"""
    {mo.as_html(fig_bar)}

    *Figure saved to `{output_dir}/{fname_bar}`*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Assessment of the Markovian Approach

    Using a Markov State Model (MSM) as a surrogate for an Agent-Based Model (ABM) to test interventions has clear pros and cons:

    **1. Strengths (Why do this?):**
    *   **Speed:** ABMs are computationally expensive (hours/days). This MSM approach runs in milliseconds. It allows for exhaustive exploration of "what-if" scenarios across all time points and parameter combinations.
    *   **Landscape Mapping:** It effectively maps the "basin of attraction" of the desired state. If the probability is high, it means the system configuration at that time is *capable* of evolving to the target under the right laws.

    **2. Weaknesses (Risks):**
    *   **The Markov Assumption (Memorylessness):** This is the critical flaw. ABMs are inherently non-Markovian; the future depends on the history (e.g., cell age, mutation accumulation). If the "State" (spatial summary stats) does not fully capture these hidden variables, the MSM will oversimplify. Two simulations might have the same spatial stats (same State) but different internal variables that make one recoverable and the other not.
    *   **The "Instant Switch" Fallacy:** Re-parameterizing an ABM (e.g., changing drug concentration) changes cell behavior, but the system's *response* might have lag (hysteresis). The MSM assumes that the moment you switch parameters, the system immediately follows the transition probabilities of the new parameter set. It ignores the transition dynamics *of the switch itself*.

    **3. Verdict:**
    This is a powerful **hypothesis generation** tool. It identifies the *theoretical* windows of opportunity. However, it **must** be validated.

    **Recommended Validation:**
    Pick 3-5 distinct points from the curve above (e.g., one high success, one at the drop-off, one low).
    Actually run the ABM starting from those saved snapshots with the new parameters ("Micro-scale initialization").
    Compare the actual ABM outcome fraction with the MSM predicted probability.
    """
    )
    return


if __name__ == "__main__":
    app.run()
