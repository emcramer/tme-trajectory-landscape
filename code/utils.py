import numpy as np 
import pandas as pd 
import itertools
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import matplotlib.pyplot as plt
from collections import Counter
import os

def ensure_directory(path):
    """
    Ensure that the given directory exists.
    If it does not exist, create it (including any necessary parent directories).
    """
    try:
        # Check if the path exists and is a directory
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)  # Create directory safely
            print(f"Directory created: {path}")
        else:
            print(f"Directory already exists: {path}")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")

def add_suffix_to_repeats(items):
    counts = {}
    result = []
    
    for item in items:
        if item not in counts:
            counts[item] = 0
        else:
            counts[item] += 1
        result.append(f"{item}_{counts[item]}")
    
    return result

def majority_vote(labels):
  most_common = Counter(labels).most_common(1)
  return most_common[0][0]

def consistent_labels(X, labels):
  # get the number of clusters 
  n_clusters = len(set(labels))
  
  # Calculate centroids for each cluster
  centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
  
  # Sort clusters based on a consistent criterion (e.g., x-coordinate of centroid)
  sorted_indices = np.argsort(centroids[:, 0])
  
  # Create a mapping from old labels to new, sorted labels
  label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
  
  # Apply the mapping to get consistent labels
  consistent_labels = np.array([label_mapping[label] for label in labels])
  return consistent_labels

def enhance_legend_markers(ax=None, marker_size=10, alpha=1.0):
    """
    Increase the marker size and opacity (alpha) of legend markers.

    Parameters:
    - ax: Matplotlib Axes object. If None, uses current axes.
    - marker_size: New size for the legend markers.
    - alpha: Opacity (0.0 to 1.0) for the legend markers.
    """
    if ax is None:
        ax = plt.gca()
    
    legend = ax.get_legend()
    if legend is None:
        print("No legend found on the given axes.")
        return

    for handle in legend.legend_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(marker_size)
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(alpha)

def get_today():
  """Returns a string of the current date in YYYYMMDD format."""
  return datetime.datetime.now().strftime("%Y%m%d")

def window_trajectory_data(processed_features: pd.DataFrame, feature_columns: list[str], window_size: int = 50) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Creates sliding windows of features from trajectory data and calculates the average value for each feature within each window.

    Args:
        processed_features: DataFrame containing the processed features, including 'sim_id', 'time_step', and feature columns.
        feature_columns: A list of column names representing the features to be windowed.
        window_size: The size of the sliding window.

    Returns:
        A tuple containing:
            - X_windows: A NumPy array where each row is a flattened window of features.
            - window_info_df: A DataFrame containing information about each window, including 'sim_id', 'start_time_step', 'end_time_step', 'window_index_in_sim', and the average value for each feature within the window.
    """
    # Concatenate windows of features into sequences, sliding window by 1

    # Combine X with sim_id and time_step for easier processing
    processed_features_subset = processed_features[['sim_id', 'time_step'] + feature_columns].dropna(axis=1)

    X_windows = []
    window_info = [] # Store info for each window

    # Group by simulation ID
    grouped_sims = processed_features_subset.groupby('sim_id')

    for sim_id, sim_df in grouped_sims:
        features_matrix = sim_df[feature_columns].to_numpy()
        time_steps_sim = sim_df['time_step'].to_numpy()

        # Create sliding windows
        for i in range(len(features_matrix) - window_size + 1):
            window = features_matrix[i : i + window_size, :]
            X_windows.append(window.flatten()) # Flatten the window into a single vector

            # Calculate the average for each feature within the window
            window_averages = np.mean(window, axis=0)

            # Store window information along with averages
            window_info.append({
                'sim_id': sim_id,
                'start_time_step': time_steps_sim[i],
                'end_time_step': time_steps_sim[i + window_size - 1],
                'window_index_in_sim': i, # Index of the window within its simulation
            })
            # Add average feature values to the dictionary
            for j, col_name in enumerate(feature_columns):
                window_info[-1][f'avg_{col_name}'] = window_averages[j]


    X_windows = np.array(X_windows)
    window_info_df = pd.DataFrame(window_info)

    print(f"Created {len(X_windows)} windows of size {window_size}.")
    return X_windows, window_info_df

def unnest_list(nested_list):
  return list(itertools.chain.from_iterable(nested_list))

def create_pmf(data_a, data_b, num_bins=50):
    # 1. Determine common bin edges across both datasets
    min_val = min(data_a.min(), data_b.min())
    max_val = max(data_a.max(), data_b.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # 2. Calculate the frequency (counts) for each array
    counts_a, _ = np.histogram(data_a, bins=bins, density=False)
    counts_b, _ = np.histogram(data_b, bins=bins, density=False)

    # 3. Normalize the counts to create Probability Mass Functions (PMFs)
    # Ensure they sum to 1 to represent probabilities
    pmf_a = counts_a / counts_a.sum()
    pmf_b = counts_b / counts_b.sum()

    return pmf_a, pmf_b

def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon Divergence (JSD) between two PMFs, p and q.
    """
    # 1. Calculate the average distribution M
    m = 0.5 * (p + q)

    # 2. Calculate the KL divergence of P from M and Q from M.
    # The 'entropy' function in scipy calculates KL divergence when two arrays are passed.
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)

    # 3. Apply the JS formula
    jsd = 0.5 * (kl_pm + kl_qm)
    
    # Note: Use np.sqrt(jsd) if you want the J-S Distance, which is bounded [0, 1] 
    # (if using log base 2 in the entropy calculation, which scipy does not default to).
    
    return jsd

def calculate_jsd_baseline(feature_array, num_simulations=100, num_bins=50, split_ratio=0.5):
    """
    Calculates the distribution of JSD scores by comparing random halves of a feature array.

    Args:
        feature_array (np.ndarray or pd.Series): The feature data to analyze.
        num_simulations (int): Number of times to split and calculate JSD.
        num_bins (int): The number of bins to use for the PMF.
        split_ratio (float): The proportion to split the array (e.g., 0.5 for half).

    Returns:
        dict: Mean and Standard Deviation of the baseline JSD scores.
    """
    if len(feature_array) < 20:
        raise ValueError("Feature array is too small to split meaningfully.")
        
    jsd_scores = []
    data = np.asarray(feature_array)
    n = len(data)
    
    for _ in range(num_simulations):
        # Randomly shuffle and split the data
        np.random.shuffle(data)
        split_point = int(n * split_ratio)
        
        # Array parts A1 and A2 are compared
        data_a1 = data[:split_point]
        data_a2 = data[split_point:]
        
        # 1. Create aligned PMFs for the two halves
        pmf_a1, pmf_a2 = create_pmf(data_a1, data_a2, num_bins=num_bins)
        
        # 2. Calculate JSD
        jsd = jensen_shannon_divergence(pmf_a1, pmf_a2)
        jsd_scores.append(jsd)

    # 3. Calculate statistics
    mean_jsd = np.mean(jsd_scores)
    std_jsd = np.std(jsd_scores)
    
    return {
        "mean_jsd_baseline": mean_jsd,
        "std_jsd_baseline": std_jsd,
        "simulations": num_simulations,
        "num_bins": num_bins
    }

def zscore_then_minmax_normalize(data):
    """
    Performs Z-score standardization followed by Min-Max normalization on a dataset.

    Parameters
    ----------
    data : array-like or pandas DataFrame
        The input data to be transformed.
        If a pandas DataFrame, it will preserve column names.

    Returns
    -------
    transformed_data : array-like or pandas DataFrame
        The data after applying both Z-score standardization and Min-Max normalization.
        The output type matches the input type (DataFrame if input was DataFrame).
    """

    if not isinstance(data, (pd.DataFrame, np.ndarray, list)):
        raise TypeError("Input 'data' must be a pandas DataFrame, numpy array, or list.")

    if isinstance(data, list):
        data = np.array(data)

    # 1. Z-score Standardization
    # Formula: z = (x - mean) / std_dev
    scaler_zscore = StandardScaler()
    zscored_data = scaler_zscore.fit_transform(data)

    # 2. Min-Max Normalization
    # Formula: x_norm = (x - min) / (max - min)
    # Scales data to a default range of [0, 1]
    scaler_minmax = MinMaxScaler()
    minmax_normalized_data = scaler_minmax.fit_transform(zscored_data)

    # If the input was a DataFrame, convert back to DataFrame with original column names
    if isinstance(data, pd.DataFrame):
        transformed_data_df = pd.DataFrame(
            minmax_normalized_data,
            columns=data.columns,
            index=data.index
        )
        return transformed_data_df
    else:
        return minmax_normalized_data

def sliding_window_average(data: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Calculates the average value within a sliding window over a 1D numpy array.

    Args:
        data: The input 1D numpy array.
        window_size: The size of the sliding window.
        step_size: The amount to slide the window by in each step.

    Returns:
        A 1D numpy array containing the average values for each window.
    """
    num_windows = (len(data) - window_size) // step_size + 1
    if num_windows <= 0:
        return np.array([])

    averaged_data = np.zeros(num_windows)
    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + window_size
        window = data[start_index:end_index]
        averaged_data[i] = np.mean(window)
    return averaged_data