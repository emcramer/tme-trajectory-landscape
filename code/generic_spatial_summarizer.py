"""
generic_spatial_summarizer.py

Purpose:
  Replicates the spatial summarization process from 'code/04_mibi_roi_spatial_analysis.marimo.py'
  using the custom 'spatialtissuepy' package.

  This module generates a spatial summary DataFrame consistent with 'spatialsummary'
  naming conventions, including graph metrics, global Moran's I, graph density,
  and a full interaction matrix (all permutations).

Naming Conventions:
  - Cell Proportions: '{CellType}'
  - Graph Metrics: '{CellType}_group_{Metric}'
  - Interaction Matrix: 'im_{TypeA}_{TypeB}' (Includes all pairs A->B and B->A)
  - Moran's I: 'moranI' (Global average across markers)
  - Graph Density: 'graph_density'

Usage:
  from code.generic_spatial_summarizer import summarize_anndata_list
  
  # summaries_df = summarize_anndata_list(adatas, cluster_key='cellTypeFunctionalLeiden')
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm

# --- Setup Path for spatialtissuepy ---
ST_PATH = Path.home() / "spatialtissuepy"
if str(ST_PATH) not in sys.path:
    sys.path.append(str(ST_PATH))

try:
    import spatialtissuepy
    from spatialtissuepy.core.spatial_data import SpatialTissueData
    from spatialtissuepy.summary import StatisticsPanel, SpatialSummary
    from spatialtissuepy.network.metrics import (
        mean_degree_centrality_by_type_metric,
        mean_closeness_centrality_by_type_metric,
        mean_clustering_by_type_metric,
        graph_density
    )
    from spatialtissuepy.summary.neighborhood import colocalization_score
    from spatialtissuepy.statistics.metrics import _morans_i_metric
    from spatialtissuepy.summary.population import cell_proportions
except ImportError as e:
    raise ImportError(f"Could not import 'spatialtissuepy'. Ensure it is located at {ST_PATH}. Error: {e}")

# Suppress warnings
warnings.filterwarnings('ignore')


def anndata_to_spatial_data(adata, cluster_key='cellTypeFunctionalLeiden'):
    """
    Converts an AnnData object to a SpatialTissueData object.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object with .obsm['spatial'] and .obs[cluster_key].
    cluster_key : str
        Column in .obs containing cell type labels.
        
    Returns
    -------
    SpatialTissueData
    """
    # Extract coordinates
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    else:
        raise ValueError("AnnData object missing .obsm['spatial']")
        
    # Extract cell types
    if cluster_key in adata.obs:
        cell_types = adata.obs[cluster_key].astype(str).values
    else:
        warnings.warn(f"Cluster key '{cluster_key}' not found in .obs. Using 'unknown'.")
        cell_types = np.array(['unknown'] * adata.n_obs)
        
    # Extract SampleID if present
    sample_ids = None
    if 'SampleID' in adata.obs:
        sample_ids = adata.obs['SampleID'].values
    elif 'sample_id' in adata.obs:
        sample_ids = adata.obs['sample_id'].values
    
    # Extract Markers (for Moran's I)
    markers = None
    if adata.n_vars > 0:
        try:
            if isinstance(adata.X, pd.DataFrame):
                markers = adata.X
            else:
                data_vals = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                markers = pd.DataFrame(data_vals, columns=adata.var_names, index=adata.obs_names)
        except Exception as e:
            warnings.warn(f"Could not extract markers from adata.X: {e}")
        
    # Create object
    return SpatialTissueData(
        coordinates=coords,
        cell_types=cell_types,
        sample_ids=sample_ids,
        markers=markers,
        coordinate_units="micrometers"
    )


def create_replication_panel():
    """
    Creates a StatisticsPanel that replicates 'spatialsummary' metrics and naming.
    """
    panel = StatisticsPanel(
        name='replication_panel',
        description='Replicates spatialsummary metrics and naming conventions'
    )
    
    # 1. Cell Proportions (Names match {CellType})
    def wrapped_props(data, **kwargs):
        res = cell_proportions(data)
        # Rename: prop_Tumor -> Tumor
        return {k.replace("prop_", ""): v for k, v in res.items()}

    panel.add_custom_function('proportions', wrapped_props, validate=False)
    
    # 2. Degree Centrality (Names match {CellType}_group_degree_centrality)
    def wrapped_degree(data, **kwargs):
        res = mean_degree_centrality_by_type_metric(data, **kwargs)
        new_res = {}
        for k, v in res.items():
            ctype = k.replace("degree_centrality_", "")
            new_res[f"{ctype}_group_degree_centrality"] = v
        return new_res

    panel.add_custom_function(
        'degree_centrality', wrapped_degree, method='delaunay', validate=False
    )
    
    # 3. Closeness Centrality (Names match {CellType}_group_closeness_centrality)
    def wrapped_closeness(data, **kwargs):
        res = mean_closeness_centrality_by_type_metric(data, **kwargs)
        new_res = {}
        for k, v in res.items():
            ctype = k.replace("closeness_centrality_", "")
            new_res[f"{ctype}_group_closeness_centrality"] = v
        return new_res

    panel.add_custom_function(
        'closeness_centrality', wrapped_closeness, method='delaunay', validate=False
    )
    
    # 4. Clustering Coefficient (Names match {CellType}_group_clustering_coefficient)
    def wrapped_clustering(data, **kwargs):
        res = mean_clustering_by_type_metric(data, **kwargs)
        new_res = {}
        for k, v in res.items():
            ctype = k.replace("clustering_", "")
            new_res[f"{ctype}_group_clustering_coefficient"] = v
        return new_res

    panel.add_custom_function(
        'clustering_coefficient', wrapped_clustering, method='delaunay', validate=False
    )
    
    # 5. Graph Density (calculated via networkx on Delaunay graph)
    def calculate_graph_density(data, **kwargs):
        import networkx as nx
        from spatialtissuepy.network.graph_construction import build_graph
        G = build_graph(data.coordinates, method='delaunay')
        return {'graph_density': nx.density(G)}

    panel.add_custom_function(
        'graph_density', calculate_graph_density, validate=False
    )
    
    # 6. Interaction Matrix (All Permutations: im_{TypeA}_{TypeB})
    def wrapped_interactions(data, **kwargs):
        # Calculate for all permutations (A->B and B->A) to ensure all keys exist
        unique_types = sorted(list(data.cell_types_unique))
        result = {}
        
        for type_a in unique_types:
            for type_b in unique_types:
                # Calculate colocalization score
                res = colocalization_score(data, type_a, type_b, radius=kwargs.get('radius', 50.0))
                
                # Rename coloc_ -> im_
                # Result key is f'coloc_{type_a}_{type_b}'
                for k, v in res.items():
                    new_key = k.replace("coloc_", "im_")
                    result[new_key] = v
                    
        return result

    panel.add_custom_function(
        'interaction_matrix', wrapped_interactions, radius=50.0, validate=False
    )
    
    # 7. Moran's I (Global average across markers, Name: moranI)
    def wrapped_moran(data, **kwargs):
        if data.markers is None or len(data.marker_names) == 0:
            return {'moranI': 0.0}
        
        scores = []
        for marker in data.marker_names:
            try:
                res = _morans_i_metric(data, marker=marker, radius=kwargs.get('radius', 50.0))
                val = res.get(f'morans_i_{marker}', np.nan)
                if not np.isnan(val):
                    scores.append(val)
            except:
                continue
        return {'moranI': np.mean(scores) if scores else 0.0}

    panel.add_custom_function(
        'global_moran_i', wrapped_moran, radius=50.0, validate=False
    )
    
    return panel


def summarize_anndata_list(adatas, cluster_key='cellTypeFunctionalLeiden', show_progress=True):
    """
    Summarizes a list of AnnData objects using spatialtissuepy.
    
    Parameters
    ----------
    adatas : list of AnnData
        List of loaded AnnData objects.
    cluster_key : str
        Column name for cell type labels.
    show_progress : bool
        Whether to show tqdm progress bar.
        
    Returns
    -------
    pd.DataFrame
        Concatenated summary statistics.
    """
    panel = create_replication_panel()
    summaries = []
    
    iterator = tqdm(adatas, desc="Summarizing samples") if show_progress else adatas
    
    for i, adata in enumerate(iterator):
        try:
            # Convert using provided cluster_key
            st_data = anndata_to_spatial_data(adata, cluster_key=cluster_key)
            
            # Compute
            summary = SpatialSummary(st_data, panel)
            
            # Convert to DataFrame row
            series = summary.to_series()
            df_row = series.to_frame().T
            
            summaries.append(df_row)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
            
    if not summaries:
        return pd.DataFrame()
        
    # Concatenate all samples
    final_df = pd.concat(summaries, ignore_index=True)
    
    # Fill NaNs with 0
    final_df = final_df.fillna(0)
    
    return final_df


if __name__ == "__main__":
    print("This module provides 'summarize_anndata_list' for spatial analysis.")
    print(f"spatialtissuepy location: {ST_PATH}")
