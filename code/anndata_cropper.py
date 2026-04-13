import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import warnings

def _validate_anndata(adata: ad.AnnData):
    """
    Validates if the anndata object contains 'spatial' key in .obsm.

    Parameters
    ----------
    adata
        The anndata object to validate.

    Raises
    ------
    ValueError
        If 'spatial' key is not found in adata.obsm.
    """
    if 'spatial' not in adata.obsm_keys():
        raise ValueError(
            "AnnData object does not contain 'spatial' coordinates in .obsm. "
            "Please ensure spatial coordinates are loaded, e.g., in adata.obsm['spatial']."
        )

def _get_spatial_coordinates(adata: ad.AnnData) -> np.ndarray:
    """
    Retrieves the spatial coordinates from the anndata object.

    Parameters
    ----------
    adata
        The anndata object.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (n_obs, 2) containing the spatial (x, y) coordinates.
    """
    return adata.obsm['spatial']

def _generate_random_crop_coordinates(
    spatial_coords: np.ndarray,
    crop_height: float,
    crop_width: float,
    random_seed: int = None
) -> tuple[float, float]:
    """
    Generates random top-left corner coordinates for cropping.

    Parameters
    ----------
    spatial_coords
        A NumPy array of shape (n_obs, 2) containing the spatial (x, y) coordinates.
    crop_height
        The desired height of the crop.
    crop_width
        The desired width of the crop.
    random_seed
        Seed for reproducibility. If None, a random seed will be used.

    Returns
    -------
    tuple[float, float]
        A tuple (x_start, y_start) representing the top-left corner of the crop.

    Raises
    ------
    ValueError
        If crop_height or crop_width are larger than the overall spatial dimensions.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)

    data_width = max_x - min_x
    data_height = max_y - min_y

    if crop_width > data_width or crop_height > data_height:
        raise ValueError(
            f"Crop dimensions (width: {crop_width}, height: {crop_height}) "
            f"cannot be larger than the data dimensions (width: {data_width}, height: {data_height})."
        )

    # Determine the possible range for the top-left corner
    # The crop's maximum x-coordinate (x_start + crop_width) must be <= max_x
    # The crop's maximum y-coordinate (y_start + crop_height) must be <= max_y
    possible_x_start_max = max_x - crop_width
    possible_y_start_max = max_y - crop_height

    # Ensure that min_x and min_y are respected if the data doesn't start at (0,0)
    x_start = np.random.uniform(min_x, possible_x_start_max)
    y_start = np.random.uniform(min_y, possible_y_start_max)

    return x_start, y_start

def _crop_anndata(
    adata: ad.AnnData,
    x_start: float,
    y_start: float,
    crop_height: float,
    crop_width: float
) -> ad.AnnData:
    """
    Crops the anndata object based on the provided bounding box coordinates.

    Parameters
    ----------
    adata
        The original anndata object.
    x_start
        The x-coordinate of the top-left corner of the crop.
    y_start
        The y-coordinate of the top-left corner of the crop.
    crop_height
        The height of the crop.
    crop_width
        The width of the crop.

    Returns
    -------
    ad.AnnData
        A new AnnData object containing only the observations within the crop.
    """
    spatial_coords = adata.obsm['spatial']

    # Define the bounding box for the crop
    x_end = x_start + crop_width
    y_end = y_start + crop_height

    # Identify observations within the bounding box
    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    within_x = (x_coords >= x_start) & (x_coords < x_end)
    within_y = (y_coords >= y_start) & (y_coords < y_end)
    
    cells_to_keep = within_x & within_y
    
    # Subset the anndata object
    cropped_adata = adata[cells_to_keep, :].copy() # .copy() ensures a new object, not a view
    
    # Store the crop coordinates in .uns
    cropped_adata.uns['crop_coordinates'] = {
        'x_start': x_start,
        'y_start': y_start,
        'crop_height': crop_height,
        'crop_width': crop_width
    }

    return cropped_adata

def random_crop_anndata(
    adata: ad.AnnData,
    crop_height: float,
    crop_width: float,
    random_seed: int = None
) -> ad.AnnData:
    """
    Produces a random crop from an anndata object based on spatial coordinates.

    The cropped anndata object will include the x and y coordinates of the
    top-left corner of the crop in the 'uns' slot as 'crop_coordinates'.

    Parameters
    ----------
    adata
        The original anndata object containing spatial coordinates in adata.obsm['spatial'].
        The 'spatial' matrix is expected to have shape (n_obs, 2), where the columns
        represent x and y coordinates respectively.
    crop_height
        The desired height of the random crop.
    crop_width
        The desired width of the random crop.
    random_seed
        An integer for reproducibility. If None, the cropping will be truly random.

    Returns
    -------
    ad.AnnData
        A new AnnData object representing the random crop.

    Raises
    ------
    ValueError
        If 'spatial' key is not found in adata.obsm.
        If crop_height or crop_width are larger than the overall spatial dimensions.

    Example
    -------
    >>> import anndata as ad
    >>> import numpy as np
    >>> # Create a dummy anndata object with spatial coordinates
    >>> n_cells = 1000
    >>> n_genes = 100
    >>> np.random.seed(42)
    >>> X = np.random.rand(n_cells, n_genes)
    >>> spatial_coords = np.random.rand(n_cells, 2) * 100 # Coordinates between 0 and 100
    >>> adata = ad.AnnData(X)
    >>> adata.obsm['spatial'] = spatial_coords

    >>> # Perform a random crop
    >>> cropped_adata = random_crop_anndata(adata, crop_height=20, crop_width=30, random_seed=0)
    >>> print(cropped_adata)
    >>> print(cropped_adata.uns['crop_coordinates'])
    """
    _validate_anndata(adata)
    spatial_coords = _get_spatial_coordinates(adata)

    x_start, y_start = _generate_random_crop_coordinates(
        spatial_coords, crop_height, crop_width, random_seed
    )

    cropped_adata = _crop_anndata(adata, x_start, y_start, crop_height, crop_width)

    return cropped_adata

def generate_multiple_random_crops(
    adata: ad.AnnData,
    crop_height: float,
    crop_width: float,
    num_crops: int = 5,
    base_random_seed: int = None
) -> list[ad.AnnData]:
    """
    Generates multiple random crops from an anndata object.

    Parameters
    ----------
    adata
        The original anndata object containing spatial coordinates in adata.obsm['spatial'].
    crop_height
        The desired height of each random crop.
    crop_width
        The desired width of each random crop.
    num_crops
        The number of random crops to generate. Defaults to 5.
    base_random_seed
        An integer seed for reproducibility of the set of crops. If provided,
        each individual crop will use a seed derived from this base seed
        (base_random_seed + i for the i-th crop). If None, each crop will be
        generated with a truly random seed.

    Returns
    -------
    list[ad.AnnData]
        A list of AnnData objects, where each object is a random crop.

    Raises
    ------
    ValueError
        If 'spatial' key is not found in adata.obsm.
        If crop_height or crop_width are larger than the overall spatial dimensions.

    Example
    -------
    >>> import anndata as ad
    >>> import numpy as np
    >>> # Create a dummy anndata object
    >>> n_cells = 1000
    >>> n_genes = 100
    >>> np.random.seed(42)
    >>> X = np.random.rand(n_cells, n_genes)
    >>> spatial_coords = np.random.rand(n_cells, 2) * 100
    >>> adata = ad.AnnData(X)
    >>> adata.obsm['spatial'] = spatial_coords

    >>> # Generate 3 random crops
    >>> list_of_crops = generate_multiple_random_crops(adata, crop_height=20, crop_width=30, num_crops=3, base_random_seed=0)
    >>> print(f"Generated {len(list_of_crops)} crops.")
    >>> for i, crop_adata in enumerate(list_of_crops):
    >>>     print(f"Crop {i+1}: {crop_adata}")
    >>>     print(f"  Crop coordinates: {crop_adata.uns['crop_coordinates']}")
    """
    _validate_anndata(adata) # Validate once for the original adata
    
    cropped_adatas = []
    for i in range(num_crops):
        current_seed = base_random_seed + i if base_random_seed is not None else None
        try:
            cropped_adata = random_crop_anndata(adata, crop_height, crop_width, random_seed=current_seed)
            if cropped_adata.n_obs == 0:
                warnings.warn(
                    f"Crop {i+1} (seed={current_seed}) resulted in an empty AnnData object. "
                    "Consider adjusting crop dimensions or the density of spatial coordinates."
                )
            cropped_adatas.append(cropped_adata)
        except ValueError as e:
            raise ValueError(f"Failed to generate crop {i+1} due to: {e}")

    return cropped_adatas

def generate_poisson_crops(
    adata: ad.AnnData,
    crop_height: float,
    crop_width: float,
    num_crops: int = 5,
    min_distance: float = None,
    random_seed: int = None,
    max_attempts: int = 1000
) -> list[ad.AnnData]:
    """
    Generates crops using a Poisson disc-like sampling strategy to ensure spatial separation.
    
    Parameters
    ----------
    adata : ad.AnnData
        The original anndata object.
    crop_height : float
        Height of the crops.
    crop_width : float
        Width of the crops.
    num_crops : int
        Number of crops to generate.
    min_distance : float, optional
        Minimum distance between crop centers. Defaults to max(crop_height, crop_width) * 0.8.
    random_seed : int, optional
        Random seed.
    max_attempts : int
        Maximum attempts to place a crop before giving up.
        
    Returns
    -------
    list[ad.AnnData]
    """
    _validate_anndata(adata)
    if random_seed is not None:
        np.random.seed(random_seed)
        
    spatial_coords = _get_spatial_coordinates(adata)
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    
    # Define valid range for top-left corners
    x_range = max_x - min_x - crop_width
    y_range = max_y - min_y - crop_height
    
    if x_range <= 0 or y_range <= 0:
         warnings.warn("Data dimensions smaller than crop size. Returning single center crop.")
         return [_crop_anndata(adata, min_x, min_y, crop_width, crop_height)]

    if min_distance is None:
        min_distance = max(crop_height, crop_width) * 0.8
        
    centers = []
    crops = []
    
    for _ in range(num_crops):
        placed = False
        for attempt in range(max_attempts):
            x_start = min_x + np.random.uniform(0, x_range)
            y_start = min_y + np.random.uniform(0, y_range)
            
            center_x = x_start + crop_width / 2
            center_y = y_start + crop_height / 2
            
            # Check distance to existing centers
            if not centers:
                accept = True
            else:
                dists = np.sqrt(np.sum((np.array(centers) - np.array([center_x, center_y]))**2, axis=1))
                if np.all(dists >= min_distance):
                    accept = True
                else:
                    accept = False
            
            if accept:
                centers.append([center_x, center_y])
                crops.append(_crop_anndata(adata, x_start, y_start, crop_height, crop_width))
                placed = True
                break
        
        if not placed:
            warnings.warn(f"Could only place {len(crops)} crops out of {num_crops} with min_distance={min_distance}")
            break
            
    return crops

class _QuadNode:
    def __init__(self, x, y, w, h, depth=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.depth = depth
        self.children = []
        
    def subdivide(self):
        hw = self.w / 2
        hh = self.h / 2
        self.children = [
            _QuadNode(self.x, self.y, hw, hh, self.depth + 1),
            _QuadNode(self.x + hw, self.y, hw, hh, self.depth + 1),
            _QuadNode(self.x, self.y + hh, hw, hh, self.depth + 1),
            _QuadNode(self.x + hw, self.y + hh, hw, hh, self.depth + 1)
        ]

def generate_quadtree_crops(
    adata: ad.AnnData,
    min_size: float = 200,
    min_cells: int = 50,
    max_depth: int = 4
) -> list[ad.AnnData]:
    """
    Decomposes the FOV using a quadtree strategy. Returns leaf nodes that satisfy criteria.
    
    Parameters
    ----------
    adata : ad.AnnData
    min_size : float
        Minimum width/height of a quadrant to allow further subdivision.
    min_cells : int
        Minimum number of cells to allow further subdivision (or to keep a leaf).
    max_depth : int
        Maximum recursion depth.
        
    Returns
    -------
    list[ad.AnnData]
    """
    _validate_anndata(adata)
    spatial_coords = _get_spatial_coordinates(adata)
    min_x, min_y = np.min(spatial_coords, axis=0)
    max_x, max_y = np.max(spatial_coords, axis=0)
    
    w = max_x - min_x
    h = max_y - min_y
    
    # Pad slightly to ensure boundary inclusion
    root = _QuadNode(min_x, min_y, w + 0.1, h + 0.1)
    nodes_to_process = [root]
    leaves = []
    
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        
        # Check cell count in this node
        x_end = node.x + node.w
        y_end = node.y + node.h
        
        mask = (spatial_coords[:, 0] >= node.x) & (spatial_coords[:, 0] < x_end) & \
               (spatial_coords[:, 1] >= node.y) & (spatial_coords[:, 1] < y_end)
        n_cells = np.sum(mask)
        
        can_subdivide = (node.w / 2 >= min_size) and (node.h / 2 >= min_size) and \
                        (n_cells >= min_cells * 4) and (node.depth < max_depth)
        
        if can_subdivide:
            node.subdivide()
            nodes_to_process.extend(node.children)
        else:
            # It's a leaf. If it has enough cells, keep it.
            if n_cells >= 10: # Arbitrary absolute minimum to avoid empty boxes
                 # Crop the actual data
                 # We use the _crop_anndata logic but efficiently
                 # Actually, simpler to just use _crop_anndata on the node box
                 leaves.append(_crop_anndata(adata, node.x, node.y, node.h, node.w))
    
    return leaves

###############################################################
# Visualization Functions
###############################################################


def visualize_single_crop(original_adata: ad.AnnData, cropped_adata: ad.AnnData, ax=None):
    """
    Visualizes a single cropped area on the original anndata object's spatial plot.

    Parameters
    ----------
    original_adata
        The original anndata object with spatial coordinates.
    cropped_adata
        The cropped anndata object, which must have 'crop_coordinates' in its .uns slot.
    ax
        A matplotlib axes object to plot on. If None, a new figure and axes will be created.

    Raises
    ------
    ValueError
        If 'spatial' key is not found in original_adata.obsm.
        If 'crop_coordinates' key is not found in cropped_adata.uns.
    """
    _validate_anndata(original_adata)
    if 'crop_coordinates' not in cropped_adata.uns:
        raise ValueError(
            "Cropped AnnData object does not contain 'crop_coordinates' in its .uns slot. "
            "Please ensure the cropped object was generated by 'random_crop_anndata'."
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot all cells from the original anndata object
    spatial_coords_original = original_adata.obsm['spatial']
    ax.scatter(spatial_coords_original[:, 0], spatial_coords_original[:, 1], s=5, alpha=0.5, label='Original Cells')

    # Get crop coordinates
    crop_info = cropped_adata.uns['crop_coordinates']
    x_start = crop_info['x_start']
    y_start = crop_info['y_start']
    crop_width = crop_info['crop_width']
    crop_height = crop_info['crop_height']

    # Draw the rectangle for the crop boundaries
    rect = patches.Rectangle(
        (x_start, y_start),
        crop_width,
        crop_height,
        linewidth=2,
        edgecolor='r',
        facecolor='none',
        label='Crop Boundary'
    )
    ax.add_patch(rect)

    ax.set_title('Original AnnData with Single Crop Boundary')
    ax.set_xlabel('Spatial X Coordinate')
    ax.set_ylabel('Spatial Y Coordinate')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')


def visualize_multiple_crops(original_adata: ad.AnnData, list_of_cropped_adatas: list[ad.AnnData]):
    """
    Visualizes multiple cropped areas on the original anndata object's spatial plot.
    Each crop boundary will be drawn in a different color.

    Parameters
    ----------
    original_adata
        The original anndata object with spatial coordinates.
    list_of_cropped_adatas
        A list of cropped anndata objects, each expected to have 'crop_coordinates'
        in its .uns slot.

    Raises
    ------
    ValueError
        If 'spatial' key is not found in original_adata.obsm.
        If any cropped_adata object in the list does not contain 'crop_coordinates'.
    """
    _validate_anndata(original_adata)

    if not isinstance(list_of_cropped_adatas, list) or not all(isinstance(a, ad.AnnData) for a in list_of_cropped_adatas):
        raise TypeError("list_of_cropped_adatas must be a list of AnnData objects.")

    if not list_of_cropped_adatas:
        print("No cropped AnnData objects provided to visualize.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot all cells from the original anndata object
    spatial_coords_original = original_adata.obsm['spatial']
    ax.scatter(spatial_coords_original[:, 0], spatial_coords_original[:, 1], s=5, alpha=0.3, label='Original Cells')

    # Define a colormap for different crop boundaries
    colors = plt.cm.get_cmap('hsv', len(list_of_cropped_adatas)) # [9, 10, 11]

    for i, cropped_adata in enumerate(list_of_cropped_adatas):
        if 'crop_coordinates' not in cropped_adata.uns:
            warnings.warn(
                f"Cropped AnnData object at index {i} does not contain 'crop_coordinates' in its .uns slot. "
                "Skipping this crop for visualization."
            )
            continue

        crop_info = cropped_adata.uns['crop_coordinates']
        x_start = crop_info['x_start']
        y_start = crop_info['y_start']
        crop_width = crop_info['crop_width']
        crop_height = crop_info['crop_height']
        
        # Draw the rectangle for the crop boundaries in a different color [2, 3, 4, 5, 6]
        color = colors(i)
        rect = patches.Rectangle(
            (x_start, y_start),
            crop_width,
            crop_height,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            label=f'Crop {i+1} Boundary'
        )
        ax.add_patch(rect)

        # Add crop number text inside the top-left corner
        # Adjust text position slightly for better visual placement
        text_x_pos = x_start + crop_width * 0.02 # 2% in from left
        text_y_pos = y_start + crop_height * 0.98 # 2% down from top
        ax.text(
            text_x_pos,
            text_y_pos,
            f'Crop {i+1}',
            color=color,
            fontsize=10,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='left'
        )

    ax.set_title('Original AnnData with Multiple Crop Boundaries')
    ax.set_xlabel('Spatial X Coordinate')
    ax.set_ylabel('Spatial Y Coordinate')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


###############################################################
# Main
###############################################################


if __name__ == '__main__':
    # Example Usage:
    # Create a dummy anndata object with spatial coordinates
    n_cells = 5000
    n_genes = 200
    np.random.seed(42)
    X = np.random.poisson(1, size=(n_cells, n_genes)).astype(float)
    
    # Simulate spatial coordinates for a tissue section, e.g., ranging from (0,0) to (500, 700)
    spatial_coords = np.random.rand(n_cells, 2) * np.array([500, 700])
    
    adata = ad.AnnData(X)
    adata.obsm['spatial'] = spatial_coords
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

    print("Original AnnData object:")
    print(adata)
    print("\nSpatial coordinates min/max:")
    print(f"X: {np.min(adata.obsm['spatial'][:, 0]):.2f} - {np.max(adata.obsm['spatial'][:, 0]):.2f}")
    print(f"Y: {np.min(adata.obsm['spatial'][:, 1]):.2f} - {np.max(adata.obsm['spatial'][:, 1]):.2f}")

    # Perform a random crop with a specified height and width and random seed
    crop_h = 150
    crop_w = 200
    seed = 42

    print(f"\nPerforming random crop with height={crop_h}, width={crop_w}, seed={seed}...")
    cropped_adata_1 = random_crop_anndata(adata, crop_height=crop_h, crop_width=crop_w, random_seed=seed)

    print("\nCropped AnnData object 1:")
    print(cropped_adata_1)
    print("Crop coordinates (x_start, y_start, crop_height, crop_width):")
    print(cropped_adata_1.uns['crop_coordinates'])
    if 'spatial' in cropped_adata_1.obsm:
        print("\nSpatial coordinates in cropped object 1 min/max:")
        print(f"X: {np.min(cropped_adata_1.obsm['spatial'][:, 0]):.2f} - {np.max(cropped_adata_1.obsm['spatial'][:, 0]):.2f}")
        print(f"Y: {np.min(cropped_adata_1.obsm['spatial'][:, 1]):.2f} - {np.max(cropped_adata_1.obsm['spatial'][:, 1]):.2f}")
    
    # Perform another random crop with a different seed to show different results
    print(f"\nPerforming another random crop with height={crop_h}, width={crop_w}, seed={seed + 1}...")
    cropped_adata_2 = random_crop_anndata(adata, crop_height=crop_h, crop_width=crop_w, random_seed=seed + 1)
    
    print("\nCropped AnnData object 2:")
    print(cropped_adata_2)
    print("Crop coordinates (x_start, y_start, crop_height, crop_width):")
    print(cropped_adata_2.uns['crop_coordinates'])
    if 'spatial' in cropped_adata_2.obsm:
        print("\nSpatial coordinates in cropped object 2 min/max:")
        print(f"X: {np.min(cropped_adata_2.obsm['spatial'][:, 0]):.2f} - {np.max(cropped_adata_2.obsm['spatial'][:, 0]):.2f}")
        print(f"Y: {np.min(cropped_adata_2.obsm['spatial'][:, 1]):.2f} - {np.max(cropped_adata_2.obsm['spatial'][:, 1]):.2f}")

    # Example of an invalid crop (crop dimensions larger than data dimensions)
    print("\nAttempting an invalid crop (should raise ValueError):")
    try:
        random_crop_anndata(adata, crop_height=800, crop_width=800, random_seed=0)
    except ValueError as e:
        print(f"Error caught: {e}")

    # Example of an anndata object without spatial coordinates
    print("\nAttempting crop on AnnData without spatial coordinates (should raise ValueError):")
    adata_no_spatial = ad.AnnData(np.random.rand(10, 10))
    try:
        random_crop_anndata(adata_no_spatial, crop_height=10, crop_width=10, random_seed=0)
    except ValueError as e:
        print(f"Error caught: {e}")