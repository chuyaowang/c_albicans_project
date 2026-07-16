import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def remove_outliers(data: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5) -> np.ndarray:
    """
    Removes outliers from a 3D array by clipping values outside the specified percentile range,
    calculated independently for each slice.

    Args:
        data (np.ndarray): Input 3D numpy array (Z, Y, X).
        lower_percentile (float): Lower bound percentile (default 0.5).
        upper_percentile (float): Upper bound percentile (default 99.5).

    Returns:
        np.ndarray: The filtered array with outliers clipped to the boundary values.
    """
    # Calculate percentiles for each slice independently
    # Result shape: (2, Z, 1, 1)
    bounds = np.percentile(data, [lower_percentile, upper_percentile], axis=(1, 2), keepdims=True)

    # Clip values using the per-slice thresholds
    filtered_data = np.clip(data, bounds[0], bounds[1])

    return filtered_data

def plot_z_stack_variance(image_stack: np.ndarray, show_plot = False):
    """
    Calculates the normalized variance for each slice in a Z-stack, clusters them
    into focused/unfocused groups using K-Means, and plots the results.
    
    Args:
        image_stack (np.ndarray): 3D numpy array with shape (Z, Height, Width).
        show_plot (boolean): If the normalized variance plot should be shown
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices of focused and unfocused slices.
    """
    if image_stack.ndim != 3:
        raise ValueError(f"Expected a 3D array (Z, Y, X), but got shape {image_stack.shape}")

    z_depth = image_stack.shape[0]
    slice_indices = np.arange(z_depth)
    normalized_variances = []

    print(f"Processing {z_depth} slices...")

    for i in slice_indices:
        current_slice = image_stack[i]
        slice_mean = np.mean(current_slice)
        slice_var = np.var(current_slice)
        
        if slice_mean > 0:
            norm_var = slice_var / slice_mean
        else:
            norm_var = 0.0
        normalized_variances.append(norm_var)
    
    normalized_variances = np.array(normalized_variances)

    # --- Clustering (K-Means) ---
    # Reshape for sklearn: (n_samples, n_features) -> (z_depth, 1)
    data = normalized_variances.reshape(-1, 1)
    
    # K-Means with 2 clusters (Focused vs Unfocused)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    
    # Determine which label corresponds to the higher variance (Focused)
    center_0 = kmeans.cluster_centers_[0][0]
    center_1 = kmeans.cluster_centers_[1][0]
    
    if center_1 > center_0:
        focused_label = 1
        threshold_est = (center_0 + center_1) / 2  # Estimate a visual threshold
    else:
        focused_label = 0
        threshold_est = (center_0 + center_1) / 2

    focused_indices = np.where(labels == focused_label)[0]
    unfocused_indices = np.where(labels != focused_label)[0]

    # --- Plotting ---
    if show_plot:
        plt.figure(figsize=(10, 6))
        
        # 1. Connecting line
        plt.plot(slice_indices, normalized_variances, linestyle='-', color='gray', alpha=0.5, zorder=1)
        
        # 2. Unfocused points
        plt.scatter(unfocused_indices, normalized_variances[unfocused_indices], 
                    color='skyblue', label='Unfocused', zorder=2)
        
        # 3. Focused points
        plt.scatter(focused_indices, normalized_variances[focused_indices], 
                    color='orange', label='Focused', zorder=2)
        
        # 4. Max indicator
        max_idx = np.argmax(normalized_variances)
        max_val = normalized_variances[max_idx]
        plt.plot(max_idx, max_val, 'r*', markersize=15, label=f'Max (Slice {max_idx})', zorder=3)

        plt.title("Normalized Variance vs. Z-Slice Number (K-Means)")
        plt.xlabel("Slice Number (Z)")
        plt.ylabel(r"Normalized Variance ($\sigma^2 / \mu$)")
        
        # Visual threshold line (midpoint between cluster centers)
        plt.axhline(y=threshold_est, color='k', linestyle=':', alpha=0.5, label='Cluster Boundary')
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    return focused_indices, unfocused_indices

def rescale_and_convert(image_float, target_dtype=np.uint16):
    """
    Min-max normalizes a float64 [z, x, y] array globally and converts it 
    to a specified unsigned integer type (uint8 or uint16).
    
    Args:
        image_float (np.ndarray): The input 3D array in float format.
        target_dtype (numpy.dtype): The desired output format (np.uint8 or np.uint16).
        
    Returns:
        np.ndarray: The rescaled integer array.
    """
    # Set the maximum limit based on the target data type
    if target_dtype == np.uint16:
        max_limit = 65535.0
    elif target_dtype == np.uint8:
        max_limit = 255.0
    else:
        raise ValueError("target_dtype must be np.uint16 or np.uint8")

    # Find the global min and max across the entire 3D stack
    img_min = image_float.min()
    img_max = image_float.max()
    
    # Safety check: Handle perfectly flat/blank images to prevent divide-by-zero errors
    if img_max == img_min:
        return np.zeros_like(image_float, dtype=target_dtype)
        
    # 1. Min-max normalize the data to a 0.0 -> 1.0 range
    normalized = (image_float - img_min) / (img_max - img_min)
    
    # 2. Scale up to the new target range (e.g., 0.0 -> 65535.0)
    scaled = normalized * max_limit
    
    # 3. Round to the nearest whole number before casting. 
    # (Just using .astype() simply chops off the decimals, which causes binning errors)
    return np.round(scaled).astype(target_dtype)