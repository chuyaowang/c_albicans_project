#!/opt/miniconda3/envs/microsam/bin/python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
# from torch_em.util.util import get_random_colors
from matplotlib import colors
from typing import List, Tuple
import re
from skimage.filters import threshold_li
from scipy.ndimage import binary_fill_holes

def get_random_colors(labels: np.ndarray) -> colors.ListedColormap:
    """Generate a random color map for a label image.

    Args:
        labels: The labels.

    Returns:
        The color map.
    """
    unique_labels = np.unique(labels)
    have_zero = 0 in unique_labels
    cmap = [[0, 0, 0]] if have_zero else []
    cmap += np.random.rand(len(unique_labels), 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap

def show_single_mask_on_ax_AIS(ax, mask, image):
    """
    Helper function to display an instance segmentation mask overlayed on an image.
    Each instance in the mask will be colored differently.
    """
    # Ensure the image is in a format that can be displayed (e.g., grayscale or RGB)
    # If the image is not already RGB, convert it for display.
    display_image = image.copy()
    if display_image.ndim == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)

    ax.imshow(display_image)
    
    # Overlay the instance segmentation mask if it contains any objects.
    if mask.max() > 0:
        cmap = get_random_colors(mask)
        ax.imshow(mask, cmap=cmap, interpolation="nearest", alpha=0.5)
    ax.axis('off')

def save_segmentation_visualization_AIS(
    processed_image: np.ndarray,
    masks: np.ndarray | None,
    output_path: Path,
    title: str
):
    """
    Saves a side-by-side comparison of an image and its segmentation mask.

    Args:
        processed_image (np.ndarray): The input image.
        masks (np.ndarray | None): The segmentation mask. If None, only the
                                   original image is saved.
        output_path (Path): The path to save the output visualization.
        title (str): The title for the entire figure.
    """
    n_images = 1 if masks is None else 2
    fig, axes = plt.subplots(1, n_images, figsize=(10 * n_images, 10), squeeze=False)
    axes = axes.flatten() # This ensures axes is always a 1D array.

    # --- Plot Original Image ---
    axes[0].imshow(processed_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # --- Plot Prediction Mask ---
    if n_images == 2 and masks is not None:
        # Assuming a helper function `show_single_mask_on_ax_AIS` exists
        # to properly overlay the mask on the image.
        show_single_mask_on_ax_AIS(axes[1], masks, processed_image)
        axes[1].set_title("Prediction")
        axes[1].axis('off')

    fig.suptitle(title, fontsize=22)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.savefig(output_path)
    plt.close(fig)

def show_single_mask_on_ax(ax, mask, image):
    """Overlays a single boolean mask on a given Matplotlib axis with a random color."""
    overlay = np.ones((*image.shape[:2], 4))
    overlay[:, :, 3] = 0
    color = np.concatenate([np.random.random(3), [0.6]])
    overlay[mask] = color
    ax.imshow(image)
    ax.imshow(overlay)

def show_points_on_ax(ax, points, image):
    """
    Overlays point prompts on a given Matplotlib axis.
    Expects points in (x, y) format.
    """
    ax.imshow(image)
    # Matplotlib's scatter function expects (x, y), which matches our point format.
    # imshow sets the axis coordinates to match image coordinates (y-axis is inverted).
    ax.scatter(points[:, 0], points[:, 1], color='green', marker='*', s=150, edgecolor='white', linewidth=1.2)

def show_boxes_on_ax(ax, boxes, image):
    """Overlays bounding box prompts on a given Matplotlib axis."""
    ax.imshow(image)
    for box in boxes:
        x0, y0, x1, y1 = box
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor='none', lw=2)
        ax.add_patch(rect)

def save_segmentation_visualization(
    processed_image: np.ndarray,
    prompts: np.ndarray,
    prompt_type: str,
    masks: list,
    scores: np.ndarray,
    logits: list,
    output_path: Path,
    title: str
):
    """
    Creates and saves a 3x3 grid showing the original image, prompt,
    the 3 generated masks, and their corresponding logits.
    """
    if not (len(masks) == len(scores) == len(logits) == 3):
        print(f"  Warning: Expected 3 masks, scores, and logits for visualization. Skipping.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(24, 24))

    # --- Top Row: Image and Prompt ---
    axes[0, 0].imshow(processed_image)
    axes[0, 0].set_title("Original Processed Image", fontsize=14)
    
    if prompt_type == 'mask':
        axes[0, 1].imshow(prompts, cmap='gray')
        axes[0, 1].set_title("Input Prompt Mask", fontsize=14)
    elif prompt_type == 'points':
        show_points_on_ax(axes[0, 1], prompts, processed_image)
        axes[0, 1].set_title("Input DAPI Point Prompts", fontsize=14)
    elif prompt_type == 'bbox':
        show_boxes_on_ax(axes[0, 1], prompts, processed_image)
        axes[0, 1].set_title("Input Bounding Box Prompts", fontsize=14)
    
    # Hide the remaining empty plots in the top row
    axes[0, 2].axis('off')

    # --- Middle and Bottom Rows: Masks and Logits ---
    # Sort everything by score in descending order to show the best one first
    sorted_data = sorted(zip(masks, scores, logits), key=lambda x: x[1], reverse=True)
    
    for i, (mask, score, logit) in enumerate(sorted_data):
        # Middle row: Masks
        ax_mask = axes[1, i]
        show_single_mask_on_ax(ax_mask, mask, processed_image)
        ax_mask.set_title(f"Mask (Score: {score:.3f})", fontsize=14)

        # Bottom row: Logits
        ax_logit = axes[2, i]
        # Display the logit map. 'viridis' is a good colormap for this.
        im = ax_logit.imshow(logit, cmap='viridis')
        fig.colorbar(im, ax=ax_logit) # Add a colorbar to show the logit value scale
        ax_logit.set_title("Raw Logit Output", fontsize=14)

    # --- Final Touches and Saving ---
    for ax in axes.flatten():
        ax.axis('off')
    
    fig.suptitle(title, fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.savefig(output_path)
    plt.close(fig)

def save_multi_mask_visualization(
    processed_image: np.ndarray,
    prompts: np.ndarray,
    prompt_type: str,
    masks: list,
    scores: list,
    logits: list,
    output_path: Path,
    title: str
):
    """
    Creates and saves a 2x2 grid for visualizing results from multiple prompts.
    Grid layout:
    - Original Image
    - Prompts on Image
    - All Masks on Image
    - Max Logits on Image
    """
    if not masks:
        print(f"  Warning: No masks provided for visualization. Skipping {title}.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    # 1. Top-left: Original Processed Image
    axes[0].imshow(processed_image)
    axes[0].set_title("Original Processed Image")

    # 2. Top-right: Input Prompts
    if prompt_type == 'points':
        show_points_on_ax(axes[1], prompts, processed_image)
        axes[1].set_title("Input Point Prompts")
    elif prompt_type == 'bbox':
        show_boxes_on_ax(axes[1], prompts, processed_image)
        axes[1].set_title("Input Bounding Box Prompts")

    # 3. Bottom-left: All generated masks combined into an instance segmentation
    # Combine list of boolean masks into a single instance-labeled mask
    instance_mask = np.zeros(masks[0].shape, dtype=np.uint32)
    for i, mask in enumerate(masks):
        # Ensure mask is boolean before assigning
        instance_mask[mask.astype(bool)] = i + 1
    
    avg_score = np.mean(scores) if scores else 0
    mask_title = f"All Generated Masks ({len(masks)} instances)\nAvg. Score: {avg_score:.3f}"
    show_single_mask_on_ax_AIS(axes[2], instance_mask, processed_image)
    axes[2].set_title(mask_title)

    # 4. Bottom-right: Maximum projection of all logit maps
    # Stack all logit tensors and find the maximum value at each pixel
    if logits:
        max_logits = np.max(np.stack(logits), axis=0)
        im = axes[3].imshow(max_logits, cmap='viridis')
        fig.colorbar(im, ax=axes[3])
        axes[3].set_title("Maximum Logit Projection")

    # --- Final Touches and Saving ---
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title, fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path)
    plt.close(fig)

def save_channel_comparison_visualization(
    all_processed_images: dict,
    all_prompts: dict,
    all_prompt_types: dict,
    all_generated_masks: dict,
    all_generated_scores: dict, # These dictionaries are now keyed by result_key
    all_generated_logits: dict,
    display_items: list, # List of ImageContainer objects
    output_path: Path,
    title: str,
    segmentation_mode: str = 'prompted'
):
    """
    Creates and saves a grid visualization comparing results across multiple channels.
    The grid has 4 rows and a column for each item in `display_items`.
    - Row 1: Original Image
    - Row 2: Prompts on Image
    - Row 3: All Masks on Image
    - Row 4: Max Logits on Image
    """
    num_columns = len(display_items)
    if num_columns == 0:
        print("No items to visualize for comparison.")
        return

    if segmentation_mode == 'automatic':
        num_rows = 2
        figsize = (5 * num_columns, 11)
    else:
        num_rows = 4
        figsize = (5 * num_columns, 20)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize, squeeze=False)
    for col_idx, container in enumerate(display_items):
        result_key = container.name
        
        # Extract the descriptive channel combination string from the container name.
        # e.g., from "basename_CY5+FITC,DAPI" -> "CY5+FITC,DAPI"
        column_title = result_key.rsplit('_', 1)[-1] if '_' in result_key else result_key

        # --- Get data for this specific result_key ---
        processed_image = all_processed_images.get(result_key)
        prompts = all_prompts.get(result_key)
        prompt_type = all_prompt_types.get(result_key)
        masks = all_generated_masks.get(result_key)
        scores = all_generated_scores.get(result_key)
        logits = all_generated_logits.get(result_key)

        if processed_image is None:  # Data not found for this result_key
            for r_idx in range(num_rows):
                axes[r_idx, col_idx].text(0.5, 0.5, "Data Not Found", ha='center', va='center')
                axes[r_idx, col_idx].axis('off')
            axes[0, col_idx].set_title(column_title, fontsize=16)
            continue

        # --- Plotting for the current column ---
        # 1. Original Image
        axes[0, col_idx].imshow(processed_image)
        axes[0, col_idx].set_title(column_title, fontsize=16)

        # --- Plot Masks (Row 1 for AIS, Row 2 for Prompted) ---
        mask_row_idx = 1 if segmentation_mode == 'automatic' else 2
        if masks is not None:
            if isinstance(masks, list) and masks: # Prompted mode returns a list of masks
                instance_mask = np.zeros(masks[0].shape, dtype=np.uint32)
                for i, mask in enumerate(masks):
                    instance_mask[mask.astype(bool)] = i + 1
                show_single_mask_on_ax_AIS(axes[mask_row_idx, col_idx], instance_mask, processed_image)
                avg_score = np.mean(scores) if scores else 0
                axes[mask_row_idx, col_idx].set_title(f"All Masks (Avg. Score: {avg_score:.3f})")
            elif isinstance(masks, np.ndarray) and masks.max() > 0: # AIS mode returns a single instance mask
                show_single_mask_on_ax_AIS(axes[mask_row_idx, col_idx], masks, processed_image)
                axes[mask_row_idx, col_idx].set_title("AIS Masks")
        else: # No masks were generated
            axes[mask_row_idx, col_idx].imshow(processed_image)
            axes[mask_row_idx, col_idx].set_title("No Masks Generated")

        # --- Plot Prompts and Logits (Only for Prompted Mode) ---
        if segmentation_mode == 'prompted':
            # 2. Prompts
            if prompts is not None and prompt_type != 'none':
                if prompt_type == 'points':
                    show_points_on_ax(axes[1, col_idx], prompts, processed_image)
                elif prompt_type == 'bbox':
                    show_boxes_on_ax(axes[1, col_idx], prompts, processed_image)
                else:  # For mask prompts
                    axes[1, col_idx].imshow(processed_image)
                    axes[1, col_idx].imshow(prompts, cmap='gray', alpha=0.5)
                axes[1, col_idx].set_title(f"Prompts ({prompt_type})")
            else:
                axes[1, col_idx].imshow(processed_image)
                axes[1, col_idx].set_title("No Prompts")

            # 4. Max Logits
            if logits:
                max_logits = np.max(np.stack(logits), axis=0)
                im = axes[3, col_idx].imshow(max_logits, cmap='viridis')
                fig.colorbar(im, ax=axes[3, col_idx], fraction=0.046, pad=0.04)
                axes[3, col_idx].set_title("Maximum Logit Projection")
            else:
                axes[3, col_idx].imshow(processed_image)
                axes[3, col_idx].set_title("No Logits")

    for ax in axes.flatten():
        ax.axis('off')

    fig.suptitle(title, fontsize=22)
    # Add vertical padding (h_pad) to prevent titles from overlapping images.
    plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=3.0)
    plt.savefig(output_path)
    plt.close(fig)

def save_subtraction_comparison_visualization(
    ais_masks: np.ndarray,
    subtracted_images_dict: dict[tuple[str, int], np.ndarray],
    output_path: Path,
    title: str,
    offsets: list[int]
):
    """
    Saves a grid visualization comparing AIS masks on various subtracted images.

    The grid layout is:
    Rows: Each row corresponds to a different offset (e.g., O1, O5, O10).
    Columns: Subtracted (Direct), AIS on Subtracted (Direct),
             Subtracted (Average), AIS on Subtracted (Average).

    Args:
        ais_masks (np.ndarray): The AIS masks generated for the original image.
        subtracted_images_dict (dict[tuple[str, int], np.ndarray]): A dictionary where keys are
                                                                    (subtraction_method, offset) tuples
                                                                    and values are the corresponding 2D subtracted images.
                                                                    Values can be None if an image was not found.
        output_path (Path): The path to save the output visualization.
        title (str): The overall title for the figure.
        offsets (list[int]): A list of offset values (e.g., [1, 5, 10]).
    """
    num_rows = len(offsets)
    num_cols = 4 # Direct, AIS on Direct, Average, AIS on Average

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), squeeze=False)

    # Column titles for the top row
    col_titles = ["Subtracted (Direct)", "AIS on Subtracted (Direct)",
                  "Subtracted (Average)", "AIS on Subtracted (Average)"]

    for r_idx, offset in enumerate(offsets):
        # Set row label (offset)
        # Using ax.text is more reliable for positioning than set_ylabel in complex layouts.
        # We place the text to the left of the first subplot in each row.
        ax = axes[r_idx, 0]
        ax.text(-0.1, 0.5, f"Offset {offset}", transform=ax.transAxes, fontsize=14, va='center', ha='right', rotation=90)

        # --- Column 1 & 2: 'direct' method ---
        direct_img = subtracted_images_dict.get(('direct', offset))
        if direct_img is not None:
            axes[r_idx, 0].imshow(direct_img, cmap='gray')
            show_single_mask_on_ax(axes[r_idx, 1], ais_masks, direct_img)
        else:
            axes[r_idx, 0].text(0.5, 0.5, "Image Not Found", horizontalalignment='center', verticalalignment='center', transform=axes[r_idx, 0].transAxes, color='red')
            axes[r_idx, 1].text(0.5, 0.5, "Image Not Found", horizontalalignment='center', verticalalignment='center', transform=axes[r_idx, 1].transAxes, color='red')

        # --- Column 3 & 4: 'average' method ---
        avg_img = subtracted_images_dict.get(('average', offset))
        if avg_img is not None:
            axes[r_idx, 2].imshow(avg_img, cmap='gray')
            show_single_mask_on_ax(axes[r_idx, 3], ais_masks, avg_img)
        else:
            axes[r_idx, 2].text(0.5, 0.5, "Image Not Found", horizontalalignment='center', verticalalignment='center', transform=axes[r_idx, 2].transAxes, color='red')
            axes[r_idx, 3].text(0.5, 0.5, "Image Not Found", horizontalalignment='center', verticalalignment='center', transform=axes[r_idx, 3].transAxes, color='red')

        # --- Set titles and turn off axes for the row ---
        for c_idx in range(num_cols):
            axes[r_idx, c_idx].axis('off')
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(col_titles[c_idx], fontsize=12)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.savefig(output_path)
    plt.close(fig)

def save_omnipose_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    flow: np.ndarray,
    boundary: np.ndarray | None,
    output_path: Path,
    title: str,
    figsize: int = 5,
    dpi: int = 300
):
    """
    Saves a visualization of Omnipose segmentation results, including the
    original image, masks, flows, and boundaries.

    Args:
        image (np.ndarray): The original input image.
        mask (np.ndarray): The instance segmentation mask.
        flow (np.ndarray): The RGB flow field visualization.
        boundary (np.ndarray | None): The boundary prediction. Can be None.
        output_path (Path): The path to save the output visualization.
        title (str): The title for the entire figure.
        figsize (int): The base size for the figure.
        dpi (int): The resolution for the saved figure.
    """
    has_boundary = boundary is not None
    n_images = 3 if has_boundary else 2
    
    fig, axes = plt.subplots(1, n_images, figsize=(figsize * n_images, figsize))
    if n_images == 2:
        ax_img, ax_flow = axes
    else:
        ax_img, ax_flow, ax_bds = axes

    # --- 1. Image with Mask Overlay ---
    ax_img.imshow(image, cmap='gray')
    show_single_mask_on_ax_AIS(ax_img, mask, image)
    ax_img.set_title("Image + Mask")
    ax_img.axis('off')

    # --- 2. Flow Field ---
    ax_flow.imshow(flow)
    ax_flow.set_title("Flow Field")
    ax_flow.axis('off')

    # --- 3. Boundaries (optional) ---
    if has_boundary:
        ax_bds.imshow(boundary, cmap='gray')
        ax_bds.set_title("Boundaries")
        ax_bds.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

def visualize_flow_magnitude_zero_regions(rgb_flow_image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Thresholds the flow field's RGB visualization to show where flow vectors are near zero.

    The brightness of the flow visualization corresponds to its magnitude. This function
    identifies the darkest regions (near-zero magnitude) using a manual threshold.

    Args:
        rgb_flow_image (np.ndarray): The RGB visualization of the flow field (e.g., flows[0]).
                                     Shape should be (H, W, 3).
        threshold (float): The manual threshold value. Pixels with magnitude below this
                           value will be considered part of the zero-flow region.

    Returns:
        np.ndarray: A single-channel grayscale image where white pixels indicate
                    regions of near-zero flow magnitude.
    """
    if rgb_flow_image.ndim != 3 or rgb_flow_image.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3).")

    # Convert the RGB flow image to grayscale to get the magnitude information.
    grayscale_magnitude = cv2.cvtColor(rgb_flow_image, cv2.COLOR_RGB2GRAY)

    # Apply the manual threshold.
    _, zero_flow_mask = cv2.threshold(grayscale_magnitude, threshold, 255, cv2.THRESH_BINARY)
    return zero_flow_mask

def fill_cell_holes(donut_mask: np.ndarray) -> np.ndarray:
    """
    Fills holes in a binary mask where cells may appear as donuts.

    This function is ideal for post-processing a thresholded flow magnitude map
    where cell bodies are white (1 or 255) but the centers are black (0),
    creating solid objects for each cell.

    Args:
        donut_mask (np.ndarray): A single-channel binary image (dtype bool or uint8)
                                 where objects have holes.

    Returns:
        np.ndarray: A binary image of the same dtype as the input, with the
                    holes filled.
    """
    # scipy.ndimage.binary_fill_holes expects a boolean array.
    # The returned mask is also boolean.
    filled_mask = binary_fill_holes(donut_mask.astype(bool))

    # If the input was a boolean mask, return the boolean result.
    if donut_mask.dtype == bool:
        return filled_mask
    # Otherwise, convert back to the original integer data type (e.g., uint8).
    else:
        return filled_mask.astype(donut_mask.dtype) * np.iinfo(donut_mask.dtype).max

def combine_thresholded_maps(
    cell_prob_map: np.ndarray,
    flow_field: np.ndarray,
    prob_threshold: float,
    flow_threshold: float
) -> np.ndarray:
    """ # noqa
    Thresholds a cell probability map and a flow field magnitude map and
    combines them with a pixel-wise AND operation.

    Args:
        cell_prob_map (np.ndarray): Grayscale image representing cell probabilities.
        flow_field (np.ndarray): RGB flow field image where brightness represents magnitude.
        prob_threshold (float): Threshold for the cell probability map.
        flow_threshold (float): Threshold for the flow field magnitude.

    Returns:
        np.ndarray: A binary mask (uint8, values 0 or 255) resulting from the
                    AND operation.
    """
    # 1. Threshold the cell probability map to get a boolean mask.
    prob_mask = cell_prob_map > prob_threshold

    # 2. Threshold the flow field magnitude to get a boolean mask.
    grayscale_magnitude = cv2.cvtColor(flow_field, cv2.COLOR_RGB2GRAY)
    flow_mask = fill_cell_holes(grayscale_magnitude > flow_threshold)

    # 3. Perform a pixel-wise AND operation and fill holes in the resulting mask.
    return fill_cell_holes(prob_mask & flow_mask)

def invert_image_colors(image: np.ndarray) -> np.ndarray:
    """
    Performs a color inversion on the input image.

    This function inverts the pixel values of an image. For an 8-bit image,
    a pixel with value `p` will become `255 - p`. For a 16-bit image,
    a pixel with value `p` will become `65535 - p`, and so on.
    It works for both grayscale and multi-channel (e.g., RGB) images.
    
    For logical matrices (boolean arrays), 0 (False) becomes 1 (True) and 
    1 (True) becomes 0 (False).

    Args:
        image (np.ndarray): The input image matrix. Can be grayscale (2D) or color (3D).

    Returns:
        np.ndarray: The color-inverted image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input 'image' must be a NumPy array.")

    # Handle logical matrices (boolean arrays)
    if image.dtype == bool:
        return np.logical_not(image)

    # cv2.bitwise_not performs a bitwise NOT operation on each pixel.
    # For an 8-bit unsigned integer (uint8), this effectively inverts the color.
    # E.g., 0 (binary 00000000) becomes 255 (binary 11111111),
    # and 255 becomes 0.
    inverted_image = cv2.bitwise_not(image)

    return inverted_image

