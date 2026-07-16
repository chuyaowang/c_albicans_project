import numpy as np
from skimage.measure import profile_line
from skimage.color import label2rgb
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from image_processing_tools.util.load_files import load_image_data
from image_processing_tools.rf_nuclei.rf_load_models import load_model
from image_processing_tools.dapi_tracing.nuclei_detection import detect_nuclei, detect_nuclei_rf, filter_nuclei


def calculate_connectivity(valid_nuclei_data, int_image, binary_mask_filled, avg_nucleus_length, path_width=3, use_orientation_penalty=True):
    """
    Computes a connectivity graph between valid nuclei.

    It evaluates the "strength" of connection between every pair of nuclei based on the
    intensity of the path between them, penalized by distance and orientation alignment.

    Args:
        valid_nuclei_data (list): List of properties for valid nuclei.
        int_image (numpy.ndarray): The intensity image used to measure signal between nuclei.
        binary_mask_filled (numpy.ndarray): Mask used to exclude the nuclei pixels themselves from the path intensity calculation.
        avg_nucleus_length (float): Used to normalize distances into "nuclei units".
        path_width (int): Width of the line profile used to sample intensity between centroids.
        use_orientation_penalty (bool): If True, penalizes connections that are not aligned with the major axes of the connected nuclei.

    Returns:
        tuple: A tuple containing:
            - adjacency_matrix (numpy.ndarray): A square matrix where [i, j] is the connectivity score (0.0 to 1.0).
            - lines (list[tuple]): Coordinates (start, end) for drawing the connections.
            - metrics (list[float]): The connectivity scores corresponding to lines.
            - masked_int_image (numpy.ndarray): The intensity image with nuclei pixels set to 0.
    """
    num_nuclei = len(valid_nuclei_data)
    adjacency_matrix = np.zeros((num_nuclei, num_nuclei))
    raw_connections = []

    masked_int_image = int_image.copy()
    masked_int_image[binary_mask_filled] = 0

    if num_nuclei < 2:
        print("Not enough nuclei to form a graph.")
    else:
        for i in range(num_nuclei):
            for j in range(i + 1, num_nuclei):
                nuc1 = valid_nuclei_data[i]
                nuc2 = valid_nuclei_data[j]

                c1 = nuc1['centroid']
                c2 = nuc2['centroid']

                dy = c2[0] - c1[0]
                dx = c2[1] - c1[1]
                dist = np.sqrt(dy**2 + dx**2)

                if dist == 0:
                    continue

                dist_in_nuclei = dist / avg_nucleus_length
                sigma = 2.5
                dist_penalty = np.exp(-((dist_in_nuclei - 5.0)**2) / (2 * sigma**2))

                profile = profile_line(masked_int_image, c1, c2, linewidth=path_width, mode='constant', cval=0)
                valid_profile = profile[profile > 0]

                if len(valid_profile) > 0:
                    mean_intensity = np.mean(valid_profile)
                else:
                    mean_intensity = 0.0

                raw_int_score = mean_intensity * path_width

                if use_orientation_penalty:
                    path_angle = np.arctan2(dx, dy)

                    def get_acute_diff(angle1, angle2):
                        diff = abs(angle1 - angle2) % np.pi
                        if diff > np.pi / 2:
                            diff = np.pi - diff
                        return diff

                    diff_1 = get_acute_diff(path_angle, nuc1['orientation'])
                    diff_2 = get_acute_diff(path_angle, nuc2['orientation'])
                    min_diff = min(diff_1, diff_2)
                    alignment_factor = np.cos(min_diff)
                    raw_int_score *= alignment_factor

                raw_connections.append({
                    'i': i, 'j': j,
                    'raw_int': raw_int_score,
                    'dist_penalty': dist_penalty,
                    'c1': c1, 'c2': c2
                })

        if raw_connections:
            all_ints = np.array([x['raw_int'] for x in raw_connections])
            int_range = all_ints.max() - all_ints.min()
            if int_range == 0:
                int_range = 1
            norm_ints = (all_ints - all_ints.min()) / int_range

            dist_penalties = np.array([x['dist_penalty'] for x in raw_connections])
            final_scores = norm_ints * dist_penalties

            lines = []
            metrics = []

            for idx, item in enumerate(raw_connections):
                score = final_scores[idx]
                i, j = item['i'], item['j']

                adjacency_matrix[i, j] = score
                adjacency_matrix[j, i] = score

                lines.append((item['c1'], item['c2']))
                metrics.append(score)
        else:
            lines = []
            metrics = []

    return adjacency_matrix, lines, metrics, masked_int_image


def plot_nuclei_analysis(seg_image, masked_int_image, binary_mask_filled, labels, valid_labels, valid_nuclei_data, int_image, lines, metrics, segmentation_method="Otsu + Watershed"):
    """
    Visualizes the entire analysis pipeline in a multi-panel figure.

    Args:
        seg_image (numpy.ndarray): The segmentation source image.
        masked_int_image (numpy.ndarray): The intensity image with nuclei masked out.
        binary_mask_filled (numpy.ndarray): The binary mask of detected nuclei.
        labels (numpy.ndarray): The raw segmentation labels.
        valid_labels (list[int]): The IDs of nuclei that passed filtering.
        valid_nuclei_data (list[dict]): Properties for drawing axes.
        int_image (numpy.ndarray): The original intensity image.
        lines (list[tuple]): Data for drawing the connectivity network lines.
        metrics (list[float]): Connectivity scores for the lines.
        segmentation_method (str): Name of the segmentation method used for the plot title.
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[:, 2])

    ax1.imshow(seg_image, cmap='gray')
    ax1.set_title("1. Segmentation Source")
    ax1.axis('off')

    ax2.imshow(masked_int_image, cmap='gray')
    ax2.set_title("2. Masked Intensity (Nuclei=0)")
    ax2.axis('off')

    ax3.imshow(binary_mask_filled, cmap='gray')
    ax3.set_title("3. Binary Mask")
    ax3.axis('off')

    if valid_labels:
        max_label = labels.max()
        keep_mask = np.zeros(max_label + 1, dtype=bool)
        keep_mask[valid_labels] = True
        filtered_labels = labels * keep_mask[labels]
    else:
        filtered_labels = np.zeros_like(labels)

    colored_labels = label2rgb(filtered_labels, bg_label=0)
    ax4.imshow(colored_labels)

    for nuc in valid_nuclei_data:
        y0, x0 = nuc['centroid']
        orientation = nuc['orientation']
        length = nuc['major_axis_length']

        dx = (length / 2) * np.sin(orientation)
        dy = (length / 2) * np.cos(orientation)

        x1, y1 = x0 - dx, y0 - dy
        x2, y2 = x0 + dx, y0 + dy

        ax4.plot([x1, x2], [y1, y2], 'black', linewidth=1.5, alpha=0.8)

    ax4.set_title(f"4. Segmentation ({segmentation_method})")
    ax4.axis('off')

    overlay_img = label2rgb(filtered_labels, image=int_image, bg_label=0, alpha=0.3, kind='overlay')
    ax5.imshow(overlay_img)
    ax5.set_title(f"5. Network (Nuclei Excluded)")
    ax5.axis('off')

    if metrics:
        plasma = plt.cm.plasma
        colors = plasma(np.linspace(0.15, 1, 256))
        bright_plasma = mcolors.LinearSegmentedColormap.from_list("bright_plasma", colors)

        norm = plt.Normalize(0, 1)

        for (c1, c2), val in zip(lines, metrics):
            color = bright_plasma(norm(val))
            ax5.plot([c1[1], c2[1]], [c1[0], c2[0]], color=color, linewidth=2.5, alpha=0.8)

        sm = plt.cm.ScalarMappable(cmap=bright_plasma, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax5, fraction=0.046, pad=0.04)
        cbar.set_label('Connectivity Score (0-1)')

    valid_centroids = [n['centroid'] for n in valid_nuclei_data]
    if valid_centroids:
        y_vals = [c[0] for c in valid_centroids]
        x_vals = [c[1] for c in valid_centroids]
        ax5.plot(x_vals, y_vals, 'o', color='white', markersize=4, markeredgecolor='black', alpha=0.9)

    plt.tight_layout()
    plt.show()


def extract_and_plot_nuclei_axis(segmentation_source, intensity_source=None,
                                 path_width=3,
                                 use_watershed=False,
                                 lower_size_factor=0.33, upper_size_factor=3.0,
                                 use_orientation_penalty=True,
                                 min_eccentricity=0.0,
                                 show_plot=True,
                                 model=None,
                                 nuclei_class=1):
    """
    Main driver function to extract nuclei and calculate connectivity using the greedy approach.

    It loads images, runs detection, filters nuclei, calculates connectivity,
    and optionally plots the results.

    Args:
        segmentation_source (str | Path | numpy.ndarray): Path to the segmentation image or the array itself.
        intensity_source (str | Path | numpy.ndarray, optional): Path to the intensity image. If None, uses segmentation_source.
        path_width (int): Width of the line profile used to sample intensity.
        use_watershed (bool): If True, uses watershed segmentation.
        lower_size_factor (float): Minimum area threshold factor.
        upper_size_factor (float): Maximum area threshold factor.
        use_orientation_penalty (bool): If True, penalizes connections based on orientation alignment.
        min_eccentricity (float): Minimum eccentricity to keep a nucleus.
        show_plot (bool): Whether to generate and show the visualization.
        model: Optional Random Forest model for segmentation. If provided, overrides thresholding.
        nuclei_class (int): Class ID for nuclei when using the model.

    Returns:
        tuple: A tuple containing:
            - adjacency_matrix (numpy.ndarray): The final connectivity matrix.
            - valid_centroids (list): Centroids of the valid nuclei.
            - valid_major_axes (list): Major axis lengths of the valid nuclei.
    """
    seg_image = load_image_data(segmentation_source)

    if intensity_source is not None:
        int_image = load_image_data(intensity_source)
        if seg_image.shape != int_image.shape:
            raise ValueError(f"Shape mismatch: Seg {seg_image.shape} vs Int {int_image.shape}")
    else:
        int_image = seg_image

    if model is not None:
        if isinstance(model, (str, Path)):
            model = load_model(Path(os.path.expanduser(str(model))))

        labels, binary_mask_filled = detect_nuclei_rf(seg_image, model, nuclei_class)
        seg_method_name = "RF Model"
    else:
        labels, binary_mask_filled = detect_nuclei(seg_image, use_watershed)
        seg_method_name = "Otsu + Watershed" if use_watershed else "Otsu Threshold"

    valid_nuclei_data, valid_labels, avg_nucleus_length = filter_nuclei(labels, lower_size_factor, upper_size_factor, min_eccentricity)

    adjacency_matrix, lines, metrics, masked_int_image = calculate_connectivity(
        valid_nuclei_data, int_image, binary_mask_filled, avg_nucleus_length,
        path_width, use_orientation_penalty
    )

    if show_plot:
        plot_nuclei_analysis(seg_image, masked_int_image, binary_mask_filled, labels, valid_labels, valid_nuclei_data, int_image, lines, metrics, seg_method_name)

    valid_centroids = [n['centroid'] for n in valid_nuclei_data]
    valid_major_axes = [n['major_axis_length'] for n in valid_nuclei_data]
    return adjacency_matrix, valid_centroids, valid_major_axes