import numpy as np
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import os
from pathlib import Path

from image_processing_tools.rf_nuclei.rf_nuclei_bg_prediction import create_pixel_features, predict_pixel_class
from image_processing_tools.rf_nuclei.rf_load_models import load_model


def detect_nuclei(seg_image, use_watershed=False):
    """
    Performs image segmentation to identify potential nuclei.

    It uses Otsu's thresholding to create a binary mask and optionally applies
    the Watershed algorithm to separate touching objects.

    Args:
        seg_image (numpy.ndarray): The grayscale input image (typically the DAPI channel).
        use_watershed (bool): If True, applies distance-transform-based watershed
                              segmentation to split connected nuclei.

    Returns:
        tuple: A tuple containing:
            - labels (numpy.ndarray): An integer mask where each detected object has a unique ID.
            - binary_mask_filled (numpy.ndarray): A boolean mask representing the foreground (nuclei) after filling holes.
    """
    thresh_val = threshold_otsu(seg_image)
    binary_mask = seg_image > thresh_val
    binary_mask_filled = ndimage.binary_fill_holes(binary_mask)

    if use_watershed:
        distance = ndimage.distance_transform_edt(binary_mask_filled)
        distance_smoothed = ndimage.gaussian_filter(distance, sigma=2)

        coords = peak_local_max(distance_smoothed, min_distance=5, labels=binary_mask_filled)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)

        labels = watershed(-distance, markers, mask=binary_mask_filled)
    else:
        labels, _ = ndimage.label(binary_mask_filled)

    return labels, binary_mask_filled


def detect_nuclei_rf(seg_image, model, nuclei_class=1):
    """
    Performs nuclei detection using a trained Random Forest model.

    Args:
        seg_image (numpy.ndarray): The input image.
        model: The trained Random Forest model.
        nuclei_class (int): The class label representing nuclei in the model output.

    Returns:
        tuple: (labels, binary_mask_filled)
    """
    features = create_pixel_features(seg_image)
    prediction = predict_pixel_class(model, features)

    binary_mask = (prediction == nuclei_class)
    binary_mask_filled = ndimage.binary_fill_holes(binary_mask)
    labels, _ = ndimage.label(binary_mask_filled)

    return labels, binary_mask_filled


def filter_nuclei_xgb(labels, min_eccentricity=0.0):
    """
    Refines a messy pixel-level segmentation into clean nuclei objects using
    `object_xgb.workers.segment_objects_worker` and returns per-nucleus dicts
    in the same shape as `filter_nuclei`, plus the refined label image so the
    caller can build a matching binary mask for `extract_graph`.

    The worker treats any nonzero pixel as foreground, fills holes, relabels
    connected components, auto-thresholds by object area (KMeans + SVM on log
    areas) to drop noise fragments, dilates, and relabels sequentially. The
    result is a clean label image on which `regionprops` produces the per-object
    dictionaries the downstream graph pipeline expects.

    Args:
        labels (numpy.ndarray): Integer-labeled segmentation from `detect_nuclei`
            or `detect_nuclei_rf`. Noisy pixel-level output is fine - the worker
            does its own re-segmentation.
        min_eccentricity (float): Optional eccentricity floor applied after the
            object-xgb cleanup. 0.0 keeps every cleaned object.

    Returns:
        tuple:
            - valid_nuclei_data (list[dict]): Per-nucleus dicts with the same keys
              as `filter_nuclei` ('centroid', 'orientation', 'major_axis_length',
              'minor_axis_length', 'area', 'eccentricity', 'perimeter', 'coords').
            - valid_labels (list[int]): Label IDs of the kept nuclei in the
              refined label image.
            - avg_nucleus_length (float): Mean `major_axis_length` across kept nuclei.
            - clean_labels (numpy.ndarray): Refined label image from the worker.
              Pass `clean_labels > 0` to `extract_graph` as the binary mask so the
              masked-out regions match the cleaned nuclei rather than the noisy
              original prediction.
    """
    from object_xgb.workers import segment_objects_worker

    clean_labels = segment_objects_worker(labels, orig_ndim=labels.ndim, layer_type='labels')

    props = regionprops(clean_labels)
    valid_nuclei_data = []
    valid_labels = []
    removed_eccentricity = 0

    for p in props:
        if p.eccentricity < min_eccentricity:
            removed_eccentricity += 1
            continue

        valid_nuclei_data.append({
            'centroid': p.centroid,
            'orientation': p.orientation,
            'major_axis_length': p.major_axis_length,
            'minor_axis_length': p.minor_axis_length,
            'area': p.area,
            'eccentricity': p.eccentricity,
            'perimeter': p.perimeter,
            'coords': p.coords,
        })
        valid_labels.append(p.label)

    avg_nucleus_length = 1.0
    if valid_nuclei_data:
        avg_nucleus_length = float(np.mean([n['major_axis_length'] for n in valid_nuclei_data]))

    print(f"[object-xgb] Refined {len(props)} objects; kept {len(valid_nuclei_data)} "
          f"(removed {removed_eccentricity} low-eccentricity).")
    if valid_nuclei_data:
        print(f"Average Nucleus Length: {avg_nucleus_length:.2f} px")

    return valid_nuclei_data, valid_labels, avg_nucleus_length, clean_labels


def filter_nuclei(labels, lower_size_factor=0.33, upper_size_factor=3.0, min_eccentricity=0.0):
    """
    Filters detected objects based on size and shape (eccentricity).

    It calculates the mean area of all objects and removes those that are significantly
    smaller or larger than the mean. It also filters based on eccentricity (roundness).
    It prints a report detailing how many nuclei were removed by each filter.

    Args:
        labels (numpy.ndarray): The labeled mask from detect_nuclei.
        lower_size_factor (float): The minimum area threshold (as a fraction of the mean area).
        upper_size_factor (float): The maximum area threshold (as a multiple of the mean area).
        min_eccentricity (float): The minimum eccentricity allowed (0 = perfect circle, 1 = line).
                                  Used to remove overly round objects if desired.

    Returns:
        tuple: A tuple containing:
            - valid_nuclei_data (list[dict]): A list of dictionaries for kept nuclei, containing 'centroid', 'orientation', and 'major_axis_length'.
            - valid_labels (list[int]): The label IDs of the kept nuclei.
            - avg_nucleus_length (float): The average major axis length of the kept nuclei.
    """
    props = regionprops(labels)
    all_areas = [p.area for p in props]
    valid_nuclei_data = []
    valid_labels = []

    avg_nucleus_length = 1.0

    if all_areas:
        mean_area = np.mean(all_areas)
        min_area = mean_area * lower_size_factor
        max_area = mean_area * upper_size_factor

        removed_small = 0
        removed_large = 0
        removed_eccentricity = 0

        for p in props:
            if p.area <= min_area:
                removed_small += 1
                continue
            elif p.area >= max_area:
                removed_large += 1
                continue

            if p.eccentricity < min_eccentricity:
                removed_eccentricity += 1
                continue

            valid_nuclei_data.append({
                'centroid': p.centroid,
                'orientation': p.orientation,
                'major_axis_length': p.major_axis_length,
                'minor_axis_length': p.minor_axis_length,
                'area': p.area,
                'eccentricity': p.eccentricity,
                'perimeter': p.perimeter,
                'coords': p.coords
            })
            valid_labels.append(p.label)

        print(f"Detected {len(props)} objects. Mean Area: {mean_area:.1f} px.")
        print(f"Kept {len(valid_nuclei_data)} nuclei (Size range: {min_area:.1f} - {max_area:.1f}).")
        print(f"Removed: {removed_small} small, {removed_large} large, {removed_eccentricity} low eccentricity.")

        if valid_nuclei_data:
            avg_nucleus_length = np.mean([n['major_axis_length'] for n in valid_nuclei_data])
            print(f"Average Nucleus Length: {avg_nucleus_length:.2f} px")
    else:
        print("No objects detected.")

    return valid_nuclei_data, valid_labels, avg_nucleus_length