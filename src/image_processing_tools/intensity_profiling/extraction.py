import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from skimage.morphology import medial_axis

from .skeleton import build_skeleton_graph, extend_skeleton
from .distance import geodesic_distance_map


def extract_intensity_profiles(dataset_folder, filename_prefix, cy_channel, pixel_size, bin_size=1):
    """Extract fluorescence and DAPI intensity profiles for all hyphae in a dataset.

    Iterates over numbered subfolders, loads images, skeletonises each labelled
    cell, identifies the hyphal tip as the skeleton endpoint furthest from the
    DAPI-stained nucleus, then bins pixel intensities along the geodesic axis.

    Parameters
    ----------
    dataset_folder : Path or str
        Root folder containing numbered subfolders (e.g. 01/, 02/, ...).
    filename_prefix : str
        Experiment identifier used to match image filenames (e.g. "CET145").
    cy_channel : str
        Channel name fragment used to glob the fluorescence TIFF
        (e.g. 'CY3.5 NAR', 'CY5').
    pixel_size : float
        Microscope calibration in µm per pixel.
    bin_size : float
        Bin width in µm for the intensity profiles.

    Returns
    -------
    df : pd.DataFrame
        Fluorescence intensity vs. distance from tip.
        Columns: label, distance, intensity.
    df_dapi : pd.DataFrame
        DAPI intensity vs. distance from tip.
        Columns: label, distance, intensity.
    """
    dataset_folder = Path(dataset_folder)
    subfolders = sorted(
        [f for f in dataset_folder.iterdir() if f.is_dir() and f.name.isdigit()]
    )

    print(f"Dataset:    {dataset_folder}")
    print(f"Subfolders: {[f.name for f in subfolders]}")

    all_records  = []
    dapi_records = []

    for subfolder in subfolders:
        imgs_cy   = sorted(subfolder.glob(f"MAX_{filename_prefix}_{cy_channel}*.tif"))
        imgs_dapi = sorted(subfolder.glob(f"MAX_{filename_prefix}_DAPI*.tif"))
        labels    = sorted(subfolder.glob(f"{filename_prefix}_CY3.5_Labels_*.tif"))

        if not imgs_cy or not imgs_dapi or not labels:
            print(f"[SKIP] {subfolder.name} — missing {cy_channel}, DAPI or Labels")
            continue

        print(f"\nProcessing {subfolder.name}/")

        zproj     = io.imread(str(imgs_cy[0]))
        dapi      = io.imread(str(imgs_dapi[0]))
        label_img = io.imread(str(labels[0]))

        # Skeletonise every labelled cell into a single combined skeleton
        skeleton_all = np.zeros_like(label_img, dtype=bool)
        for lab in np.unique(label_img)[1:]:
            mask = label_img == lab
            skeleton_all |= medial_axis(mask)

        # Extend skeleton endpoints so the skeleton reaches the cell boundary
        G               = build_skeleton_graph(skeleton_all.astype(int))
        pixel_neighbors = {node: len(list(G.neighbors(node))) for node in G.nodes}
        endpoints       = [node for node, n in pixel_neighbors.items() if n == 1]

        extended_skeleton = skeleton_all.copy().astype(int)
        for ep in endpoints:
            extended_skeleton = extend_skeleton(extended_skeleton, ep, max_steps=20)
        extended_skeleton = np.where(label_img > 0, extended_skeleton, 0)

        # Rebuild graph on the extended skeleton
        G2               = build_skeleton_graph(extended_skeleton)
        pixel_neighbors2 = {node: len(list(G2.neighbors(node))) for node in G2.nodes}
        endpoints2       = [node for node, n in pixel_neighbors2.items() if n == 1]

        # Tip = skeleton endpoint furthest from the DAPI-stained nucleus
        max_per_label = {}
        for label_id in np.unique(label_img)[1:]:
            cell_mask = label_img == label_id

            cell_endpoints = [ep for ep in endpoints2 if cell_mask[ep[0], ep[1]]]
            if len(cell_endpoints) < 2:
                continue

            dapi_within = dapi * cell_mask
            dapi_coords = np.argwhere(
                dapi_within > np.percentile(dapi_within[cell_mask], 75)
            )
            if len(dapi_coords) == 0:
                print(f"  [WARN] label {label_id} — no DAPI signal found, skipping")
                continue
            nucleus_centroid = dapi_coords.mean(axis=0)

            distances_to_nucleus = {
                ep: np.hypot(ep[0] - nucleus_centroid[0], ep[1] - nucleus_centroid[1])
                for ep in cell_endpoints
            }
            tip_coord = max(distances_to_nucleus, key=distances_to_nucleus.get)
            max_per_label[label_id] = (tip_coord, distances_to_nucleus[tip_coord])

            print(
                f"  [TIP] label {label_id} — tip at {tip_coord}, "
                f"dist from nucleus: {distances_to_nucleus[tip_coord]:.1f}px"
            )

        # Geodesic distance maps from tip to every foreground pixel
        max_distance  = 0
        distance_maps = {}
        for label_id, (tip_coord, _) in max_per_label.items():
            cell_mask = label_img == label_id
            if not np.any(cell_mask):
                continue
            dist_map = geodesic_distance_map(cell_mask, tip_coord, pixel_size_um=pixel_size)
            distance_maps[label_id] = dist_map
            max_distance = max(max_distance, dist_map.max())

        # Bin mean intensities along the geodesic axis
        bins        = np.arange(0, max_distance + bin_size, bin_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for label_id, dist_map in distance_maps.items():
            cell_mask        = label_img == label_id
            coords           = np.argwhere(cell_mask)
            distances        = dist_map[coords[:, 0], coords[:, 1]]
            intensities      = zproj[coords[:, 0], coords[:, 1]]
            dapi_intensities = dapi[coords[:, 0], coords[:, 1]]
            bin_indices      = np.digitize(distances, bins) - 1

            label = f"{subfolder.name}_{label_id}"
            for i, center in enumerate(bin_centers):
                in_bin = bin_indices == i
                if np.any(in_bin):
                    all_records.append({
                        "label":     label,
                        "distance":  center,
                        "intensity": np.mean(intensities[in_bin]),
                    })
                    dapi_records.append({
                        "label":     label,
                        "distance":  center,
                        "intensity": np.mean(dapi_intensities[in_bin]),
                    })

    df      = pd.DataFrame(all_records)
    df_dapi = pd.DataFrame(dapi_records)
    print(f"\nTotal hyphae: {df['label'].nunique()} across {len(subfolders)} fields of view")
    return df, df_dapi
