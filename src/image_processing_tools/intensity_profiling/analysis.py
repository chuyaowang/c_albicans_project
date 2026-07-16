import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu


def normalize_profiles(df):
    """Normalize each hypha's intensity values to its own maximum.

    Rescales all profiles to [0, 1] so spatial localization patterns are
    comparable across cells with different absolute expression levels.

    Parameters
    ----------
    df : pd.DataFrame
        Intensity DataFrame with columns: label, distance, intensity.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with intensity per hypha rescaled to [0, 1].
    """
    collect = []
    for _, group in df.groupby("label"):
        group = group.copy()
        group["intensity"] = group["intensity"] / group["intensity"].max()
        collect.append(group)
    return pd.concat(collect, ignore_index=True)


def flip_profile_directions(df, flip_labels):
    """Reverse the distance axis for specified hyphae.

    Used to correct hyphae whose tip was mis-identified during extraction,
    causing the intensity profile to appear backwards (high signal at large
    distances rather than near zero).

    Parameters
    ----------
    df : pd.DataFrame
        Intensity DataFrame with columns: label, distance, intensity.
    flip_labels : list of str
        Label identifiers to flip (format "{subfolder}_{label_id}",
        e.g. ["02_5", "04_3"]).

    Returns
    -------
    pd.DataFrame
        Modified copy of *df* with selected profiles mirrored along the
        distance axis.
    """
    df = df.copy()
    for flip_label in flip_labels:
        mask = df["label"] == flip_label
        if not mask.any():
            print(f"[WARN] {flip_label} not found in DataFrame")
            continue
        max_dist = df.loc[mask, "distance"].max()
        df.loc[mask, "distance"] = max_dist - df.loc[mask, "distance"]
        print(f"[FLIP] {flip_label} flipped, max distance was {max_dist:.2f} µm")
    return df


def compute_dapi_metrics(df_dapi, dapi_threshold=None):
    """Quantify nuclear position along each hypha from DAPI intensity profiles.

    Computes two metrics per hypha:
    - **DAPI spread** — µm range where DAPI signal exceeds the threshold.
    - **DAPI centroid** — intensity-weighted mean distance of DAPI signal from
      the tip; larger values indicate the nucleus is further from the tip.

    Parameters
    ----------
    df_dapi : pd.DataFrame
        DAPI intensity DataFrame with columns: label, distance, intensity.
    dapi_threshold : float or None
        Intensity threshold to define DAPI-positive bins. If None, the Otsu
        threshold is computed automatically from all DAPI intensity values.

    Returns
    -------
    df_metrics : pd.DataFrame
        One row per hypha; columns: label, dapi_spread_um, dapi_centroid_um.
    dapi_threshold : float
        Threshold actually applied (useful when dapi_threshold=None).
    """
    if dapi_threshold is None:
        dapi_threshold = threshold_otsu(df_dapi["intensity"].values)
        print(f"DAPI threshold: {dapi_threshold:.0f} (auto Otsu)")
    else:
        print(f"DAPI threshold: {dapi_threshold} (manual override)")

    metrics = []
    for hypha_id, group in df_dapi.groupby("label"):
        dapi_above  = group[group["intensity"] > dapi_threshold]
        dapi_spread = (
            dapi_above["distance"].max() - dapi_above["distance"].min()
            if len(dapi_above) > 1
            else np.nan
        )
        dapi_centroid = (
            np.average(group["distance"], weights=group["intensity"])
            if group["intensity"].sum() > 0
            else np.nan
        )
        metrics.append({
            "label":            hypha_id,
            "dapi_spread_um":   dapi_spread,
            "dapi_centroid_um": dapi_centroid,
        })

    df_metrics = pd.DataFrame(metrics)
    print(f"Median DAPI spread:   {df_metrics['dapi_spread_um'].median():.2f} µm")
    print(f"Median DAPI centroid: {df_metrics['dapi_centroid_um'].median():.2f} µm from tip")
    return df_metrics, dapi_threshold


def compute_tip_body_ratio(df, tip_zone=(0, 5), body_zone=(5, 10)):
    """Compute the tip-to-body mean intensity ratio for each hypha.

    Compares mean fluorescence in the tip zone against the body zone.
    A ratio > 1 indicates tip-enriched mRNA localization.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (unnormalized) intensity DataFrame with columns: label, distance,
        intensity. Use the unnormalized DataFrame so that absolute intensity
        differences between zones are captured.
    tip_zone : (float, float)
        (start, end) µm range defining the tip zone. Default (0, 5).
    body_zone : (float, float)
        (start, end) µm range defining the body zone. Default (5, 10).
        Should cover the same width as tip_zone.

    Returns
    -------
    pd.DataFrame
        Columns: label, tip_mean, body_mean, ratio.
        Hyphae shorter than body_zone[1] are excluded.
    """
    tip_start,  tip_end  = tip_zone
    body_start, body_end = body_zone

    ratios = []
    for hypha_id, group in df.groupby("label"):
        tip_intensity = group[
            (group["distance"] >= tip_start) & (group["distance"] <= tip_end)
        ]["intensity"].mean()
        body_intensity = group[
            (group["distance"] >= body_start) & (group["distance"] <= body_end)
        ]["intensity"].mean()

        if pd.notna(tip_intensity) and pd.notna(body_intensity) and body_intensity > 0:
            ratios.append({
                "label":     hypha_id,
                "tip_mean":  tip_intensity,
                "body_mean": body_intensity,
                "ratio":     tip_intensity / body_intensity,
            })

    df_ratio   = pd.DataFrame(ratios)
    n_excluded = df["label"].nunique() - len(df_ratio)
    print(f"Hyphae included: {len(df_ratio)} / {df['label'].nunique()} total")
    if n_excluded:
        print(f"  ({n_excluded} excluded — shorter than {body_end} µm, no body zone)")
    print(f"Median tip:body ratio: {df_ratio['ratio'].median():.2f}")
    return df_ratio
