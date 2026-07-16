import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def _build_intensity_matrix(df, bin_size):
    """Build a (n_hyphae × n_bins) intensity matrix sorted by hypha length.

    Returns
    -------
    matrix : ndarray
        Shape (n_hyphae, n_bins), NaN where a hypha has no data for that bin.
    bin_centers : ndarray
        Centre of each distance bin in µm.
    sorted_labels : list of str
        Hypha labels in the same row order as *matrix* (shortest first).
    """
    max_distance = df.groupby("label")["distance"].max().max()
    bins         = np.arange(0, max_distance + bin_size, bin_size)
    bin_centers  = (bins[:-1] + bins[1:]) / 2
    hypha_ids    = sorted(df["label"].unique())
    num_bins     = len(bin_centers)
    num_hyphae   = len(hypha_ids)

    matrix = np.full((num_hyphae, num_bins), np.nan)
    for i, hypha_id in enumerate(hypha_ids):
        group = df[df["label"] == hypha_id]
        for _, row in group.iterrows():
            j = np.searchsorted(bin_centers, row["distance"])
            if 0 <= j < num_bins:
                matrix[i, j] = row["intensity"]

    lengths      = (~np.isnan(matrix)).sum(axis=1)
    sorted_order = np.argsort(lengths)
    return matrix[sorted_order], bin_centers, [hypha_ids[i] for i in sorted_order]


def plot_intensity_profiles(df, mRNA, filename_prefix, dataset_folder,
                            xlim=(0, 40), ylim=None, normalized=False):
    """Plot per-hypha fluorescence intensity as overlaid semi-transparent lines.

    Parameters
    ----------
    df : pd.DataFrame
        Intensity DataFrame (label, distance, intensity). Pass df_norm for
        normalized data.
    mRNA : str
        mRNA label shown in the plot title.
    filename_prefix : str
        Used in output filenames and the plot title.
    dataset_folder : Path or str
        Directory where output PDFs/PNGs are saved.
    xlim : (float, float)
        x-axis limits in µm.
    ylim : (float, float) or None
        y-axis limits. If None, matplotlib autoscales.
    normalized : bool
        If True, appends "_normalized" to the output filename.
    """
    dataset_folder = Path(dataset_folder)
    suffix         = "_intensity_profiles_normalized" if normalized else "_intensity_profiles"

    n_hyphae = df["label"].nunique()
    n_fov    = df["label"].str.split("_").str[0].nunique()
    print(f"Plotting {n_hyphae} hyphae across {n_fov} fields of view")
    print(f"Distance range: 0 — {df['distance'].max():.1f} µm")
    print(f"Intensity range: {df['intensity'].min():.3f} — {df['intensity'].max():.3f}")

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df, x="distance", y="intensity", hue="label",
        estimator=None, alpha=0.5, legend=False,
    )
    plt.xlabel("Distance from tip (µm)")
    plt.ylabel("Average intensity per bin")
    title_suffix = " normalized" if normalized else ""
    plt.title(
        f"{mRNA} intensity profiles — {filename_prefix} ({n_hyphae} hyphae){title_suffix}"
    )
    plt.grid(True)
    plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.savefig(dataset_folder / f"{filename_prefix}{suffix}.pdf", bbox_inches="tight")
    plt.savefig(
        dataset_folder / f"{filename_prefix}{suffix}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_dapi_heatmap(df_dapi, filename_prefix, dataset_folder, bin_size=1, xlim=(0, 40)):
    """Plot a DAPI intensity heatmap sorted by hypha length.

    Used as a sanity check: DAPI signal (nuclear DNA) should be concentrated
    at large distances from the tip (near the yeast body), not near zero.
    Any hypha with DAPI peaking near zero may have a reversed direction and
    should be added to FLIP_LABELS.

    Parameters
    ----------
    df_dapi : pd.DataFrame
        DAPI intensity DataFrame (label, distance, intensity).
    filename_prefix : str
    dataset_folder : Path or str
    bin_size : float
    xlim : (float, float)
    """
    dataset_folder = Path(dataset_folder)
    matrix, bin_centers, sorted_labels = _build_intensity_matrix(df_dapi, bin_size)
    num_hyphae = len(sorted_labels)

    plt.figure(figsize=(12, max(6, num_hyphae * 0.3)))
    im = plt.imshow(
        matrix, aspect="auto",
        extent=[0, bin_centers[-1], 0, num_hyphae],
        origin="lower", cmap="Blues",
    )
    plt.xlim(*xlim)
    plt.colorbar(im, label="DAPI Intensity")
    plt.xlabel("Distance from tip (µm)")
    plt.title(f"{filename_prefix} — DAPI sanity check (nucleus should peak at base)")

    tick_positions = np.arange(num_hyphae) + 0.5
    plt.yticks(tick_positions, sorted_labels, fontsize=7)

    plt.savefig(dataset_folder / f"{filename_prefix}_DAPI_sanity.pdf", bbox_inches="tight")
    plt.savefig(
        dataset_folder / f"{filename_prefix}_DAPI_sanity.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_dapi_metrics(df_dapi_metrics, filename_prefix, dataset_folder, dapi_threshold):
    """Plot DAPI spread and centroid position as jittered scatter plots with median.

    Parameters
    ----------
    df_dapi_metrics : pd.DataFrame
        Output of compute_dapi_metrics() with columns:
        label, dapi_spread_um, dapi_centroid_um.
    filename_prefix : str
    dataset_folder : Path or str
    dapi_threshold : float
        Threshold used for DAPI binarisation (shown in plot title).
    """
    dataset_folder = Path(dataset_folder)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, col, ylabel, title in zip(
        axes,
        ["dapi_spread_um", "dapi_centroid_um"],
        ["DAPI spread (µm)", "DAPI centroid distance from tip (µm)"],
        ["DAPI spread", "DAPI centroid position"],
    ):
        data = df_dapi_metrics[col].dropna()
        x    = np.ones(len(data)) + np.random.uniform(-0.05, 0.05, len(data))
        ax.scatter(x, data, alpha=0.6, edgecolors="black", linewidths=0.5, zorder=3)
        ax.hlines(
            data.median(), 0.85, 1.15, colors="red", linewidths=2,
            label=f"median: {data.median():.2f}",
        )
        ax.set_xticks([1])
        ax.set_xticklabels([filename_prefix])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"{filename_prefix} — DAPI metrics (threshold: {dapi_threshold:.0f})", y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        dataset_folder / f"{filename_prefix}_dapi_metrics.pdf", bbox_inches="tight"
    )
    plt.savefig(
        dataset_folder / f"{filename_prefix}_dapi_metrics.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_intensity_heatmap(df_norm, mRNA, filename_prefix, dataset_folder,
                           bin_size=1, xlim=(0, 40), vmax=1, show_labels=False):
    """Plot a normalized fluorescence intensity heatmap sorted by hypha length.

    Parameters
    ----------
    df_norm : pd.DataFrame
        Normalized intensity DataFrame (label, distance, intensity).
    mRNA : str
        mRNA label shown in the plot title and colorbar.
    filename_prefix : str
    dataset_folder : Path or str
    bin_size : float
        Bin width in µm. Use a coarser value (e.g. 2) when show_labels=True
        to keep the matrix manageable.
    xlim : (float, float)
    vmax : float
        Colour scale maximum. Default 1 (for normalized data).
    show_labels : bool
        If True, prints hypha ID labels on the y-axis (useful for identifying
        reversed hyphae by name) and saves directly to dataset_folder.
        If False, saves to a heatmaps/ subfolder.
    """
    dataset_folder = Path(dataset_folder)
    sns.set_context("talk")
    matrix, bin_centers, sorted_labels = _build_intensity_matrix(df_norm, bin_size)
    num_hyphae = len(sorted_labels)

    print(f"Max intensity in matrix:  {np.nanmax(matrix):.2f}")
    print(f"95th percentile:          {np.nanpercentile(matrix, 95):.2f}")

    fig_height = max(6, num_hyphae * 0.3) if show_labels else 4
    fig, ax = plt.subplots(figsize=(12 if show_labels else 8, fig_height))

    im = ax.imshow(
        matrix, aspect="auto",
        extent=[0, bin_centers[-1], 0, num_hyphae],
        origin="lower", cmap="plasma" if not show_labels else "magma",
        vmin=0, vmax=vmax,
        interpolation="None",
    )
    ax.set_xlim(*xlim)
    plt.colorbar(im, ax=ax, label=f"{mRNA} Signal Intensity")
    ax.set_xlabel("Distance from tip (µm)")
    ax.set_ylabel("Hyphae (sorted by length)")
    ax.set_title(f"{filename_prefix} {mRNA} intensity heatmap")

    if show_labels:
        tick_positions = np.arange(num_hyphae) + 0.5
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(sorted_labels, fontsize=7)
        save_dir = dataset_folder
        stem     = f"{filename_prefix}_heatmap"
    else:
        save_dir = dataset_folder / "heatmaps"
        save_dir.mkdir(exist_ok=True)
        stem = f"{filename_prefix}_heatmap_vmax{vmax}"

    plt.savefig(save_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(save_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved heatmap to {save_dir}")


def plot_tip_body_ratio(df_ratio, filename_prefix, dataset_folder,
                        tip_zone=(0, 5), body_zone=(5, 10)):
    """Plot tip:body intensity ratio as a jittered scatter with median line.

    Parameters
    ----------
    df_ratio : pd.DataFrame
        Output of compute_tip_body_ratio(); columns: label, tip_mean,
        body_mean, ratio.
    filename_prefix : str
    dataset_folder : Path or str
    tip_zone : (float, float)
    body_zone : (float, float)
    """
    dataset_folder = Path(dataset_folder)
    plt.figure(figsize=(6, 4))
    x = np.ones(len(df_ratio)) + np.random.uniform(-0.05, 0.05, len(df_ratio))
    plt.scatter(
        x, df_ratio["ratio"],
        alpha=0.6, edgecolors="black", linewidths=0.5, zorder=3,
    )
    plt.axhline(
        df_ratio["ratio"].median(), color="red", linestyle="-", linewidth=1.5,
        label=f"median: {df_ratio['ratio'].median():.2f}",
    )
    plt.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks([1], [filename_prefix])
    plt.ylabel("Tip:body intensity ratio")
    plt.title(
        f"{filename_prefix} — tip:body ratio\n"
        f"(tip {tip_zone[0]}–{tip_zone[1]} µm vs body {body_zone[0]}–{body_zone[1]} µm)"
    )
    plt.legend()
    plt.savefig(
        dataset_folder / f"{filename_prefix}_tip_body_ratio.pdf", bbox_inches="tight"
    )
    plt.savefig(
        dataset_folder / f"{filename_prefix}_tip_body_ratio.png", dpi=300, bbox_inches="tight"
    )
    plt.show()