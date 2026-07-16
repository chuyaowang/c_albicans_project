"""Convert TIFF microscopy images to OME-Zarr (OME-NGFF v0.4).

Handles 2D (YX), 3D (ZYX), and multi-channel (CYX / ZCYX), with or without a
time axis (T). The goal is lossless on-disk size reduction: the full-resolution
data is stored unchanged but compressed with zstd + bit-shuffle, which typically
shrinks uncompressed 16-bit microscopy stacks several-fold.

A multiscale pyramid is OFF by default (it adds ~1/3 more data before
compression). Turn it on with ``--levels N`` when fast multi-resolution viewing
matters more than minimum size.

Examples
--------
    python -m image_processing_tools.util.tiff_to_omezarr input.tif
    python -m image_processing_tools.util.tiff_to_omezarr input.tif --levels 4
    tiff-to-omezarr input.tif --output out.ome.zarr --channel-names CY5 FITC DAPI
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from numcodecs import blosc as _blosc
from tqdm import tqdm

# OME-NGFF canonical dimension order. tifffile image series always end in Y, X;
# the leading axes are some permutation of T, C, Z which we normalise to this.
CANONICAL_ORDER = "TCZYX"
SPACE_UNIT = "micrometer"

AXIS_TYPE = {
    "t": "time",
    "c": "channel",
    "z": "space",
    "y": "space",
    "x": "space",
}

# Default display colours (RGB hex) for common fluorescence channels, matched on
# channel name; used only for the optional `omero` rendering hints.
DYE_COLORS = {
    "dapi": "0000FF",
    "fitc": "00FF00",
    "gfp": "00FF00",
    "cy3": "FFFF00",
    "tritc": "FFFF00",
    "cy5": "FF00FF",
    "phal": "FF0000",
}
# Fallback palette cycled by channel index when the name is unknown.
FALLBACK_COLORS = ["FFFFFF", "00FF00", "FF00FF", "00FFFF", "FFFF00", "FF0000"]


def build_compressor(clevel: int) -> Blosc:
    """Lossless zstd + bit-shuffle, well suited to 16-bit microscopy."""
    return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)


def set_blosc_threads(n_threads: int) -> None:
    """Set the number of threads Blosc uses *internally* per chunk.

    Blosc only parallelises blocks within a single chunk, which barely helps
    here, so the converter keeps this at 1 and instead parallelises across
    chunks with a thread pool. Exposed mainly for benchmarking. The setting is
    global/process-wide and applies to both compression and decompression.
    """
    n_threads = max(1, int(n_threads))
    # Force numcodecs to honour the explicit thread count instead of its
    # automatic single-/multi-thread heuristic.
    _blosc.use_threads = n_threads > 1
    _blosc.set_nthreads(n_threads)


def parse_pixel_metadata(tif: tifffile.TiffFile) -> dict:
    """Best-effort extraction of pixel size, z-step and channel names.

    Returns a dict with keys ``pixel_size`` (XY, float), ``z_step`` (float) and
    ``channel_names`` (list[str] or None). Missing values are returned as None.
    """
    info = ""
    meta = tif.imagej_metadata or {}
    if isinstance(meta, dict):
        info = meta.get("Info", "") or ""

    pixel_size = None
    m = re.search(r"Physical pixel size\s*=\s*\(([\d.eE+-]+)", info)
    if not m:
        m = re.search(r"Calibration\s*=\s*\(([\d.eE+-]+)", info)
    if m:
        pixel_size = float(m.group(1))
    if pixel_size is None:
        # Fall back to the TIFF resolution tag (pixels per unit -> size = 1/res).
        try:
            xres = tif.pages[0].tags["XResolution"].value
            num, den = xres
            if num:
                pixel_size = den / num
        except Exception:
            pass

    z_step = None
    m = re.search(r"Z increment[Vv]alue\s*=\s*(-?[\d.eE+-]+)", info)
    if m:
        z_step = abs(float(m.group(1)))
    if z_step is None and isinstance(meta, dict) and meta.get("spacing"):
        z_step = abs(float(meta["spacing"]))

    channel_names = None
    names = re.findall(r"Channel name #\d+\s*=\s*(.+)", info)
    if names:
        channel_names = [n.strip() for n in names]

    return {
        "pixel_size": pixel_size,
        "z_step": z_step,
        "channel_names": channel_names,
    }


def normalise_axes(src_axes: str):
    """Map tifffile axes to canonical NGFF order.

    Returns ``(dest_axes, transpose)`` where ``dest_axes`` is the lowercase axis
    string in canonical order and ``transpose`` reorders the source array's
    leading (non-YX) axes into that order.
    """
    src_axes = src_axes.upper()
    if not src_axes.endswith("YX"):
        raise ValueError(
            f"Unsupported axis layout {src_axes!r}; expected image axes ending in YX."
        )
    dest_axes = "".join(a for a in CANONICAL_ORDER if a in src_axes)
    transpose = [src_axes.index(a) for a in dest_axes]
    return dest_axes.lower(), transpose


def downsample_xy(plane: np.ndarray) -> np.ndarray:
    """2x block-mean downsample of a 2D YX plane (for pyramid levels only)."""
    h, w = plane.shape
    h2, w2 = h // 2, w // 2
    cropped = plane[: h2 * 2, : w2 * 2]
    reduced = cropped.reshape(h2, 2, w2, 2).mean(axis=(1, 3))
    return reduced.astype(plane.dtype)


def channel_metadata(channel_names, n_channels, dtype):
    """Build the `omero` block describing channel display settings."""
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    win_min = float(info.min) if info else 0.0
    win_max = float(info.max) if info else 1.0

    channels = []
    for i in range(n_channels):
        name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"
        color = None
        for key, hexcol in DYE_COLORS.items():
            if key in name.lower():
                color = hexcol
                break
        if color is None:
            color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        channels.append(
            {
                "label": name,
                "color": color,
                "active": True,
                "window": {"min": win_min, "max": win_max, "start": win_min, "end": win_max},
            }
        )
    return {"channels": channels, "rdefs": {"model": "color"}}


def convert(
    input_path: Path,
    output_path: Path,
    levels: int = 1,
    pixel_size: float | None = None,
    z_step: float | None = None,
    channel_names: list[str] | None = None,
    chunk_xy: int = 1024,
    clevel: int = 5,
    threads: int | None = None,
    progress: bool = True,
) -> Path:
    """Convert ``input_path`` TIFF to an OME-Zarr store at ``output_path``."""
    n_workers = max(1, threads or os.cpu_count() or 1)
    # Parallelism comes from the chunk-level thread pool below; keep Blosc's own
    # per-chunk threading off so the two don't oversubscribe the cores.
    set_blosc_threads(1)
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        src_axes = series.axes
        dtype = series.dtype
        dest_axes, transpose = normalise_axes(src_axes)

        meta = parse_pixel_metadata(tif)
        if pixel_size is None:
            pixel_size = meta["pixel_size"]
        if z_step is None:
            z_step = meta["z_step"]
        if channel_names is None:
            channel_names = meta["channel_names"]
        pixel_size = pixel_size or 1.0
        z_step = z_step or 1.0

        # Lazy, chunked view of the TIFF so we never load the whole stack at once.
        source = zarr.open(series.aszarr(), mode="r")
        src_shape = source.shape
        dest_shape = tuple(src_shape[i] for i in transpose)

        n_lead = len(dest_axes) - 2  # leading (non-YX) dims in dest order
        base_h, base_w = dest_shape[-2], dest_shape[-1]

        store = zarr.storage.LocalStore(str(output_path))
        root = zarr.open_group(store=store, mode="w", zarr_format=2)
        compressor = build_compressor(clevel)

        # Create one array per resolution level.
        arrays = []
        level_dims = []
        for lvl in range(levels):
            h = max(1, base_h >> lvl)
            w = max(1, base_w >> lvl)
            shape = dest_shape[:-2] + (h, w)
            chunks = (1,) * n_lead + (min(chunk_xy, h), min(chunk_xy, w))
            arr = root.create_array(
                name=str(lvl),
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressors=compressor,
            )
            arrays.append(arr)
            level_dims.append((h, w))

        # Read each YX plane once (from the main thread; the tifffile store is
        # not thread-safe), then hand compression + write to a worker. Pyramid
        # levels are derived from the plane in memory (~10 MB at a time).
        yx = (slice(None), slice(None))

        def read_plane(dest_lead):
            src_index = [0] * len(src_shape)
            for d, val in enumerate(dest_lead):
                src_index[transpose[d]] = val
            return np.asarray(source[tuple(src_index[:-2]) + yx])

        def write_plane(dest_lead, plane):
            arrays[0][dest_lead + yx] = plane
            cur = plane
            for lvl in range(1, levels):
                cur = downsample_xy(cur)
                arrays[lvl][dest_lead + yx] = cur

        lead_ranges = [range(dest_shape[i]) for i in range(n_lead)]
        total_planes = math.prod(len(r) for r in lead_ranges)
        bar = (
            tqdm(total=total_planes, unit="plane", desc="Compressing")
            if progress
            else None
        )

        if n_workers == 1:
            for dest_lead in product(*lead_ranges):
                write_plane(dest_lead, read_plane(dest_lead))
                if bar is not None:
                    bar.update(1)
        else:
            # Distinct planes write to distinct chunks (z/c chunked at 1), so
            # concurrent writes are safe. Bound in-flight work to cap memory.
            max_inflight = n_workers * 2
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                pending = deque()
                for dest_lead in product(*lead_ranges):
                    plane = read_plane(dest_lead)
                    pending.append(ex.submit(write_plane, dest_lead, plane))
                    if len(pending) >= max_inflight:
                        pending.popleft().result()
                        if bar is not None:
                            bar.update(1)
                while pending:
                    pending.popleft().result()
                    if bar is not None:
                        bar.update(1)
        if bar is not None:
            bar.close()

    # Per-axis scale (size of one pixel). XY doubles each pyramid level.
    base_scale = {
        "t": 1.0,
        "c": 1.0,
        "z": float(z_step),
        "y": float(pixel_size),
        "x": float(pixel_size),
    }
    axes_meta = []
    for a in dest_axes:
        entry = {"name": a, "type": AXIS_TYPE[a]}
        if entry["type"] == "space":
            entry["unit"] = SPACE_UNIT
        axes_meta.append(entry)

    datasets = []
    for lvl in range(levels):
        scale = []
        for a in dest_axes:
            s = base_scale[a]
            if a in ("x", "y"):
                s *= 2 ** lvl
            scale.append(s)
        datasets.append(
            {
                "path": str(lvl),
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": input_path.stem,
            "axes": axes_meta,
            "datasets": datasets,
        }
    ]
    if "c" in dest_axes:
        n_channels = dest_shape[dest_axes.index("c")]
        root.attrs["omero"] = channel_metadata(channel_names, n_channels, dtype)

    return output_path


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a TIFF image to OME-Zarr (OME-NGFF v0.4), losslessly compressed."
    )
    parser.add_argument(
        "input", type=Path,
        help="Input .tif/.tiff file. Quote the path if it contains spaces.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .ome.zarr path (default: <input stem>.ome.zarr next to input)",
    )
    parser.add_argument(
        "--levels", type=int, default=1,
        help="Number of resolution levels. 1 (default) = no pyramid; >1 builds an "
             "XY-downsampled pyramid (increases size, speeds up viewing).",
    )
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="XY pixel size in micrometers (default: read from metadata)")
    parser.add_argument("--z-step", type=float, default=None,
                        help="Z step in micrometers (default: read from metadata)")
    parser.add_argument("--channel-names", nargs="+", default=None,
                        help="Channel names (default: read from metadata)")
    parser.add_argument("--chunk-xy", type=int, default=1024,
                        help="XY chunk size in pixels (default: 1024)")
    parser.add_argument("--clevel", type=int, default=5,
                        help="zstd compression level 1-9 (default: 5)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Parallel compression workers, one plane each "
                             "(default: all CPU cores; ~4-8 is the sweet spot)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output store if it already exists")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress the progress bar")
    args, extra = parser.parse_known_args(argv)

    if extra:
        parser.error(
            f"Unexpected extra arguments: {extra}. "
            "If your file path contains spaces, wrap it in quotes, e.g.\n"
            '  tiff-to-omezarr "my image, with spaces.tif"'
        )

    input_path = args.input
    if not input_path.is_file():
        parser.error(f"Input not found: {input_path}")
    if args.levels < 1:
        parser.error("--levels must be >= 1")

    output = args.output
    if output is None:
        output = input_path.with_name(input_path.stem + ".ome.zarr")
    if output.exists():
        if not args.overwrite:
            parser.error(f"Output exists (use --overwrite): {output}")
        import shutil
        shutil.rmtree(output)

    convert(
        input_path,
        output,
        levels=args.levels,
        pixel_size=args.pixel_size,
        z_step=args.z_step,
        channel_names=args.channel_names,
        chunk_xy=args.chunk_xy,
        clevel=args.clevel,
        threads=args.threads,
        progress=not args.quiet,
    )

    in_size = input_path.stat().st_size
    out_size = _dir_size(output)
    print(f"Wrote {output}")
    print(f"Input:  {in_size / 1e9:.3f} GB")
    print(f"Output: {out_size / 1e9:.3f} GB  ({in_size / out_size:.2f}x smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())