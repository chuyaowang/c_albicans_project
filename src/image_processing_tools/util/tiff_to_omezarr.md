# Converting TIFF to OME-Zarr — a practical guide

`tiff_to_omezarr.py` converts microscopy TIFF images to **OME-Zarr** (OME-NGFF
v0.4): a chunked, compressed, cloud-friendly array format. It handles 2D, 3D,
and multi-channel stacks, with the primary goal of **shrinking files losslessly**
while keeping the data fast to access region-by-region.

This document explains how to run it, what every parameter does, how OME-Zarr
compression actually works (chunks, blocks, lossless vs lossy, pyramids,
multithreading, the compress/decompress path), and the benchmarks that justify
the defaults.

---

## 1. Quick start

```bash
# Activate the environment that has zarr/numcodecs/tifffile/tqdm
conda activate microsam

# Simplest: convert in place (output written next to the input)
tiff-to-omezarr "/path/to/My Image, 3 channels.tif"

# Equivalent module form (no console script needed)
python -m image_processing_tools.util.tiff_to_omezarr "/path/to/image.tif"
```

The output is a directory named `<input-stem>.ome.zarr` next to the input file.

> **Quote paths with spaces.** The tool takes a single input path. If you don't
> quote a path containing spaces, the shell splits it into several arguments and
> you get a clear error telling you to add quotes.

---

## 2. Installation / requirements

Dependencies (all declared in `pyproject.toml`): `numpy`, `tifffile`, `zarr`
(v3), `numcodecs`, `tqdm`. No `ome-zarr-py` is required — the tool writes the
NGFF v0.4 metadata directly, which avoids pinning/downgrading `zarr`.

```bash
pip install -e .          # installs the package + the `tiff-to-omezarr` command
# or just make sure the deps above are importable
```

---

## 3. The CLI parameters

```
tiff-to-omezarr INPUT [options]
```

| Parameter | Default | What it does |
|---|---|---|
| `INPUT` (positional) | — | Input `.tif`/`.tiff` file. **Quote if it has spaces.** |
| `-o`, `--output PATH` | `<stem>.ome.zarr` next to input | Output store path. |
| `--levels N` | `1` | Number of resolution levels. `1` = no pyramid. `>1` builds an XY-downsampled pyramid (faster viewing, **larger** on disk). |
| `--pixel-size µm` | read from metadata | XY pixel size in micrometers, written into the scale metadata. |
| `--z-step µm` | read from metadata | Z step (slice spacing) in micrometers. |
| `--channel-names ...` | read from metadata | Space-separated channel names (e.g. `CY5 FITC DAPI`). |
| `--chunk-xy N` | `1024` | XY chunk size in pixels. See §5.2. |
| `--clevel N` | `5` | zstd compression level, 1–9. Higher = smaller + slower, always lossless. |
| `--threads N` | all CPU cores | Parallel compression workers (one plane each). ~4–8 is the sweet spot. |
| `--overwrite` | off | Overwrite the output store if it already exists. |
| `-q`, `--quiet` | off | Suppress the progress bar. |

### Common recipes

```bash
# Smallest possible file (slowest), use a moderate worker count
tiff-to-omezarr image.tif --clevel 9 --threads 6

# Fast conversion, default compression
tiff-to-omezarr image.tif --threads 8

# Build a 4-level pyramid for smooth multi-resolution viewing in napari/Fiji
tiff-to-omezarr image.tif --levels 4

# Override metadata that the file doesn't carry
tiff-to-omezarr image.tif --pixel-size 0.065 --z-step 0.2 --channel-names CY5 FITC DAPI

# Scripting / logging: quiet, explicit output
tiff-to-omezarr image.tif -o /data/out.ome.zarr --quiet --overwrite
```

### What it does automatically

- **Detects the axis layout** (`YX`, `ZYX`, `CYX`, `ZCYX`, `TCZYX`, …) from the
  TIFF and normalises it to the canonical NGFF order `t, c, z, y, x`.
- **Reads pixel size, z-step and channel names** from ImageJ/OME metadata when
  present (overridable by the flags above).
- **Writes `omero` display metadata** (channel names + sensible colors: DAPI→blue,
  FITC/GFP→green, CY5→magenta, etc.) when the image has a channel axis.
- **Streams plane by plane**, so a 1.7 GB stack never has to fit in RAM all at
  once.

---

## 4. What OME-Zarr is

OME-Zarr stores an image not as one monolithic file but as a **directory** of
many small compressed files plus JSON metadata, following the
[OME-NGFF](https://ngff.openmicroscopy.org/) specification (this tool writes
**v0.4**, the most widely supported version, readable by napari, Fiji/BigDataViewer,
QuPath, vizarr, and the validators).

A converted store looks like:

```
image.ome.zarr/
├── .zgroup            # marks this as a Zarr group (format v2)
├── .zattrs            # OME-NGFF metadata: "multiscales" + "omero"
└── 0/                 # resolution level 0 (full resolution)
    ├── .zarray        # array spec: shape, chunks, dtype, compressor
    ├── 0.0.0.0        # one compressed chunk (file name = chunk grid index)
    ├── 0.0.0.1
    └── ...
```

- **`.zattrs` (group level)** holds the OME metadata:
  - `multiscales`: the axes (with `type` = time/channel/space and `unit` =
    micrometer for spatial axes), and one `dataset` per resolution level with a
    `coordinateTransformations` **scale** = the physical size of one pixel along
    each axis. The XY scale doubles at each pyramid level.
  - `omero`: per-channel display hints (label, color, intensity window).
- **`0/`, `1/`, …** are the resolution levels (only `0/` exists unless you ask
  for a pyramid). Each is a Zarr array.

---

## 5. How the compression works

### 5.1 Lossless vs lossy

- **Lossless** (what this tool does): the decompressed pixels are *bit-for-bit
  identical* to the original. Nothing is thrown away; the file is smaller purely
  because redundancy is encoded compactly. Verified here — single-thread and
  multi-thread outputs are identical, and match the source TIFF exactly.
- **Lossy** (NOT used): formats like JPEG, or tricks like dropping to 8-bit or
  downscaling, achieve smaller files by discarding information. The tool never
  does this; the only knobs are *how hard* the lossless compressor works.

The pyramid levels (`--levels > 1`) are downsampled and therefore *are* a lossy
*representation* — but they are **extra** data for fast viewing; the
full-resolution level 0 always remains lossless.

### 5.2 Chunks

A Zarr array is split into a regular grid of **chunks**. Each chunk is
compressed independently and stored as its own file. Here the chunking is:

```
(t=1, c=1, z=1, y=1024, x=1024)
```

i.e. one channel × one z-slice × a 1024×1024 XY tile. A 2304×2304 plane
therefore becomes a 3×3 grid of chunk files per (channel, slice). Set the XY
tile with `--chunk-xy`.

Why chunks matter:

- **Partial reads.** A viewer that needs one z-slice, one channel, or a small
  crop reads only the overlapping chunks — not the whole image. This is the main
  performance advantage of Zarr over a flat TIFF.
- **Parallelism.** Independent chunks can be compressed (and read) concurrently.
- **Chunk size trade-off.** Bigger chunks → slightly better compression ratio
  (more context) and fewer files, but coarser partial reads and more edge
  padding when the chunk size doesn't divide the image. ~1–2 MB per chunk
  (1024² uint16 ≈ 2 MB) is a good general default. Chunk size affects total file
  size only modestly (a few %); pixel content dominates.

### 5.3 Blocks (inside a chunk)

The compressor used is **Blosc** — a *meta-compressor* that wraps a codec
(here **zstd**) plus an optional byte/bit filter. Blosc internally splits each
chunk into smaller **blocks** and can compress those blocks with multiple
threads. Block size is chosen automatically.

The practical consequence (see benchmarks): a single 2 MB chunk doesn't split
into enough blocks to keep many cores busy, so **Blosc's internal threading
barely helps**. Real parallelism comes from processing many chunks at once
(§5.5), not from block threading. The tool keeps Blosc-internal threads at 1 to
avoid oversubscribing cores.

### 5.4 The codec: zstd + bit-shuffle

Each chunk is encoded as:

```text
raw uint16 pixels  →  [bit-shuffle filter]  →  [zstd entropy coding]  →  bytes on disk
```

- **Bit-shuffle** reorders the bits so that the high-order bits of all pixels sit
  together, the next bits together, and so on. Microscopy backgrounds and smooth
  gradients have lots of repeated high bits, so after shuffling the stream is
  far more compressible. This is lossless (a pure reordering) and is especially
  effective on 16-bit data.
- **zstd** then finds and compactly encodes repeated patterns. The **`--clevel`**
  controls how hard zstd searches: higher = smaller file, more CPU time,
  **never** lossy. Decompression speed is essentially independent of the level.

**Decompression** is the exact inverse: read chunk file → zstd decode →
un-shuffle → original pixels. Because each chunk is self-contained, a reader can
decode just the chunks it needs.

### 5.5 Multithreading

Compression is **CPU-bound**, so the tool parallelises it with a **thread pool
over planes**: the main thread reads each YX plane from the TIFF (sequentially —
the TIFF reader is not thread-safe), then hands the *compress + write* to a
worker. Because zstd releases the Python GIL, workers genuinely run in parallel.
In-flight work is bounded to `2 × workers` planes so memory stays capped.
Control the worker count with `--threads` (default = all cores).

Distinct planes write to distinct chunks (z and c are chunked at 1), so
concurrent writes never collide. Output is **identical regardless of thread
count** (lossless and deterministic).

Decompression, by contrast, is **memory-bandwidth / overhead bound**, not
CPU-bound — zstd decode is already very fast — so multithreading reads gives
little benefit and the tool does not parallelise reading. (If you read through
**dask**, you get chunk-parallel reads for free at the framework level.)

---

## 6. Benchmarks

All measured on the project's test stack:
`Cocu_CET155_24_Phal_Cell_Calc_01_CY5, FITC, DAPI.tif` — **ZCYX, 54×3×2304×2304,
uint16, 1.72 GB uncompressed ImageJ TIFF**, on a 12-core machine, files on an
external drive. Timings with "warm cache" had the data already in the OS page
cache, isolating CPU cost from disk I/O.

### 6.1 File size (lossless)

| Setting | Output | Ratio |
|---|---|---|
| `--clevel 5` (default) | 1.19 GB | 1.44× smaller |
| `--clevel 9` | 1.17 GB | 1.47× smaller |

**Conclusion:** zstd + bit-shuffle shrinks this dense stack ~1.4×, losslessly.
Going from level 5→9 saves only ~2% more but costs much more time (§6.4). Files
with more empty background compress more.

### 6.2 Read speed: TIFF vs OME-Zarr (warm cache)

| Access pattern | TIFF | OME-Zarr | Winner |
|---|---|---|---|
| Full stack → RAM | 5.45 s | 7.78 s | TIFF ~1.4× |
| One z-plane, one channel (2304²) | 30.5 ms | 11.2 ms | Zarr ~2.7× |
| 256×256 crop | 4.7 ms | 5.2 ms | ~tie |

**Conclusion:** uncompressed TIFF wins *bulk* reads (no decode cost — basically a
memcpy). OME-Zarr wins *selective* reads (one slice/channel/region touches only
the relevant chunks). On slow/remote storage the full-read gap narrows because
Zarr reads ~1.4× fewer bytes off disk.

### 6.3 Decompression threading

| Approach | 1 thread | 12 threads |
|---|---|---|
| Blosc internal threads | 5.57 s | 5.46 s (no gain) |
| Chunk-level thread pool | 4.14 s | ~3.4 s (~1.2×, no scaling past 2) |

**Conclusion:** decompression is memory-bandwidth/overhead bound, not CPU-bound.
Threading reads barely helps, so the tool does not parallelise reading.

### 6.4 Compression threading (the win)

Data already in RAM, isolating compression cost:

| | sequential | pool(4) | pool(8) | pool(12) |
|---|---|---|---|---|
| `--clevel 5` | 34.2 s | **12.1 s (2.8×)** | 13.7 s | 14.9 s |
| `--clevel 9` | 200.5 s | 113.4 s | **109.2 s (1.8×)** | 109.1 s |

Full pipeline (including TIFF read), default clevel 5:
`--threads 1` → **36.5 s**, `--threads 6` → **12.0 s (~3×)**.

**Conclusions:**

- Chunk-level parallel compression scales well; **Blosc-internal threading does
  not** (§5.3). The tool uses the former.
- **Sweet spot ≈ 4–8 workers.** Beyond that, memory bandwidth caps gains.
- Higher `--clevel` is more CPU-heavy per chunk, so it scales a bit worse
  (1.8× vs 2.8×) but benefits most in absolute seconds saved.

---

## 7. Choosing settings

- **Default (`--clevel 5`, no pyramid, all cores):** good balance — ~1.4× smaller,
  fast, lossless.
- **Minimise size:** `--clevel 9`. Expect only a few % smaller than level 5 on
  dense data, at several× the time; pair with `--threads 6`–`8`.
- **Interactive viewing / very large images:** add `--levels 3`–`4` for a
  pyramid (smooth zoom in napari/Fiji), accepting ~⅓ more data before
  compression.
- **Tiled ML / patch pipelines:** keep `--chunk-xy` near your patch size so each
  patch read touches few chunks; read via dask for parallelism.
- **Throughput on many files:** `--threads 4`–`8` per file; don't exceed core
  count across concurrent jobs.

---

## 8. Reading the output back

```python
import zarr
g = zarr.open_group("image.ome.zarr", mode="r")
arr = g["0"]                      # full-resolution level, axis order c,z,y,x (here)
plane = arr[1, 27]               # channel 1, z-slice 27 -> 2304x2304, decodes ~9 chunks
crop  = arr[1, 27, 1000:1256, 1000:1256]   # decodes ~1 chunk

print(g.attrs["multiscales"])    # axes, scales (physical pixel sizes), levels
print(g.attrs.get("omero"))      # channel names + display colors
```

napari: `napari image.ome.zarr` (with the napari-ome-zarr plugin), or
`File → Open` the folder. Fiji: *Plugins → BigDataViewer → OME-Zarr*, or the
*HCS/OME-Zarr* readers.

---

## 9. Summary

- OME-Zarr = chunked + compressed + self-describing. Smaller files and fast
  partial access, at the cost of CPU on read.
- This tool writes **lossless** OME-NGFF v0.4 with **zstd + bit-shuffle**,
  auto-detecting axes/scale/channels and streaming to bound memory.
- **Compression** is parallelised over planes (~3× on 6 workers); **decompression**
  is not CPU-bound and isn't parallelised.
- Tune with `--clevel` (size vs time), `--threads` (4–8), `--chunk-xy` (access
  granularity), and `--levels` (pyramid for viewing).
