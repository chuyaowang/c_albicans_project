import numpy as np
from tifffile import TiffFile, imwrite
from pathlib import Path
import cv2
import sys
import argparse
from typing import Dict, Optional, Tuple

def split_channels(
    input_tif_path: Path,
    output_dir: Path,
    num_channels: int,
    save_split_channels: bool = False,
    output_dtype: np.dtype = np.uint16
) -> Optional[Dict[int, np.ndarray]]:
    """
    Splits a multi-channel, multi-z-stack TIFF file into separate Z-stacks per channel.

    Args:
        input_tif_path (Path): Path to the input multi-channel, multi-z-stack TIFF file.
        output_dir (Path): Directory to save the split channel files.
        num_channels (int): The number of channels in the input TIFF file.
        save_split_channels (bool): If True, saves each channel's Z-stack to a file.
        output_dtype (np.dtype): Data type for saving output files.

    Returns:
        Optional[Dict[int, np.ndarray]]: A dictionary mapping channel index to its
                                         Z-stack numpy array, or None on failure.
    """
    print(f"--- Splitting Channels for {input_tif_path.name} ---")

    try:
        with TiffFile(input_tif_path) as tif:
            data = tif.series[0].asarray()
            axes = tif.series[0].axes
            print(f"  Detected TIFF axes: {axes}, with shape: {data.shape}, and dtype: {data.dtype}")

            if data.ndim == 4 and axes.upper() == 'CZYX':
                print("  Interpreting data as CZYX format.")
                if data.shape[0] != num_channels:
                    print(f"  Error: Number of channels in file ({data.shape[0]}) does not match user-provided value ({num_channels}).")
                    return None
                data = data.transpose((1, 0, 2, 3))

            elif data.ndim == 4 and axes.upper() == 'ZCYX':
                print("  Interpreting data as ZCYX format.")
                if data.shape[1] != num_channels:
                    print(f"  Error: Number of channels in file ({data.shape[1]}) does not match user-provided value ({num_channels}).")
                    return None

            elif data.ndim == 3:
                print("  Interpreting data as a flat 3D stack (Z*C, H, W).")
                num_pages, height, width = data.shape
                if num_pages % num_channels != 0:
                    print(f"  Error: Total pages ({num_pages}) is not divisible by the number of channels ({num_channels}).")
                    return None
                num_z_stacks = num_pages // num_channels
                print(f"  Inferred {num_z_stacks} Z-stacks for {num_channels} channels.")
                data = data.reshape(num_z_stacks, num_channels, height, width)
            else:
                print(f"  Error: Unsupported dimension order '{axes}' with {data.ndim} dimensions.")
                return None

            channel_stacks = {i: data[:, i, :, :] for i in range(num_channels)}

            if save_split_channels:
                base_name = input_tif_path.stem
                for i, stack in channel_stacks.items():
                    output_filename = f"C{i+1}_{base_name}.tif"
                    output_path = output_dir / output_filename
                    imwrite(output_path, stack.astype(output_dtype))
                    print(f"  -> Saved split channel {i+1} to {output_path}")

            return channel_stacks

    except Exception as e:
        print(f"An error occurred while processing {input_tif_path}: {e}")
        return None

def filter_outliers(image: np.ndarray, percentile: float) -> np.ndarray:
    """
    Clips image pixel values to the given percentile range to remove outliers.

    Args:
        image (np.ndarray): The input 2D image.
        percentile (float): The percentage of darkest and brightest pixels to ignore.

    Returns:
        np.ndarray: The image with outlier pixels clipped.
    """
    if percentile > 0 and image.size > 0:
        min_val, max_val = np.percentile(image, (percentile, 100 - percentile))
        return np.clip(image, min_val, max_val)
    return image

def project_stack(
    channel_stack: np.ndarray,
    projection_mode: str = 'max',
    sharpness_method: str | int = 'laplacian',
    outlier_percentile: float = 0.35,
    slice_denominator: int = 3
) -> Optional[Tuple[np.ndarray, Optional[int]]]:
    """
    Computes a 2D projection from a Z-stack.

    Args:
        channel_stack (np.ndarray): The input Z-stack (Z, H, W).
        projection_mode (str): 'max' or 'sharpest'.
        sharpness_method (str | int): 'laplacian', 'normalized_variance', or 'combined' for sharpest mode,
                                       or an integer to specify the slice index manually.
        outlier_percentile (float): The percentage of pixels to ignore as outliers before sharpness calculation.
        slice_denominator (int): Determines the fraction of middle slices to use (e.g., 3 for middle 1/3).
                                 Set to 1 to use all slices.

    Returns:
        Optional[Tuple[np.ndarray, Optional[int]]]: A tuple containing:
            - The 2D projected image.
            - The absolute index of the sharpest slice (if applicable), otherwise None.
        Returns None on failure.
    """
    if channel_stack.ndim != 3 or channel_stack.shape[0] == 0:
        print("  Error: Invalid channel stack for projection.")
        return None

    # --- Determine which slices to use for projection ---
    num_slices = channel_stack.shape[0]
    if slice_denominator > 1 and num_slices > slice_denominator:
        # Use a fraction of the middle slices
        slices_to_use = num_slices // slice_denominator
        start_index = (num_slices - slices_to_use) // 2
        end_index = start_index + slices_to_use
        processing_stack = channel_stack[start_index:end_index]
        print(f"    -> Using middle 1/{slice_denominator} of slices (index {start_index} to {end_index-1}) for projection...")
    else:
        # Use all slices
        start_index, end_index = 0, num_slices
        processing_stack = channel_stack
        print(f"    -> Using all {num_slices} slices for projection...")

    if projection_mode == 'max':
        return np.max(processing_stack, axis=0), None

    elif projection_mode == 'sharpest':
        if isinstance(sharpness_method, int):
            manual_index = sharpness_method
            if 0 <= manual_index < num_slices:
                print(f"    -> Using manually specified sharpest slice at absolute index {manual_index}")
                return channel_stack[manual_index], manual_index
            else:
                print(f"  Error: Manual slice index {manual_index} is out of bounds for stack with {num_slices} slices.")
                return None
        sharpest_slice_index = -1
        max_sharpness = -1
        for z_index, slice_img in enumerate(processing_stack):
            # Filter outliers before calculating sharpness for more robust results
            filtered_slice = filter_outliers(slice_img, outlier_percentile)

            sharpness = 0
            if sharpness_method == 'laplacian':
                # Variance of Laplacian: Measures high-frequency edges.
                slice_float = filtered_slice.astype(np.float32)
                sharpness = cv2.Laplacian(slice_float, cv2.CV_32F).var()
            elif sharpness_method == 'normalized_variance':
                # Normalized Variance: Measures contrast relative to brightness.
                slice_float = filtered_slice.astype(np.float32)
                mean = np.mean(slice_float)
                if mean > 0:
                    sharpness = np.std(slice_float) / mean
            elif sharpness_method == 'combined':
                # Combined metric: normalized_variance * mean(|Laplacian|)
                slice_float = filtered_slice.astype(np.float32)
                mean = np.mean(slice_float)
                if mean > 0:
                    # Normalized variance is (std/mean)^2, or var/mean^2
                    normalized_var = np.var(slice_float) / (mean**2) # Using var() is faster than std()
                    # Mean of absolute Laplacian
                    lap = cv2.Laplacian(slice_float, cv2.CV_32F)
                    mean_abs_lap = np.mean(np.abs(lap))
                    # Combine the two metrics
                    sharpness = normalized_var * mean_abs_lap

            if sharpness > max_sharpness:
                max_sharpness = sharpness
                sharpest_slice_index = z_index

        if sharpest_slice_index != -1:
            absolute_slice_index = start_index + sharpest_slice_index
            print(f"    -> Sharpest slice found at relative index {sharpest_slice_index} (absolute: {absolute_slice_index}) with score {max_sharpness:.2f}")
            return processing_stack[sharpest_slice_index], absolute_slice_index
        else:
            print("  Warning: Could not determine sharpest slice, returning None.")
            return None
    else:
        print(f"  Error: Unknown projection_mode '{projection_mode}'.")
        return None

def subtract_adjacent_slices(
    channel_stack: np.ndarray,
    sharpness_method: str | int = 'laplacian',
    subtraction_method: str = 'direct',
    slice_offset: int = 1,
    outlier_percentile: float = 0.35,
    slice_denominator: int = 3
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Finds the sharpest slice and performs a subtraction using adjacent slices.

    Args:
        channel_stack (np.ndarray): The input Z-stack (Z, H, W).
        sharpness_method (str | int): The focus metric to find the sharpest slice, or an integer
                                       to specify the slice index manually.
        subtraction_method (str): 'direct' (slice_above - slice_below) or
                                  'average' (sharpest - 0.5*(above+below)).
        slice_offset (int): The number of slices to go above and below the sharpest slice for subtraction.
        outlier_percentile (float): The percentage of pixels to ignore as outliers.
        slice_denominator (int): Determines the fraction of middle slices to use for finding the sharpest slice.
                                 Set to 1 to use all slices.

    Returns:
        Optional[Tuple[np.ndarray, int]]: A tuple containing:
            - The subtracted image (as float32).
            - The absolute index of the center (sharpest) slice.
        Returns None on failure or if the sharpest slice is at an edge.
    """
    # First, find the sharpest slice to determine the center point
    sharpest_result = project_stack(channel_stack, 'sharpest', sharpness_method, outlier_percentile, slice_denominator)

    if sharpest_result and sharpest_result[1] is not None:
        sharpest_slice, slice_idx = sharpest_result
        num_slices = channel_stack.shape[0]
        # Ensure we are not at the very edge of the stack
        if slice_idx < slice_offset or slice_idx >= num_slices - slice_offset:
            print(f"  Warning: Cannot perform subtraction. The sharpest slice is at index {slice_idx}, "
                  f"which is too close to the edge for an offset of {slice_offset}. "
                  f"The stack has {num_slices} slices (indices 0 to {num_slices-1}).")
            print("           Please consider using a smaller --slice_offset value.")
            return None

        slice_above = channel_stack[slice_idx + slice_offset].astype(np.float32)
        slice_below = channel_stack[slice_idx - slice_offset].astype(np.float32)

        if subtraction_method == 'direct':
            # Original method: slice_above - slice_below
            subtracted_image = cv2.subtract(slice_above, slice_below)
        elif subtraction_method == 'average':
            # New method: sharpest - 0.5 * (above + below)
            sharpest_slice_float = sharpest_slice.astype(np.float32)
            subtracted_image = sharpest_slice_float - 0.5 * (slice_above + slice_below)
        else:
            print(f"  Error: Unknown subtraction_method '{subtraction_method}'.")
            return None
        return subtracted_image, slice_idx
    
    return None

def normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    """
    Normalizes a float array with potentially negative values to the uint16 range.

    Args:
        image (np.ndarray): The input float image.

    Returns:
        np.ndarray: The image normalized and cast to uint16.
    """
    min_val = image.min()
    max_val = image.max()
    
    if max_val > min_val:
        # Rescale the image from [min_val, max_val] to [0, 65535]
        normalized_image = 65535 * (image - min_val) / (max_val - min_val)
    else:
        # Handle the case of a flat image (all pixels the same)
        normalized_image = np.zeros_like(image)
        
    return normalized_image.astype(np.uint16)

def main():
    parser = argparse.ArgumentParser(
        description="Split multi-channel TIFFs and/or compute Z-projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_path", type=Path,
        help="Path to the input TIFF file or a directory containing TIFF files."
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, default=None,
        help="Directory to save output files. Defaults to a folder named after the input file."
    )
    parser.add_argument(
        "-c", "--num_channels", type=int, required=True,
        help="The number of channels in the TIFF file."
    )
    parser.add_argument(
        "--sharpness_method", nargs='+', default=['normalized_variance'],
        help="The sharpness metric for sharpest projection. Can be one of: 'laplacian', 'normalized_variance', 'combined'. "
             "Alternatively, provide a list of integers to manually specify the sharpest slice index for each channel (e.g., 21 25 31)."
    )
    parser.add_argument(
        "--subtraction_method", type=str, default='direct', choices=['direct', 'average'],
        help="The subtraction method to use. 'direct' is (slice_above - slice_below). 'average' is (sharpest - 0.5*(above+below))."
    )
    parser.add_argument(
        "--slice_offset", type=int, default=1,
        help="Number of slices to offset from the sharpest slice for subtraction."
    )
    parser.add_argument(
        "--outlier_percentile", type=float, default=0.35,
        help="Percentage of darkest/brightest pixels to ignore for robust sharpness calculation. Set to 0 to disable."
    )
    parser.add_argument(
        "--projection_slice_denominator", type=int, default=3,
        help="Denominator for the fraction of middle slices to use for projections (e.g., 3 for 1/3, 4 for 1/4). Set to 1 to use all slices."
    )
    parser.add_argument(
        '--run-split', action='store_true',
        help="Run channel splitting and save the resulting Z-stacks."
    )
    parser.add_argument(
        '--run-max', action='store_true',
        help="Run MAX Z-projection and save the resulting 2D images."
    )
    parser.add_argument(
        '--run-sharpest', action='store_true',
        help="Run SHARPEST slice Z-projection and save the resulting 2D images."
    )
    parser.add_argument(
        '--run-subtract', action='store_true',
        help="Run SHARPEST slice subtraction (slice_above - slice_below) and save the result."
    )

    args = parser.parse_args()

    # --- Validate sharpness_method argument ---
    is_manual_sharpness = False
    sharpness_methods = []
    try:
        # Check if all inputs are integers for manual mode
        sharpness_methods = [int(i) - 1 for i in args.sharpness_method] # Convert from 1-based to 0-based index
        is_manual_sharpness = True
        if len(sharpness_methods) != args.num_channels:
            print(f"Error: The number of manual slice indices ({len(sharpness_methods)}) must match the number of channels ({args.num_channels}).")
            sys.exit(1)
        print(f"Using manually specified 1-based sharpest slice indices: {[i+1 for i in sharpness_methods]}")
    except ValueError:
        # Otherwise, treat as a single string method
        if len(args.sharpness_method) > 1:
            print(f"Error: --sharpness_method must be a single method name (e.g., 'laplacian') or a list of integer indices.")
            sys.exit(1)
        sharpness_methods = args.sharpness_method[0]

    # If no specific action is requested, default to running all
    run_split = args.run_split
    run_max = args.run_max
    run_sharpest = args.run_sharpest
    run_subtract = args.run_subtract
    if not any([run_split, run_max, run_sharpest, run_subtract]):
        print("No specific action requested. Defaulting to run all: split, max, sharpest, and subtract.")
        run_split = True
        run_max = True
        run_sharpest = True
        run_subtract = True

    if not args.input_path.exists():
        print(f"Error: The specified input path does not exist: {args.input_path}")
        sys.exit(1)

    # --- Discover files to process ---
    files_to_process = []
    if args.input_path.is_dir():
        print(f"Input is a directory. Searching for TIFF files in: {args.input_path}")
        files_to_process = sorted([p for p in args.input_path.glob('*') if p.suffix.lower() in ('.tif', '.tiff')])
        if not files_to_process:
            print("No TIFF files found in the directory.")
            sys.exit(0)
        print(f"Found {len(files_to_process)} files to process.")
    elif args.input_path.is_file():
        files_to_process.append(args.input_path)
    
    # --- Process each file ---
    for i, file_path in enumerate(files_to_process):
        print(f"\n{'='*20} Processing file {i+1}/{len(files_to_process)}: {file_path.name} {'='*20}")

        # Determine output directory for the current file
        if args.output_dir:
            # If an output dir name is given, create it in the same parent dir as the input file.
            current_output_dir = file_path.parent / args.output_dir
        else:
            # Default: create a folder named after the file, in the same directory.
            current_output_dir = file_path.parent / file_path.stem
        
        current_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output for this file will be saved to: {current_output_dir}")

        # --- Main Logic per file ---
        channel_stacks = None
        # Only run projections if at least one is requested
        run_any_projection = run_max or run_sharpest or run_subtract
        
        # Decide whether to perform a fresh split from the source file.
        perform_fresh_split = False
        if run_split:
            all_splits_exist = all((current_output_dir / f"C{ch_idx+1}_{file_path.stem}.tif").exists() for ch_idx in range(args.num_channels))
            if all_splits_exist:
                print(f"Split files for {file_path.name} already exist in {current_output_dir}.")
                overwrite = input("Do you want to overwrite them? [y/n]: ").lower().strip()
                if overwrite == 'y':
                    print("Overwriting existing split files...")
                    perform_fresh_split = True
                else:
                    print("Skipping overwrite. Existing split files will be used for projections if needed.")
            else:
                # Files don't exist, so we must split.
                perform_fresh_split = True

        if perform_fresh_split:
            print("Splitting channels from source file and saving...")
            channel_stacks = split_channels(
                file_path, current_output_dir, args.num_channels, save_split_channels=True
            )

        # If stacks haven't been loaded yet and projections are needed, load them now.
        if channel_stacks is None and run_any_projection:
            try:
                print("Checking for existing split channel files...")
                channel_stacks = {
                    i: TiffFile(current_output_dir / f"C{i+1}_{file_path.stem}.tif").asarray() for i in range(args.num_channels)
                }
                print("Found existing split files. Loading them directly.")
            except FileNotFoundError:
                print("Existing split files not found or incomplete. Splitting from source.")
                # If loading fails, we will split in-memory without saving.
                channel_stacks = split_channels(file_path, current_output_dir, args.num_channels, save_split_channels=False)

        if channel_stacks is None:
            print(f"Could not obtain channel stacks for {file_path.name}. Skipping.")
            continue

        # --- Run Projections ---
        projection_tasks = []
        if run_max:
            projection_tasks.append(('max', 'MAX', None))
        if run_sharpest:
            projection_tasks.append(('sharpest', 'SHARPEST', sharpness_methods))

        # --- Run Subtraction (handled separately as it needs the full stack) ---
        if run_subtract:
            print(f"\n--- Running Slice Subtraction ---")
            for ch_idx, stack in channel_stacks.items():
                print(f"  Processing Channel {ch_idx+1}...")
                current_sharpness_method = sharpness_methods[ch_idx] if is_manual_sharpness else sharpness_methods
                subtraction_result = subtract_adjacent_slices(stack, current_sharpness_method, args.subtraction_method, args.slice_offset, args.outlier_percentile, args.projection_slice_denominator)

                if subtraction_result:
                    subtracted_image, slice_idx = subtraction_result
                    # Save the result
                    method_desc = "manual" if is_manual_sharpness else current_sharpness_method
                    prefix = (f"SUBTRACT-{args.subtraction_method}-{method_desc}-"
                              f"S{slice_idx+1}-O{args.slice_offset}")

                    # Normalize the float image (with negative values) to uint16 range for proper saving
                    saveable_image = normalize_to_uint16(subtracted_image)
                    output_filename = f"{prefix}_C{ch_idx+1}_{file_path.stem}.tif"
                    output_path = current_output_dir / output_filename
                    imwrite(output_path, saveable_image)
                    print(f"    -> Saved subtracted image to {output_path}")

        for mode, prefix, method_arg in projection_tasks:
            print(f"\n--- Projecting Channels (Mode: {mode}) ---")
            for ch_idx, stack in channel_stacks.items():
                print(f"  Processing Channel {ch_idx+1}...")
                
                current_method = None
                if mode == 'sharpest':
                    current_method = method_arg[ch_idx] if is_manual_sharpness else method_arg
                
                projection_result = project_stack(stack, mode, current_method, args.outlier_percentile, args.projection_slice_denominator)

                if projection_result is not None:
                    projected_image, slice_idx = projection_result
                    
                    filename_prefix = prefix # Start with base prefix ('MAX' or 'SHARPEST')
                    if mode == 'sharpest' and slice_idx is not None:
                        # Construct a descriptive prefix for the sharpness method used
                        if is_manual_sharpness:
                            filename_prefix = f"{prefix}-manual-S{slice_idx+1}"
                        else:
                            filename_prefix = f"{prefix}-{current_method}-S{slice_idx+1}"

                    output_filename = f"{filename_prefix}_C{ch_idx+1}_{file_path.stem}.tif"
                    output_path = current_output_dir / output_filename
                    imwrite(output_path, projected_image.astype(np.uint16))
                    print(f"    -> Saved projected image to {output_path}")

if __name__ == '__main__':
    main()
    print("\nProcessing complete.")