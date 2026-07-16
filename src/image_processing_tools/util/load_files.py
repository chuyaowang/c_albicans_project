from pathlib import Path
import logging
from typing import List, Optional, Union
import tifffile
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def find_files_by_pattern(search_paths: Union[str, Path, List[Union[str, Path]]], file_pattern: str, verbose: bool = False) -> List[Path]:
    """
    Finds files matching a specific pattern within one or more search paths.

    Args:
        search_paths (Union[str, Path, List[Union[str, Path]]]): A single search path or a list of paths.
        file_pattern (str): The glob pattern to match files (e.g., '*.tif').
        verbose (bool): If True, prints the found files.

    Returns:
        List[Path]: A sorted list of Path objects for the found files.
    """
    if not isinstance(search_paths, list):
        search_paths = [search_paths]

    all_files = []
    for p in search_paths:
        base_path = Path(p).expanduser()
        if base_path.is_dir():
            files = sorted(base_path.glob(file_pattern))
            all_files.extend(files)
            if verbose:
                print(f"Found {len(files)} files in {base_path}:")
                for f in files:
                    print(f"  - {f.name}")
    
    if not all_files:
        logger.warning(f"No files found for pattern '{file_pattern}' in paths: {search_paths}")

    return all_files

def find_dapi_channel_file(file_list: List[Path]) -> Optional[int]:
    """
    Identifies the DAPI channel file from a list of TIFF files and returns its index.

    It assumes that the channel names are listed at the end of the filename, separated by commas,
    and that the corresponding file contains a 'C<n>' identifier, where 'n' is the 1-based index
    of 'DAPI' in the list.

    Args:
        file_list (List[Path]): A list of Path objects for the image set.

    Returns:
        Optional[int]: The index of the DAPI file in the list, or None if not found.
    """
    if not file_list:
        return None

    ref_filename = file_list[0].name
    try:
        channel_list_str = ref_filename.rsplit('_', 1)[-1].rsplit('.', 1)[0]
        channels = [ch.strip().upper() for ch in channel_list_str.split(',')]
        dapi_channel_number = channels.index('DAPI') + 1
        dapi_identifier = f"C{dapi_channel_number}"
    except (IndexError, ValueError):
        logger.error(f"Could not determine DAPI channel from filename: {ref_filename}")
        return None

    for i, file_path in enumerate(file_list):
        if dapi_identifier in file_path.name:
            logger.info(f"Found DAPI channel file: {file_path.name}")
            return i

    logger.warning(f"DAPI file with identifier '{dapi_identifier}' not found in the list.")
    return None

def load_tiff_stack(file_path: Path) -> np.ndarray:
    """
    Loads a multi-page TIFF microscopy stack into a numpy array.

    Args:
        file_path (Path): A pathlib.Path object pointing to the TIFF file.

    Returns:
        np.ndarray: A numpy array containing the image data. 
                    Shape is typically (frames, height, width) for stacks,
                    or (frames, channels, height, width) depending on the file.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        ValueError: If the file is not a valid file.
    """
    # Ensure the input is a Path object
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    if not file_path.is_file():
        raise ValueError(f"The path {file_path} is not a file.")

    try:
        # tifffile.imread handles Path objects directly and is optimized 
        # for microscopy data (ImageJ TIFFs, OME-TIFFs, etc.)
        image_stack = tifffile.imread(file_path)
        return image_stack
    except Exception as e:
        raise RuntimeError(f"Failed to load TIFF file: {e}")

def load_image_data(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Loads an image from a file path or returns the array if already numpy.
    Converts to grayscale if 3D array.
    """
    if isinstance(source, (str, Path)):
        source = str(source)
        img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {source}")
        return img
    elif isinstance(source, np.ndarray):
        if source.ndim == 3:
            return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        return source
    else:
        raise TypeError("Input must be a file path or a numpy array.")