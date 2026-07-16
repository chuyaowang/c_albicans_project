import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import tifffile
from scipy.ndimage import find_objects, shift
import re
import os

from image_processing_tools.util.load_files import find_dapi_channel_file

logger = logging.getLogger(__name__)


def _sum_and_stretch(images: List[np.ndarray], dtype: np.dtype) -> np.ndarray:
    """Combine images into one and min-max stretch it to the full range of `dtype`.

    Summing is accumulated in float64 so saturated inputs cannot overflow. Note
    that summing and averaging are interchangeable here: min-max is invariant to
    a positive scale factor and sum == n * mean, so the two give identical
    output. The stretch is the part that matters -- combining channels compresses
    everything toward mid-range, and this restores the contrast.

    Callers are expected to hand in channels that have already been through
    `_get_high_contrast_16bit` (i.e. percentile-clipped), so the min/max here are
    not driven by hot pixels.
    """
    summed = np.sum(np.stack(images, axis=0).astype(np.float64), axis=0)
    min_val, max_val = summed.min(), summed.max()
    if max_val > min_val:
        summed = (summed - min_val) / (max_val - min_val) * np.iinfo(dtype).max
    else:
        summed = np.zeros_like(summed)
    return summed.astype(dtype)


class _SingleChannel:
    """
    Private helper class to manage a single image file.
    Handles lazy loading and preprocessing for one channel.
    """

    def __init__(self, image_path: Path, config: Dict[str, Any], is_label: bool = False):
        self.path = image_path
        self.config = config
        self.is_label = is_label
        self.proc_config = self.config.get("preprocessing", {})
        self._image_16bit: np.ndarray | None = None
        self._image_8bit: np.ndarray | None = None
        self._resized_8bit: np.ndarray | None = None
        self._resized_16bit: np.ndarray | None = None
        self.scale_factor = 1.0

    @property
    def image_16bit(self) -> np.ndarray:
        if self._image_16bit is None:
            logger.debug(f"Lazily loading image from {self.path}")
            img = tifffile.imread(str(self.path))
            if img is None:
                raise IOError(f"Failed to load image from {self.path}")
            self._image_16bit = img.astype(np.uint16)

            dic_shift = self.proc_config.get("correct_DIC_shift") or (0, 0)
            if sum(dic_shift) != 0 and "DIC" in self.path.name.upper():
                if isinstance(dic_shift, bool):
                    raise TypeError(f"Invalid `correct_DIC_shift` value: {dic_shift}. Expected a list of dimensions.")
                if self._image_16bit.ndim == 3:
                    # Use last 2 elements as [dy, dx] — works for both [dy, dx] and [dz, dy, dx] inputs
                    spatial_shift = list(dic_shift)[-2:]
                    logger.info(f"Correcting DIC shift for {self.path.name} per slice with shift {spatial_shift}")
                    for z in range(self._image_16bit.shape[0]):
                        self._image_16bit[z] = shift(self._image_16bit[z], shift=spatial_shift, mode='nearest')
                else:
                    logger.info(f"Correcting DIC shift for {self.path.name} with shift {list(dic_shift)}")
                    self._image_16bit = shift(self._image_16bit, shift=list(dic_shift), mode='nearest')
        return self._image_16bit

    def _get_high_contrast_16bit(self) -> np.ndarray:
        outlier_percentile = self.proc_config.get("outlier_percentile", 0.35)
        if outlier_percentile <= 0:
            return self.image_16bit

        img = self.image_16bit.copy().astype(np.float32)
        min_val, max_val = np.percentile(img, (outlier_percentile, 100 - outlier_percentile))

        if max_val > min_val:
            img = np.clip(img, min_val, max_val)
            img = ((img - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        else:
            img = np.zeros(img.shape, dtype=np.uint16)
        return img

    @property
    def image_8bit(self) -> np.ndarray:
        if self._image_8bit is None:
            img_16bit_hc = self._get_high_contrast_16bit()
            self._image_8bit = (img_16bit_hc / 257).astype(np.uint8)
        return self._image_8bit

    @property
    def resized_8bit(self) -> np.ndarray:
        if self._resized_8bit is None:
            image = self.image_8bit
            if self.proc_config.get("resize_image", True):
                max_dim = self.proc_config.get("max_dim", 1024)
                is_3d = image.ndim == 3 and image.shape[0] > 1
                height, width = (image.shape[1], image.shape[2]) if is_3d else image.shape[:2]
                longest_edge = max(height, width)
                if longest_edge != max_dim:
                    self.scale_factor = max_dim / longest_edge
                    new_h = int(height * self.scale_factor)
                    new_w = int(width * self.scale_factor)
                    interpolation = cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_LINEAR
                    logger.info(f"Resizing 8-bit image from ({height},{width}) to ({new_h},{new_w}) (scale={self.scale_factor:.4f})")
                    if is_3d:
                        self._resized_8bit = np.stack(
                            [cv2.resize(image[z], (new_w, new_h), interpolation=interpolation) for z in range(image.shape[0])],
                            axis=0,
                        )
                    else:
                        self._resized_8bit = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
                    return self._resized_8bit
            self._resized_8bit = image
        return self._resized_8bit

    @property
    def resized_16bit(self) -> np.ndarray:
        if self._resized_16bit is None:
            image = self._get_high_contrast_16bit()
            if self.proc_config.get("resize_image", True):
                max_dim = self.proc_config.get("max_dim", 1024)
                is_3d = image.ndim == 3 and image.shape[0] > 1
                height, width = (image.shape[1], image.shape[2]) if is_3d else image.shape[:2]
                longest_edge = max(height, width)
                if longest_edge != max_dim:
                    self.scale_factor = max_dim / longest_edge
                    new_h = int(height * self.scale_factor)
                    new_w = int(width * self.scale_factor)
                    interpolation = cv2.INTER_AREA if self.scale_factor < 1.0 else cv2.INTER_LINEAR
                    logger.info(f"Resizing 16-bit image from ({height},{width}) to ({new_h},{new_w}) (scale={self.scale_factor:.4f})")
                    if is_3d:
                        self._resized_16bit = np.stack(
                            [cv2.resize(image[z], (new_w, new_h), interpolation=interpolation) for z in range(image.shape[0])],
                            axis=0,
                        )
                    else:
                        self._resized_16bit = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
                    return self._resized_16bit
            self._resized_16bit = image
        return self._resized_16bit

    def get_image_for_processing(self) -> np.ndarray:
        if self.is_label:
            image = self.image_16bit
            if self.proc_config.get("resize_image", True):
                max_dim = self.proc_config.get("max_dim", 1024)
                is_3d = image.ndim == 3 and image.shape[0] > 1
                height, width = (image.shape[1], image.shape[2]) if is_3d else image.shape[:2]
                if max(height, width) != max_dim:
                    scale = max_dim / max(height, width)
                    new_h, new_w = int(height * scale), int(width * scale)
                    if is_3d:
                        image = np.stack(
                            [cv2.resize(image[z], (new_w, new_h), interpolation=cv2.INTER_NEAREST) for z in range(image.shape[0])],
                            axis=0,
                        )
                    else:
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return image

        quantization = self.proc_config.get("quantization", "8bit")
        resize = self.proc_config.get("resize_image", True)
        if quantization == "8bit":
            return self.resized_8bit if resize else self.image_8bit
        elif quantization == "16bit":
            return self.resized_16bit if resize else self._get_high_contrast_16bit()
        else:
            logger.warning(f"Unknown quantization '{quantization}'. Defaulting to 8-bit.")
            return self.resized_8bit if resize else self.image_8bit

    def set_processed_image(self, image: np.ndarray):
        """Directly sets image attributes, bypassing lazy loading."""
        is_resized = self.proc_config.get("resize_image", True)
        if image.dtype == np.uint16:
            if is_resized:
                # The summed image is a resized image.
                self._resized_16bit = image
                self.scale_factor = self.proc_config.get("max_dim", 1024) / max(image.shape)
            else:
                # The summed image is a full-size image.
                self._image_16bit = image
        elif image.dtype == np.uint8:
            if is_resized:
                # The summed image is a resized image.
                self._resized_8bit = image
                self.scale_factor = self.proc_config.get("max_dim", 1024) / max(image.shape)
            else:
                # The summed image is a full-size image.
                self._image_8bit = image


class ImageContainer:
    """
    A unified class to represent and process single or multi-channel images.

    This class handles lazy loading, preprocessing, composition, and prompt generation.
    It can be initialized with a single file path for a mono-channel image or a list
    of paths for a multi-channel composite.
    """

    def __init__(self, structure: Union[Path, List[Any]], config: Dict[str, Any], is_label: bool = False):
        self.config = config
        self.is_label = is_label
        self.proc_config = self.config.get("preprocessing", {})
        self._source_paths: List[Path] = []
        
        source_path_set = set()

        if isinstance(structure, Path):
            # Base case: initialize with a single path
            self.channels: List[_SingleChannel] = [_SingleChannel(structure, config, is_label=is_label)]
            source_path_set.add(structure)
        else:
            # Handle complex structures like [[c1, c2], c3]
            processed_channels = []
            for item in structure:
                if isinstance(item, list):
                    # This is a group to be summed.
                    channels_to_sum = []
                    for sub_item in item:
                        if isinstance(sub_item, ImageContainer):
                            source_path_set.update(sub_item._source_paths)
                            channels_to_sum.extend(sub_item.channels)
                        elif isinstance(sub_item, Path):
                            source_path_set.add(sub_item)
                            channels_to_sum.append(_SingleChannel(sub_item, config, is_label=is_label))
                        else:
                            raise TypeError(f"Unsupported type in summation list: {type(sub_item)}")

                    if channels_to_sum:
                        summed_channel = self._sum_channels(channels_to_sum)
                        processed_channels.append(summed_channel)

                elif isinstance(item, ImageContainer):
                    source_path_set.update(item._source_paths)
                    processed_channels.extend(item.channels)
                elif isinstance(item, Path):
                    source_path_set.add(item)
                    processed_channels.append(_SingleChannel(item, config, is_label=is_label))
                else:
                    raise TypeError(f"Unsupported type in ImageContainer structure: {type(item)}")
            self.channels = processed_channels
        
        self._source_paths = list(source_path_set)
        # Attributes for storing generated prompts
        self.prompts: np.ndarray | None = None
        self.prompt_type: str | None = None
        self.dapi_channel_index: Optional[int] = None
        self.dapi_slice: Optional[int] = None
        self.seed_slice: Optional[int] = None

    @property
    def name(self) -> str:
        """
        Returns a descriptive name for the container.
        - If it contains one channel, it returns the channel's filename.
        - If it contains multiple channels, it creates a composite name.
        """
        # --- Helper function to get the short name (e.g., 'DAPI') from a path ---
        def get_channel_short_name(p: Path) -> str:
            try:
                # e.g., from "..._CY5,FITC,DAPI.tif" -> "CY5,FITC,DAPI"
                channel_list_str = p.name.rsplit('_', 1)[-1].rsplit('.', 1)[0]
                channels = [ch.strip() for ch in channel_list_str.split(',')]
                
                # Find which C-number this path corresponds to, e.g., "C3"
                match = re.search(r'C(\d+)', p.name)
                if match:
                    channel_num = int(match.group(1))
                    if 1 <= channel_num <= len(channels):
                        return channels[channel_num - 1] # 1-based to 0-based index
            except (IndexError, ValueError):
                pass # Fallback if parsing fails
            return p.stem # Fallback to the stem if a short name can't be parsed

        # --- Find all unique source paths that make up this container ---
        # This is now reliably populated by the constructor.
        if not self._source_paths:
            return "+".join(ch.path.stem for ch in self.channels)

        # --- Generate the name ---
        if len(self._source_paths) == 1:
            return self._source_paths[0].stem

        # Find the common base name by removing channel-specific parts (_C1_, _C2_, etc.)
        base_name = re.sub(r'_C\d_', '_', self._source_paths[0].stem)

        # --- Generate the structure part of the name (e.g., "CY5+FITC,DAPI") ---
        structure_parts = []
        for ch in self.channels:
            # The stem can be a single stem or a "+"-joined string of stems for summed channels.
            stems = ch.path.stem.split('+')
            part_names = []
            for stem in stems:
                # Find the original Path object from the complete list of source paths.
                found = False
                for p in self._source_paths:
                    if p.stem == stem:
                        part_names.append(get_channel_short_name(p))
                        found = True
                        break
                if not found:
                    # This fallback should ideally not be reached if __init__ is correct.
                    part_names.append(stem)
            structure_parts.append("+".join(part_names))
        
        structure_string = ','.join(structure_parts)
        return f"{base_name}_{structure_string}"

    def _sum_channels(self, channel_group: List[_SingleChannel]) -> _SingleChannel:
        """Sums a group of channels into a single new _SingleChannel object."""
        images_to_sum = [ch.get_image_for_processing() for ch in channel_group]
        summed_image = _sum_and_stretch(images_to_sum, images_to_sum[0].dtype)

        # Create an informative name by joining the stems of the channels being summed.
        summed_channel_name = "+".join([ch.path.stem for ch in channel_group])
        # Use the config of the first channel for the config of the summed channel.
        summed_channel = _SingleChannel(Path(summed_channel_name), channel_group[0].config)

        summed_channel.set_processed_image(summed_image)
        return summed_channel

    def __add__(self, other: "ImageContainer") -> "ImageContainer":
        """
        Combines two ImageContainer objects by creating a new container
        that holds the structure of both. For example, if `self` represents
        `[c1, c3]` and `other` represents `[c2]`, the result is equivalent to
        `ImageContainer([[c1, c3], c2], config)`.
        """
        return ImageContainer([self, other], self.config)

    def __getitem__(self, index: int) -> _SingleChannel:
        """Allows accessing individual channels by index."""
        return self.channels[index]

    def _is_3d(self) -> bool:
        """Returns True if the first channel is a multi-slice z-stack."""
        if not self.channels:
            return False
        try:
            return self.channels[0].image_16bit.ndim == 3
        except Exception:
            return False

    def _merge_3d(self) -> np.ndarray:
        """
        Merges 3D channel volumes into (Z, H, W, 3) for SAM.
        - 1 channel  → replicate × 3
        - 2 channels → sum + stretch, then replicate × 3
        - 3 channels → use as-is
        - 4+ channels → ch1, ch2 unchanged; ch3..chN summed + stretched into ch3

        Combining channels always goes through `_sum_and_stretch`, the same
        reduction `_sum_channels` uses, so a combined channel is contrast-stretched
        rather than left compressed toward mid-range.
        """
        volumes = [ch.get_image_for_processing() for ch in self.channels]
        n = len(volumes)
        dtype = volumes[0].dtype

        if n == 1:
            ch = volumes[0]
            result = np.stack([ch, ch, ch], axis=-1)
            logger.info("3D merge: 1 channel replicated to (Z, H, W, 3).")
        elif n == 2:
            merge_mode = self.config.get("preprocessing", {}).get("two_channel_merge_mode", "average_replicate")
            if merge_mode == "passthrough":
                result = np.stack(volumes, axis=-1)
                logger.info("3D merge: 2 channels passed through as (Z, H, W, 2); micro_sam will pad zero blue channel.")
            else:
                combined = _sum_and_stretch(volumes, dtype)
                result = np.stack([combined, combined, combined], axis=-1)
                logger.info("3D merge: 2 channels summed, stretched and replicated to (Z, H, W, 3).")
        elif n == 3:
            result = np.stack(volumes, axis=-1)
            logger.info("3D merge: 3 channels stacked to (Z, H, W, 3).")
        else:
            ch3 = _sum_and_stretch(volumes[2:], dtype)
            result = np.stack([volumes[0], volumes[1], ch3], axis=-1)
            logger.info(f"3D merge: {n} channels → (Z, H, W, 3), ch3..ch{n} summed and stretched into ch3.")

        return result

    def _get_dapi_slice_container(self, z: int) -> 'ImageContainer':
        """
        Builds a temporary 2D ImageContainer for a single z-slice, using the
        same config as the parent so the DAPI prompt slice is processed at the
        same resolution as the volume slices passed to inference.
        """
        temp_container = ImageContainer(self.channels[0].path, self.config, is_label=self.is_label)
        temp_channels = []
        for ch in self.channels:
            temp_ch = _SingleChannel(ch.path, self.config, is_label=self.is_label)
            temp_ch._image_16bit = ch.image_16bit[z]
            temp_channels.append(temp_ch)
        temp_container.channels = temp_channels
        temp_container.dapi_channel_index = self.dapi_channel_index
        return temp_container

    def merge(self) -> np.ndarray:
        """
        Merges the channel(s) into a single NumPy array.

        For 3D z-stacks (auto-detected from file shape): returns (Z, H, W, 3).
        For 2D images:
          - 1 channel  → (H, W)
          - >1 channels → (H, W, C) via cv2.merge

        This combines channels; it does not reduce them. To get a single 2D
        channel out of several, group them in the constructor structure so
        `_sum_channels` reduces them first -- `ImageContainer([[c1, c2]], config)`
        holds one summed channel, and merge() then returns it as (H, W).
        """
        if not self.channels:
            raise ValueError("Cannot merge an empty ImageContainer.")

        if self._is_3d():
            return self._merge_3d()

        images = [ch.get_image_for_processing() for ch in self.channels]

        if len(images) == 1:
            logger.info("Returning single channel image.")
            return images[0]

        if len(set(img.shape for img in images)) > 1:
            logger.warning("Channel images have different shapes. Resizing to the first channel's shape.")
            target_shape = images[0].shape
            images = [cv2.resize(img, (target_shape[1], target_shape[0])) if img.shape != target_shape else img for img in images]

        merged_image = cv2.merge(images)
        logger.info(f"Merged {len(self.channels)} channels into a {merged_image.shape} image.")
        return merged_image

    def _normalize_for_display(self, image: np.ndarray) -> np.ndarray:
        """Normalizes an image to a displayable format (float 0-1)."""
        if image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            return image
        
        min_val, max_val = image.min(), image.max()
        if max_val > 1.0:
            return (image - min_val) / (max_val - min_val)
        return image

    def display(self, title: str = "Image", cmap: str = "gray", ax=None):
        """Displays the merged image, handling 1, 2, or 3 channels correctly."""
        import matplotlib.pyplot as plt

        merged_image = self.merge()
        
        if merged_image.ndim == 3:
            num_ch = merged_image.shape[2]
            if num_ch == 2:
                logger.info("Displaying 2-channel image by mapping to Red and Green.")
                blue_channel = np.zeros_like(merged_image[:, :, 0])
                display_image = cv2.merge([merged_image[:, :, 0], merged_image[:, :, 1], blue_channel])
            elif num_ch == 3:
                display_image = merged_image
            else:
                logger.warning(f"Displaying first channel of {num_ch}-channel image.")
                display_image = merged_image[:, :, 0]
        else: # Grayscale
            display_image = merged_image

        # display_image = self._normalize_for_display(display_image)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            show = True
        else:
            show = False

        cmap = cmap if display_image.ndim == 2 else None
        ax.imshow(display_image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

        if show:
            plt.show()

    def generate_prompts(self):
        """
        Generates prompts (points, bboxes, or masks) for this image object
        by automatically finding the DAPI channel within its own channels.
        The results are stored in `self.prompts` and `self.prompt_type`.

        For 3D z-stacks, prompts are generated from a single seed slice.
        Set `self.seed_slice` to an integer to fix the slice; otherwise the
        middle slice (Z // 2) is used automatically and stored back to
        `self.seed_slice` for the pipeline to retrieve.

        To override automatic DAPI detection, set `self.dapi_channel_index`
        to the desired channel index before calling this method.
        """
        logger.info("Generating prompts for image container by finding DAPI channel...")

        if self.dapi_channel_index is not None:
            if not (0 <= self.dapi_channel_index < len(self.channels)):
                logger.error(
                    f"dapi_channel_index {self.dapi_channel_index} is out of range "
                    f"(container has {len(self.channels)} channel(s)). Prompt generation failed."
                )
                return
            dapi_index = self.dapi_channel_index
            logger.info(f"Using manually set dapi_channel_index: {dapi_index}")
        else:
            all_paths = [ch.path for ch in self.channels]
            dapi_index = find_dapi_channel_file(all_paths)
            if dapi_index is None:
                logger.error("Could not find DAPI channel. Prompt generation failed.")
                return

        if self._is_3d():
            z_size = self.channels[0].image_16bit.shape[0]
            if self.dapi_slice is None:
                dapi_z = z_size // 2
                logger.info(f"No dapi_slice set; using middle slice z={dapi_z} of {z_size} for prompt generation.")
            else:
                dapi_z = self.dapi_slice
                if not (0 <= dapi_z < z_size):
                    logger.error(f"dapi_slice {dapi_z} out of range for volume with {z_size} slices.")
                    return
            self.dapi_slice = dapi_z
            prompt_container = self._get_dapi_slice_container(dapi_z)
            logger.info(f"Generating prompts from DAPI slice z={dapi_z}.")
        else:
            prompt_container = self

        prompt_gen_helper = _PromptGeneratorHelper(prompt_container, dapi_index, self.config)
        self.prompts, self.prompt_type = prompt_gen_helper.get_prompts()
        logger.info(f"Generated and stored '{self.prompt_type}' prompts.")


class _PromptGeneratorHelper:
    """
    Internal helper class to contain the logic for generating prompts.
    This is instantiated and used by `ImageContainer.generate_prompts`.
    """

    def __init__(
        self,
        image_container: ImageContainer,
        dapi_channel_index: int,
        config: Dict[str, Any],
    ):
        self.image_container = image_container
        self.config = config
        self.prompt_config = config.get("prompting", {})
        self.proc_config = config.get("preprocessing", {})
        self.dapi_channel = image_container[dapi_channel_index]
        
        self.cell_centers, self.nucleus_markers = self._find_cell_centers()
        self.median_radius = self._calculate_median_radius(self.nucleus_markers)

    def _get_image_for_prompting(self, channel: _SingleChannel) -> np.ndarray:
        """
        Gets the appropriate image from a channel (resized or full-size) based on the config
        and ensures it is uint8. Most OpenCV functions for thresholding and morphological
        operations, which are used in prompt generation, work best with 8-bit images.
        """
        resize = self.proc_config.get("resize_image", True)
        quantization = self.proc_config.get("quantization", "8bit")

        if quantization == "16bit":
            img_16bit = channel.resized_16bit if resize else channel._get_high_contrast_16bit()
            return (img_16bit / 257).astype(np.uint8)
        else:  # Default to 8bit
            return channel.resized_8bit if resize else channel.image_8bit

    def _find_cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Finds cell centroids from the DAPI channel using a watershed algorithm."""
        logger.info("Finding cell centers from DAPI channel...")
        dapi_8bit = self._get_image_for_prompting(self.dapi_channel)
        _, binary_mask = cv2.threshold(dapi_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        min_area = self.prompt_config.get("min_mask_area", 100)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        cleaned_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255

        # Watershed algorithm
        dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cv2.dilate(cleaned_mask, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0

        watershed_source = cv2.cvtColor(dapi_8bit, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_source, markers)

        centroids = []
        for label in np.unique(markers):
            if label <= 1: continue
            label_mask = np.zeros(markers.shape, dtype="uint8")
            label_mask[markers == label] = 255
            M = cv2.moments(label_mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))

        logger.info(f"Found {len(centroids)} cell centers.")
        return np.array(centroids), markers

    def _calculate_median_radius(self, nucleus_markers: np.ndarray) -> float:
        """Calculates the median radius from the watershed-separated nuclei mask."""
        unique_labels = [l for l in np.unique(nucleus_markers) if l > 0]
        if not unique_labels:
            return 0.0

        areas = [np.sum(nucleus_markers == label) for label in unique_labels]
        median_radius = np.median(np.sqrt(np.array(areas) / np.pi))
        logger.debug(f"Median radius of detected DAPI blobs: {median_radius:.2f} pixels.")
        return median_radius

    def _get_seeded_watershed_mask(self) -> np.ndarray:
        """Performs a seeded watershed on the target channel(s) to get instance masks."""
        # Prepare the source image for watershed (uses maximum 3-channel)
        num_ch = len(self.image_container.channels)
        if num_ch == 1:
            watershed_source = cv2.cvtColor(self._get_image_for_prompting(self.image_container.channels[0]), cv2.COLOR_GRAY2BGR)
        elif num_ch == 2:
            ch1, ch2 = self.image_container.channels
            img1_8bit, img2_8bit = self._get_image_for_prompting(ch1), self._get_image_for_prompting(ch2)
            blue_ch = np.zeros_like(img1_8bit)
            watershed_source = cv2.merge([img1_8bit, img2_8bit, blue_ch])
        elif num_ch >= 3:
            # Use the first 3 channels
            ch1, ch2, ch3 = self.image_container.channels[:3]
            imgs_8bit = [self._get_image_for_prompting(c) for c in [ch1, ch2, ch3]]
            watershed_source = cv2.merge(imgs_8bit)

        # Create markers from DAPI centroids
        markers = np.zeros(watershed_source.shape[:2], dtype=np.int32)
        for i, (x, y) in enumerate(self.cell_centers):
            if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
                markers[y, x] = i + 1
        markers = cv2.dilate(markers.astype(np.uint8), np.ones((3,3)), iterations=2).astype(np.int32)

        # Perform watershed
        markers = cv2.watershed(watershed_source, markers)
        
        instance_mask = np.zeros_like(markers, dtype=np.uint16)
        for label in np.unique(markers):
            if label > 1:
                instance_mask[markers == label] = label - 1
        
        return instance_mask

    def _generate_point_prompts(self) -> np.ndarray:
        """Returns the DAPI-derived cell centers as point prompts."""
        logger.info(f"Generated {len(self.cell_centers)} point prompts.")
        return self.cell_centers

    def _generate_mask_prompts(self) -> np.ndarray:
        """Generates an instance segmentation mask using the seeded watershed method."""
        logger.info("Generating mask prompts via seeded watershed.")
        return self._get_seeded_watershed_mask()

    def _generate_bbox_prompts(self) -> np.ndarray:
        """Generates bounding box prompts."""
        use_seeded_watershed = self.prompt_config.get("use_seeded_watershed_for_bbox", True)

        if use_seeded_watershed:
            logger.info("Generating bounding box prompts using seeded watershed.")
            instance_mask = self._get_seeded_watershed_mask()
            if instance_mask.max() == 0:
                return np.array([])

            objects = find_objects(instance_mask)
            bboxes = []
            for i, slc in enumerate(objects):
                if slc is not None:
                    y_slice, x_slice = slc
                    bboxes.append([x_slice.start, y_slice.start, x_slice.stop, y_slice.stop])
        else:
            logger.info("Generating bounding box prompts using DAPI radius multiplier.")
            multiplier = self.prompt_config.get("bbox_radius_multiplier", 2.0)
            bboxes = []
            if self.cell_centers.size > 0 and self.median_radius > 0:
                box_half_size = int(multiplier * self.median_radius)
                for (x, y) in self.cell_centers:
                    bboxes.append([x - box_half_size, y - box_half_size, x + box_half_size, y + box_half_size])

        logger.info(f"Generated {len(bboxes)} bounding box prompts.")
        return np.array(bboxes)

    def get_prompts(self) -> tuple[np.ndarray | None, str | None]:
        """
        Generates the appropriate type of prompts based on the configuration.
        """
        prompt_mode = self.prompt_config.get("prompt_mode", "points")
        if prompt_mode == "points":
            return self._generate_point_prompts(), "points"
        elif prompt_mode == "bbox":
            return self._generate_bbox_prompts(), "bbox"
        elif prompt_mode == "mask":
            return self._generate_mask_prompts(), "mask"
        else:
            logger.warning(f"Unsupported prompt_mode: {prompt_mode}. No prompts generated.")
            return None, None
