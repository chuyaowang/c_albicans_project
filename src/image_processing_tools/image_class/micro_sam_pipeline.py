import logging
import os
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.inference import batched_inference, batched_tiled_inference
from micro_sam.instance_segmentation import (AMGBase,
                                             InstanceSegmentationWithDecoder,
                                             get_instance_segmentation_generator, get_decoder)
from micro_sam.util import get_sam_model, SamPredictor, precompute_image_embeddings, set_precomputed
from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume, merge_instance_segmentation_3d
import tifffile
from pathlib import Path
from image_processing_tools.image_class.image_container import ImageContainer
from image_processing_tools.util.visualize import (save_channel_comparison_visualization, save_multi_mask_visualization, save_segmentation_visualization_AIS)
 
# Get a logger instance for this module.
# The configuration (level, handlers, format) should be set by the application's entry point (e.g., the notebook).
logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a JSON configuration file.

    Args:
        config_path (Union[str, Path]): The path to the JSON config file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class MicroSAMPipeline:
    """
    A class to encapsulate the entire image segmentation pipeline using microSAM.

    This class handles preprocessing, prompt-based segmentation, and result visualization.

    Args:
        image_containers (Union[ImageContainer, List[ImageContainer]]): One or more
            pre-built ImageContainer objects to be processed by the pipeline.
        config (Dict[str, Any]): A dictionary containing all parameters for the pipeline.
            Required keys:
                - 'model_type' (str): The microSAM model architecture (e.g., 'vit_l_lm').
                - 'checkpoint_path' (str): Path to the microSAM model checkpoint.
                - 'base_input_dir' (str): Base directory of input images, used for structuring the output folder.

            Optional/Conditional keys:
                - 'segmentation_mode' (str): 'prompted' (default) or 'automatic'.
                - 'decoder_checkpoint_path' (str): Path to the decoder checkpoint (required for 'automatic' mode).
                - 'preprocessing' (dict): Dictionary of parameters for `preprocessing.preprocess_image`.
                - 'prompting' (dict): Dictionary of parameters for prompt generation within `preprocess_image`.
    """

    def __init__(self, image_containers: Union['ImageContainer', List['ImageContainer']], config: Dict[str, Any]):
        """
        Initializes the MicroSAMPipeline with pre-built ImageContainer objects.

        Args:
            image_containers (Union[ImageContainer, List[ImageContainer]]): One or more
                ImageContainer objects defining the images to process.
            config (Dict[str, Any]): A dictionary containing all parameters for the pipeline.
        """
        # Validate required config keys
        required_keys = ["model_type", "checkpoint_path", "base_input_dir"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Configuration dictionary must contain the key: '{key}'")

        self.config = config
        logger.info(f"Initializing MicroSAMPipeline with config: {self.config}")

        if isinstance(image_containers, ImageContainer):
            self.run_containers: List['ImageContainer'] = [image_containers]
        else:
            self.run_containers = list(image_containers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_mode = self.config.get("segmentation_mode", "prompted")

        self.predictor: Optional[SamPredictor] = None
        self.segmenter: Optional[Union[AMGBase, InstanceSegmentationWithDecoder]] = None
        self._initialize_models()
        self.model_tag = self._model_tag()

        self.results: Dict[str, Dict[str, Any]] = {}

    def _initialize_models(self):
        """Loads the SAM predictor and/or segmenter based on the configuration."""
        model_type = self.config["model_type"]
        checkpoint_path = Path(self.config["checkpoint_path"]).expanduser()
        logger.info(f"Loading model: {model_type}...")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        if self.segmentation_mode in ('automatic', 'combined'):
            decoder_path = self.config.get("decoder_checkpoint_path")
            if not decoder_path:
                msg = "`decoder_checkpoint_path` must be provided for automatic/combined segmentation."
                logger.error(msg)
                raise ValueError(msg)

            decoder_path = Path(decoder_path).expanduser()
            if not decoder_path.exists():
                msg = f"Decoder checkpoint not found at: {decoder_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)

            tiling_config = self.config.get("tiling", {})
            _tile_shape = tiling_config.get("tile_shape")
            _halo = tiling_config.get("halo")
            is_tiled = _tile_shape is not None and _halo is not None

            # Logic from get_predictor_and_segmenter
            predictor, state = get_sam_model(
                model_type=model_type, device=self.device, checkpoint_path=checkpoint_path, return_state=True
            )
            state["decoder_state"] = torch.load(str(decoder_path), map_location=self.device)

            if "decoder_state" not in state:
                raise RuntimeError("Automatic segmentation requires a model with a segmentation decoder.")

            decoder_state = state["decoder_state"]
            decoder = get_decoder(image_encoder=predictor.model.image_encoder, decoder_state=decoder_state, device=self.device)
            self.predictor = predictor
            self.segmenter = get_instance_segmentation_generator(predictor=predictor, is_tiled=is_tiled, decoder=decoder)
            logger.info(f"Models for AIS loaded on device: {self.predictor.device} (is_tiled={is_tiled})")

        elif self.segmentation_mode == 'prompted':
            self.predictor = get_sam_model(
                model_type=model_type,
                device=self.device,
                checkpoint_path=checkpoint_path
            )
            logger.info(f"Model for prompted segmentation loaded on device: {self.predictor.device}")
        else:
            msg = f"Unknown segmentation_mode: {self.segmentation_mode}"
            logger.error(msg)
            raise ValueError(msg)

        if self.predictor is None:
            raise RuntimeError("Model initialization failed.")

    def _model_tag(self) -> str:
        """Short identifier for the loaded model, used to keep embedding caches and
        saved outputs from different models apart.

        Two models never share a cache: the tag combines the checkpoint stem, the
        checkpoint file's modification date (YYYY-MM-DD), and an 8-char hash of the
        full checkpoint/decoder/model_type/mtime identity. The stem and date keep
        the tag human-readable; the hash guards against a same-day retrain that
        reuses the same filename. Without this, whichever model ran first on an
        image cached its embeddings and every later model silently reused them,
        which fed a finetuned decoder the base encoder's features.
        """
        ckpt = Path(self.config["checkpoint_path"]).expanduser()
        decoder = self.config.get("decoder_checkpoint_path")
        decoder = str(Path(decoder).expanduser()) if decoder else ""
        try:
            mtime = ckpt.stat().st_mtime
        except OSError:
            mtime = 0.0
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d") if mtime else "nodate"
        identity = f"{ckpt}|{decoder}|{self.config['model_type']}|{mtime}"
        digest = hashlib.md5(identity.encode()).hexdigest()[:8]
        return f"{ckpt.stem}_{date_str}_{digest}"

    def _get_embedding_path(self, image_container: 'ImageContainer') -> Path:
        """Returns the per-container zarr embedding path inside microsam_outputs/.

        The model tag is part of the filename so a cache is never reused across
        models. The tile shape and halo are also encoded, so different tiling
        settings (and no-tiling) each keep their own cache and never overwrite one
        another. A tiled embedding is incompatible with a differently-tiled or
        untiled run, so they must not share a filename.
        """
        tiling_config = self.config.get("tiling", {})
        tile_shape = tiling_config.get("tile_shape")
        halo = tiling_config.get("halo")
        if tile_shape is not None and halo is not None:
            th, tw = tuple(tile_shape)
            hy, hx = tuple(halo)
            suffix = f"_tile{th}x{tw}_halo{hy}x{hx}"
        else:
            suffix = ""
        return (image_container._source_paths[0].parent / "microsam_outputs"
                / f"{image_container.name}__{self.model_tag}{suffix}.zarr")

    def _rescale_to_original(self, arr: np.ndarray, container: 'ImageContainer', is_float: bool = False) -> np.ndarray:
        """
        Rescales arr from processed resolution back to original image resolution.

        Supports (H, W) and (D, H, W) inputs. Uses INTER_NEAREST for label/mask
        arrays and INTER_LINEAR for continuous float maps.
        """
        orig_hw = container.channels[0].image_16bit.shape[-2:]
        if arr.shape[-2:] == orig_hw:
            return arr
        interp = cv2.INTER_LINEAR if is_float else cv2.INTER_NEAREST
        dsize = (orig_hw[1], orig_hw[0])  # cv2 expects (W, H)
        if arr.ndim == 2:
            return cv2.resize(arr, dsize, interpolation=interp)
        return np.stack(
            [cv2.resize(arr[i], dsize, interpolation=interp) for i in range(arr.shape[0])],
            axis=0,
        )

    def _make_prompt_viz(self, result_data: dict, container: 'ImageContainer', ndim: int) -> Optional[np.ndarray]:
        """
        Creates a labeled prompt-annotation image at original resolution.

        Seed slice (or the only slice for 2D): prompt geometry drawn with per-object
        label values — stars for points, outlines for bboxes, filled regions for masks.
        All non-seed slices in 3D are left as zero so this can be used as a pure
        prompt annotation layer on top of _masks.tif.

        Returns (H, W) uint16 for 2D, (Z, H, W) uint16 for 3D, or None if unavailable.
        """
        prompts = result_data.get("prompts")
        prompt_type = result_data.get("prompt_type")
        if prompts is None or prompt_type is None:
            return None

        orig_hw = container.channels[0].image_16bit.shape[-2:]
        H, W = int(orig_hw[0]), int(orig_hw[1])
        scale_factor = container.channels[0].scale_factor
        coord_scale = 1.0 / scale_factor if scale_factor > 0 else 1.0

        canvas = np.zeros((H, W), dtype=np.uint16)

        if prompt_type == "points":
            marker_size = max(20, int(min(H, W) * 0.02))
            for obj_id, (x, y) in enumerate(prompts, start=1):
                cx = int(round(x * coord_scale))
                cy = int(round(y * coord_scale))
                cv2.drawMarker(canvas, (cx, cy), color=int(obj_id),
                               markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2)

        elif prompt_type == "bbox":
            for obj_id, (x1, y1, x2, y2) in enumerate(prompts, start=1):
                pt1 = (int(round(x1 * coord_scale)), int(round(y1 * coord_scale)))
                pt2 = (int(round(x2 * coord_scale)), int(round(y2 * coord_scale)))
                cv2.rectangle(canvas, pt1, pt2, color=int(obj_id), thickness=3)

        elif prompt_type == "mask":
            canvas = self._rescale_to_original(prompts.astype(np.uint16), container, is_float=False)

        if ndim == 3 and result_data.get("seed_slice") is not None:
            n_slices = container.channels[0].image_16bit.shape[0]
            viz = np.zeros((n_slices, H, W), dtype=np.uint16)
            viz[result_data["seed_slice"]] = canvas
            return viz

        return canvas

    def _process_single_image(self, image_container: 'ImageContainer'):
        """
        Processes a single run specification (mono-channel or composite) for prompted segmentation.

        This method orchestrates the analysis for a single ImageContainer.
        It prepares the analysis image, generates prompts, scales them, and runs inference.

        Args:
            image_container (ImageContainer): The container for the image(s) to be processed.

        Returns:
            A tuple containing the generated result key and the dictionary of result data.
        """
        file_name_key = image_container.name

        # Step 1: Prepare the final analysis image by merging the container's channels.
        # This handles resizing, quantization, and composition automatically.
        processed_image = image_container.merge()
        
        # Step 2: Generate prompts. The container handles DAPI detection and prompt creation internally.
        image_container.generate_prompts()
        prompts, prompt_type = image_container.prompts, image_container.prompt_type

        if prompts is None or prompts.size == 0:
            logger.warning(f"Skipping {file_name_key} due to prompt generation error.")
            return None

        # Save intermediate outputs if debug mode is enabled
        debug_config = self.config.get("debug_mode", {})
        if debug_config.get("save_prompts"):
            debug_dir = Path(debug_config.get("output_dir", "debug_outputs")) / "prompts"
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_path = debug_dir / f"{file_name_key}_prompts.npy"
            np.save(save_path, prompts)
            logger.debug(f"Saved prompts to {save_path}")

        # Step 3: Run batched inference with the prepared image and prompts.
        tiling_config = self.config.get("tiling", {})
        tile_shape = tiling_config.get("tile_shape")
        halo = tiling_config.get("halo")
        if tile_shape is not None:
            tile_shape = tuple(tile_shape)
        if halo is not None:
            halo = tuple(halo)
        use_tiling = tile_shape is not None and halo is not None

        if not use_tiling:
            self.predictor.set_image(processed_image)

        inference_kwargs = dict(
            predictor=self.predictor,
            image=processed_image,
            batch_size=len(prompts),
            multimasking=False,
            return_instance_segmentation=False,
        )
        if use_tiling:
            inference_kwargs.update(tile_shape=tile_shape, halo=halo)
            run_inference = batched_tiled_inference
        else:
            run_inference = batched_inference

        if prompt_type == 'points':
            mask_data = run_inference(
                **inference_kwargs,
                points=prompts[:, None, :],
                point_labels=np.ones((len(prompts), 1)),
            )
        elif prompt_type == 'bbox':
            mask_data = run_inference(**inference_kwargs, boxes=prompts)
        else:
            logger.warning(f"Prompt type '{prompt_type}' is not supported for batched inference.")
            return None

        masks = [d['segmentation'].cpu().numpy() for d in mask_data]
        scores = [d['predicted_iou'] for d in mask_data]
        logits = [np.squeeze(d['logits'].cpu().numpy()) for d in mask_data]

        result_data = {
            "processed_image": processed_image,
            "prompts": prompts,
            "prompt_type": prompt_type,
            "masks": masks,
            "scores": scores,
            "logits": logits,
        }
        return file_name_key, result_data


    def _run_ais_on_image(self, image_container: 'ImageContainer'):
        """
        Processes a single run specification (mono-channel or composite) for automatic segmentation.

        Args:
            image_container (ImageContainer): The container for the image(s) to be processed.

        Returns:
            A tuple containing the generated result key and a dictionary of result data.
            If processing fails, the result data will be None.
        """
        file_name_key = image_container.name

        # Prepare the analysis image (resizing and channel merging/replication).
        processed_image = image_container.merge()

        if processed_image is None:
            logger.warning(f"Skipping {file_name_key} due to preprocessing error.")
            return file_name_key, None

        tiling_config = self.config.get("tiling", {})
        tile_shape = tiling_config.get("tile_shape")
        halo = tiling_config.get("halo")
        if tile_shape is not None:
            tile_shape = tuple(tile_shape)
        if halo is not None:
            halo = tuple(halo)

        ndim = self.config.get("ndim", 2)
        embedding_path = self._get_embedding_path(image_container)
        prediction = automatic_instance_segmentation(
            predictor=self.predictor, segmenter=self.segmenter, input_path=processed_image, ndim=ndim,
            tile_shape=tile_shape, halo=halo, embedding_path=embedding_path,
        )
        logger.debug(f"Generated AIS mask with shape {prediction.shape if prediction is not None else 'None'}")

        result_data = {"processed_image": processed_image, "masks": prediction}
        if isinstance(self.segmenter, InstanceSegmentationWithDecoder):
            result_data["foreground"] = self.segmenter._foreground.copy()
            result_data["center_distances"] = self.segmenter._center_distances.copy()
            result_data["boundary_distances"] = self.segmenter._boundary_distances.copy()
        return file_name_key, result_data

    def _process_3d_image(self, image_container: 'ImageContainer'):
        """
        Runs prompted 3D segmentation on a single ImageContainer.

        Generates prompts from the seed slice, precomputes 3D embeddings, runs
        batched inference on the seed slice, then propagates each mask through
        the volume using segment_mask_in_volume.

        Args:
            image_container: Container holding a multi-slice z-stack.

        Returns:
            A tuple (result_key, result_data), or None if processing fails.
        """
        file_name_key = image_container.name
        logger.info(f"[3D] Starting '{file_name_key}'")

        # Step 1: Build 3D volume and generate prompts from the seed slice.
        logger.info(f"[3D] '{file_name_key}' — step 1/4: merging channels...")
        volume = image_container.merge()  # (Z, H, W, 3)
        n_slices = volume.shape[0]
        spatial_shape = volume.shape[1:3]  # (H, W)
        logger.info(f"[3D] '{file_name_key}' — merged volume shape: {volume.shape}")

        # Config seed_slice takes precedence over any value already on the container.
        config_seed = self.config.get("segmentation_3d", {}).get("seed_slice")
        if config_seed is not None:
            image_container.seed_slice = config_seed

        logger.info(f"[3D] '{file_name_key}' — generating prompts from DAPI slice...")
        image_container.generate_prompts()
        prompts, prompt_type = image_container.prompts, image_container.prompt_type

        # If seed_slice was not configured, fall back to the DAPI prompt slice.
        if image_container.seed_slice is None:
            image_container.seed_slice = image_container.dapi_slice
        seed_z = image_container.seed_slice
        logger.info(f"[3D] '{file_name_key}' — {len(prompts) if prompts is not None else 0} '{prompt_type}' prompts from DAPI z={image_container.dapi_slice}, seed z={seed_z}")

        if prompts is None or prompts.size == 0:
            logger.warning(f"Skipping {file_name_key} due to prompt generation error.")
            return None

        # Step 2: Precompute 3D embeddings slice-by-slice.
        embedding_path = self._get_embedding_path(image_container)
        tiling_config = self.config.get("tiling", {})
        tile_shape = tuple(tiling_config["tile_shape"]) if tiling_config.get("tile_shape") else None
        halo = tuple(tiling_config["halo"]) if tiling_config.get("halo") else None
        use_tiling = tile_shape is not None and halo is not None

        logger.info(f"[3D] '{file_name_key}' — step 2/4: precomputing embeddings (tiling={use_tiling}, save_path={embedding_path})...")
        image_embeddings = precompute_image_embeddings(
            predictor=self.predictor,
            input_=volume,
            save_path=embedding_path,
            ndim=3,
            tile_shape=tile_shape,
            halo=halo,
        )
        logger.info(f"[3D] '{file_name_key}' — embeddings ready.")

        # Step 3: Run batched inference on the seed slice.
        # Tiled and non-tiled embeddings use incompatible storage formats,
        # so inference must match the embedding type.
        logger.info(f"[3D] '{file_name_key}' — step 3/4: seed-slice inference on z={seed_z}...")
        inference_kwargs = dict(
            batch_size=len(prompts),
            multimasking=False,
            return_instance_segmentation=False,
        )
        if prompt_type == 'points':
            inference_kwargs.update(points=prompts[:, None, :], point_labels=np.ones((len(prompts), 1)))
        elif prompt_type == 'bbox':
            inference_kwargs.update(boxes=prompts)
        else:
            logger.warning(f"Prompt type '{prompt_type}' is not supported for 3D inference.")
            return None

        if use_tiling:
            mask_data = batched_tiled_inference(
                predictor=self.predictor,
                image=None,
                image_embeddings=image_embeddings,
                i=seed_z,
                **inference_kwargs,
            )
        else:
            set_precomputed(self.predictor, image_embeddings, i=seed_z)
            mask_data = batched_inference(
                predictor=self.predictor,
                image=None,
                **inference_kwargs,
            )
        logger.info(f"[3D] '{file_name_key}' — seed-slice inference done: {len(mask_data)} masks.")

        # Step 4: Propagate each seed mask through the volume.
        logger.info(f"[3D] '{file_name_key}' — step 4/4: propagating {len(mask_data)} masks through {n_slices} slices...")
        seg_3d_config = self.config.get("segmentation_3d", {})
        iou_threshold = seg_3d_config.get("iou_threshold", 0.5)
        projection = seg_3d_config.get("projection", "mask")
        stop_lower = seg_3d_config.get("stop_lower", False)
        stop_upper = seg_3d_config.get("stop_upper", False)
        box_extension = seg_3d_config.get("box_extension", 0.0)

        final_volume = np.zeros((n_slices, *spatial_shape), dtype=np.uint16)

        for obj_id, mask_dict in enumerate(mask_data, start=1):
            binary_mask_2d = mask_dict['segmentation'].cpu().numpy()
            if binary_mask_2d.ndim == 3:
                binary_mask_2d = binary_mask_2d[0]
            binary_mask_2d = (binary_mask_2d > 0).astype(np.uint8)

            seed_seg = np.zeros((n_slices, *spatial_shape), dtype=np.uint8)
            seed_seg[seed_z] = binary_mask_2d

            propagated, _ = segment_mask_in_volume(
                segmentation=seed_seg,
                predictor=self.predictor,
                image_embeddings=image_embeddings,
                segmented_slices=np.array([seed_z]),
                stop_lower=stop_lower,
                stop_upper=stop_upper,
                iou_threshold=iou_threshold,
                projection=projection,
                box_extension=box_extension,
            )
            final_volume[propagated == 1] = obj_id

        logger.info(f"[3D] '{file_name_key}' — propagation done. Unique objects: {len(np.unique(final_volume)) - 1}")

        result_data = {
            "processed_image": volume,
            "prompts": prompts,
            "prompt_type": prompt_type,
            "masks": final_volume,
            "seed_slice": seed_z,
        }
        return file_name_key, result_data

    def run_3d_ais(self, image_container: 'ImageContainer') -> Optional[str]:
        """
        Runs decoder-based automatic instance segmentation on a 3D z-stack.

        Processes each slice individually to capture raw decoder outputs (cell
        probability, center distances, boundary distances), then merges per-slice
        segmentations into a 3D instance mask. Results are stored in self.results
        for retrieval via save_results().

        Args:
            image_container: Container holding a multi-slice z-stack.

        Returns:
            The result key (file name stem), or None if processing fails.
        """
        file_name_key = image_container.name
        embedding_path = self._get_embedding_path(image_container)

        volume = image_container.merge()  # (Z, H, W, 3)
        n_slices = volume.shape[0]

        tiling_config = self.config.get("tiling", {})
        tile_shape = tuple(tiling_config["tile_shape"]) if tiling_config.get("tile_shape") else None
        halo = tuple(tiling_config["halo"]) if tiling_config.get("halo") else None
        use_tiling = tile_shape is not None and halo is not None

        logger.info(f"Precomputing 3D embeddings for {file_name_key} (tiling={use_tiling}, save_path={embedding_path})...")
        image_embeddings = precompute_image_embeddings(
            predictor=self.predictor,
            input_=volume,
            save_path=embedding_path,
            ndim=3,
            tile_shape=tile_shape,
            halo=halo,
        )

        ais_cfg = self.config.get("ais_generate", {})
        generate_kwargs = {
            "center_distance_threshold": ais_cfg.get("center_distance_threshold", 0.5),
            "boundary_distance_threshold": ais_cfg.get("boundary_distance_threshold", 0.5),
            "foreground_threshold": ais_cfg.get("foreground_threshold", 0.5),
            "foreground_smoothing": ais_cfg.get("foreground_smoothing", 1.0),
            "distance_smoothing": ais_cfg.get("distance_smoothing", 1.6),
            "min_size": ais_cfg.get("min_size", 0),
        }
        if use_tiling:
            generate_kwargs.update(tile_shape=tile_shape, halo=halo)

        init_kwargs = {"image_embeddings": image_embeddings}
        if use_tiling:
            init_kwargs["batch_size"] = 1

        # Slice loop — capture raw decoder outputs per slice.
        seg_volume = np.zeros((n_slices, *volume.shape[1:3]), dtype=np.uint32)
        foreground_stack, center_dist_stack, boundary_dist_stack = [], [], []
        offset = 0

        for z in range(n_slices):
            self.segmenter.initialize(volume[z], i=z, **init_kwargs)
            seg = self.segmenter.generate(**generate_kwargs)
            foreground_stack.append(self.segmenter._foreground.copy())
            center_dist_stack.append(self.segmenter._center_distances.copy())
            boundary_dist_stack.append(self.segmenter._boundary_distances.copy())

            max_id = int(seg.max())
            if max_id > 0:
                seg[seg != 0] += offset
                offset += max_id
            seg_volume[z] = seg

        # Merge 2D per-slice segmentations into a coherent 3D instance mask.
        instance_mask = merge_instance_segmentation_3d(seg_volume, beta=0.5, with_background=True)

        # Stack raw maps into (Z, H, W) volumes.
        foreground_3d = np.stack(foreground_stack, axis=0).astype(np.float32)
        center_dist_3d = np.stack(center_dist_stack, axis=0).astype(np.float32)
        boundary_dist_3d = np.stack(boundary_dist_stack, axis=0).astype(np.float32)

        self.results[file_name_key] = {
            "masks": instance_mask,
            "foreground": foreground_3d,
            "center_distances": center_dist_3d,
            "boundary_distances": boundary_dist_3d,
            "processed_image": volume,
        }
        logger.info(f"Stored 3D AIS results for '{file_name_key}' — call save_results() to write to disk.")
        return file_name_key

    def _run_combined_3d(self, image_container: 'ImageContainer') -> Optional[str]:
        """
        Runs combined AIS + propagation 3D segmentation on a single ImageContainer.

        Runs AIS on the seed slice to obtain automatic instance masks, then propagates
        each instance through the full volume using segment_mask_in_volume.

        Args:
            image_container: Container holding a multi-slice z-stack.

        Returns:
            The result key (file name stem), or None if no instances were found.
        """
        file_name_key = image_container.name
        logger.info(f"[3D-combined] Starting '{file_name_key}'")

        # Step 1: Build the 3D volume.
        logger.info(f"[3D-combined] '{file_name_key}' — step 1/4: merging channels...")
        volume = image_container.merge()  # (Z, H, W, 3)
        n_slices = volume.shape[0]
        spatial_shape = volume.shape[1:3]
        logger.info(f"[3D-combined] '{file_name_key}' — volume shape: {volume.shape}")

        # Step 2: Determine seed slice (config → container.seed_slice → dapi_slice → middle).
        config_seed = self.config.get("segmentation_3d", {}).get("seed_slice")
        if config_seed is not None:
            seed_z = int(config_seed)
        elif image_container.seed_slice is not None:
            seed_z = image_container.seed_slice
        elif image_container.dapi_slice is not None:
            seed_z = image_container.dapi_slice
        else:
            seed_z = n_slices // 2
        image_container.seed_slice = seed_z
        logger.info(f"[3D-combined] '{file_name_key}' — seed_z={seed_z}")

        # Step 3: Precompute 3D embeddings.
        embedding_path = self._get_embedding_path(image_container)
        tiling_config = self.config.get("tiling", {})
        tile_shape = tuple(tiling_config["tile_shape"]) if tiling_config.get("tile_shape") else None
        halo = tuple(tiling_config["halo"]) if tiling_config.get("halo") else None
        use_tiling = tile_shape is not None and halo is not None

        logger.info(f"[3D-combined] '{file_name_key}' — step 2/4: precomputing embeddings (tiling={use_tiling})...")
        image_embeddings = precompute_image_embeddings(
            predictor=self.predictor,
            input_=volume,
            save_path=embedding_path,
            ndim=3,
            tile_shape=tile_shape,
            halo=halo,
        )
        logger.info(f"[3D-combined] '{file_name_key}' — embeddings ready.")

        # Step 4: Run AIS on the seed slice.
        logger.info(f"[3D-combined] '{file_name_key}' — step 3/4: AIS on seed slice z={seed_z}...")
        ais_cfg = self.config.get("ais_generate", {})
        generate_kwargs = {
            "center_distance_threshold": ais_cfg.get("center_distance_threshold", 0.5),
            "boundary_distance_threshold": ais_cfg.get("boundary_distance_threshold", 0.5),
            "foreground_threshold": ais_cfg.get("foreground_threshold", 0.5),
            "foreground_smoothing": ais_cfg.get("foreground_smoothing", 1.0),
            "distance_smoothing": ais_cfg.get("distance_smoothing", 1.6),
            "min_size": ais_cfg.get("min_size", 0),
        }
        if use_tiling:
            generate_kwargs.update(tile_shape=tile_shape, halo=halo)

        init_kwargs = {"image_embeddings": image_embeddings, "i": seed_z}
        if use_tiling:
            init_kwargs["batch_size"] = 1
        self.segmenter.initialize(volume[seed_z], **init_kwargs)
        seed_seg_2d = self.segmenter.generate(**generate_kwargs)  # (H, W) instance-labeled

        n_instances = int(seed_seg_2d.max())
        logger.info(f"[3D-combined] '{file_name_key}' — AIS found {n_instances} instances on seed slice.")

        if n_instances == 0:
            logger.warning(f"[3D-combined] '{file_name_key}' — no instances found on seed slice. Aborting.")
            return None

        # Step 5: Propagate each instance through the volume.
        logger.info(f"[3D-combined] '{file_name_key}' — step 4/4: propagating {n_instances} masks through {n_slices} slices...")
        seg_3d_config = self.config.get("segmentation_3d", {})
        iou_threshold = seg_3d_config.get("iou_threshold", 0.5)
        projection = seg_3d_config.get("projection", "mask")
        stop_lower = seg_3d_config.get("stop_lower", False)
        stop_upper = seg_3d_config.get("stop_upper", False)
        box_extension = seg_3d_config.get("box_extension", 0.0)

        final_volume = np.zeros((n_slices, *spatial_shape), dtype=np.uint16)
        unique_ids = np.unique(seed_seg_2d)
        unique_ids = unique_ids[unique_ids != 0]

        for obj_id in unique_ids:
            binary_mask_2d = (seed_seg_2d == obj_id).astype(np.uint8)
            seed_seg_vol = np.zeros((n_slices, *spatial_shape), dtype=np.uint8)
            seed_seg_vol[seed_z] = binary_mask_2d

            propagated, _ = segment_mask_in_volume(
                segmentation=seed_seg_vol,
                predictor=self.predictor,
                image_embeddings=image_embeddings,
                segmented_slices=np.array([seed_z]),
                stop_lower=stop_lower,
                stop_upper=stop_upper,
                iou_threshold=iou_threshold,
                projection=projection,
                box_extension=box_extension,
            )
            final_volume[propagated == 1] = int(obj_id)

        logger.info(f"[3D-combined] '{file_name_key}' — done. Unique objects: {len(np.unique(final_volume)) - 1}")

        self.results[file_name_key] = {
            "processed_image": volume,
            "masks": final_volume,
            "seed_slice": seed_z,
        }
        logger.info(f"Stored 3D combined results for '{file_name_key}' — call save_results() to write to disk.")
        return file_name_key

    def run(self):
        """
        Executes the segmentation pipeline on all ImageContainer objects provided at initialization.
        """
        ndim = self.config.get("ndim", 2)

        # --- Execute Prompted Segmentation Workflow ---
        if self.segmentation_mode == 'prompted':
            logger.info(f"Starting prompted segmentation for {len(self.run_containers)} run(s)...")
            for container in self.run_containers:
                if ndim == 3:
                    result = self._process_3d_image(container)
                else:
                    result = self._process_single_image(container)
                if result is not None:
                    result_key, result_data = result
                    if result_data:
                        self.results[result_key] = result_data
        # --- Execute Automatic Segmentation (AIS) Workflow ---
        elif self.segmentation_mode == 'automatic':
            logger.info(f"Starting automatic segmentation for {len(self.run_containers)} run(s)...")
            for container in self.run_containers:
                if ndim == 3:
                    self.run_3d_ais(container)
                else:
                    result_key, result_data = self._run_ais_on_image(container)
                    if result_data:
                        self.results[result_key] = result_data
                        logger.info(f"  Generated automatic instance segmentation.")
            logger.info("Finished generating masks for all files.")
        # --- Execute Combined (AIS seed + propagation) Workflow ---
        elif self.segmentation_mode == 'combined':
            logger.info(f"Starting combined segmentation for {len(self.run_containers)} run(s)...")
            if ndim != 3:
                logger.warning("Combined segmentation mode requires ndim=3. No containers processed.")
                return
            for container in self.run_containers:
                self._run_combined_3d(container)
            logger.info("Finished combined segmentation for all containers.")

    def get_masks(self, image_name: str) -> Optional[Union[List[np.ndarray], np.ndarray]]:
        """
        Retrieves the segmentation masks for a specific image.

        Args:
            image_name (str): The name/key of the image (usually the filename).

        Returns:
            The masks (format depends on segmentation_mode) or None if not found.
        """
        if image_name in self.results:
            return self.results[image_name].get("masks")
        logger.warning(f"No results found for image: {image_name}")
        return None

    def extract_objects(self, image_name: str) -> List[np.ndarray]:
        """
        Extracts individual objects from the image based on segmentation masks,
        cropped to their bounding boxes.

        Args:
            image_name (str): The name/key of the image.

        Returns:
            List[np.ndarray]: A list of arrays, each containing one isolated object cropped to its bounding box.
        """
        if image_name not in self.results:
            logger.warning(f"No results found for image: {image_name}")
            return []

        result_data = self.results[image_name]
        image = result_data["processed_image"]
        masks = result_data["masks"]
        extracted_objects = []

        def _crop_and_mask(binary_mask):
            # Find bounding box indices
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                return None
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop image and mask to the bounding box
            if image.ndim == 3:
                crop_img = image[rmin:rmax+1, cmin:cmax+1, :]
                crop_mask = binary_mask[rmin:rmax+1, cmin:cmax+1, None]
            else:
                crop_img = image[rmin:rmax+1, cmin:cmax+1]
                crop_mask = binary_mask[rmin:rmax+1, cmin:cmax+1]
            
            # Apply mask (set background to 0)
            return crop_img * crop_mask

        if isinstance(masks, list):  # Prompted mode
            for mask_arr in masks:
                # Handle (1, H, W) vs (H, W) shapes
                binary_mask = mask_arr[0] > 0 if (mask_arr.ndim == 3 and mask_arr.shape[0] == 1) else mask_arr > 0
                obj = _crop_and_mask(binary_mask)
                if obj is not None: extracted_objects.append(obj)
        elif isinstance(masks, np.ndarray):  # Automatic mode
            unique_labels = np.unique(masks)
            for label in unique_labels[unique_labels != 0]:
                obj = _crop_and_mask(masks == label)
                if obj is not None: extracted_objects.append(obj)

        return extracted_objects

    def save_results(self):
        """
        Saves all segmentation results for every container to:
            <source_file_parent>/microsam_outputs/

        The segmentation mode and the model tag are embedded in every filename so
        results from different modes or different models on the same image never
        overwrite each other. With `tag = {mode}_{model_tag}`:

            {name}_{tag}_masks.tif      — instance labels at original resolution
            {name}_{tag}_raw.tif        — (3, …) float32 [fg, center_dist, boundary_dist]
            {name}_{tag}_prompts.npy    — raw prompt coordinates (prompted mode)
            {name}_{tag}_viz.tif        — labeled annotation: seed slice shows prompt
                                          geometry, all other slices are zero (prompted mode)
        """
        if not self.results:
            logger.warning("No results to save. Please run the pipeline first.")
            return

        ndim = self.config.get("ndim", 2)
        mode = self.segmentation_mode  # "ais" or "prompted"

        for container in self.run_containers:
            result_key = container.name
            result_data = self.results.get(result_key)
            if not result_data:
                continue

            masks = result_data.get("masks")
            if masks is None:
                logger.warning(f"No masks for '{result_key}' — skipping.")
                continue

            output_dir = container._source_paths[0].parent / "microsam_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            # Model tag in the stem keeps results from different models side by
            # side instead of overwriting each other on the same image.
            stem = f"{result_key}_{mode}_{self.model_tag}"

            # --- Instance-labeled mask at original resolution ---
            if isinstance(masks, list):
                # 2D prompted: list of binary per-object masks → single instance label image
                m_sample = masks[0][0] if (masks[0].ndim == 3 and masks[0].shape[0] == 1) else masks[0]
                instance = np.zeros(m_sample.shape, dtype=np.uint16)
                for obj_id, m in enumerate(masks, start=1):
                    m2d = m[0] if (m.ndim == 3 and m.shape[0] == 1) else m
                    instance[m2d > 0] = obj_id
            else:
                instance = masks.astype(np.uint16)

            tifffile.imwrite(
                str(output_dir / f"{stem}_masks.tif"),
                self._rescale_to_original(instance, container, is_float=False),
            )

            # --- AIS raw decoder outputs: (3, H, W) or (3, Z, H, W) float32 ---
            if "foreground" in result_data:
                raw_out = np.stack([
                    self._rescale_to_original(result_data["foreground"].astype(np.float32),         container, is_float=True),
                    self._rescale_to_original(result_data["center_distances"].astype(np.float32),   container, is_float=True),
                    self._rescale_to_original(result_data["boundary_distances"].astype(np.float32), container, is_float=True),
                ], axis=0)
                tifffile.imwrite(str(output_dir / f"{stem}_raw.tif"), raw_out)

            # --- Prompted: prompts .npy + prompt viz ---
            if mode == "prompted":
                prompts = result_data.get("prompts")
                if prompts is not None:
                    np.save(str(output_dir / f"{stem}_prompts.npy"), prompts)

                viz = self._make_prompt_viz(result_data, container, ndim)
                if viz is not None:
                    tifffile.imwrite(str(output_dir / f"{stem}_viz.tif"), viz)

            logger.info(f"Saved {mode} results for '{result_key}' to {output_dir}")

    def visualize_results(self, visualization_mode: str = 'single'):
        """
        Saves visualizations of the segmentation results.

        Args:
            visualization_mode (str): 'single' for one plot per image,
                                      'channel_comparison' for a combined plot.
        """
        if not self.results:
            logger.warning("No results to visualize. Please run the pipeline first.")
            return

        base_input_dir = Path(self.config["base_input_dir"]).expanduser()

        if visualization_mode == 'single':
            # Handle 'single' mode: one plot per individual result (mono or composite)
            for container in self.run_containers:
                result_key = container.name
                result_data = self.results.get(result_key)
                if not result_data: continue

                # Determine output directory suffix and base name for this result
                is_composite = True if len(container._source_paths) > 1 else False # comp means composite of multiple channels
                dir_suffix = "_comp" if is_composite else ""
                if self.segmentation_mode == 'automatic':
                    base_output_dir = Path(f"output/microSAM_AIS{dir_suffix}")
                else:  # prompted
                    prompt_mode = self.config.get("prompting", {}).get("prompt_mode", "none")
                    base_output_dir = Path(f"output/microSAM_{prompt_mode}_prompts{dir_suffix}")

                # Create output directory structure
                try:
                    relative_dir = container._source_paths[0].parent.relative_to(base_input_dir)
                except ValueError:
                    relative_dir = container._source_paths[0].parent.name

                current_output_dir = base_output_dir / relative_dir
                current_output_dir.mkdir(parents=True, exist_ok=True)

                # Determine output filename (using result_key as stem)
                output_filename = current_output_dir / f"{result_key}_visualization.png"

                # --- Call the appropriate visualization function for 'single' mode ---
                logger.info(f"Creating visualization for '{result_key}'...")
                if self.segmentation_mode == 'automatic':
                    save_segmentation_visualization_AIS(
                        **result_data,
                        output_path=output_filename,
                        title=f"microSAM AIS Output for {result_key}"
                    )
                else:  # prompted
                    save_multi_mask_visualization(
                        **result_data,
                        output_path=output_filename,
                        title=f"microSAM Output for {result_key}"
                    )
                logger.info(f"  -> Saved to {output_filename}")

        elif visualization_mode == 'channel_comparison':
            # Handle 'channel_comparison' mode: create one plot comparing all results
            # Aggregate all data from self.results
            fov_all_processed_images = {}
            fov_all_prompts = {}
            fov_all_prompt_types = {}
            fov_all_generated_masks = {}
            fov_all_generated_scores = {}
            fov_all_generated_logits = {}
            
            # Collect all results
            for container in self.run_containers:
                result_key = container.name
                result_data = self.results.get(result_key)
                if not result_data: continue
                
                # Populate data for the plot, handling missing keys for AIS mode
                fov_all_processed_images[result_key] = result_data.get("processed_image")
                fov_all_generated_masks[result_key] = result_data.get("masks")

                if self.segmentation_mode == 'prompted':
                    fov_all_prompts[result_key] = result_data.get("prompts")
                    fov_all_prompt_types[result_key] = result_data.get("prompt_type")
                    fov_all_generated_scores[result_key] = result_data.get("scores")
                    fov_all_generated_logits[result_key] = result_data.get("logits")
                else: # AIS mode doesn't have these
                    fov_all_prompts[result_key] = None
                    fov_all_prompt_types[result_key] = 'none'
                    fov_all_generated_scores[result_key] = None
                    fov_all_generated_logits[result_key] = None

            if not self.run_containers:
                logger.warning("No valid results to create a channel comparison plot.")
                return

            # Determine common FOV name and output path
            ref_path_for_fov_dir = self.run_containers[0]._source_paths[0]
            try:
                relative_dir = ref_path_for_fov_dir.parent.relative_to(base_input_dir)
            except ValueError:
                relative_dir = ref_path_for_fov_dir.parent.name

            # Check if any of the containers in the run represent a composite image.
            any_composite = any(len(container._source_paths) > 1 for container in self.run_containers)
            dir_suffix = "_comp" if any_composite else ""

            if self.segmentation_mode == 'automatic':
                base_output_dir = Path(f"output/microSAM_AIS_comparison{dir_suffix}")
            else:  # prompted
                prompt_mode = self.config.get("prompting", {}).get("prompt_mode", "none")
                base_output_dir = Path(f"output/microSAM_{prompt_mode}_prompts_comparison{dir_suffix}")

            current_output_dir = base_output_dir / relative_dir
            current_output_dir.mkdir(parents=True, exist_ok=True)

            # The title and filename should reflect the common FOV.
            # We derive this from the result key of the first container, which
            # already calculates the common base name.
            first_result_key = self.run_containers[0].name
            fov_name = first_result_key.rsplit('_', 1)[0] if '_' in first_result_key else first_result_key
            output_filename = current_output_dir / f"{fov_name}_channel_comparison.png"

            logger.info(f"Creating channel comparison visualization for FOV '{fov_name}'...")
            save_channel_comparison_visualization(
                all_processed_images=fov_all_processed_images,
                all_prompts=fov_all_prompts,
                all_prompt_types=fov_all_prompt_types,
                all_generated_masks=fov_all_generated_masks,
                all_generated_scores=fov_all_generated_scores,
                all_generated_logits=fov_all_generated_logits,
                display_items=self.run_containers, # Pass the list of container objects
                output_path=output_filename,
                title=f"Micro-SAM Channel Comparison for {fov_name}",
                segmentation_mode=self.segmentation_mode
            )
            logger.info(f"  -> Saved to {output_filename}")
        else:
            logger.warning(f"Unknown visualization mode: {visualization_mode}")

        logger.info("All visualizations have been saved.")
