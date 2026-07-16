import logging
import os
import json
import itertools
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.inference import batched_inference
from micro_sam.instance_segmentation import (AMGBase,
                                             InstanceSegmentationWithDecoder,
                                             get_amg, get_decoder)
from micro_sam.util import get_sam_model, SamPredictor
from pathlib import Path
from image_processing_tools.image_class.image_container import ImageContainer
from image_processing_tools.util.load_files import find_files_by_pattern
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

    This class handles file loading, preprocessing, prompt-based segmentation,
    and result visualization.

    Args:
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

    def __init__(self, search_path: str, file_pattern: str, config: Dict[str, Any]):
        """
        Initializes the MicroSAMPipeline with a given configuration.

        Args:
            search_path (str): The directory to search for image files.
            file_pattern (str): The glob pattern to match files.
            config (Dict[str, Any]): A dictionary containing all parameters for the pipeline.
        """
        # Validate required config keys
        required_keys = ["model_type", "checkpoint_path", "base_input_dir"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Configuration dictionary must contain the key: '{key}'")

        self.config = config
        logger.info(f"Initializing MicroSAMPipeline with config: {self.config}")

        # Discover files and automatically determine the compute device.
        self.file_paths = find_files_by_pattern(search_path, file_pattern, verbose=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_mode = self.config.get("segmentation_mode", "prompted")
        
        self.predictor: Optional[SamPredictor] = None
        self.segmenter: Optional[Union[AMGBase, InstanceSegmentationWithDecoder]] = None
        self._initialize_models()

        self.results: Dict[str, Dict[str, Any]] = {}
        # Cache for single-channel ImageContainer objects to avoid re-reading files.
        self.image_cache: Dict[Path, 'ImageContainer'] = {}
        # List of composite containers created for the current run.
        self.run_containers: List['ImageContainer'] = []

    def _initialize_models(self):
        """Loads the SAM predictor and/or segmenter based on the configuration."""
        model_type = self.config["model_type"]
        checkpoint_path = Path(self.config["checkpoint_path"]).expanduser()
        logger.info(f"Loading model: {model_type}...")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        if self.segmentation_mode == 'automatic':
            decoder_path = self.config.get("decoder_checkpoint_path")
            if not decoder_path:
                msg = "`decoder_checkpoint_path` must be provided for automatic segmentation."
                logger.error(msg)
                raise ValueError(msg)
            
            decoder_path = Path(decoder_path).expanduser()
            if not decoder_path.exists():
                msg = f"Decoder checkpoint not found at: {decoder_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            
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
            self.segmenter = get_amg(predictor=predictor, is_tiled=False, decoder=decoder)
            logger.info(f"Models for AIS loaded on device: {self.predictor.device}")

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


    def _get_image_container(self, image_path: Path) -> 'ImageContainer':
        """
        Retrieves an ImageContainer object from the cache or creates a new one.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            ImageContainer: The cached or newly created ImageContainer object.
        """
        if image_path not in self.image_cache:
            logger.debug(f"Caching new ImageContainer for: {image_path.name}")
            self.image_cache[image_path] = ImageContainer(image_path, self.config)
        return self.image_cache[image_path]

    def _process_single_image(self, image_container: 'ImageContainer'):
        """
        Processes a single run specification (mono-channel or composite) for prompted segmentation.

        This method orchestrates the analysis for a single item from the `run_spec`.
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
        self.predictor.set_image(processed_image)

        if prompt_type == 'points':
            points_for_batch = prompts[:, None, :]
            mask_data = batched_inference(
                predictor=self.predictor,
                image=processed_image,
                batch_size=len(prompts),
                points=points_for_batch,
                point_labels=np.ones((len(prompts), 1)),
                multimasking=False,
                return_instance_segmentation=False
            )
        elif prompt_type == 'bbox':
            mask_data = batched_inference(
                predictor=self.predictor,
                image=processed_image,
                batch_size=len(prompts),
                boxes=prompts,
                multimasking=False,
                return_instance_segmentation=False
            )
        else:
            # add support for mask later
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

        prediction = automatic_instance_segmentation(
            predictor=self.predictor, segmenter=self.segmenter, input_path=processed_image, ndim=2
        )
        logger.debug(f"Generated AIS mask with shape {prediction.shape if prediction is not None else 'None'}")

        return file_name_key, {"processed_image": processed_image, "masks": prediction}

    def run(self, run_spec: List[Any]):
        """
        Executes the segmentation pipeline based on a user-provided run specification.

        This is the main entry point for running analysis. It takes a `run_spec` that
        uses integer indices to refer to the files discovered during initialization.

        Args:
            run_spec (List[Any]): A list defining analysis runs. Each item can be:
                - An integer index for a mono-channel run (e.g., `0`).
                - A list of indices for a simple composite (e.g., `[1, 2, 3]`).
                - A nested list for channel summation (e.g., `[[1, 2], 3]`).
        """
        # --- Helper to recursively parse the run_spec and build a structure of ImageContainers ---
        def get_structure_from_spec(spec_item):
            if isinstance(spec_item, int):
                # Base case: get a cached single-channel container.
                return self._get_image_container(self.file_paths[spec_item])
            elif isinstance(spec_item, list):
                # Recursive step: build a list of containers.
                return [get_structure_from_spec(sub_item) for sub_item in spec_item]
            else:
                raise TypeError(f"Unsupported type in run_spec: {type(spec_item)}")

        # --- Validate all indices in the run_spec before processing ---
        flat_indices = [item for item in itertools.chain.from_iterable(run_spec) if isinstance(item, int)]
        max_index = len(self.file_paths) - 1
        if any(i > max_index for i in flat_indices):
            raise IndexError(f"Invalid index provided. All indices must be between 0 and {max_index}.")

        # --- Create composite ImageContainer objects for each run in the spec ---
        self.run_containers = []
        for spec_item in run_spec:
            # Build a structure of cached single-channel containers.
            structure = get_structure_from_spec(spec_item)
            # The ImageContainer constructor handles composition from this structure.
            self.run_containers.append(ImageContainer(structure, self.config))

        # --- Execute Prompted Segmentation Workflow ---
        if self.segmentation_mode == 'prompted':
            logger.info(f"Starting prompted segmentation for {len(run_spec)} run(s)...")
            for container in self.run_containers:
                result_key, result_data = self._process_single_image(container)
                if result_data:
                    self.results[result_key] = result_data
        # --- Execute Automatic Segmentation (AIS) Workflow ---
        elif self.segmentation_mode == 'automatic':
            logger.info(f"Starting automatic segmentation for {len(run_spec)} run(s)...")
            for container in self.run_containers:
                result_key, result_data = self._run_ais_on_image(container)
                if result_data:
                    self.results[result_key] = result_data
                    logger.info(f"  Generated automatic instance segmentation.")
            logger.info("Finished generating masks for all files.")

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
