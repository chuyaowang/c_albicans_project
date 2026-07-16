from napari_rf.features import FeatureCreator
from napari_rf.RF import RF
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

def create_pixel_features(image: np.ndarray, indices = None) -> np.ndarray:
    """
    Generates pixel-wise features for a given image using the FeatureCreator.

    Args:
        image (np.ndarray): Input 2D image.

    Returns:
        np.ndarray: Feature stack of shape (Y, X, n_features).
    """
    feature_creator = FeatureCreator()
    feature_gen = feature_creator.make_simple_features(image, indices=indices)
    
    features = None
    for item in feature_gen:
        if isinstance(item, np.ndarray):
            features = item
            
    return features

def predict_pixel_class(model, features: np.ndarray) -> np.ndarray:
    """
    Predicts the class of each pixel using the loaded Random Forest model.

    Args:
        model: The loaded scikit-learn Random Forest model or RF object.
        features (np.ndarray): Feature stack of shape (Y, X, n_features).

    Returns:
        np.ndarray: Predicted class mask (0 for background, 1 for nuclei).
    """
    # Check if the model is already an instance of RF (has predict_segmenter)
    # or if it's a raw sklearn classifier that needs wrapping.
    if hasattr(model, "predict_segmenter"):
        rf = model
    else:
        rf = RF(clf=model)
    
    # predict_segmenter returns probability maps of shape (n_classes, Y, X)
    prediction_probs = rf.predict_segmenter(features)
    
    # Convert probabilities to class labels (argmax along channel axis)
    prediction_mask = np.argmax(prediction_probs, axis=0).astype(np.uint8)
    
    return prediction_mask

def predict_z_stack_and_plot(image_stack: np.ndarray, model, output_filename: Path):
    """
    Iterates through the z-slices of a stack, predicts pixel classes, 
    and saves a grid plot with the predictions overlayed.
    
    Args:
        image_stack (np.ndarray): Input stack (Z, Y, X).
        model: Trained Random Forest model.
        output_filename (Path): File path to save the plot.
    """
    z_depth = image_stack.shape[0]
    
    # Determine grid size (approx square)
    cols = int(math.ceil(math.sqrt(z_depth)))
    rows = int(math.ceil(z_depth / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i in range(z_depth):
        ax = axes[i]
        image_slice = image_stack[i]
        
        # Predict
        features = create_pixel_features(image_slice)
        prediction = predict_pixel_class(model, features)
        
        # Plot Image
        # Normalize for display if float data is outside [0, 1] range
        display_slice = image_slice
        if display_slice.dtype.kind == 'f' and display_slice.max() > 1.0:
            v_min, v_max = display_slice.min(), display_slice.max()
            if v_max > v_min:
                display_slice = (display_slice - v_min) / (v_max - v_min)
        ax.imshow(display_slice, cmap='gray_r')
        
        # Overlay Prediction (Red for Nuclei)
        # Create RGBA overlay: Red channel=1, Alpha=1 where class is 1
        overlay = np.zeros(image_slice.shape + (4,))
        overlay[prediction == 1] = [1, 0, 0, 1] 
        
        ax.imshow(overlay)
        ax.set_title(f"Slice {i}")
        ax.axis('off')
        
    # Hide empty subplots
    for j in range(z_depth, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()