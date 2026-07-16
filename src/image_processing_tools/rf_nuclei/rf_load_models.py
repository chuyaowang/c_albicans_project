import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import math

def load_model(model_path: Path):
    """
    Loads a saved scikit-learn model from a .joblib file.

    Args:
        model_path (Path): Path to the .joblib model file.

    Returns:
        The loaded model object.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)
