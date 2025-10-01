import os
from pathlib import Path
from PIL import Image
import json
import pandas as pd

# Base path: repo root / data
BASE_DATA_PATH = Path(__file__).resolve().parents[2] / "data"
IMAGES_PATH = BASE_DATA_PATH / "images"

def get_path(filename, subdir=None):
    """
    Resolve a file path inside the data/ directory.
    
    Args:
        filename (str): Name of the file.
        subdir (str, optional): Subdirectory inside data/. Defaults to None.
    Returns:
        pathlib.Path: Absolute path to the file.
    """
    path = IMAGES_PATH / filename if subdir is None else BASE_DATA_PATH / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

def load_image(filename):
    """
    Load an image from data/images.
    
    Args:
        filename (str): Image filename or full path.
    Returns:
        PIL.Image.Image: Loaded image in RGB format.
    """
    path = Path(filename)
    if not path.exists():
        # fallback to data/images folder
        path = IMAGES_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return Image.open(path).convert("RGB")

def load_csv(filename, subdir="annotations"):
    """
    Load a CSV annotation file from data/annotations.
    
    Args:
        filename (str): CSV filename.
    Returns:
        pandas.DataFrame: Dataframe with CSV content.
    """
    path = BASE_DATA_PATH / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
