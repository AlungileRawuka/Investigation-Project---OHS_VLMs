import os
from pathlib import Path
from PIL import Image
import json
import pandas as pd

# Base path: repo root / data
BASE_DATA_PATH = Path(__file__).resolve().parents[2] / "data"

def get_path(filename, subdir="raw"):
    """
    Resolve a file path inside the data/ directory.
    
    Args:
        filename (str): Name of the file 
        subdir (str): Subdirectory inside data/ (raw, processed, annotations).
    Returns:
        pathlib.Path: Absolute path to the file.
    """
    path = BASE_DATA_PATH / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

def load_image(filename, subdir="raw"):
    """
    Load an image from the data/ directory.
    
    Args:
        filename (str): Image filename.
        subdir (str): Subdirectory ("raw" or "processed").
    Returns:
        PIL.Image.Image: Loaded image in RGB format.
    """
    path = get_path(filename, subdir=subdir)
    return Image.open(path).convert("RGB")

def load_json(filename, subdir="annotations"):
    """
    Load a JSON annotation file from data/annotations.
    
    Args:
        filename (str): JSON filename.
    Returns:
        dict: Parsed JSON content.
    """
    path = get_path(filename, subdir=subdir)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(filename, subdir="annotations"):
    """
    Load a CSV annotation file from data/annotations.
    
    Args:
        filename (str): CSV filename.
    Returns:
        pandas.DataFrame: Dataframe with CSV content.
    """
    path = get_path(filename, subdir=subdir)
    return pd.read_csv(path)
