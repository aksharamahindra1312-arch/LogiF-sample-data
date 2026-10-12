"""
General utility functions for spatial calculations, configuration loading, and logging.
"""

import math
import yaml
import os
import numpy as np

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the project YAML configuration file."""
    if not os.path.exists(config_path):
        # Check parent directory
        parent_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    if any(np.isnan([lat1, lon1, lat2, lon2])):
        return np.nan
    try:
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
        c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
        return 6371.0 * c
    except Exception:
        return np.nan

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized numpy calculation of great-circle distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0 * c
