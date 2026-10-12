"""
Live and historical weather connector for Open-Meteo API.
"""

import os
import json
import urllib.request
import pandas as pd
from typing import Dict, Any, Tuple

_WEATHER_CACHE: Dict[Tuple[float, float], Dict[str, Any]] = {}

def fetch_open_meteo_weather(lat: float, lng: float) -> Dict[str, Any]:
    """Fetches real-time hourly weather conditions from Open-Meteo API with offline fallback and caching."""
    if pd.isna(lat) or pd.isna(lng):
        return {'rain_mm': 0.0, 'temp_c': 28.0, 'wind_kmph': 12.0, 'severe_flag': False, 'fog_flag': False, 'source': 'DEFAULT'}
    
    grid_key = (round(float(lat), 1), round(float(lng), 1))
    if os.environ.get('LF_LIVE_WEATHER') != '1':
        return {'rain_mm': 0.0, 'temp_c': 28.0, 'wind_kmph': 12.0, 'severe_flag': False, 'fog_flag': False, 'source': 'FALLBACK_CALIBRATED'}
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={grid_key[0]}&longitude={grid_key[1]}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
        req = urllib.request.Request(url, headers={'User-Agent': 'LogiFreight-System/1.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            curr = data.get('current', {})
            rain = float(curr.get('precipitation', 0.0))
            temp = float(curr.get('temperature_2m', 28.0))
            wind = float(curr.get('wind_speed_10m', 12.0))
            wcode = int(curr.get('weather_code', 0))
            severe = bool(rain > 10.0 or wind > 40.0 or wcode in [95, 96, 99])
            fog = bool(wcode in [45, 48])
            result = {'rain_mm': rain, 'temp_c': temp, 'wind_kmph': wind, 'severe_flag': severe, 'fog_flag': fog, 'source': 'LIVE_OPEN_METEO'}
            _WEATHER_CACHE[grid_key] = result
            return result
    except Exception:
        fallback = {'rain_mm': 0.0, 'temp_c': 28.0, 'wind_kmph': 12.0, 'severe_flag': False, 'fog_flag': False, 'source': 'FALLBACK_CALIBRATED'}
        _WEATHER_CACHE[grid_key] = fallback
        return fallback
