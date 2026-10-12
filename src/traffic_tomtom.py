"""
Traffic flow connector for TomTom Traffic Flow API.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
from typing import Dict, Any

def fetch_tomtom_traffic(lat: float, lng: float, api_key: str = None) -> Dict[str, Any]:
    """Queries TomTom Traffic Flow API using environment variable or direct key."""
    key = api_key or os.environ.get("TOMTOM_API_KEY")
    if key and pd.notna(lat) and pd.notna(lng):
        try:
            url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json?point={lat},{lng}&key={key}"
            req = urllib.request.Request(url, headers={'User-Agent': 'LogiFreight-System/1.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                flow = data.get('flowSegmentData', {})
                curr_speed = float(flow.get('currentSpeed', 50.0))
                free_speed = float(flow.get('freeFlowSpeed', 60.0))
                ratio = float(np.clip(curr_speed / max(1.0, free_speed), 0.2, 1.0))
                sev = 'SEVERE' if ratio < 0.60 else ('MODERATE' if ratio < 0.80 else 'LOW')
                return {'ratio': ratio, 'severity': sev, 'source': 'LIVE_TOMTOM_API'}
        except Exception:
            pass
    return {'ratio': 0.88, 'severity': 'LOW', 'source': 'CORRIDOR_BASELINE'}
