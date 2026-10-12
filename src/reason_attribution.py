"""
Evidence-based root cause delay attribution engine.
"""

def attribute_delay_reason(is_delayed: bool, features: dict) -> str:
    """Maps empirical telemetry, weather, traffic, and civic evidence into structured operational delay reasons."""
    if not is_delayed:
        return 'ON_TRACK'
    elif features.get('Civic_Disruption_Alert') == 1:
        return 'CIVIC_POLITICAL_DISRUPTION'
    elif features.get('Weather_Severe_Flag') == 1 or features.get('Weather_Rain_MM', 0) > 10.0 or features.get('Weather_Fog_Flag') == 1:
        return 'SEVERE_WEATHER'
    elif features.get('Traffic_Congestion_Ratio', 1.0) < 0.65:
        return 'TRAFFIC_CONGESTION'
    elif features.get('Telemetry_Freshness_Min', 0) > 180.0:
        return 'EXTENDED_STOPPAGE'
    elif features.get('Elapsed_Frac', 0) > 1.2:
        return 'ORIGIN_DISPATCH_DELAY'
    elif features.get('Route_Type') == 'LOCAL_PLANT_TRANSFER':
        return 'PLANT_GATE_DWELL'
    else:
        return 'TRANSIT_DWELL_ACCUMULATION'
