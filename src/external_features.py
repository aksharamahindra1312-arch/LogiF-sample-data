"""
Spatio-temporal external feature fusion (Weather + Traffic + Civic Alerts).
"""

from src.weather_openmeteo import fetch_open_meteo_weather
from src.traffic_tomtom import fetch_tomtom_traffic
from src.civic_gdelt import fetch_gdelt_civic_disruptions

def get_fused_external_context(lat: float, lng: float, tomtom_key: str = None) -> dict:
    """Fuses real-time external environmental features."""
    w = fetch_open_meteo_weather(lat, lng)
    t = fetch_tomtom_traffic(lat, lng, api_key=tomtom_key)
    g = fetch_gdelt_civic_disruptions(lat, lng)
    
    return {
        'Weather_Rain_MM': w['rain_mm'],
        'Weather_Temp_C': w['temp_c'],
        'Weather_Wind_Kmph': w['wind_kmph'],
        'Weather_Severe_Flag': int(w['severe_flag']),
        'Weather_Fog_Flag': int(w['fog_flag']),
        'Traffic_Congestion_Ratio': t['ratio'],
        'Traffic_Severity': t['severity'],
        'Civic_Disruption_Alert': int(g['active_alert']),
        'Civic_Disruption_Severity': g['severity'],
        'Weather_Source': w['source'],
        'Traffic_Source': t['source'],
        'GDELT_Source': g['source']
    }
