"""
Feature engineering pipeline for point-in-time in-transit snapshots.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils import haversine_distance
from src.external_features import get_fused_external_context
from src.geo_resolver import (
    resolve_location, classify_route_type, calculate_route_distance,
    resolve_remaining_distance
)

# Load historical empirical priors if available
_PRIORS_CACHE = None

def _get_priors() -> dict:
    global _PRIORS_CACHE
    if _PRIORS_CACHE is None:
        priors_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'historical_priors.json')
        if os.path.exists(priors_file):
            try:
                with open(priors_file, 'r', encoding='utf-8') as f:
                    _PRIORS_CACHE = json.load(f)
            except Exception:
                _PRIORS_CACHE = {}
        else:
            _PRIORS_CACHE = {}
    return _PRIORS_CACHE

def extract_point_in_time_features(latest_event: dict, as_of_dt: datetime, dispatch_dt: datetime, original_eta_dt: datetime, tomtom_key: str = None) -> dict:
    """Constructs ML features strictly from information known at As_Of."""
    lat = pd.to_numeric(latest_event.get('lat', latest_event.get('Latitude')), errors='coerce')
    lng = pd.to_numeric(latest_event.get('lng', latest_event.get('Longitude')), errors='coerce')
    has_gps = int(pd.notna(lat) and pd.notna(lng))
    
    orig_city = str(latest_event.get('consignor_city', ''))
    dest_city = str(latest_event.get('consignee_city', ''))
    event_loc = str(latest_event.get('event_location', ''))
    shipper = str(latest_event.get('Shipper', latest_event.get('Customer Name', 'UNKNOWN_SHIPPER')))
    ba = str(latest_event.get('BA_Name', latest_event.get('BA NAME', 'UNKNOWN_BA')))
    
    # 1. Resolve coordinates using depot aliases and city dictionary
    orig_coords = resolve_location(event_loc if 'outbound' in event_loc.lower() or 'plant' in event_loc.lower() else None, orig_city)
    dest_coords = resolve_location(event_loc if 'inbound' in event_loc.lower() or 'destination' in event_loc.lower() else None, dest_city)
    
    elapsed_hours = max(0.1, (as_of_dt - dispatch_dt).total_seconds() / 3600.0)
    planned_hours = max(0.2, (original_eta_dt - dispatch_dt).total_seconds() / 3600.0)
    elapsed_frac = elapsed_hours / planned_hours
    hours_to_eta = (original_eta_dt - as_of_dt).total_seconds() / 3600.0
    
    # Fallback coordinates if current lat/lng is missing
    if pd.isna(lat) or pd.isna(lng):
        curr_coords = resolve_location(event_loc, orig_city if elapsed_frac < 0.3 else dest_city)
        if curr_coords:
            lat, lng = curr_coords
            
    # 2. Classify Route Type
    route_type = latest_event.get('Route_Type')
    if not route_type:
        route_type = classify_route_type(orig_city, dest_city, orig_coords, dest_coords)
        
    # 3. Calculate Total Route Distance & Planned Speed
    total_route_km = calculate_route_distance(orig_coords, dest_coords, route_type, planned_hours)
    planned_speed = total_route_km / planned_hours
    
    # 4. Resolve Remaining KM & Traversed KM dynamically (Eliminate 320 KM default)
    if pd.notna(latest_event.get('Remaining_KM')) and float(latest_event.get('Remaining_KM')) != 320.0:
        remaining_km = float(latest_event.get('Remaining_KM'))
        traversed_km = float(latest_event.get('Traversed_KM', max(0.0, total_route_km - remaining_km)))
    else:
        remaining_km, traversed_km = resolve_remaining_distance(
            lat, lng, orig_coords, dest_coords, route_type, total_route_km, elapsed_frac
        )
        
    route_progress = float(np.clip(traversed_km / max(1.0, traversed_km + remaining_km), 0.0, 1.0))
    
    req_hours_left = max(0.1, hours_to_eta)
    req_speed = remaining_km / req_hours_left
    eff_speed = traversed_km / elapsed_hours
    
    # Buffer hours against normal operating speed
    norm_speed = 25.0 if route_type == 'LOCAL_PLANT_TRANSFER' else (40.0 if route_type == 'REGIONAL_SHORT_HAUL' else 45.0)
    buffer_hours = hours_to_eta - (remaining_km / norm_speed)
    
    # Route-type appropriate Behind_Pace calculation & Proximity Buffer
    if route_type == 'LOCAL_PLANT_TRANSFER':
        behind_pace = bool(elapsed_frac > 1.0 or (req_speed > 35.0 and remaining_km > 5.0 and buffer_hours < -0.5))
        prox_buffer = int(remaining_km <= 5.0 and hours_to_eta >= 0.5)
    elif route_type == 'REGIONAL_SHORT_HAUL':
        behind_pace = bool(elapsed_frac > 1.0 or (req_speed > 55.0 and remaining_km > 15.0 and buffer_hours < -1.5))
        prox_buffer = int(remaining_km <= 10.0 and hours_to_eta >= 1.0)
    else:
        behind_pace = bool(elapsed_frac > 1.0 or (req_speed > 60.0 and remaining_km > 25.0 and buffer_hours < -2.5))
        prox_buffer = int(remaining_km <= 20.0 and hours_to_eta >= 1.5)
        
    # Origin yard dwell hours before physical dispatch
    c_creat_dt = pd.to_datetime(latest_event.get('Consignment_created_at', latest_event.get('Trip_Created_at')), errors='coerce')
    origin_dwell_hours = float(np.clip((dispatch_dt - c_creat_dt).total_seconds() / 3600.0, 0.0, 168.0)) if pd.notna(c_creat_dt) else 0.0
    
    # Empirical historical delay priors
    priors = _get_priors()
    global_prior = priors.get('global_delay_prior', 0.31)
    shipper_rate = priors.get('shipper_priors', {}).get(shipper, global_prior)
    ba_rate = priors.get('ba_priors', {}).get(ba, global_prior)
    corridor_rate = priors.get('corridor_priors', {}).get(f"{orig_city}->{dest_city}", global_prior)
    
    # External features
    ext = get_fused_external_context(lat if pd.notna(lat) else 19.0, lng if pd.notna(lng) else 74.0, tomtom_key=tomtom_key)
    
    stage = str(latest_event.get('event_type', latest_event.get('Stage', 'IN_TRANSIT')))
    stage_group = 'Mid_Transit (40-60%)' if 0.4 <= elapsed_frac <= 0.7 else ('Early_Journey (15-35%)' if elapsed_frac < 0.4 else 'Late_Journey (65-80%)')
    
    return {
        'Stage': stage,
        'Route_Type': route_type,
        'Shipper': shipper,
        'BA_Name': ba,
        'Origin_City': str(latest_event.get('consignor_city', 'UNKNOWN_ORIGIN')),
        'Origin_State': str(latest_event.get('consignor_state', 'UNKNOWN_STATE')),
        'Dest_City': str(latest_event.get('consignee_city', 'UNKNOWN_DEST')),
        'Dest_State': str(latest_event.get('consignee_state', 'UNKNOWN_STATE')),
        'Vehicle_Type': str(latest_event.get('Vehicle Type', 'UNKNOWN_VEHICLE')),
        'Event_Source': str(latest_event.get('Event_Source', 'Manifest')),
        'Eventable_Type': str(latest_event.get('eventable_type', 'Manifest')),
        'Traffic_Severity': ext['Traffic_Severity'],
        'Civic_Disruption_Severity': ext['Civic_Disruption_Severity'],
        'Journey_Stage_Group': stage_group,
        'Latitude': lat if pd.notna(lat) else 19.0,
        'Longitude': lng if pd.notna(lng) else 74.0,
        'Has_GPS_Coordinates': has_gps,
        'Elapsed_Hours': round(elapsed_hours, 2),
        'Planned_Hours': round(planned_hours, 2),
        'Elapsed_Frac': round(elapsed_frac, 4),
        'Remaining_KM': round(remaining_km, 2),
        'Traversed_KM': round(traversed_km, 2),
        'Total_Route_KM': round(total_route_km, 2),
        'Corridor_Planned_Speed_Kmph': round(planned_speed, 2),
        'Route_Progress_Ratio': round(route_progress, 4),
        'Required_Speed_Kmph': round(req_speed, 2),
        'Effective_Speed_Kmph': round(eff_speed, 2),
        'Behind_Pace': int(behind_pace),
        'Origin_Dwell_Hours': round(origin_dwell_hours, 2),
        'Buffer_Hours': round(buffer_hours, 2),
        'Proximity_Buffer_Flag': prox_buffer,
        'Pings_Last_3H': int(latest_event.get('pings_3h', 3 if has_gps else 0)),
        'Pings_Last_6H': int(latest_event.get('pings_6h', 6 if has_gps else 0)),
        'Telemetry_Freshness_Min': float(latest_event.get('telemetry_freshness', 15.0 if has_gps else 240.0)),
        'Hour_Of_Day': as_of_dt.hour,
        'Day_Of_Week': as_of_dt.weekday(),
        'Month': as_of_dt.month,
        'BA_Historical_Delay_Rate': round(ba_rate, 4),
        'BA_Historical_Trips': 25,
        'Shipper_Historical_Delay_Rate': round(shipper_rate, 4),
        'Corridor_Baseline_Hours': round(total_route_km / 35.0, 2),
        'Corridor_Historical_Delay_Rate': round(corridor_rate, 4),
        'Weather_Rain_MM': ext['Weather_Rain_MM'],
        'Weather_Temp_C': ext['Weather_Temp_C'],
        'Weather_Wind_Kmph': ext['Weather_Wind_Kmph'],
        'Weather_Severe_Flag': ext['Weather_Severe_Flag'],
        'Weather_Fog_Flag': ext['Weather_Fog_Flag'],
        'Traffic_Congestion_Ratio': ext['Traffic_Congestion_Ratio'],
        'Civic_Disruption_Alert': ext['Civic_Disruption_Alert']
    }
