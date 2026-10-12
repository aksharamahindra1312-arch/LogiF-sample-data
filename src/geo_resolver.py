"""
Geographic coordinate resolution and depot alias normalization service.
Maps industrial hubs, plant depots, and cities to real geographic coordinates.
"""

import re
import math
from typing import Tuple, Optional, Dict, Any
from src.utils import haversine_distance

# Coordinates for major Indian automotive plants, depots, and logistics hubs
DEPOT_ALIASES: Dict[str, Tuple[float, float]] = {
    # Chakan Industrial Area / M&M Auto Sector Plants
    'AS CHAKAN PLANT': (18.7428, 73.8554),
    'CK01': (18.7428, 73.8554),
    'MLLB_ASOB_CK01': (18.7428, 73.8554),
    'IUTN-CK01-MLLB': (18.7428, 73.8554),
    'CHAKAN PLANT': (18.7428, 73.8554),
    'CHAKAN': (18.7562, 73.8554),
    'TALEGAON MIDC': (18.7300, 73.6800),
    'KANHE': (18.7615, 73.8351),

    # Kandivali / Mumbai Plant (A002)
    'AS KANDIVILI PLANT': (19.2042, 72.8673),
    'AS KANDIVILI PLANT(A002)': (19.2042, 72.8673),
    'A002': (19.2042, 72.8673),
    'MLLB_ASOB_A002': (19.2013, 72.8530),
    'KANDIVALI': (19.2062, 72.8530),
    'TALOJA': (19.0688, 73.1114),
    'TALOJA OFFICE': (19.0688, 73.1114),
    'CORPORATE OFFICE, MUMBAI': (19.0176, 72.8300),

    # Nashik / Satpur MIDC Plant (A003)
    'AS NASIK PLANT': (20.0021, 73.7898),
    'AS NASIK PLANT-A003-MLLB': (20.0021, 73.7898),
    'A003': (20.0021, 73.7898),
    'MLLB_ASOB_A003': (20.0021, 73.7898),
    'NASHIK': (19.9975, 73.7898),
    'NASIK': (19.9975, 73.7898),

    # Igatpuri Engine Plant (A004)
    'AS IGATPURI': (19.6866, 73.5447),
    'AS IGATPURI-A004-MLLB': (19.6866, 73.5447),
    'IUTN-A004-MLLB': (19.6866, 73.5447),
    'A004': (19.6866, 73.5447),
    'IGATPURI': (19.6967, 73.5594),

    # Zaheerabad Plant (A005, Telangana)
    'AS ZAHEERABAD PLANT': (17.6795, 77.6083),
    'AS ZAHEERABAD PLANT-A005-MLLB': (17.6795, 77.6083),
    'A005': (17.6795, 77.6083),
    'ZAHEERABAD': (17.6833, 77.6083),

    # Haridwar Plant (A010, Uttarakhand)
    'AS HARIDWAR PLANT': (29.9490, 78.1642),
    'MLLB_ASOB_A010': (29.9490, 78.1642),
    'A010': (29.9490, 78.1642),
    'HARIDWAR': (29.9457, 78.1642),

    # Mohali Tractor / Swaraj Division (FSPB, Punjab)
    'MLLB_SDOB_FSPB': (30.7264, 76.7179),
    'MOHALI': (30.7046, 76.7179),
    'SWARAJ': (30.7046, 76.7179),

    # Nagpur Farm Equipment Sector (F003)
    'FES - NAGPUR -F003-MLLB': (21.1284, 79.0882),
    'F003': (21.1284, 79.0882),
    'NAGPUR': (21.1458, 79.0882),

    # Jaipur Farm Equipment Sector (F007)
    'FES - JAIPUR -F007-MLLB': (26.9194, 75.7873),
    'F007': (26.9194, 75.7873),
    'JAIPUR': (26.9124, 75.7873),

    # Belgaum / Netalkar Works
    'NETALKAR WORKS ENGG-DN016-MLLB': (15.8442, 74.4977),
    'BELGAUM': (15.8497, 74.4977),
    'BELAGAVI': (15.8497, 74.4977),

    # Railway / Freight Terminals
    'FATUHA(FUT)': (25.5017, 85.3120),
    'FATUHA': (25.5017, 85.3120),
    'FUT': (25.5017, 85.3120),
    'CHITPUR(CP)': (22.5666, 88.3740),
    'CHITPUR': (22.5666, 88.3740),
    'CP': (22.5666, 88.3740),
}

# Exhaustive coordinate mapping for Indian logistics cities
CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    'chakan': (18.7562, 73.8554),
    'pune': (18.5204, 73.8567),
    'mumbai': (19.0760, 72.8777),
    'nashik': (19.9975, 73.7898),
    'nasik': (19.9975, 73.7898),
    'kandivali': (19.2062, 72.8530),
    'igatpuri': (19.6967, 73.5594),
    'zaheerabad': (17.6833, 77.6083),
    'haridwar': (29.9457, 78.1642),
    'mohali': (30.7046, 76.7179),
    'nagpur': (21.1458, 79.0882),
    'jaipur': (26.9124, 75.7873),
    'kolhapur': (16.7050, 74.2433),
    'ahmedabad': (23.0225, 72.5714),
    'bhopal': (23.2599, 77.4126),
    'patna': (25.5941, 85.1376),
    'fatuha': (25.5017, 85.3120),
    'kolkata': (22.5726, 88.3639),
    'chitpur': (22.5666, 88.3740),
    'ranchi': (23.3441, 85.3096),
    'lucknow': (26.8467, 80.9462),
    'hyderabad': (17.3850, 78.4867),
    'indore': (22.7196, 75.8577),
    'guwahati': (26.1445, 91.7362),
    'visakhapatnam': (17.6868, 83.2185),
    'trivandrum': (8.5241, 76.9366),
    'kishanganj': (26.0740, 87.9405),
    'darbhanga': (26.1542, 85.8918),
    'muzaffarpur': (26.1209, 85.3647),
    'gaya': (24.7914, 85.0002),
    'nawada': (24.8856, 85.5434),
    'jalandhar': (31.3260, 75.5762),
    'ernakulam': (9.9816, 76.2999),
    'agartala': (23.8315, 91.2868),
    'dispur': (26.1433, 91.7898),
    'jhansi': (25.4484, 78.5685),
    'lalitpur': (24.6896, 78.4118),
    'mehsana': (23.5880, 72.3693),
    'dehradun': (30.3165, 78.0322),
    'ghaziabad': (28.6692, 77.4538),
    'delhi': (28.6139, 77.2090),
    'new delhi': (28.6139, 77.2090),
    'noida': (28.5355, 77.3910),
    'gurgaon': (28.4595, 77.0266),
    'faridabad': (28.4089, 77.3178),
    'belgaum': (15.8497, 74.4977),
    'belagavi': (15.8497, 74.4977),
    'bhubaneshwar': (20.2961, 85.8245),
    'cuttack': (20.4625, 85.8828),
    'chennai': (13.0827, 80.2707),
    'bengaluru': (12.9716, 77.5946),
    'bangalore': (12.9716, 77.5946),
    'vadodara': (22.3072, 73.1812),
    'surat': (21.1702, 72.8311),
    'rajkot': (22.3039, 70.8022),
    'siliguri': (26.7271, 88.3953),
    'solapur': (17.6599, 75.9064),
    'aurangabad': (19.8762, 75.3433),
    'sangli': (16.8524, 74.5815),
    'satara': (17.6805, 73.9997),
    'amritsar': (31.6340, 74.8723),
    'ludhiana': (30.9010, 75.8573),
    'kanpur': (26.4499, 80.3319),
    'varanasi': (25.3176, 82.9739),
    'agra': (27.1767, 78.0081),
    'bareilly': (28.3670, 79.4304),
    'gorakhpur': (26.7606, 83.3732),
    'allahabad': (25.4358, 81.8463),
    'prayagraj': (25.4358, 81.8463),
    'gwalior': (26.2183, 78.1828),
    'jabalpur': (23.1815, 79.9864),
    'raipur': (21.2514, 81.6296),
    'durg': (21.1904, 81.2849),
    'bilaspur': (22.0797, 82.1409),
    'jamshedpur': (22.8046, 86.2029),
    'dhanbad': (23.7957, 86.4304),
    'asansol': (23.6739, 86.9524),
    'coimbatore': (11.0168, 76.9558),
    'madurai': (9.9252, 78.1198),
    'salem': (11.6643, 78.1460),
    'tiruchirappalli': (10.7905, 78.7047),
    'vijayawada': (16.5062, 80.6480),
    'guntur': (16.3067, 80.4365),
    'warangal': (17.9689, 79.5941),
    'kochi': (9.9312, 76.2673),
    'calicut': (11.2588, 75.7804),
    'thrissur': (10.5276, 76.2144),
    'kollam': (8.8932, 76.6141),
    'kottayam': (9.5916, 76.5222),
    'kanhe': (18.7615, 73.8351),
    'talegaon midc': (18.7300, 73.6800),
    'penukonda': (14.0815, 77.5927),
    'roorkee': (29.8543, 77.8880),
    'rudrapur': (28.9800, 79.4000),
    'haldwani': (29.2183, 79.5130),
    'panvel': (18.9894, 73.1175),
    'thane': (19.2183, 72.9781),
    'bhiwandi': (19.2967, 73.0631),
    'navi mumbai': (19.0330, 73.0297),
    'vasai': (19.3919, 72.8397),
    'alwar': (27.5530, 76.6346),
    'bikaner': (28.0229, 73.3119),
    'jodhpur': (26.2389, 73.0243),
    'udaipur': (24.5854, 73.7125),
    'kota': (25.2138, 75.8648),
    'bhilwara': (25.3475, 74.6408),
    'morbi': (22.8120, 70.8385),
    'jamnagar': (22.4707, 70.0577),
    'bhavnagar': (21.7645, 72.1519),
    'gandhidham': (23.0753, 70.1337),
    'anand': (22.5645, 72.9289),
    'bharuch': (21.7051, 72.9959),
    'ankleshwar': (21.6264, 73.0034),
    'silchar': (24.8333, 92.7789),
    'dibrugarh': (27.4728, 94.9120),
    'shillong': (25.5788, 91.8933),
    'imphal': (24.8170, 93.9368),
    'aizwal': (23.7271, 92.7176),
    'dimapur': (25.9068, 93.7274),
    'itarsi': (22.6127, 77.7615),
    'dewas': (22.9676, 76.0534),
    'ujjain': (23.1765, 75.7885),
    'satna': (24.6005, 80.8322),
    'sagar': (23.8388, 78.7378),
    'panipat': (29.3909, 76.9635),
    'karnal': (29.6857, 76.9905),
    'ambala': (30.3782, 76.7767),
    'hisar': (29.1492, 75.7217),
    'rohtak': (28.8955, 76.6066),
    'sonipat': (28.9931, 77.0151),
    'shimla': (31.1048, 77.1734),
    'solan': (30.9084, 77.0999),
    'mandi': (31.7087, 76.9320),
    'jammu': (32.7266, 74.8570),
    'srinagar': (34.0837, 74.7973),
    'bathinda': (30.2110, 74.9455),
    'patiala': (30.3398, 76.3869),
    'hosur': (12.7409, 77.8253),
    'hubli': (15.3647, 75.1240),
    'mysore': (12.2958, 76.6394),
    'mangalore': (12.9141, 74.8560),
    'puducherry': (11.9416, 79.8083),
}

def normalize_text(text: Optional[str]) -> str:
    """Normalizes location/city text by stripping whitespace and punctuation."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r'[^\w\s]', ' ', text).strip().lower()

def resolve_depot_coordinates(location_str: Optional[str]) -> Optional[Tuple[float, float]]:
    """Resolves coordinates from depot aliases or event location strings."""
    if not location_str or not isinstance(location_str, str):
        return None
    
    clean_upper = location_str.upper()
    for alias, coords in DEPOT_ALIASES.items():
        if alias in clean_upper:
            return coords
            
    return None

def resolve_city_coordinates(city_name: Optional[str], state_name: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Resolves coordinates for a given city name with normalization."""
    if not city_name or not isinstance(city_name, str):
        return None
        
    norm_city = normalize_text(city_name)
    if norm_city in CITY_COORDINATES:
        return CITY_COORDINATES[norm_city]
        
    # Check partial match
    for c, coords in CITY_COORDINATES.items():
        if c in norm_city or norm_city in c:
            return coords
            
    return None

def resolve_location(location_str: Optional[str] = None, city_name: Optional[str] = None, state_name: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Hierarchically resolves coordinates using depot aliases first, then city lookup."""
    if location_str:
        depot_coords = resolve_depot_coordinates(location_str)
        if depot_coords:
            return depot_coords
            
    if city_name:
        city_coords = resolve_city_coordinates(city_name, state_name)
        if city_coords:
            return city_coords
            
    return None

def classify_route_type(origin_city: str, dest_city: str, origin_coords: Optional[Tuple[float, float]] = None,
                        dest_coords: Optional[Tuple[float, float]] = None, route_distance_km: Optional[float] = None) -> str:
    """
    Classifies a logistics journey into:
    - LOCAL_PLANT_TRANSFER: Same city, intra-plant shuttle, or distance <= 40 km.
    - REGIONAL_SHORT_HAUL: Distance between 40 km and 250 km.
    - INTERSTATE_HIGHWAY: Distance > 250 km.
    - UNKNOWN: Missing metadata.
    """
    norm_orig = normalize_text(origin_city)
    norm_dest = normalize_text(dest_city)
    
    if not norm_orig and not norm_dest:
        return 'UNKNOWN'
        
    # Same city / Intra-plant transfers
    if norm_orig and norm_dest and norm_orig == norm_dest:
        return 'LOCAL_PLANT_TRANSFER'
        
    if not origin_coords and norm_orig:
        origin_coords = resolve_city_coordinates(norm_orig)
    if not dest_coords and norm_dest:
        dest_coords = resolve_city_coordinates(norm_dest)
        
    # Check distance if available
    dist = route_distance_km
    if dist is None and origin_coords and dest_coords:
        dist = 1.25 * haversine_distance(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1])
        
    if dist is not None and not math.isnan(dist):
        if dist <= 40.0:
            return 'LOCAL_PLANT_TRANSFER'
        elif dist <= 250.0:
            return 'REGIONAL_SHORT_HAUL'
        else:
            return 'INTERSTATE_HIGHWAY'
            
    # Default based on origin/dest known
    if norm_orig and norm_dest:
        return 'INTERSTATE_HIGHWAY'
        
    return 'UNKNOWN'

def calculate_route_distance(orig_coords: Optional[Tuple[float, float]], dest_coords: Optional[Tuple[float, float]],
                             route_type: str, planned_hours: float = 24.0) -> float:
    """
    Calculates physically realistic total route distance in kilometers.
    Applies a 1.25 highway curvature factor over great-circle distance.
    For local plant transfers, returns empirical shuttle distance (8-15 km).
    """
    if route_type == 'LOCAL_PLANT_TRANSFER':
        if orig_coords and dest_coords:
            gc = haversine_distance(orig_coords[0], orig_coords[1], dest_coords[0], dest_coords[1])
            if not math.isnan(gc) and 0.0 < gc <= 40.0:
                return round(max(5.0, gc * 1.25), 1)
        return 12.0  # Representative intra-plant/same-city transfer distance
        
    if orig_coords and dest_coords:
        gc = haversine_distance(orig_coords[0], orig_coords[1], dest_coords[0], dest_coords[1])
        if not math.isnan(gc):
            return round(max(10.0, gc * 1.25), 1)
            
    # Safe speed-based corridor estimation if coordinates unknown
    # Local: ~20 km/h, Regional/Highway: ~35 km/h
    avg_speed = 20.0 if route_type == 'LOCAL_PLANT_TRANSFER' else (30.0 if route_type == 'REGIONAL_SHORT_HAUL' else 38.0)
    est_dist = planned_hours * avg_speed
    return round(max(15.0, est_dist), 1)

def resolve_remaining_distance(current_lat: Optional[float], current_lng: Optional[float],
                               orig_coords: Optional[Tuple[float, float]], dest_coords: Optional[Tuple[float, float]],
                               route_type: str, total_route_km: float, elapsed_frac: float) -> Tuple[float, float]:
    """
    Resolves remaining KM and traversed KM without defaulting blindly to 320 KM.
    Uses GPS coordinates if available; otherwise uses route progress ratio.
    """
    # If valid current GPS is present and destination coordinates are known
    if current_lat is not None and current_lng is not None and not math.isnan(current_lat) and not math.isnan(current_lng):
        if dest_coords:
            rem_gc = haversine_distance(current_lat, current_lng, dest_coords[0], dest_coords[1])
            if not math.isnan(rem_gc):
                road_factor = 1.15 if route_type == 'LOCAL_PLANT_TRANSFER' else 1.25
                rem_km = max(0.5, round(rem_gc * road_factor, 1))
                if orig_coords:
                    trav_gc = haversine_distance(orig_coords[0], orig_coords[1], current_lat, current_lng)
                    trav_km = max(0.0, round(trav_gc * road_factor, 1)) if not math.isnan(trav_gc) else max(0.0, total_route_km - rem_km)
                else:
                    trav_km = max(0.0, round(total_route_km - rem_km, 1))
                return rem_km, trav_km
                
    # If GPS is not available or at origin, use elapsed fraction and total route distance
    safe_frac = max(0.0, min(1.0, elapsed_frac))
    if route_type == 'LOCAL_PLANT_TRANSFER':
        trav_km = round(safe_frac * total_route_km, 1)
        rem_km = max(1.0, round(total_route_km - trav_km, 1))
    else:
        trav_km = round(safe_frac * total_route_km, 1)
        rem_km = max(5.0, round(total_route_km - trav_km, 1))
        
    return rem_km, trav_km
