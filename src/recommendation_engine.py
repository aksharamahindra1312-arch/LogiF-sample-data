"""
Prescriptive decision and recommendation policy engine.
"""

def generate_recommendation(delay_reason: str, risk_level: str, features: dict) -> str:
    """Generates actionable prescriptive operational recommendations."""
    if delay_reason == 'ON_TRACK':
        return 'NO_ACTION_REQUIRED: Vehicle progressing normally on schedule.'
    elif delay_reason == 'CIVIC_POLITICAL_DISRUPTION':
        return 'RECOMMENDATION: Activate alternate highway corridor bypass to avoid active civic blockade.'
    elif delay_reason == 'SEVERE_WEATHER':
        rain = features.get('Weather_Rain_MM', 0)
        return f"RECOMMENDATION: Notify consignee of weather slowdown (Precipitation: {rain:.1f}mm). Update dock receiving window."
    elif delay_reason == 'TRAFFIC_CONGESTION':
        ratio = features.get('Traffic_Congestion_Ratio', 0.5)
        return f"RECOMMENDATION: Highway congestion detected (Speed ratio: {ratio:.2f}). Monitor choke point clearance."
    elif delay_reason == 'EXTENDED_STOPPAGE':
        return 'RECOMMENDATION: Contact Transporter BA immediately to investigate vehicle stoppage and driver status.'
    elif delay_reason == 'ORIGIN_DISPATCH_DELAY':
        return 'RECOMMENDATION: Origin dispatch delayed past planned window. Reallocate downstream warehouse slot.'
    elif delay_reason == 'PLANT_GATE_DWELL':
        return 'RECOMMENDATION: Gate/dock congestion at plant. Coordinate with yard manager to expedite bay allocation.'
    else:
        return 'RECOMMENDATION: Dwell accumulation en-route. Request expedited milestone tracking from carrier.'
