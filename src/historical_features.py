"""
Leakage-safe expanding window historical aggregate features.
"""

class HistoricalPriorsEngine:
    """Stores precomputed carrier, shipper, and corridor reliability priors."""
    def __init__(self):
        self.ba_priors = {}
        self.shipper_priors = {}
        self.corridor_priors = {}
        
    def get_priors(self, ba_name: str, shipper: str, origin: str, dest: str) -> dict:
        ba_str = str(ba_name).strip().lower()
        sh_str = str(shipper).strip().lower()
        corr_key = f"{str(origin).strip().lower()}->{str(dest).strip().lower()}"
        
        return {
            'BA_Historical_Delay_Rate': self.ba_priors.get(ba_str, {}).get('delay_rate', 0.28),
            'BA_Historical_Trips': self.ba_priors.get(ba_str, {}).get('total_trips', 25),
            'Shipper_Historical_Delay_Rate': self.shipper_priors.get(sh_str, {}).get('delay_rate', 0.25),
            'Corridor_Baseline_Hours': self.corridor_priors.get(corr_key, 24.0)
        }
