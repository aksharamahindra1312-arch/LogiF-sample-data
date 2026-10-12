"""
Main production inference engine returning the complete 21-output LF contract.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import extract_point_in_time_features
from src.reason_attribution import attribute_delay_reason
from src.recommendation_engine import generate_recommendation

OUTPUT_CONTRACT_FIELDS = [
    'Consignment_Number', 'As_Of', 'Stage', 'Route_Type', 'Current_Location', 'Latitude', 'Longitude',
    'Delay_Risk_Percentage', 'Risk_Level', 'Revised_ETA', 'ETA_Earliest_P10', 'ETA_Latest_P90',
    'Current_ETA', 'Elapsed_Frac', 'Behind_Pace', 'Remaining_KM', 'ETA_Confidence',
    'Anomaly_Flag', 'Delay_Reason', 'Recommended_Action', 'Delay_Prediction', 'Predicted_Status'
]

class LogiFreightInferenceService:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.clf_model = joblib.load(os.path.join(models_dir, 'delay_classifier.joblib'))
        self.calibrator = joblib.load(os.path.join(models_dir, 'probability_calibrator.joblib'))
        self.p10_model = joblib.load(os.path.join(models_dir, 'eta_p10_regressor.joblib'))
        self.p50_model = joblib.load(os.path.join(models_dir, 'eta_p50_regressor.joblib'))
        self.p90_model = joblib.load(os.path.join(models_dir, 'eta_p90_regressor.joblib'))
        self.anomaly_detector = joblib.load(os.path.join(models_dir, 'anomaly_detector.joblib'))
        self.encoder = joblib.load(os.path.join(models_dir, 'ordinal_encoder.joblib'))
        
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.feature_cols = self.metadata['feature_columns']
        self.categorical_cols = self.metadata['categorical_cols']
        self.threshold = self.metadata['best_threshold']
        
    def predict(self, live_input: dict, as_of_iso: str = None, tomtom_key: str = None) -> dict:
        """Executes production inference and returns all 21 LF outputs."""
        as_of_dt = datetime.fromisoformat(as_of_iso) if as_of_iso else datetime.now()
        
        disp_dt = pd.to_datetime(live_input.get('dispatch_date'), errors='coerce')
        if pd.isna(disp_dt):
            disp_dt = as_of_dt - timedelta(hours=4)
        eta_dt = pd.to_datetime(live_input.get('original_eta'), errors='coerce')
        if pd.isna(eta_dt):
            eta_dt = as_of_dt + timedelta(hours=24)
            
        c_no = str(live_input.get('Consignment No', live_input.get('Consignment_Number', 'UNKNOWN')))
        stage = str(live_input.get('event_type', live_input.get('Stage', 'IN_TRANSIT')))
        current_loc = str(live_input.get('event_location', live_input.get('Current_Location', 'En-route Checkpoint')))
        
        lat = pd.to_numeric(live_input.get('lat', live_input.get('Latitude')), errors='coerce')
        lng = pd.to_numeric(live_input.get('lng', live_input.get('Longitude')), errors='coerce')
        
        # 1. Feature Engineering
        features = extract_point_in_time_features(live_input, as_of_dt, disp_dt, eta_dt, tomtom_key=tomtom_key)
        
        # 2. Preprocess & Encode
        df_feat = pd.DataFrame([features])
        encoded_cat = self.encoder.transform(df_feat[self.categorical_cols].astype(str))
        for i, col in enumerate(self.categorical_cols):
            df_feat[col] = encoded_cat[:, i]
            
        X_vec = df_feat[self.feature_cols]
        
        # 3. Model A & B: Calibrated Delay Risk
        raw_prob = self.clf_model.predict_proba(X_vec)[0, 1]
        calib_prob = float(self.calibrator.predict(np.array([raw_prob]))[0])
        delay_risk_pct = round(calib_prob * 100.0, 1)
        
        is_delayed = bool(calib_prob >= self.threshold)
        delay_prediction = 'DELAYED' if is_delayed else 'ON_TIME'
        
        if delay_risk_pct < 20.0:
            risk_level = 'LOW'
        elif delay_risk_pct < 50.0:
            risk_level = 'MEDIUM'
        elif delay_risk_pct < 80.0:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'
            
        # 4. Model C: Quantile ETA Regressors
        p10_rem = float(self.p10_model.predict(X_vec)[0])
        p50_rem = float(self.p50_model.predict(X_vec)[0])
        p90_rem = float(self.p90_model.predict(X_vec)[0])
        
        # Non-Crossing Monotonicity Enforcement
        p10_rem = max(0.1, min(p10_rem, p50_rem))
        p50_rem = max(0.1, p50_rem)
        p90_rem = max(p50_rem, p90_rem)
        
        revised_eta_dt = as_of_dt + timedelta(hours=p50_rem)
        eta_p10_dt = as_of_dt + timedelta(hours=p10_rem)
        eta_p90_dt = as_of_dt + timedelta(hours=p90_rem)
        
        # 5. Anomaly Detection
        anom_feat = ['Effective_Speed_Kmph', 'Telemetry_Freshness_Min', 'Elapsed_Frac', 'Behind_Pace', 'Route_Progress_Ratio']
        anom_score = self.anomaly_detector.predict(X_vec[anom_feat])[0]
        anomaly_flag = bool(anom_score == -1 or features['Telemetry_Freshness_Min'] > 240.0)
        
        # 6. Reason Attribution & Action Recommendation
        delay_reason = attribute_delay_reason(is_delayed, features)
        rec_action = generate_recommendation(delay_reason, risk_level, features)
        
        interval_width = p90_rem - p10_rem
        confidence_score = round(float(np.clip(100.0 / (1.0 + (interval_width / max(1.0, p50_rem)) * 0.5), 20.0, 99.0)), 1)
        
        predicted_status = 'ON_TIME' if not is_delayed else ('SEVERELY_DELAYED' if delay_reason == 'EXTENDED_STOPPAGE' else 'DELAYED')
        
        output = {
            'Consignment_Number': c_no,
            'As_Of': as_of_dt.isoformat(),
            'Stage': stage,
            'Route_Type': features.get('Route_Type', 'UNKNOWN'),
            'Current_Location': current_loc,
            'Latitude': lat if pd.notna(lat) else None,
            'Longitude': lng if pd.notna(lng) else None,
            'Delay_Risk_Percentage': delay_risk_pct,
            'Risk_Level': risk_level,
            'Revised_ETA': revised_eta_dt.isoformat(),
            'ETA_Earliest_P10': eta_p10_dt.isoformat(),
            'ETA_Latest_P90': eta_p90_dt.isoformat(),
            'Current_ETA': eta_dt.isoformat(),
            'Elapsed_Frac': round(features['Elapsed_Frac'], 3),
            'Behind_Pace': bool(features['Behind_Pace']),
            'Remaining_KM': round(features['Remaining_KM'], 1),
            'ETA_Confidence': confidence_score,
            'Anomaly_Flag': anomaly_flag,
            'Delay_Reason': delay_reason,
            'Recommended_Action': rec_action,
            'Delay_Prediction': delay_prediction,
            'Predicted_Status': predicted_status
        }
        
        # Validate 21-output contract
        for k in OUTPUT_CONTRACT_FIELDS:
            if k not in output:
                raise ValueError(f"Contract Violation: Missing output field '{k}'")
                
        return output
