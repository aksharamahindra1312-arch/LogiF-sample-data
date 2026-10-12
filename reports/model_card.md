# Logi Freight Model Card

## Model Overview
Logi Freight is a production-grade multi-task machine learning ensemble designed for real-time logistics forecasting and delivery risk tracking.

## Intended Use
- Real-time in-transit delay probability estimation.
- Quantile arrival time forecasting (P10, P50, P90).
- Telemetry movement anomaly detection.
- Root cause delay attribution and prescriptive operational guidance.

## Model Architecture
- **Delay Classifier**: LightGBM Gradient Boosted Decision Tree (`objective='binary'`, `num_leaves=45`).
- **Probability Calibrator**: Non-parametric Isotonic Regression (`CalibratedClassifierCV`).
- **ETA Regressors**: 3 separate LightGBM Quantile Regressors with Pinball loss ($\\alpha = 0.10, 0.50, 0.90$) with strict non-crossing monotonicity enforcement ($P10 \\le P50 \\le P90$).
- **Anomaly Detector**: Scikit-Learn `IsolationForest` on spatial movement pace and telemetry freshness.
- **Reason Attribution & Recommender**: TreeSHAP + multi-factor causal engine.

## Performance Summary
- **Historical Holdout Test (May - Sep 2025)**: 95.39% Accuracy | ROC-AUC: 0.9887 | ETA MedAE: 3.29 hours.
- **External Real-World Test (live data test- ct.csv)**: 92.50% Snapshot Accuracy | 85.00% Consignment Accuracy | Recall: 100.0% | ROC-AUC: 0.9700.

## Known Limitations & Considerations
- **Early-Journey Uncertainty**: Delay prediction uncertainty is naturally higher during the first 15-35% of a journey ($F_1 \\approx 0.78$) and narrows smoothly as the vehicle approaches destination ($F_1 \\approx 0.97$).
- **External API Fallbacks**: When live GPS coordinates are unavailable or external APIs timeout, the system gracefully reverts to empirical corridor baseline speed and regional climate averages.
