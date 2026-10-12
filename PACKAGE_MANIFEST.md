# Logi Freight Package Manifest

| File / Path | Component | Description |
|---|---|---|
| `README.md` | Documentation | Complete project documentation, architecture, CLI usage, and API guide. |
| `requirements.txt` | Dependencies | Production Python package dependencies. |
| `config.yaml` | Configuration | Model thresholds, risk tiers, and external API endpoint configurations. |
| `.gitignore` | Version Control | Standard gitignore excluding raw telemetry data and temporary files. |
| `PACKAGE_MANIFEST.md` | Manifest | This inventory of package contents. |
| `src/__init__.py` | Source Package | Package initialization. |
| `src/utils.py` | Core Utilities | Haversine distance, configuration loader, and math helpers. |
| `src/data_validation.py` | Ingestion | 26-column LF input schema validation. |
| `src/data_preprocessing.py`| Data Prep | Stream cleaning, timestamp parsing, and GPS sanitation. |
| `src/leakage_prevention.py`| Governance | Automated point-in-time leakage audit checks. |
| `src/snapshot_generator.py`| Data Prep | Reconstructs realistic point-in-time in-transit live snapshots. |
| `src/target_definition.py` | Targets | Mathematical ground-truth target calculation. |
| `src/weather_openmeteo.py` | External Data | Live Open-Meteo current weather connector with safe fallbacks. |
| `src/traffic_tomtom.py` | External Data | Live TomTom Traffic Flow API connector. |
| `src/civic_gdelt.py` | External Data | GDELT regional civic/political disruption index connector. |
| `src/external_features.py` | Features | Fuses environmental and civic signals into model features. |
| `src/historical_features.py`| Features | Precomputed carrier, shipper, and corridor baseline priors. |
| `src/feature_engineering.py`| Features | Constructs the complete 41-feature predictive vector. |
| `src/delay_model.py` | ML Models | LightGBM Binary Classifier builder. |
| `src/calibration.py` | ML Models | Isotonic probability calibrator builder. |
| `src/eta_models.py` | ML Models | P10, P50, P90 Quantile Regressors builder. |
| `src/anomaly_detection.py` | ML Models | Isolation Forest movement anomaly detector. |
| `src/train.py` | Training | End-to-end multi-task training workflow. |
| `src/validate.py` | Validation | Threshold optimization on validation partition. |
| `src/test.py` | Evaluation | Benchmark evaluation metrics computation. |
| `src/reason_attribution.py`| XAI | Evidence-based root-cause delay attribution. |
| `src/recommendation_engine.py`| Action Engine | Prescriptive operational recommendation policy engine. |
| `src/inference.py` | Production | Core live inference engine returning the 21 LF outputs. |
| `src/pipeline.py` | Production | Batch processing pipeline for multi-shipment datasets. |
| `models/` | Serialized Models| Calibrated LightGBM models, encoders, and `model_metadata.json`. |
| `scripts/predict_live.py` | CLI | Main production command-line inference tool. |
| `scripts/train_model.py` | CLI | Standalone training runner. |
| `scripts/evaluate_model.py`| CLI | Standalone evaluation runner. |
| `scripts/validate_dataset.py`| CLI | Standalone schema validation runner. |
| `tests/` | Test Suite | Automated pytest suite for contract, inference, and leakage guards. |
| `examples/` | Examples | Sample input payload and reference 21-output contract JSON. |
| `reports/` | Reports | `model_card.md`, `model_metrics.json`, and `feature_metadata.json`. |
