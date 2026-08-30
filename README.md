# Machine Failure Prediction (MFP)

Production-grade predictive maintenance system for industrial machines. Sensor forecasting → failure-risk classification.

## Architecture

```
Telemetry (CSV) → Ingestion+Validation → Feature+Label Pipeline → Training (Walk-forward CV)
                                                                  ↓
MLflow Registry ← Versioned Artifacts (model+scaler+threshold) ← Baseline Gates
                                                                  ↓
                                              ┌────────────────── Serving (FastAPI) ──────────────────┐
                                              │ /v1/forecast /v1/risk /v1/predict /v1/explain /v1/jobs │
                                              │ JWT/API-key auth │ Rate limiting │ OOD detection      │
                                              │ Fallback ladder  │ Audit log → Redis → DB              │
                                              └────────────────── Monitoring ───────────────────────────┘
                                                        ↓
                                               Prometheus/Grafana → Drift jobs → Retrain triggers
                                                        ↓
                                               CI/CD: Canary rollout → Auto-rollback
```

## Quickstart

```bash
# Install
pip install -e .

# Train (Phase 0-2)
mfp train --data machine_data.csv --artifacts ./artifacts

# Serve (Phase 4+)
# uvicorn mfp.serving.api.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
mfp/
├── pyproject.toml              # Package config, deps, tools
├── configs/                    # Hydra/pydantic-settings configs
├── src/mfp/
│   ├── core/                   # Config, seeds, logging, exceptions
│   ├── data/                   # Ingest, Pandera validation, horizon-based labeling
│   ├── features/               # Train-only scaling, purged temporal sequences
│   ├── models/                 # LSTM forecaster, Risk classifier (sigmoid+BCE), baselines
│   ├── explain/                # SHAP + Integrated Gradients (Phase 3)
│   ├── serving/                # FastAPI, auth, OOD, fallback (Phase 4-5)
│   ├── monitoring/             # Prometheus metrics, drift detection (Phase 6)
│   └── cli.py                  # `mfp train|evaluate|backfill`
├── tests/                      # Unit, integration, model, load, chaos tests
├── docker/                     # Multi-stage Dockerfiles, compose
└── .github/workflows/          # CI/CD pipelines
```

## Phase 0 — Correctness Fixes (Complete)

| Bug | Fix |
|-----|-----|
| C1 Random `Status` labels | Horizon-based risk labels: `risk[t] = 1` if failure in `(t, t+horizon]` |
| C2 Scaler fit on full data | `FeatureScaler.fit()` on train split only |
| C3 Overlapping sequences | `temporal_split_with_purge()` with `purge_gap = seq_len + horizon` |
| C4 Test double-dipped | 3-way temporal split: train/val/test (test touched once) |
| C5 Wrong loss/metrics | Forecaster: MSE+MAE; Classifier: BCE + PR-AUC/ROC-AUC/Precision/Recall |
| H1-H8 Engineering bugs | Fixed tuple unpack, attr names, time-ordered splits, no `plt.show()` |

## Key Design Decisions

- **Labeling**: Rule-based horizon labels (threshold breach within H steps) — deterministic fallback + XAI oracle
- **Forecaster**: Predicts 12 sensor values only (no Status); classifier trains on *actual* readings
- **Split**: Purged walk-forward CV; embargo of `seq_len + horizon` at every boundary
- **Baselines**: Persistence (naive) + Ridge; LSTM must beat persistence to justify existence
- **Artifacts**: Atomic versioned bundle = {model + scaler + feature list + threshold + manifest hash}

## Configuration

All hyperparameters, thresholds, paths in `src/mfp/core/config.py` (pydantic-settings, env-overridable):

```python
# Example overrides via env
MFP_DATA_TEMP_THRESHOLD=90
MFP_SPLIT_PURGE_GAP=50
MFP_FORECASTER_LSTM_UNITS=[128,128]
```

## Testing

```bash
# Unit + correctness tests
pytest tests/test_correctness.py -v

# Model tests
pytest tests/test_models.py -v

# All tests
pytest -v
```

## CI/CD Pipeline

`.github/workflows/ci.yml`:
1. Lint (ruff, black, mypy)
2. Unit + integration tests
3. Build Docker images
4. Security scan (trivy)
5. Model tests (directional, invariance, baseline gates, XAI sanity)

## Monitoring (Phase 6)

- **RED metrics**: RPS, latency p50/p95/p99, error rates
- **Model metrics**: Prediction dist, % risk, OOD rate, calibration error, fallback activations
- **Drift**: PSI per feature, prediction-distribution KL vs reference
- **Alerts**: P1 (error rate >1%, fallback >0), P2 (PSI >0.2, DLQ >100), P3 (calibration drift)

## Deployment

```bash
# Build
docker build -f docker/Dockerfile.api -t mfp-api .
docker build -f docker/Dockerfile.train -t mfp-train .

# Compose (API + Redis + Prometheus + Grafana)
docker compose -f docker/compose.yaml up
```

## License

MIT