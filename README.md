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

## Phase 1 — Engineering Hygiene (Complete)

| Item | Implementation |
|------|----------------|
| **Exact package versions** | All deps pinned in `pyproject.toml` (e.g., `pandas==2.3.3`, `numpy==2.3.5`, `tensorflow==2.21.0`, `pandera==0.33.0`, `pydantic==2.12.2`, `structlog==26.1.0`, `mlflow==3.15.2`, `dvc==3.67.1`, `optuna==4.9.0`, `shap==0.52.0`, `pytest==8.4.2`, `ruff==0.11.10`, `black==26.3.1`, `mypy==1.16.1`, `pre-commit==4.2.0`) |
| **Pre-commit hooks** | `.pre-commit-config.yaml`: ruff (lint+format), black, mypy, isort, trailing-whitespace, end-of-file-fixer, check-yaml/toml, check-merge-conflict, detect-private-key, debug-logger |
| **CI Pipeline** | `.github/workflows/ci.yml`: lint → test → model-tests → build → docker → security (trivy) |
| **Pydantic Settings** | `src/mfp/core/config.py`: structured config with env prefixes (`MFP_DATA_`, `MFP_SPLIT_`, `MFP_FORECASTER_`, `MFP_CLASSIFIER_`) |
| **Structured Logging** | `src/mfp/core/logging.py`: structlog JSON output with request_id, model_version context |
| **Deterministic Seeds** | `src/mfp/core/seeds.py`: `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`, global seed for random/numpy/TF |
| **Typer CLI** | `src/mfp/cli.py`: `mfp train`, `mfp evaluate`, `mfp backfill` commands |
| **Unit Tests** | `tests/test_correctness.py`: labeler (horizon logic, sanity report), sequences (shape, values), temporal splits (ratios, purge gap, leakage detection), scaler (train-only fit, persistence) |

## CI/CD Pipeline

`.github/workflows/ci.yml`:
1. **Lint**: ruff, black, mypy (strict), pre-commit hooks
2. **Unit & Integration Tests**: pytest with coverage
3. **Model Correctness Tests**: Baseline gate (LSTM MSE < Persistence MSE), XAI sanity (Phase 3)
4. **Build Package**: `python -m build` → wheel/sdist artifacts
5. **Docker Build**: API + Train images (continue-on-error for missing Dockerfiles)
6. **Security Scan**: Trivy filesystem scan → SARIF upload to GitHub Security

## Pre-commit Setup

```bash
pip install pre-commit==4.2.0
pre-commit install
pre-commit run --all-files
```

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