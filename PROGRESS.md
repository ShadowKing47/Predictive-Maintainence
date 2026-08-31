# MFP Project Progress Tracker

## Project Overview
**Machine Failure Prediction (MFP)** - Production-grade predictive maintenance system for industrial machines.
Sensor forecasting → failure-risk classification.

---

## Completed Phases

### ✅ Phase 0 — Correctness Fixes (Complete)
| Bug | Fix |
|-----|-----|
| C1 Random `Status` labels | Horizon-based risk labels: `risk[t] = 1` if failure in `(t, t+horizon]` |
| C2 Scaler fit on full data | `FeatureScaler.fit()` on train split only |
| C3 Overlapping sequences | `temporal_split_with_purge()` with `purge_gap = seq_len + horizon` |
| C4 Test double-dipped | 3-way temporal split: train/val/test (test touched once) |
| C5 Wrong loss/metrics | Forecaster: MSE+MAE; Classifier: BCE + PR-AUC/ROC-AUC/Precision/Recall |
| H1-H8 Engineering bugs | Fixed tuple unpack, attr names, time-ordered splits, no `plt.show()` |

### ✅ Phase 1 — Engineering Hygiene (Complete)
| Item | Implementation |
|------|----------------|
| **Exact package versions** | All deps pinned in `pyproject.toml` |
| **Pre-commit hooks** | `.pre-commit-config.yaml`: ruff, black, mypy, isort, etc. |
| **CI Pipeline** | `.github/workflows/ci.yml`: lint → test → model-tests → build → docker → security |
| **Pydantic Settings** | `src/mfp/core/config.py`: structured config with env prefixes |
| **Structured Logging** | `src/mfp/core/logging.py`: structlog JSON output with request_id, model_version |
| **Deterministic Seeds** | `src/mfp/core/seeds.py`: `TF_DETERMINISTIC_OPS=1`, global seed |
| **Typer CLI** | `src/mfp/cli.py`: `mfp train`, `mfp evaluate`, `mfp backfill` |
| **Unit Tests** | `tests/test_correctness.py`: labeler, sequences, temporal splits, scaler |

### ✅ Phase 2 — Hyperparameter Tuning & Experiment Tracking (Complete)
| Component | Implementation |
|-----------|----------------|
| **Optuna Integration** | `src/mfp/models/tuning.py`: TPE sampler + Median pruner for forecaster & classifier |
| **Hyperparameter Spaces** | LSTM units, dropout, LR, gradient clip (forecaster); hidden units, dropout, LR (classifier) |
| **MLflow Tracking** | Experiment logging for models, metrics, params, artifacts |
| **Walk-Forward CV** | `walk_forward_cv()` + `run_walk_forward_training()` with purge gaps |
| **Config Extensions** | `OptunaConfig` in `src/mfp/core/config.py` (enabled, n_trials, timeout, sampler, pruner, direction, metric) |
| **CLI Enhancements** | `--optuna`, `--walk-forward`, `--n-splits` flags on `mfp train` |

---

## New Files Added (Phase 2)
- `src/mfp/models/tuning.py` — Optuna objectives, MLflow logging, walk-forward CV
- `.github/workflows/ci.yml` — CI/CD pipeline
- `.pre-commit-config.yaml` — Pre-commit hooks
- `tests/test_models.py` — Model unit tests
- `tests/test_correctness.py` — Data pipeline correctness tests

## Modified Files (Phase 2)
- `src/mfp/core/config.py` — Added `OptunaConfig` class
- `src/mfp/models/__init__.py` — Exported tuning functions
- `src/mfp/models/train.py` — Integrated Optuna tuning + MLflow + walk-forward
- `src/mfp/cli.py` — Added `--optuna`, `--walk-forward`, `--n-splits` options
- `src/mfp/models/forecast.py` — Fixed Path-based save/load (ruff PTH fixes)
- `src/mfp/models/risk.py` — Fixed Path-based save/load (ruff PTH fixes)
- `pyproject.toml` — Added N803/N806 to ruff ignore

---

## Pending Phases

### ⏳ Phase 3 — Explainability
- [ ] SHAP explanations (TreeExplainer for classifier, DeepExplainer for forecaster)
- [ ] Integrated Gradients for forecaster
- [ ] `src/mfp/explain/` module
- [ ] XAI sanity checks in CI

### ⏳ Phase 4 — Serving (FastAPI)
- [ ] FastAPI app with `/v1/forecast`, `/v1/risk`, `/v1/predict`, `/v1/explain`, `/v1/jobs`
- [ ] JWT/API-key authentication
- [ ] OOD detection (Mahalanobis distance on latent space)
- [ ] Fallback ladder (persistence → ridge → LSTM)
- [ ] Rate limiting (slowapi)
- [ ] Circuit breaker (pybreaker)

### ⏳ Phase 5 — Production Hardening
- [ ] Audit logging → Redis → DB
- [ ] Request/response logging with correlation IDs
- [ ] Health checks + readiness probes

### ⏳ Phase 6 — Monitoring & Drift Detection
- [ ] Prometheus metrics (RED + model metrics)
- [ ] PSI per feature, prediction-distribution KL
- [ ] Alert rules (P1/P2/P3)
- [ ] Drift jobs → retrain triggers

### ⏳ Deployment
- [ ] Multi-stage Dockerfiles (`docker/Dockerfile.api`, `docker/Dockerfile.train`)
- [ ] `docker/compose.yaml` (API + Redis + Prometheus + Grafana)

---

## Commands Reference

```bash
# Install
pip install -e .

# Train (single split)
mfp train --data machine_data.csv --artifacts ./artifacts

# Train with Optuna tuning
mfp train --data machine_data.csv --artifacts ./artifacts --optuna

# Train with walk-forward CV
mfp train --data machine_data.csv --artifacts ./artifacts --walk-forward --n-splits 5

# Combine both
mfp train --data machine_data.csv --artifacts ./artifacts --optuna --walk-forward --n-splits 5

# Run tests
pytest tests/test_correctness.py -v
pytest tests/test_models.py -v
pytest -v

# Lint & format
ruff check src/ --fix
black src/
mypy src/
```

---

## Key Configuration (env-overridable)
```bash
# Data
MFP_DATA_TEMP_THRESHOLD=90
MFP_DATA_PRESS_THRESHOLD=110
MFP_DATA_HORIZON=30

# Splits
MFP_SPLIT_TRAIN_RATIO=0.7
MFP_SPLIT_VAL_RATIO=0.15
MFP_SPLIT_TEST_RATIO=0.15
MFP_SPLIT_SEQ_LEN=10
MFP_SPLIT_PURGE_GAP=40

# Forecaster
MFP_FORECASTER_LSTM_UNITS=[128,128]
MFP_FORECASTER_DROPOUT=0.2
MFP_FORECASTER_EPOCHS=50
MFP_FORECASTER_LEARNING_RATE=1e-3

# Classifier
MFP_CLASSIFIER_HIDDEN_UNITS=[128,64]
MFP_CLASSIFIER_DROPOUT=0.3

# Optuna
MFP_OPTUNA_ENABLED=true
MFP_OPTUNA_N_TRIALS=50
MFP_OPTUNA_TIMEOUT=3600
MFP_OPTUNA_DIRECTION=minimize
MFP_OPTUNA_METRIC=val_loss
```

---

## Architecture Summary
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