# implementation.md — Productionizing the Machine Failure Prediction System

**Project:** Predictive maintenance for industrial machines (sensor forecasting → failure-risk classification)
**Current state:** Single-script Colab notebook (`TimeSeriesPred` LSTM → `predictions.csv` → `Regression` MLP)
**Target state:** Versioned, tested, monitored, explainable ML service with auth, retries, drift detection, and safe rollouts.

---

## 1. Executive Summary

The project currently mixes data prep, labeling, training, prediction, and plotting in one script with no tests, no config, no persistence, no monitoring, and two correctness bugs that invalidate results. This document provides:

1. A **current-state audit** (bugs ranked by severity)
2. A **target architecture** and repo structure
3. A **7-phase implementation plan** with deliverables and Definitions of Done
4. Detailed designs for **request handling, auth, retries/resilience, explainability**
5. A **stage-by-stage failure-mode analysis** (model + system) with mitigations
6. **Monitoring, testing, and rollout** strategy

**Guiding principle:** a model trained on random labels, with leakage, served with no monitoring, is worse than no model — it produces confident wrong answers silently. We fix correctness → reproducibility → observability → scale, in that order.

---

## 2. Current-State Audit

### 2.1 Critical (invalidates results)

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| C1 | **`Status` label is random noise** | `preprocess_data()` — `random.randint(0,1)` | Model learns nothing; ~50% "accuracy" is coin-flip. Downstream `Regression` model trains on noise. |
| C2 | **Data leakage — scaler fit on full data** | `self.scaler.fit_transform(self.a)` before split | Test-set statistics leak into training; inflated metrics. |
| C3 | **Leakage — overlapping sequences across split boundary** | `create_seq` then `X[:train_size]` | Train sequences contain timestamps that appear in the test region. Must purge a gap of `seq_len + horizon` rows at the boundary. |
| C4 | **Test set double-dipped** | `validation_data=(X_test, y_test)` and evaluation on same set | No untouched holdout. Need 3-way split: train / val / test (test touched once). |
| C5 | **Wrong loss/metric pairing** | Both models: `loss='mse', metrics=['accuracy']` | "Accuracy" on regression outputs is meaningless. Classifier uses MSE + 0.5 threshold instead of sigmoid + BCE. |

### 2.2 High (breaks engineering)

| # | Issue | Impact |
|---|-------|--------|
| H1 | `self.X_train, y_train, X_test, y_test = None,...` in `__init__` — tuple-unpack bug; `self.y_train` never initialized | Latent `AttributeError`; fragile class state |
| H2 | `self.Model` vs `self.model` inconsistent attribute names | Same class of latent bug |
| H3 | `self.a = dataframe` aliases (no copy); hardcoded columns; magic numbers | Mutation risk, zero configurability |
| H4 | `Regression.preprocess_data` uses shuffled `train_test_split` on **time-ordered** data | More leakage |
| H5 | Forecasting `Status` (a binary label) as a continuous LSTM output, then thresholding at 0.5 | Conceptually wrong — forecast sensors, classify risk separately |
| H6 | Chained design trains classifier on **LSTM's predictions**, compounding error with no measurement of that propagation | Should train on real data; evaluate error propagation explicitly |
| H7 | `plt.show()` in pipeline | Crashes headless environments |
| H8 | No seeds for numpy/TF, no artifact persistence, no logging, no error handling, 5% test split | Non-reproducible, unrunnable in prod |

### 2.3 Design decision (replaces C1)

**New labeling strategy — horizon-based risk labels** (standard predictive-maintenance framing): *"Will the machine enter a failure state within the next H timesteps?"* If real failure logs exist, join them instead.

```python
def add_risk_labels(df, temp_threshold=85.0, press_threshold=110.0, horizon=30):
    """risk[t] = 1 if a threshold breach (failure) occurs in (t, t+horizon]."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    failure = (df["Temperature1"] > temp_threshold) | (df["Pressure1"] > press_threshold)
    df["risk"] = (
        failure.rolling(window=horizon).max().shift(-horizon).fillna(0).astype(int)
    )
    return df
```

Bonus: rule-based labels give us a **deterministic fallback classifier for production** (see §7.5) and a **validation oracle for explainability** (see §8.5).

---

## 3. Target Architecture

### 3.1 System diagram

```
 Telemetry (CSV batch / Kafka stream)
        │
        ▼
 ┌──────────────┐   raw data (S3 + DVC versioning)
 │ Ingestion +  │──────────────►
 │ Validation   │
 │ (Pandera)    │
 └──────┬───────┘
        ▼
 Feature + Labeling pipeline ──► Feature checks (schema, ranges, drift)
        │
        ▼
 Training (Keras) ──► Walk-forward validation ──► Baseline gates
        │
        ▼
 MLflow Registry ──► versioned artifact = {model + scaler + feature list + manifest hash}
        │
        ▼
 ┌─────────────────────── Serving (FastAPI in Docker, N replicas) ───────────────────────┐
 │ /healthz /readyz /v1/forecast /v1/risk /v1/predict /v1/explain /v1/jobs               │
 │ JWT/API-key auth │ rate limiting │ idempotency keys │ OOD check │ timeout budget      │
 │ semaphore-guarded inference │ fallback ladder │ audit log → Redis queue → DB          │
 └──────────────┬────────────────────────────────────────────────────────────────────────┘
                ▼
 Monitoring (Prometheus/Grafana, OpenTelemetry) ──► Drift jobs ──► Retrain triggers
                ▼
 CI/CD (canary rollout, auto-rollback, champion/challenger promotion)
```

### 3.2 Repository structure

```
mfp/
├── pyproject.toml            # package metadata, deps, tool configs
├── configs/                  # base.yaml, prod.yaml (Hydra or pydantic-settings)
├── src/mfp/
│   ├── core/                 # config.py, logging.py (structlog), exceptions.py, seeds.py
│   ├── data/                 # ingest.py, validate.py (Pandera schemas), label.py
│   ├── features/             # scaling.py, sequences.py
│   ├── models/               # forecast.py (LSTM), risk.py (classifier), baselines.py,
│   │                         # train.py, registry.py (MLflow)
│   ├── explain/              # calibration.py, shap_explainer.py, ig_explainer.py
│   ├── serving/              # api/main.py, schemas.py, auth.py, middleware.py,
│   │                         # ood.py, fallback.py, jobs.py
│   ├── monitoring/           # metrics.py, drift.py
│   └── cli.py                # typer: `mfp train`, `mfp evaluate`, `mfp backfill`
├── tests/                    # unit/ integration/ contract/ model_tests/ load/
├── docker/                   # Dockerfile.api, Dockerfile.train, compose.yaml
├── .github/workflows/        # ci.yml, cd.yml
├── notebooks/                # exploration ONLY — never an import target
└── implementation.md
```

---

## 4. Phase-Wise Implementation Plan

### Phase 0 — Correctness Fixes (Days 1–3) 🚨 *nothing else matters until this is done*

**Goal:** Make the numbers real.

- Replace random `Status` with horizon-based risk labels (§2.3); produce a label sanity report (distribution, temporal autocorrelation — real failures cluster, random labels don't).
- Fit scaler **on train only**; build sequences **within** each split; purge `seq_len + horizon` rows at split boundaries.
- 3-way temporal split: train / val / test. Test is touched exactly once.
- Restructure pipeline: **LSTM forecasts sensors only** (drop Status from forecast targets); **RiskClassifier** = sigmoid output + `binary_crossentropy`, metrics = PR-AUC, ROC-AUC, precision, recall, F1. Forecaster metrics = MSE, MAE (delete `accuracy`).
- Train classifier on **actual** sensor readings; separately evaluate error propagation when fed LSTM forecasts.
- Fix H1–H8 bugs; set global seeds (`random`, `numpy`, `tf`, `TF_DETERMINISTIC_OPS=1`); `matplotlib.use("Agg")` + `savefig`.
- Add persistence (naive) baseline; LSTM must beat it to justify existing.

**DoD:** unit test asserts scaler stats derive from train rows only; baseline vs LSTM comparison report exists; label report shows non-random structure.

### Phase 1 — Engineering Hygiene (Week 1)

**Goal:** From script to package.

- `pyproject.toml`, `pip install -e .`, `ruff` + `black` + `mypy` + `pre-commit`.
- `pydantic-settings` config — no hardcoded paths, columns, thresholds, epochs anywhere.
- Structured JSON logging with `structlog` (replace all `print`); run IDs on every log line.
- `pytest` unit tests for: `create_seq` (shape, boundary), labeler (known synthetic cases), splitter (no overlap, purged gap), scaler (train-only fit).
- `typer` CLI replacing `__main__`; plotting behind a `--plot` flag writing PNGs.

**DoD:** `pytest` green; one YAML drives a full run; CI runs lint+tests on every PR.

### Phase 2 — ML Hardening (Weeks 2–3)

**Goal:** Trustworthy, reproducible, registered models.

- **Data validation:** Pandera schemas at ingest (required columns, dtypes, physical ranges, timestamp monotonicity); quarantine + alert on violation.
- **Validation:** walk-forward (rolling-origin) CV instead of a single split.
- **Gates:** persistence + ridge baselines; promotion requires LSTM > baseline by configured margin.
- **Training robustness:** `EarlyStopping` (on val), `ModelCheckpoint` (resumable), gradient clipping, `ReduceLROnPlateau`.
- **Class imbalance:** class weights / focal loss; decision threshold tuned on validation PR curve (not 0.5).
- **MLflow tracking + registry.** Artifact = model + scaler + feature list + manifest:

```json
{
  "name": "risk_classifier", "version": "2025.1.0",
  "weights_sha256": "…", "scaler_sha256": "…",
  "features": ["Temperature1", "Temperature2", "Pressure1", "Pressure2"],
  "training_data_dvc_hash": "…",
  "metrics": {"pr_auc": 0.91, "recall@chosen_threshold": 0.88},
  "decision_threshold": 0.42
}
```

- **DVC** for dataset versioning; **Optuna** (small budget) for hyperparameter search.
- Refactor both models into `build_forecaster()` / `build_risk_classifier()` factories with config-driven architectures.

**DoD:** any run reproducible from (data DVC hash + config + code SHA); promotion gate script blocks regressions; every experiment in MLflow.

### Phase 3 — Explainability (Week 4)

**Goal:** Every prediction ships with "why". See §8 for full design.

- Probability **calibration** (isotonic/Platt) — uncalibrated scores aren't worth explaining.
- SHAP (`DeepExplainer`/`KernelExplainer`) for the risk classifier; **Integrated Gradients** (alibi) for per-timestep LSTM attributions; optional attention layer.
- `explain` module with caching; global reports (SHAP summary, PDP) generated per release, stored with the artifact.
- **XAI validation tests** — because labels are rule-based, attributions must recover the rules (§8.5).

**DoD:** top-3 driver features per prediction with signed impacts; attribution sanity tests pass in CI.

### Phase 4 — Serving Layer (Week 5)

**Goal:** Model behind a real API. See §5.

- FastAPI: `/healthz`, `/readyz`, `/v1/forecast`, `/v1/risk`, `/v1/predict` (chained), `/v1/explain`, `/v1/jobs` (async batch).
- Pydantic request/response schemas; model+scaler loaded at startup with **hash verification** and warm-up inference; model version echoed in every response.
- Docker (multi-stage, non-root, healthcheck); compose with Redis; Uvicorn workers.
- Idempotency keys, request IDs (propagated to logs), batch endpoint, Celery/Redis async job mode for backfills.

**DoD:** load test (k6) at 100 RPS with p95 < 300 ms on `/v1/predict`; contract tests against OpenAPI spec; graceful shutdown under SIGTERM verified.

### Phase 5 — Auth, Retries & Resilience (Week 6)

**Goal:** Survive hostile traffic and failing dependencies. See §6–§7.

- JWT (RS256) for users, API keys for machine clients, OAuth2 client-credentials for service-to-service; scopes (`predict:read`, `models:manage`); hashed key storage + rotation.
- Rate limiting (token bucket per key), quotas, load shedding (503 + `Retry-After`).
- `tenacity` retries with exponential backoff + jitter on outbound calls; circuit breakers; per-hop timeouts within a total request budget; DLQ for poison messages.
- Secrets via Vault/AWS Secrets Manager (never in images or git); audit log for every request.
- Chaos test: kill Redis/DB → predictions still served (degraded mode); expire tokens; malformed payloads.

**DoD:** 401/403/429 paths tested; retry storm simulation doesn't cascade; fail-closed auth; fallback ladder demoed.

### Phase 6 — MLOps: Monitoring, Drift, CI/CD (Weeks 7–8)

**Goal:** See problems before users do; fix them without downtime. See §10, §12.

- Prometheus metrics + Grafana dashboards (RED + model metrics); OpenTelemetry traces.
- Drift jobs: PSI on feature distributions, prediction distribution监控, KS tests; alert thresholds.
- CI/CD: lint → test → build → scan (trivy) → push → **canary deploy with auto-rollback** on SLO breach.
- Retraining triggers: scheduled (weekly) OR drift (PSI > 0.2) OR performance decay (PR-AUC drop > 5 pts); champion/challenger promotion gate.
- Runbooks per alert; SLOs: availability 99.9%, p95 sync predict < 300 ms.

**DoD:** injected drift fires alert and triggers retrain pipeline; canary auto-rollback demonstrated; on-call runbook reviewed.

### Phase 7 — Scale & Advanced (Optional, Week 9+)

Kafka streaming ingestion → real-time risk scoring; K8s HPA; per-machineID models or grouped training; active-learning loop with human-in-the-loop label review; Temporal Fusion Transformer (native attention) as challenger.

---

## 5. Production Request Handling

### 5.1 Request lifecycle

```
Client ──▶ LB/Gateway ──▶ [auth ─▶ rate limit ─▶ request-id] ──▶ FastAPI
  ──▶ Pydantic validation (422 on bad payload)
  ──▶ OOD / staleness check on input (§9-G)
  ──▶ semaphore-guarded, timeout-wrapped inference (asyncio.to_thread)
  ──▶ optional explanation (cached / async)
  ──▶ response {prediction, risk, model_version, explanation?}
  ──▶ audit record → Redis queue → DB (fire-and-forget, retried offline)
```

### 5.2 Serving patterns (choose per use case)

| Pattern | Use when | Contract |
|---|---|---|
| **Sync** | Single sequence window, low latency | 200 + result, p95 < 300 ms |
| **Async job** | Batch backfills, heavy explain, long horizons | `202 Accepted` + `job_id`; poll `/v1/jobs/{id}` or webhook |
| **Streaming** | Continuous machine telemetry | Kafka consumer → push risk scores; alerting topic for high risk |

### 5.3 Core service skeleton

```python
# src/mfp/serving/api/main.py (condensed)
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
import asyncio, hashlib

@asynccontextmanager
async def lifespan(app: FastAPI):
    art = registry.load(settings.model_version)          # model + scaler + manifest
    verify_sha256(art)                                    # corrupt/wrong artifact ⇒ refuse to boot
    art.warmup()                                          # avoids first-request latency spike
    app.state.art = art
    yield

app = FastAPI(lifespan=lifespan)
sem = asyncio.Semaphore(8)                                # bound concurrent inferences

@app.post("/v1/predict", dependencies=[Depends(verify_jwt), Depends(enforce_quota)])
async def predict(req: RiskRequest, request: Request):
    x = app.state.art.prepare(req.window)                 # validate seq_len, finiteness, ranges
    async with sem:
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(app.state.art.predict, x), timeout=0.3)
        except asyncio.TimeoutError:
            raise HTTPException(503, "inference timeout", headers={"Retry-After": "1"})
    await audit.enqueue(request, out)                     # queue-backed, non-blocking (§7)
    return PredictResponse(**out, model_version=app.state.art.version)
```

### 5.4 Concurrency, scaling, shutdown

- **Stateless replicas** (model is small — ship inside each pod); horizontal scale on CPU/latency/queue depth.
- **Timeout budget (total 500 ms):** gateway 10 · validation 5 · inference 150 · explain 200 (optional) · serialize 20 · slack 115. Enforce per hop; never let one stage eat the budget.
- **Load shedding:** when semaphore wait > threshold → `503` + `Retry-After` (protects p99 at the cost of rejects).
- **Graceful shutdown:** SIGTERM → readiness false (pull from LB) → drain in-flight (bounded) → flush audit queue → exit. Kubernetes `terminationGracePeriodSeconds` aligned.
- **Warm-up** on boot to avoid cold-start p99 spikes; readiness probe stays `503` until model verified + warmed.

---

## 6. Authentication & Authorization

### 6.1 Scheme selection

| Scheme | Use for | Notes |
|---|---|---|
| **API keys** | Internal scripts, batch clients | Store **hashed** (like passwords); key prefix identifies client for rotation; short prefix in logs, never full key |
| **JWT (RS256)** | User-facing apps | Short-lived access (15 min) + refresh tokens; validate `aud`, `iss`, `exp`; JWKS cached with kid-based refresh |
| **OAuth2 client-credentials** | Service-to-service | Token endpoint per service; scopes gate endpoints |
| **mTLS** | In-cluster / partner integrations | Certs + SPIFFE-style identity; rotation automation |

**This project:** JWT for the dashboard/operator UI, API keys for plant-floor batch clients, mTLS between services.

### 6.2 Enforcement details

- **RBAC scopes:** `predict:read`, `explain:read`, `models:manage`, `jobs:submit`. Scope check at dependency level (below).
- **Fail closed:** if the JWKS endpoint is unreachable, external endpoints return `503` rather than letting unverified tokens through. Internal calls ride mTLS as a second factor.
- **Secrets:** Vault / AWS Secrets Manager; no secrets in env files committed, images, or logs; CI uses OIDC, not static keys.
- **Audit:** every request logs `(actor_id, scopes, endpoint, model_version, request_id, timestamp)` — immutable sink.
- **Rotation & revocation:** key versions supported; revocation list checked; alert on keys nearing expiry.

```python
bearer = HTTPBearer(auto_error=True)
_jwks = PyJWKClient(settings.jwks_url, cache_keys=True)

def verify_jwt(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> Claims:
    try:
        key = _jwks.get_signing_key_from_jwt(creds.credentials).key
        claims = jwt.decode(creds.credentials, key, algorithms=["RS256"],
                            audience=settings.audience, issuer=settings.issuer)
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid or expired token")
    if "predict:read" not in claims.get("scp", []):
        raise HTTPException(403, "insufficient scope")
    return claims
```

Rate limiting (slowapi / token bucket per API key or principal claim): `@limiter.limit("100/minute")` plus daily quotas per client tier; `429` carries `Retry-After`.

---

## 7. Retries, Resilience & Degradation

### 7.1 Retry policy matrix

| Failure | Retry? | Strategy |
|---|---|---|
| Network timeout / conn reset | Yes | Exponential backoff + **full jitter**, max 4 attempts |
| `429 Too Many Requests` | Yes | Honor server `Retry-After` exactly |
| `503` from dependency | Yes (bounded) | Backoff; circuit breaker trips after repeated failures |
| `400/401/403/422` | **No** | Fix client; retrying is waste |
| Non-idempotent POST | Only with **idempotency key** | Server dedupes by key |
| Model inference timeout | No (client-side choice) | Return 503 + Retry-After; don't retry server-side |

**Retry budget:** retries may add ≤ 20% extra load — prevents retry storms amplifying an outage. All backoff includes jitter (`rand(0, min(cap, base·2ⁿ))`).

### 7.2 Implementation

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from pybreaker import CircuitBreaker

class AuditSink:
    @retry(wait=wait_exponential_jitter(initial=0.3, max=5), stop=stop_after_attempt(4),
           retry=retry_if_exception_type((TimeoutError, ConnectionError)))
    async def write(self, record: dict): ...

audit_breaker = CircuitBreaker(fail_max=5, reset_timeout=30)

async def audit(request, out):                       # never fail the prediction on audit
    try:
        await audit_breaker.call(audit_sink.write, make_record(request, out))
    except CircuitBreakerError:
        metrics.audit_dropped.inc()                  # alert on this counter
```

### 7.3 Idempotency

Client sends `Idempotency-Key: <uuid>`; server stores `(key → response, TTL 24h)` in Redis. Duplicate retries (same key) return the stored response instead of double-scoring. Critical for the batch endpoint.

### 7.4 Queue consumers (async jobs / streaming)

- Ack after successful processing; visibility/redelivery timeout > worst-case processing time.
- **Poison messages** (fail parse/validation N times) → **DLQ** + alert; human inspection; never infinite retry.
- Consumer concurrency bounded; backpressure by pausing partition reads when queue depth high.

### 7.5 Fallback ladder (degradation, not downtime)

When inference fails, degrade in this order:

1. **Fresh model** (normal path)
2. **Last-good model version** kept warm in memory (reject only if artifact hash mismatch)
3. **Rule-based fallback** — the §2.3 labeling thresholds as a deterministic classifier (`risk = temp>T or pressure>P`). Response gains `"degraded": "rules_v1"` so clients know confidence semantics changed.
4. **503 + Retry-After** — honest failure, never a fabricated prediction.

---

## 8. Explainability (XAI)

### 8.1 Why here specifically

Predictive maintenance fails organizationally when operators don't trust alerts. Every risk alert must answer: *which sensor, which direction, over what recent window?*

### 8.2 Model-level techniques

| Component | Technique | Output |
|---|---|---|
| Risk classifier (tabular) | **SHAP** (`DeepExplainer`, background = 100 train samples) | Signed contribution per sensor: "Pressure1 +0.31 → pushed risk up" |
| LSTM forecaster (sequence) | **Integrated Gradients** (alibi) over the input window | Per-timestep × per-feature attribution heatmap: "temps from 6–10 steps ago drove this forecast" |
| Architecture option | Add **attention** layer over LSTM timesteps | Native, cheap attention weights (approximate but inspectable) |
| Global views | Aggregated SHAP summary, PDP/ICE, per-machineID breakdown | Release-time report attached to the MLflow artifact |

### 8.3 Calibration first

Apply isotonic/Platt calibration before explaining — an explanation of an uncalibrated score is an explanation of a number that doesn't mean what it claims.

### 8.4 Serving explanations

```python
@app.post("/v1/explain", dependencies=[Depends(verify_jwt)])
async def explain(req: RiskRequest):
    key = hash_window(req.window)                      # cache identical windows
    if (hit := explain_cache.get(key)): return hit
    pred = await predict_raw(req.window)
    out = {
        "risk": pred.risk,
        "top_drivers": shap_top_k(req.window, k=3),    # [{feature, impact}]
        "temporal_attribution": ig_window(req.window), # per-timestep weights
        "calibrated": True,
    }
    explain_cache.put(key, out, ttl=600)
    return out
```

**Performance rules:** explanations are cached by input hash; SHAP background sampled (not full train); heavy explanation jobs run via `/v1/jobs` (async). Sync path budget: 200 ms. If the explainer throws, return the prediction with `"explanation": {"available": false}` — **never fail the prediction because XAI failed**.

### 8.5 Validating the explanations (the smart part)

Because Phase-0 labels are **rule-based**, ground truth for explanations exists: SHAP attributions **must rank the rule's features (Temperature1, Pressure1) on top** for breach cases. This becomes an automated CI test — an XAI unit test. If attributions don't recover the known rules, the explainer (or model) is broken. Additionally run the deletion test: zero out the top-attributed feature → prediction must move materially.

---

## 9. Failure-Mode Analysis by Stage (Model + System)

Detection = how we notice; Mitigation = how we recover. Each row is an item to implement or test.

### A. Data ingestion

| Failure | Detection | Mitigation |
|---|---|---|
| Missing/renamed columns, schema drift | Pandera schema at ingest; fail fast | Quarantine file, 422/alert to data owner; block training on bad batch |
| NaNs, duplicates, non-monotonic timestamps | Row-level checks | ffill with limit + gap flag; dedupe; sort by timestamp; reject gap > X |
| Sensor stuck (frozen value, sensor fault) | Rolling std == 0 over window | Flag `sensor_fault`, exclude from scoring, alert maintenance |
| Physically impossible values (e.g., 9999°C) | Range validation | Clip + flag; never silently feed to model |
| Unit change (°C→°F) mid-stream | Distribution shift check / unit sanity | Alert + versioned unit config; block silent ingestion |
| Corrupt/empty file | Checksum, min row count | Retry from source → DLQ after N failures |

### B. Labeling

| Failure | Detection | Mitigation |
|---|---|---|
| Random/degenerate labels (current C1) | Label autocorrelation + mutual-info report per training run | Hard gate: refuse to train if label–feature mutual info ≈ 0 |
| Extreme class imbalance | Label distribution report | Class weights / focal loss; threshold tuned on PR curve; report PR-AUC (not accuracy) |
| Label lookahead leakage | Label-time audit test | Horizon labeling only uses (t, t+H]; CI test asserts labels never reference > H ahead |

### C. Preprocessing / features

| Failure | Detection | Mitigation |
|---|---|---|
| Scaler fit on full data (leak) | Unit test: scaler stats computed from train rows only | Fit on train; persisted inside artifact |
| Train/serve skew (different scaler than model) | Artifact manifest carries scaler hash; verified at load | Bundle model + scaler + feature list + threshold as one atomic versioned artifact |
| Input window shorter than `seq_len` / NaN / inf | Pydantic + `np.isfinite` check | 422 with precise field errors; never pad silently |
| Sequence spans a time gap (missing telemetry) | Timestamp diff > threshold ⇒ window invalid | Reject window with `reason: "gap_in_window"` |

### D. Training

| Failure | Detection | Mitigation |
|---|---|---|
| NaN/Inf loss (exploding gradients) | Callback aborts on non-finite loss | Gradient clipping, LR decay, abort + alert (never ship a NaN checkpoint) |
| OOM kill | Container memory metrics | Smaller batch, mixed precision, gradient accumulation; memory limit + alert at 80% |
| Overfitting | Val-loss curve divergence | EarlyStopping + checkpoint-best; dropout; L2 |
| Mid-run crash | Process exit | Epoch checkpoints; resume-from-checkpoint support |
| Non-reproducibility | Re-run diff test in CI | All seeds fixed; `TF_DETERMINISTIC_OPS=1`; log env + code SHA per run |

### E. Evaluation & promotion

| Failure | Detection | Mitigation |
|---|---|---|
| Optimistic metrics (leakage/split bugs) | Walk-forward CV + purged gaps | Embargo of `seq_len + horizon` at every split boundary |
| Fancy model ≤ dumb baseline | Mandatory baseline gate (persistence, ridge) | Auto-reject promotion; keep the baseline as the shipped fallback challenger |
| Overfit to one machine/regime | Group-aware split by machineID + per-group metrics | Grouped CV; report worst-group, not just average |

### F. Registry / artifacts

| Failure | Detection | Mitigation |
|---|---|---|
| Corrupt model file | SHA-256 verify at load | Immutable object storage; refuse boot; readiness stays false |
| Wrong version loaded at startup | Version + hash logged and asserted vs. desired tag | Pin via env/manifest; fail fast on mismatch |
| Scaler/model/threshold mismatch | Manifest contains all hashes; verified together | Atomic artifact (§4 Phase 2) |

### G. Serving / inference

| Failure | Detection | Mitigation |
|---|---|---|
| Cold-start latency spike | First-request p99 in dashboards | Lifespan eager-load + warmup call before readiness |
| Malformed/adversarial payload | Pydantic validation; field ranges | 422 with clear errors; request size cap; auth (§6) |
| **Out-of-distribution input** (sensor regime never seen in training) | Mahalanobis distance / autoencoder reconstruction error vs. train reference | Flag `"ood": true` + low-confidence marker; log for retraining corpus; optionally reject |
| Model artifact missing at pod boot | Readiness probe fails | Retry download with backoff; baked-in last-good model in image as fallback; pod stays NotReady (K8s restarts) |
| Slow inference under load | Latency histogram + semaphore wait time | Timeout (§5.4), micro-batching, scale out, load shedding |
| Memory leak over days/weeks | RSS gauge per pod | Rolling restarts (smart default for long-lived model servers), profiling |
| Retry storm from clients | Retry-rate metric | 429s with Retry-After early; jitter requirement documented in API docs |

### H. Explainability runtime

| Failure | Detection | Mitigation |
|---|---|---|
| SHAP too slow at traffic peak | Explain latency histogram | Cache by input hash; async job mode; sampled background |
| Explainer crash on edge input | Exception counter | Return prediction without explanation + `"available": false` (prediction never fails due to XAI) |
| Unstable/nonsense attributions | §8.5 sanity tests in CI | Seed explainer; average over k samples; alert if sanity tests fail in prod shadow mode |

### I. Infrastructure dependencies

| Failure | Detection | Mitigation |
|---|---|---|
| Audit DB down | Sink health + breaker open | Queue in Redis (bounded), retry, drop-with-audit-counter if non-critical — **predictions continue** |
| Redis down | Health check | Circuit-break; disable idempotency cache + rate-limit fallback to per-pod local limiter; alert |
| Auth provider down | JWKS fetch failures | Fail closed externally; mTLS continues internally; alert P1 |
| DLQ filling | DLQ depth gauge | Alert + runbook; poison messages reviewed manually |
| Cert/key expiry | Expiry monitoring (30-day alert) | Automated rotation |

### J. Post-deployment (the silent killers)

| Failure | Detection | Mitigation |
|---|---|---|
| Data drift (sensors drift with machine age) | PSI/KS per feature, weekly | Alert at PSI > 0.1, retrain trigger at PSI > 0.2 |
| Concept drift (same sensors, new failure mode) | Live PR-AUC on delayed ground truth; prediction-distribution shift | Champion/challenger; scheduled retrain; human review queue |
| Silent degradation (no errors, worse answers) | Shadow-mode comparison vs. fallback rules; calibration drift | Canary + auto-rollback; alert on calibration error growth |
| Feedback loop (alerts → operator intervention → labels change) | Alert-response event log | Log interventions; treat post-intervention data as censored for labeling |

---

## 10. Monitoring & Observability

**Metrics (Prometheus):**

| Category | Metrics |
|---|---|
| HTTP/RED | RPS, latency p50/p95/p99 per endpoint, error rate by status code, 429 rate |
| Model | Prediction distribution, % flagged risk, OOD rate, calibration error, explanation availability, fallback activations (counter — **any nonzero rate alerts**) |
| Drift | PSI per feature, prediction-distribution KL vs. reference window |
| Infra | CPU/mem RSS per pod, semaphore wait, queue depth, DLQ size, audit drops |

**Logs:** structured JSON; every line carries `request_id`, `actor_id`, `model_version`. **Traces:** OpenTelemetry spans gateway → validation → inference → audit, so a slow request is attributable to a stage in one glance.

**Alerts (SLO-based, severity-tiered to avoid fatigue):**

- P1: error rate > 1% over 5 min; fallback activations > 0; readiness failing on >1 replica
- P2: PSI > 0.2 on any feature; DLQ depth > 100; p95 latency > SLO burn rate
- P3: calibration drift, audit drops > 0 (non-critical path)

---

## 11. Testing Strategy

| Layer | Examples |
|---|---|
| Unit | `create_seq` shapes/boundaries; labeler on synthetic fixtures; splitter purged-gap assertion; scaler train-only fit |
| Integration | End-to-end train (2 epochs, tiny synthetic data) → register → load → predict, in CI |
| Contract | API responses match OpenAPI schema; artifact manifest hash verification |
| **Model tests** | *Directional:* pushing Temperature1 past threshold must raise risk. *Invariance:* unit-consistent rescaling shouldn't flip risk. *Minimum performance:* PR-AUC ≥ baseline + margin. *XAI sanity:* §8.5 rules-recovery test |
| Load | k6: 100 RPS sustained, spike to 300; assert p95 + shedding behavior |
| Chaos | Kill Redis/DB mid-traffic (expect degraded-but-serving); corrupt model file (expect boot refusal); expire JWT mid-flight |

---

## 12. Deployment, Rollout & Rollback

1. **Shadow:** new model scores live traffic; predictions logged, not served; compare vs. champion.
2. **Canary:** 5% → 25% → 100% traffic; auto-rollback if error rate, latency, or flagged-risk rate exits expected band (drastic prediction-distribution shift is itself a red flag).
3. **Blue-green for model swaps:** new version in warm standby; flip via config; instant flip-back.
4. **Versioning:** every response includes `model_version`; clients may pin; old versions kept warm for `rollback_window`.
5. **Freeze rule:** no model rollouts during active incidents.

CI/CD: PR → lint + type + unit + integration → build image → scan (trivy) → push registry → staging deploy + model tests → canary in prod → promote.

---

## 13. Timeline & Definition of Done

| Phase | Week | Exit criteria |
|---|---|---|
| 0 Correctness | 1 (days 1–3) | Real labels; leakage tests pass; baselines reported |
| 1 Hygiene | 1 | Package + config + logging + tests + CI |
| 2 ML hardening | 2–3 | Walk-forward CV; MLflow registry; promotion gates; DVC |
| 3 Explainability | 4 | Calibrated model; per-prediction drivers; XAI sanity tests in CI |
| 4 Serving | 5 | Dockerized API; 100 RPS p95 < 300 ms; idempotency; async jobs |
| 5 Auth & resilience | 6 | JWT/API keys + scopes; 429s; retries + breakers; fallback ladder; chaos tests |
| 6 MLOps | 7–8 | Dashboards, drift alerts, canary + auto-rollback, runbooks |
| 7 Advanced | 9+ | Streaming, autoscaling, active learning |

**Overall DoD:** a new dataset can be dropped in, and — with no code changes — the system validates it, retrains, gates against baselines, registers the artifact, explains its predictions, ships behind auth with rate limits, survives dependency outages in degraded mode, detects drift, and rolls back automatically when it misbehaves. That is "more than a Colab notebook."

---

*Notes on assumptions:* real failure logs (if available) should replace threshold-based labels in §2.3; thresholds themselves must come from domain expertise or equipment specs, not defaults; if multiple machines are in the data, add `machine_id` as a grouping key for splits, per-machine metrics, and possibly per-machine models (Phase 7).