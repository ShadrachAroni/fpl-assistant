# FPL Assistant – Intelligent Squad Planner

End-to-end system that ingests Fantasy Premier League data, enriches it with third-party stats, trains a LightGBM ensemble, and serves actionable recommendations through a FastAPI backend. The stack combines a predictive model, a Monte-Carlo backed planner, chip timing heuristics, and rich reporting for every gameweek.

---

## Architecture

### Data layer
- **FPL API bootstrap**: base player metadata, fixtures, prices, ownership, chips, manager history.
- **Advanced stats connectors** (`app/api_client/*`): Understat, FBRef, SportMonks, PL injury feeds, BBC squads.
- **Pipelines** (`app/data/pipeline.py`):
  1. Merge raw feeds into a feature-complete player table.
  2. Engineer rolling form, xG, xA, risk, EO, fixture difficulty, price volatility.
  3. Persist historical caches inside `data/cache`.

### Modeling layer
- **Trainer** (`app/models/trainer.py`): builds LightGBM regressors with cross-validation, logs feature importance, and exports `models/lightgbm_model.joblib` plus `training_metrics.json`.
- **Predictor** (`app/models/predictor.py`): loads ensembles, applies risk/form/fixture/ownership adjustments, produces confidence intervals and multi-GW horizons.

### Planning layer
- **Transfer simulator** (`app/planner/simulator.py`): evaluates transfers under budget, risk, EO, chip horizon, price changes, and Monte-Carlo distributions.
- **Captaincy selector**: EO-aware captain/vice reasoning with template vs differential strategies.
- **Chip engine**: heuristic scoring for TC, BB, FH, WC with DGW/BGW awareness and formation health checks.

### API layer
- **FastAPI app** (`app/main.py`): orchestrates fetching, modeling, simulation, and response serialization using Pydantic schemas in `app/schemas.py`.
- **Scripts** (`scripts/*.py`): convenient wrappers for training, clearing caches, and generating recommendations from the CLI.

---

## Requirements
- Python 3.11+
- Recommended: virtual environment (`python -m venv venv && venv\Scripts\activate` on Windows).
- `config.yaml` populated with API tokens (SportMonks, Understat credentials, etc).
- Optional `.env` for sensitive manager cookies when hitting private endpoints.

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Configuration
1. Copy `config.yaml` and adjust:
   - `training`: CV splits, rolling window, feature flags.
   - `simulation`: planning horizon, Monte-Carlo parameters, risk tolerances.
   - `api`: external service keys.
2. Provide session information if you want private manager data:
   - Either set `FPL_SESSION` cookie in `.env`.
   - Or add username/password (not recommended) in config.

---

## Training workflow
```bash
python -m scripts.clear_cache_and_test   # optional sanity checks
python -m scripts.run_train              # builds LightGBM models
```
Artifacts land in `models/` and will be loaded automatically by the API.

---

## Running the API
```bash
uvicorn app.main:app --reload
```

Key routes:
| Endpoint | Description |
| --- | --- |
| `GET /` | Capability overview. |
| `GET /health` | Status + feature flags. |
| `POST /train` | Kick off retraining (uses config defaults). |
| `GET /recommend/{manager_id}` | Full recommendation package for a manager id. |
| `GET /squad/{manager_id}` | Lightweight squad summary (risk, ownership, price moves). |
| `GET /docs` | Swagger UI with schema details. |

To call the recommender without spinning up FastAPI you can run:
```bash
python scripts/run_recommend.py --manager-id 123456
```

### Rotation monitor (auto-update)

The new rotation monitor scores managers based on how often they change their starting XI over the last few gameweeks.  
It pulls official FPL match histories, flags high-volatility teams, and writes the cache that the data pipeline consumes when computing rotation risk.

Single run:
```bash
python -m scripts.rotation_watch
```

Continuous background refresh (every 3 hours):
```bash
python -m scripts.rotation_watch --interval-minutes 180
```

Outputs land in `data/rotation_watch.json`; the pipeline automatically ingests this file and scales rotation risk by the live volatility score.

---

## Response anatomy (`/recommend/{manager_id}`)

The endpoint returns a `DetailedRecommendation` object containing:
- **Summary**: bank trajectory, transfer counts, risk/price/ownership aggregates, Monte-Carlo insight, and now a flat list of recommended pick names for quick scanning.
- **Player picks**: enriched per-player intelligence (fixture difficulty, EO, risk, price change).
- **Transfers**: recommended moves with gain estimates, risk warnings, ownership and price alerts.
- **Captaincy**: captain/vice selection reasoned by EO strategy and risk profile.
- **Chip advice**: selected chip (or explicit “No chip recommended” with `optimal_gw = null`).
- **Formation analysis**: bench strength, coverage, admissible formations.
- **Instructions**: bullet list summarizing transfers, chip, captain, warnings, and player pick names.

---

## Development tips
- **Testing**: lint/test scripts can be added under `scripts/` (e.g. extend `clear_cache_and_test.py`).
- **Data refresh**: delete `data/cache` if the bootstrap becomes stale.
- **Model updates**: run `python -m scripts.run_train` after changing feature engineering.
- **Logging**: verbose INFO logs describe every pipeline stage; adjust log level if needed.

---

## Troubleshooting
- Missing model file → run training.
- 403/401 from FPL → refresh session cookie or login info.
- DGW/BGW detection relies on fixture data; ensure `fpl.fixtures()` is reachable.
- Memory-heavy predictions → tune `simulation.planning_horizon` or disable Monte-Carlo in `config.yaml`.

---

Happy tinkering and green arrows! 🚀
