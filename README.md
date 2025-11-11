# FPL Assistant — Multi-GW optimizer and recommendations

## What this does
- Fetches Fantasy Premier League public data and (optionally) your private manager data (if you supply cookies/login).
- Enriches players with shots/key-passes/xG from Understat or FBRef (or SportMonks paid API).
- Builds per-player rolling features (form, shots per 90, xG per 90, team strength, rotation risk).
- Trains a LightGBM model using historical per-GW points (cross-validated).
- Provides predictions and simulates multi-gameweek transfer paths (subject to budget & transfer limits).
- Exposes a small FastAPI endpoint to request analysis for your manager/team.

## Important notes
- To fetch private manager data, you must provide your FPL session cookie or username/password. See `config.yaml` and `.env.example`.
- xG sources:
  - Free community option: Understat python wrapper (scrapes understat). See docs.
  - Paid/official: SportMonks or other commercial providers (requires an API key).
- The code is a full working blueprint. You must run it locally and the app will fetch up-to-date data at runtime.

Sources and community docs:
- FPL API documentation and community guides. :contentReference[oaicite:1]{index=1}
- Understat Python package docs. :contentReference[oaicite:2]{index=2}
- SportMonks xG API (paid) available if you prefer official xG endpoints. :contentReference[oaicite:3]{index=3}

## Quickstart
1. Create a python venv and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt


## Configure config.yaml (see file comments).

- Run training:
   ```bash
    python -m scripts.run_train

- Start API:
    ```bash
    uvicorn app.main:app --reload

- Query recommendations:
    ```bash
    python scripts/run_recommend.py --manager-id FPL_TEAM_ID (3184916)

| Goal                                      | Command                                                         | Explanation                                      |
| ----------------------------------------- |-----------------------------------------------------------------| ------------------------------------------------ |
| 🧮 Run the actual FPL optimizer algorithm | `python -m app.main`                                            | Runs everything (fetch + ML + recommendations)   |
| 🌐 View results on local web dashboard    | `uvicorn app.main:app --reload`                                 | Starts FastAPI server at `http://127.0.0.1:8000` |
| 🧾 Check detailed outputs                 | Open `output/fpl_analysis_report.html` or use the web dashboard |                                                  |


🚀 Main API Endpoints
| **URL**                                                                                        | **Description**                                            |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/)                                             | Displays a welcome message and lists all available routes. |
| [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)                                     | Opens interactive API documentation (Swagger UI).          |
| [`http://127.0.0.1:8000/health`](http://127.0.0.1:8000/health)                                 | Performs a health check to ensure the API is running.      |
| [`http://127.0.0.1:8000/train`](http://127.0.0.1:8000/train)                                   | Triggers model training.                                   |
| [`http://127.0.0.1:8000/recommend/{manager_id}`](http://127.0.0.1:8000/recommend/{manager_id}) | Generates recommendations for the specified manager.       |
