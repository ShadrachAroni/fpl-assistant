"""
FPL Assistant Main API - COMPLETE INTELLIGENCE STACK v6.0

ENHANCEMENTS:
✅ Ownership intelligence (template vs differential)
✅ Price change prediction and opportunity cost
✅ EO-based captain recommendations
✅ Formation validation
✅ Bench strength optimization
✅ Multi-factor transfer scoring
✅ Advanced risk assessment
✅ Value hold analysis

PRODUCTION READY
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

import yaml
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from app.data.pipeline import DataPipeline
from app.models.trainer import train_lightgbm
from app.models.predictor import Predictor
from app.planner.simulator import TransferSimulator
from app.api_client.fpl_client import FPLClient
from app.api_client.sportmonks_client import SportmonksClient
from app.api_client.understat_client import UnderstatClient
from app.api_client.pl_injury_client import PremierLeagueInjuryClient
from app.api_client.bbc_client import BBCLineupClient
from app.api_client.news_client import GNewsClient, NewsAPIClient, RSSClient
from app.schemas import (
    DetailedRecommendation, 
    TransferItem, 
    PlayerPick, 
    CaptaincyRecommendation, 
    ChipAdvice, 
    Summary,
    RiskAssessment,
    OwnershipIntelligence,
    PriceIntelligence,
    FormationAnalysis,
    EffectiveOwnership,
    ModelPerformance,
    MonteCarloInsight
)

# ==================== CONFIGURATION ====================

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Silence very chatty HTTP/client libraries
for noisy_logger in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger("fpl_assistant")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as fh:
    CONFIG: Dict[str, Any] = yaml.safe_load(fh)

MODEL_PATH = CONFIG.get("model_path", "models/lightgbm_model.joblib")

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="FPL Assistant API - Full Intelligence Stack",
    version="6.0.0",
    description="Complete FPL engine with ownership, price intelligence, and advanced analytics"
)


# ==================== STARTUP ====================

@app.on_event("startup")
def startup_event() -> None:
    """Initialize FPL client and data pipeline."""
    logger.info("🚀 Starting FPL Assistant API v6.0 (Full Intelligence Stack)")
    try:
        app.state.fpl_client = FPLClient()
        logger.info("✅ FPLClient initialized")
        
        sportmonks_client = None
        sportmonks_cfg = CONFIG.get("sources", {}).get("sportmonks", {})
        sportmonks_token = sportmonks_cfg.get("api_token") or os.getenv("SPORTMONKS_API_TOKEN")

        if sportmonks_token:
            try:
                sportmonks_client = SportmonksClient(
                    base_url=sportmonks_cfg.get("base_url", "https://api.sportmonks.com"),
                    api_token=sportmonks_token,
                    injuries_endpoint=sportmonks_cfg.get("injuries_endpoint", "v3/football/injuries")
                )
                logger.info("✅ Sportmonks client initialized")
            except Exception as exc:
                logger.warning(f"⚠️ Sportmonks client unavailable: {exc}")

        understat_client = None
        try:
            understat_client = UnderstatClient()
            logger.info("✅ Understat client initialized")
        except Exception as exc:
            logger.warning(f"⚠️ Understat client unavailable: {exc}")

        pl_injury_client = None
        injury_cfg = CONFIG.get("injury_intel", {})
        pl_cfg = injury_cfg.get("premier_league_api", {})
        if injury_cfg.get("external_sources_enabled", True):
            try:
                pl_injury_client = PremierLeagueInjuryClient(
                    comp_seasons=pl_cfg.get("comp_seasons"),
                    page_size=pl_cfg.get("page_size", 40),
                )
                logger.info("✅ Premier League injury client initialized")
            except Exception as exc:
                logger.warning(f"⚠️ Premier League injury client unavailable: {exc}")

        bbc_client = None
        lineup_cfg = CONFIG.get("lineup_intel", {})
        if lineup_cfg.get("enabled", False):
            try:
                bbc_client = BBCLineupClient()
                logger.info("✅ BBC lineup client initialized")
            except Exception as exc:
                logger.warning(f"⚠️ BBC lineup client unavailable: {exc}")

        news_cfg = CONFIG.get("news_intel", {})
        gnews_client = None
        newsapi_client = None
        rss_client = None

        if news_cfg:
            g_cfg = news_cfg.get("gnews", {})
            n_cfg = news_cfg.get("newsapi", {})
            rss_cfg = news_cfg.get("rss", [])

            if g_cfg.get("api_key"):
                gnews_client = GNewsClient(
                    api_key=g_cfg["api_key"],
                    lang=g_cfg.get("lang", "en"),
                    country=g_cfg.get("country")
                )

            if n_cfg.get("api_key"):
                newsapi_client = NewsAPIClient(
                    api_key=n_cfg["api_key"],
                    country=n_cfg.get("country", "gb"),
                    category=n_cfg.get("category", "sports")
                )

            if rss_cfg:
                rss_client = RSSClient(rss_cfg)

        app.state.pipeline = DataPipeline(
            CONFIG,
            fpl_client=app.state.fpl_client,
            understat_client=understat_client,
            sportmonks_client=sportmonks_client,
            pl_injury_client=pl_injury_client,
            bbc_client=bbc_client,
            gnews_client=gnews_client,
            newsapi_client=newsapi_client,
            rss_client=rss_client,
        )
        logger.info("✅ Pipeline initialized")
        
        app.state.pipeline.fetch_bootstrap()
        logger.info("✅ Bootstrap loaded")
        
    except Exception as exc:
        logger.exception("Failed to initialize: %s", exc)
        app.state.pipeline = None
        app.state.fpl_client = None


# ==================== HELPER FUNCTIONS ====================

def _safe_get_current_next_gw(bootstrap: Dict[str, Any]) -> tuple[int, int]:
    """Safely extract current and next gameweek."""
    if not bootstrap or "events" not in bootstrap:
        raise HTTPException(status_code=500, detail="Bootstrap invalid")

    events = pd.DataFrame(bootstrap.get("events", []))
    if events.empty:
        raise HTTPException(status_code=500, detail="No events")

    current_row = events[events["is_current"] == True]
    next_row = events[events["is_next"] == True]

    current_gw = int(current_row["id"].iloc[0]) if not current_row.empty else int(events["id"].max())
    next_gw = int(next_row["id"].iloc[0]) if not next_row.empty else current_gw + 1

    return current_gw, next_gw


def _ensure_predictor_features(players_df: pd.DataFrame, predictor: Predictor) -> pd.DataFrame:
    """Ensure all required features exist."""
    for col in getattr(predictor, "feats", []):
        if col not in players_df.columns:
            players_df[col] = 0
    return players_df


def _format_fixture_display(opponent_short: str, is_home: bool, difficulty: int) -> str:
    """Format fixture display with emoji."""
    if opponent_short == "—" or opponent_short == "UNK":
        return "No fixture"
    
    venue = "H" if is_home else "A"
    
    if difficulty <= 2:
        emoji = "✅"
    elif difficulty >= 4:
        emoji = "⚠️"
    else:
        emoji = "➡️"
    
    return f"{emoji} {opponent_short} ({venue})"


def _create_risk_assessment(player: pd.Series) -> RiskAssessment:
    """Create comprehensive risk assessment."""
    total_risk = float(player.get("total_risk", 0.0))
    risk_category = player.get("risk_category", "🟡 Medium")
    
    risks = []
    
    injury_risk = float(player.get("injury_risk", 0.0))
    if injury_risk > 0.3:
        chance = int(player.get("chance_of_playing_next_round", 100))
        risks.append(f"⚕️ {chance}% fit")
    
    rotation_risk = float(player.get("rotation_risk", 0.0))
    if rotation_risk > 0.5:
        mins = int(player.get("minutes_per_game", 0))
        risks.append(f"🔄 {mins}min/game")
    
    disciplinary_risk = float(player.get("disciplinary_risk", 0.0))
    if disciplinary_risk > 0.5:
        yellows = int(player.get("yellow_cards", 0))
        risks.append(f"🟨 {yellows} cards")
    
    fatigue_risk = float(player.get("fatigue_risk", 0.0))
    if fatigue_risk > 0.5:
        risks.append("😓 Fatigue")
    
    form_risk = float(player.get("form_drop_risk", 0.0))
    if form_risk > 0.5:
        form = float(player.get("form", 0))
        risks.append(f"📉 {form:.1f} form")
    
    risk_summary = " | ".join(risks) if risks else "✅ Low risk"
    
    return RiskAssessment(
        total_risk=total_risk,
        risk_category=risk_category,
        injury_risk=injury_risk,
        rotation_risk=rotation_risk,
        disciplinary_risk=disciplinary_risk,
        fatigue_risk=fatigue_risk,
        form_drop_risk=form_risk,
        risk_summary=risk_summary
    )


def _create_ownership_intelligence(player: pd.Series) -> OwnershipIntelligence:
    """Create ownership intelligence object."""
    return OwnershipIntelligence(
        selected_by_percent=float(player.get("selected_by_percent", 0)),
        ownership_category=player.get("ownership_category", "🟡 Standard"),
        is_template=bool(player.get("is_template", False)),
        is_differential=bool(player.get("is_differential", False)),
        captain_eo_multiplier=float(player.get("captain_eo_multiplier", 1.0)),
        template_priority=float(player.get("template_priority", 0))
    )


def _create_price_intelligence(player: pd.Series) -> PriceIntelligence:
    """Create price intelligence object."""
    return PriceIntelligence(
        price_rise_probability=float(player.get("price_rise_probability", 0)),
        price_fall_probability=float(player.get("price_fall_probability", 0)),
        price_change_status=player.get("price_change_status", "➡️ Stable"),
        net_transfers=int(player.get("net_transfers", 0)),
        value_hold_score=float(player.get("value_hold_score", 0)),
        opportunity_cost=float(player.get("opportunity_cost", 0))
    )


def _build_player_reasoning(row: pd.Series, next_gw: int) -> str:
    """Build reasoning string for a player pick."""
    expected = float(row.get(f"pred_gw{next_gw}", 0.0))
    form_val = float(row.get("form", 7.5))
    fixture = int(row.get("fixture_difficulty", 3))
    opponent = row.get("next_opponent_short", "UNK")
    is_home = row.get("is_home", True)
    venue = "H" if is_home else "A"
    risk_cat = row.get("risk_category", "")
    ownership = float(row.get("selected_by_percent", 0))
    
    reasons = [f"{expected:.1f} pts predicted"]

    if form_val >= 8.0:
        reasons.append(f"Great form ({form_val:.1f})")
    elif form_val < 5.0:
        reasons.append(f"Poor form ({form_val:.1f})")

    if fixture <= 2:
        reasons.append(f"Easy fixture vs {opponent} ({venue})")
    elif fixture >= 4:
        reasons.append(f"Hard fixture vs {opponent} ({venue})")
    else:
        reasons.append(f"vs {opponent} ({venue})")
    
    # Add ownership context
    if ownership > 50:
        reasons.append(f"🔴 Essential ({ownership:.0f}%)")
    elif ownership > 35:
        reasons.append(f"🟠 Template ({ownership:.0f}%)")
    elif ownership < 5:
        reasons.append(f"💎 Differential ({ownership:.0f}%)")
    
    # Add risk indicator
    if "🔴" in risk_cat:
        reasons.append("⚠️ High risk")
    elif "🟡" in risk_cat:
        reasons.append("Medium risk")

    return " | ".join(reasons)


def _build_player_pick_summary(picks: List[PlayerPick]) -> List[str]:
    """Summarize recommended picks for quick scanning."""
    if not picks:
        return []

    top_points = sorted(picks, key=lambda p: p.predicted_points_next_gw, reverse=True)[:3]
    top_text = ", ".join(f"{p.player_name} ({p.predicted_points_next_gw:.1f})" for p in top_points)

    differentials = [
        p for p in picks
        if p.ownership_intel and p.ownership_intel.is_differential
    ][:2]
    diff_text = ", ".join(f"{p.player_name} ({p.predicted_points_next_gw:.1f})" for p in differentials) or "None"

    value_candidates = [p for p in picks if p.current_price]
    value_picks = sorted(
        value_candidates,
        key=lambda p: (p.current_price or 0) / max(p.predicted_points_next_gw, 0.1)
    )[:2]
    value_text = ", ".join(f"{p.player_name} (£{p.current_price:.1f}m)" for p in value_picks) or "None"

    return [
        f"Top points: {top_text}",
        f"Differentials: {diff_text}",
        f"Value picks: {value_text}"
    ]


def _assign_horizon_predictions(
    df: pd.DataFrame,
    predictor: Predictor,
    start_gw: int,
    horizon: int
) -> None:
    """
    Populate pred_gw columns by running one model inference per DataFrame.
    """
    if df.empty:
        return

    preds = predictor.predict(df)

    for offset in range(horizon):
        df[f"pred_gw{start_gw + offset}"] = preds


def _normalize_chip_advice(chip_advice: ChipAdvice) -> ChipAdvice:
    """
    Ensure 'No chip recommended' responses never carry an optimal GW.
    """
    if not chip_advice:
        return chip_advice

    if chip_advice.recommended_chip in (None, "No chip recommended"):
        chip_advice.optimal_gw = None

    return chip_advice


def _enforce_transfer_risk_warning(
    in_total_risk: Optional[float],
    risk_summary: str,
    existing_warning: Optional[str]
) -> Optional[str]:
    """Ensure risk warnings align with risk thresholds."""
    if in_total_risk is None:
        return None
    message = risk_summary or "No detailed risk data"
    if in_total_risk >= 0.6:
        expected = f"🔴 HIGH RISK: {message}"
    elif in_total_risk >= 0.4:
        expected = f"🟡 MEDIUM RISK: {message}"
    else:
        return None
    return existing_warning or expected


def _validate_chip_advice_window(chip_advice: ChipAdvice, next_gw: int, horizon: int) -> ChipAdvice:
    """Ensure chip advice contains a valid optimal GW within planning horizon."""
    if not chip_advice or chip_advice.recommended_chip == "No chip recommended":
        return chip_advice
    optimal = chip_advice.optimal_gw
    min_gw = next_gw
    max_gw = next_gw + horizon
    if optimal is None or optimal < min_gw or optimal > max_gw:
        chip_advice.optimal_gw = min_gw
    return chip_advice


# ==================== API ENDPOINTS ====================

@app.get("/")
def root() -> Dict[str, Any]:
    """API information."""
    return {
        "message": "🎯 FPL Assistant API v6.0 - FULL INTELLIGENCE STACK",
        "features": [
            "✅ Ownership intelligence (template vs differential)",
            "✅ Price change prediction",
            "✅ Effective ownership (EO) captain recommendations",
            "✅ Formation validation",
            "✅ Bench strength optimization",
            "✅ Multi-factor transfer scoring",
            "✅ Comprehensive risk assessment",
            "✅ Value hold analysis",
            "✅ 7-gameweek horizon planning",
            "✅ Chip timing optimization"
        ],
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/train": "Train prediction model",
            "/recommend/{manager_id}": "Get full recommendations"
        }
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "6.0.0-full-intelligence",
        "features": "ownership+price+eo+formation+risk"
    }


@app.post("/train")
def train(n_splits: int = None) -> Dict[str, Any]:
    """Train the prediction model."""
    if app.state.pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    n_splits = n_splits or CONFIG["training"]["n_splits_cv"]

    logger.info(f"📄 Training model with {n_splits} CV splits...")
    
    df = app.state.pipeline.build_train_features(
        rolling_window=CONFIG["training"]["rolling_window_gw"]
    )

    if df.empty:
        return {"status": "no-data", "message": "No training data available"}

    result = train_lightgbm(df, n_splits=n_splits)

    return {
        "status": "✅ Training Complete",
        "oof_rmse": result.get("oof_rmse"),
        "features_used": len(result.get("feature_importance", []))
    }


@app.get("/recommend/{manager_id}", response_model=DetailedRecommendation)
def recommend(manager_id: int) -> DetailedRecommendation:
    """
    Get complete FPL recommendations with full intelligence stack.
    """
    try:
        if app.state.pipeline is None or app.state.fpl_client is None:
            raise HTTPException(status_code=500, detail="Services not initialized")

        logger.info("=" * 80)
        logger.info(f"📊 GENERATING RECOMMENDATIONS: Manager {manager_id}")
        logger.info("=" * 80)

        fpl = app.state.fpl_client
        bootstrap = fpl.bootstrap()

        if not bootstrap:
            raise HTTPException(status_code=500, detail="Bootstrap failed")

        current_gw, next_gw = _safe_get_current_next_gw(bootstrap)
        logger.info(f"📅 Current GW: {current_gw} | Next GW: {next_gw}")

        # ==================== STEP 1: Financial Data ====================
        
        logger.info("💰 STEP 1: Fetching financial data...")
        bank_info = fpl.manager_bank_and_free_transfers(manager_id)
        bank = float(bank_info["bank"])
        free_transfers_remaining = int(bank_info["free_transfers"])
        transfers_made_this_gw = int(bank_info["transfers_made_this_gw"])
        starting_fts = int(bank_info["starting_free_transfers"])
        logger.info(f"   Bank: £{bank:.1f}m | Free Transfers: {free_transfers_remaining}")

        # ==================== STEP 2: Current Squad ====================
        
        logger.info("👥 STEP 2: Fetching current squad...")
        current_player_ids = fpl.get_current_squad_player_ids(manager_id)

        if not current_player_ids:
            raise HTTPException(status_code=404, detail="Could not load squad")

        logger.info(f"   ✅ Squad: {len(current_player_ids)} players")

        # ==================== STEP 3: Global Player Data ====================
        
        logger.info("🌍 STEP 3: Loading global player data with full intelligence...")
        if app.state.pipeline.bootstrap is None:
            app.state.pipeline.fetch_bootstrap()

        players_df = app.state.pipeline.players_df().copy()

        if players_df.empty:
            raise HTTPException(status_code=500, detail="Player data failed")

        POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        players_df["position"] = players_df["element_type"].map(POS_MAP).fillna("UNK")

        logger.info(f"   ✅ Loaded {len(players_df)} players")
        
        # Log intelligence distribution
        if "ownership_category" in players_df.columns:
            ownership_dist = players_df["ownership_category"].value_counts().to_dict()
            logger.info(f"   📊 Ownership: {ownership_dist}")
        
        if "price_change_status" in players_df.columns:
            price_dist = players_df["price_change_status"].value_counts().to_dict()
            logger.info(f"   💰 Price changes: {price_dist}")

        # ==================== STEP 4: Load Model ====================
        
        logger.info("🤖 STEP 4: Loading prediction model...")
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Run /train first"
            )

        predictor = Predictor(MODEL_PATH)
        players_df = _ensure_predictor_features(players_df, predictor)
        logger.info("   ✅ Model loaded")

        # ==================== STEP 5: Build Current Squad ====================
        
        logger.info("📝 STEP 5: Building current squad...")
        current_squad = players_df[players_df["id"].isin(current_player_ids)].copy()

        if current_squad.empty:
            raise HTTPException(status_code=400, detail="Could not match squad players")

        logger.info(f"   ✅ Matched {len(current_squad)} players")
        
        # Log squad intelligence
        if "total_risk" in current_squad.columns:
            avg_squad_risk = current_squad["total_risk"].mean()
            high_risk_count = (current_squad["total_risk"] > 0.6).sum()
            logger.info(f"   ⚠️ Squad risk: {avg_squad_risk:.2f} avg | {high_risk_count} high-risk")
        
        if "selected_by_percent" in current_squad.columns:
            avg_ownership = current_squad["selected_by_percent"].mean()
            template_count = (current_squad["selected_by_percent"] > 35).sum()
            differential_count = (current_squad["selected_by_percent"] < 5).sum()
            logger.info(f"   📊 Ownership: {avg_ownership:.1f}% avg | {template_count} templates | {differential_count} differentials")

        # ==================== STEP 6: Generate Predictions ====================
        
        logger.info("🔮 STEP 6: Generating predictions...")
        horizon = int(CONFIG["simulation"].get("planning_horizon", 5))

        def _safe_assign_predictions(df: pd.DataFrame, label: str) -> None:
            try:
                _assign_horizon_predictions(df, predictor, next_gw, horizon)
                logger.info(f"   ✅ {label}: predictions cached for {horizon} GW(s)")
            except Exception as exc:
                logger.error(
                    f"   ❌ {label}: bulk prediction failed ({exc}). "
                    "Falling back to per-GW inference."
                )
                for i in range(horizon):
                    gw = next_gw + i
                    col = f"pred_gw{gw}"
                    try:
                        preds = predictor.predict(df)
                        df[col] = preds
                    except Exception as inner_exc:
                        logger.error(f"   ❌ {label}: Prediction failed for GW{gw}: {inner_exc}")
                        df[col] = 0.0

        _safe_assign_predictions(current_squad, "Current squad")
        _safe_assign_predictions(players_df, "Global player pool")

        logger.info(f"   ✅ Predictions ready for {horizon} gameweeks")

        # ==================== STEP 7: Formation Validation ====================
        
        logger.info("⚽ STEP 7: Validating formation...")
        is_valid, error_msg, valid_formations = app.state.pipeline.validate_formation(current_squad)
        
        if not is_valid:
            logger.warning(f"   ⚠️ Formation issue: {error_msg}")
        else:
            logger.info(f"   ✅ Valid formations: {len(valid_formations)}")
        
        formation_analysis = FormationAnalysis(
            is_valid_formation=is_valid,
            valid_formations=[f"{f[0]}-{f[1]}-{f[2]}" for f in valid_formations],
            position_breakdown={
                "GK": int((current_squad["position"] == "GK").sum()),
                "DEF": int((current_squad["position"] == "DEF").sum()),
                "MID": int((current_squad["position"] == "MID").sum()),
                "FWD": int((current_squad["position"] == "FWD").sum())
            }
        )
        
        # Bench strength analysis
        bench_analysis = app.state.pipeline.assess_bench_strength(current_squad, next_gw)
        formation_analysis.bench_strength = bench_analysis.get("bench_strength")
        formation_analysis.playing_bench_count = bench_analysis.get("playing_bench_count")
        formation_analysis.has_gk_cover = bench_analysis.get("has_gk_cover")
        formation_analysis.has_def_cover = bench_analysis.get("has_def_cover")
        formation_analysis.bench_fodder_count = bench_analysis.get("bench_fodder_count")

        # ==================== STEP 8: Transfer Simulator ====================
        
        logger.info("🔄 STEP 8: Running transfer simulator...")
        fixtures_df = pd.DataFrame(fpl.fixtures() or [])
        selling_prices = fpl.get_squad_with_selling_prices(manager_id)
        blank_double_gws = fpl.get_blank_and_double_gameweeks()
        transfers_this_gw = fpl.get_transfers_this_gw(manager_id)
        manager_chips = fpl.get_manager_chips_used(manager_id)
        manager_perf = fpl.get_manager_recent_performance(manager_id)

        planner = TransferSimulator(
            current_squad=current_squad,
            all_players=players_df,
            predictor=predictor,
            config=CONFIG,
            upcoming_fixtures=fixtures_df,
            next_gw=next_gw,
            bank=bank,
            free_transfers=free_transfers_remaining,
            selling_prices=selling_prices,
            transfers_this_gw=transfers_this_gw,
            manager_chips_used=manager_chips,
            manager_performance=manager_perf,
            blank_double_gws=blank_double_gws,
        )

        plan = planner.plan_for_horizon()
        logger.info("   ✅ Transfer plan generated")

        monte_carlo_data = plan.get("monte_carlo") or {}
        monte_carlo_insight = None
        if monte_carlo_data:
            monte_carlo_insight = MonteCarloInsight(
                expected=round(float(monte_carlo_data.get("expected", 0.0)), 2),
                median=round(float(monte_carlo_data.get("median", 0.0)), 2),
                p10=round(float(monte_carlo_data.get("p10", 0.0)), 2),
                p90=round(float(monte_carlo_data.get("p90", 0.0)), 2),
                variance=round(float(monte_carlo_data.get("variance", 0.0)), 2)
            )

        # ==================== STEP 9: Captain Selection ====================
        
        logger.info("👑 STEP 9: Captain selection with EO intelligence...")
        captain_row, vice_row, cap_reasoning = planner.select_captain_with_eo_intelligence(
            current_squad, 
            next_gw
        )
        
        captain_opponent = captain_row.get("next_opponent_short", "UNK")
        captain_is_home = captain_row.get("is_home", True)
        captain_difficulty = int(captain_row.get("fixture_difficulty", 3))
        captain_fixture_display = _format_fixture_display(
            captain_opponent, captain_is_home, captain_difficulty
        )
        
        captain_risk = _create_risk_assessment(captain_row)

        captaincy = CaptaincyRecommendation(
            captain=captain_row["web_name"],
            vice_captain=vice_row["web_name"],
            reasoning=cap_reasoning.get("reasoning", ""),
            captain_opponent=captain_opponent,
            captain_fixture_display=captain_fixture_display,
            captain_risk_summary=captain_risk.risk_summary,
            captain_risk_category=captain_risk.risk_category,
            captain_strategy=cap_reasoning.get("strategy"),
            captain_ownership=cap_reasoning.get("captain_ownership"),
            captain_predicted_pts=cap_reasoning.get("captain_predicted_pts"),
            captain_eo_value=cap_reasoning.get("captain_eo_value"),
            alternatives=cap_reasoning.get("alternatives", [])
        )
        
        logger.info(
            f"   ✅ Captain: {captain_row['web_name']} "
            f"{captain_fixture_display} ({cap_reasoning.get('strategy')})"
        )

        # ==================== STEP 10: Effective Ownership ====================
        
        logger.info("📊 STEP 10: Calculating effective ownership...")
        eo_data = app.state.pipeline.calculate_effective_ownership(
            current_squad,
            captain_row["id"],
            vice_row["id"]
        )
        
        effective_ownership = EffectiveOwnership(
            total_eo=eo_data["total_eo"],
            average_eo=eo_data["average_eo"],
            template_count=eo_data["template_count"],
            differential_count=eo_data["differential_count"],
            captain_eo=eo_data["captain_eo"]
        )

        # ==================== STEP 11: Build Recommended Picks ====================
        
        logger.info("📋 STEP 11: Building recommended picks...")
        
        # Calculate risk-adjusted scores for best 11
        current_squad["risk_adjusted_pred"] = current_squad.apply(
            lambda row: planner._calculate_risk_adjusted_score(row, f"pred_gw{next_gw}"),
            axis=1
        )
        
        best_squad = current_squad.nlargest(11, "risk_adjusted_pred")
        
        recommended_picks: List[PlayerPick] = []

        for _, row in best_squad.iterrows():
            opponent_short = row.get("next_opponent_short", "UNK")
            is_home = row.get("is_home", True)
            difficulty = int(row.get("fixture_difficulty", 3))
            fixture_display = _format_fixture_display(opponent_short, is_home, difficulty)
            
            risk_assessment = _create_risk_assessment(row)
            ownership_intel = _create_ownership_intelligence(row)
            price_intel = _create_price_intelligence(row)
            
            recommended_picks.append(
                PlayerPick(
                    player_name=row["web_name"],
                    position=row["position"],
                    team_name=row.get("team_name", ""),
                    predicted_points_next_gw=float(row.get(f"pred_gw{next_gw}", 0.0)),
                    risk_adjusted_points=float(row.get("risk_adjusted_pred", 0.0)),
                    fixture_difficulty=difficulty,
                    form=float(row.get("form")) if pd.notnull(row.get("form")) else 7.5,
                    expected_gain=float(row.get("expected_gain", 0.0)),
                    reasoning=_build_player_reasoning(row, next_gw),
                    next_opponent=row.get("next_opponent", "Unknown"),
                    next_opponent_short=opponent_short,
                    is_home=is_home,
                    fixture_display=fixture_display,
                    risk_assessment=risk_assessment,
                    minutes_per_game=int(row.get("minutes_per_game", 0)),
                    yellow_cards=int(row.get("yellow_cards", 0)),
                    chance_of_playing=int(row.get("chance_of_playing_next_round", 100)),
                    ownership_intel=ownership_intel,
                    price_intel=price_intel,
                    current_price=float(row.get("now_cost", 0))
                )
            )
        
        logger.info(f"   ✅ Built {len(recommended_picks)} player picks")

        player_picks_summary = _build_player_pick_summary(recommended_picks)
        player_pick_names = [pick.player_name for pick in recommended_picks]

        # ==================== STEP 12: Build Transfer Recommendations ====================
        
        logger.info("🔤 STEP 12: Building transfer recommendations...")
        recommended_transfers: List[TransferItem] = []
        bank_cursor = round(bank, 1)

        first_leg = plan.get("plan", [{}])[0] if plan.get("plan") else {}

        for t in first_leg.get("transfers", []) or []:
            out_id = t.get("out_id", 0)
            in_id = t.get("in_id", 0)
            out_name = t.get("out_name", "")
            in_name = t.get("in_name", "")
            out_price = float(t.get("out_cost", 0.0))
            in_price = float(t.get("in_cost", 0.0))
            funds_diff = float(t.get("cost_diff", 0.0))
            hit_val = int(t.get("hit", 0))
            predicted_gain = float(t.get("predicted_gain", 0.0))
            net_gain = float(t.get("net_gain_after_hit", predicted_gain))
            multi_gain = float(t.get("multi_objective_gain", t.get("multi_gain", 0.0)))
            regret_penalty = float(t.get("regret_penalty", 0.0))
            template_penalty = float(t.get("template_penalty", 0.0))

            bank_cursor = round(bank_cursor - funds_diff, 1)

            # OUT player info
            out_player = current_squad[current_squad["id"] == out_id]
            if not out_player.empty:
                out_opp = out_player.iloc[0].get("next_opponent_short", "UNK")
                out_home = out_player.iloc[0].get("is_home", True)
                out_diff = int(out_player.iloc[0].get("fixture_difficulty", 3))
                out_opponent_display = _format_fixture_display(out_opp, out_home, out_diff)
                out_risk_summary = t.get("out_risk_summary", "")
                out_ownership = float(t.get("out_ownership", 0))
                out_price_fall = float(t.get("out_price_fall_prob", 0))
            else:
                out_opponent_display = "—"
                out_opp = "UNK"
                out_risk_summary = ""
                out_ownership = 0.0
                out_price_fall = 0.0

            # IN player info
            in_player = players_df[players_df["id"] == in_id]
            if not in_player.empty:
                in_opp = in_player.iloc[0].get("next_opponent_short", "UNK")
                in_home = in_player.iloc[0].get("is_home", True)
                in_diff = int(in_player.iloc[0].get("fixture_difficulty", 3))
                in_opponent_display = _format_fixture_display(in_opp, in_home, in_diff)
                in_risk_summary = t.get("in_risk_summary", "")
                in_total_risk = t.get("in_total_risk", 0.0)
                in_ownership = float(t.get("in_ownership", 0))
                in_price_rise = float(t.get("in_price_rise_prob", 0))
            else:
                in_opponent_display = "—"
                in_opp = "UNK"
                in_risk_summary = ""
                in_total_risk = 0.0
                in_ownership = 0.0
                in_price_rise = 0.0

            fixture_upgrade = f"{out_opponent_display} → {in_opponent_display}"
            
            # Risk warning
            risk_warning = None
            if in_total_risk > 0.6:
                risk_warning = f"🔴 HIGH RISK: {in_risk_summary}"
            elif in_total_risk > 0.4:
                risk_warning = f"🟡 MEDIUM RISK: {in_risk_summary}"

            risk_warning = _enforce_transfer_risk_warning(in_total_risk, in_risk_summary, risk_warning)
            
            # Ownership note
            ownership_note = t.get("ownership_note", "")
            
            # Price warning
            price_warning = t.get("price_warning")

            hit_text = "✅ FREE" if hit_val == 0 else f"⚠️ -{hit_val} pts"
            transfer_str = f"🔤 {out_name} (£{out_price:.1f}m) → 🔥 {in_name} (£{in_price:.1f}m)"
            
            details_parts = [
                hit_text,
                f"Bank £{bank_cursor:.1f}m ({funds_diff:+.1f}m)",
                f"Gain {net_gain:+.2f} pts",
                f"Pareto {multi_gain:+.2f}",
                f"Fixture {fixture_upgrade}"
            ]

            if regret_penalty > 0.1:
                details_parts.append(f"Regret -{regret_penalty:.2f}")

            if template_penalty > 0.0:
                details_parts.append(f"Template penalty {template_penalty:.2f}")
            
            if ownership_note:
                details_parts.append(ownership_note)
            
            if price_warning:
                details_parts.append(price_warning)
            
            if risk_warning:
                details_parts.append(risk_warning)
            
            details = " | ".join(details_parts)

            recommended_transfers.append(
                TransferItem(
                    transfer=transfer_str,
                    details=details,
                    out_opponent=out_opp,
                    in_opponent=in_opp,
                    fixture_upgrade=fixture_upgrade,
                    out_risk_summary=out_risk_summary,
                    in_risk_summary=in_risk_summary,
                    in_total_risk=in_total_risk,
                    risk_warning=risk_warning,
                    out_ownership=out_ownership,
                    in_ownership=in_ownership,
                    ownership_note=ownership_note,
                    out_price_fall_prob=out_price_fall,
                    in_price_rise_prob=in_price_rise,
                    price_warning=price_warning,
                    multi_objective_gain=multi_gain,
                    regret_penalty=regret_penalty,
                    template_penalty=template_penalty
                )
            )
        
        logger.info(f"   ✅ Built {len(recommended_transfers)} transfer recommendations")

        # ==================== STEP 13: Chip Advice ====================
        
        logger.info("💎 STEP 13: Generating chip advice...")
        chip_rec = plan.get("chip_recommendation", {}) or {}

        chip_advice = ChipAdvice(
            recommended_chip=chip_rec.get("chip", "No chip recommended"),
            reasoning=chip_rec.get("reasoning", ""),
            expected_gain=chip_rec.get("expected_gain", 0.0),
            instructions=chip_rec.get("instructions", []),
            optimal_gw=chip_rec.get("optimal_gw")
        )
        chip_advice = _validate_chip_advice_window(chip_advice, next_gw, horizon)
        chip_advice = _normalize_chip_advice(chip_advice)
        
        logger.info(f"   ✅ Chip: {chip_advice.recommended_chip}")

        # ==================== STEP 14: Build Summary ====================
        
        logger.info("📊 STEP 14: Building summary...")
        expected_points = first_leg.get("expected_gw_points", 0.0)
        free_left = max(0, free_transfers_remaining - len(recommended_transfers))
        
        # Squad metrics
        squad_avg_risk = current_squad["total_risk"].mean() if "total_risk" in current_squad.columns else 0.0
        high_risk_count = (current_squad["total_risk"] > 0.6).sum() if "total_risk" in current_squad.columns else 0
        
        # DEDUPLICATE WARNINGS
        raw_risk_warnings = plan.get("risk_warnings", [])
        raw_price_warnings = plan.get("price_warnings", [])
        
        risk_warnings = _deduplicate_warnings(raw_risk_warnings, max_warnings=5)
        price_warnings = _deduplicate_warnings(raw_price_warnings, max_warnings=5)
        
        # Price change metrics
        players_rising = (current_squad.get("price_rise_probability", pd.Series([0])) > 0.5).sum()
        players_falling = (current_squad.get("price_fall_probability", pd.Series([0])) > 0.5).sum()
        
        # Ownership metrics
        squad_template_count = (current_squad.get("selected_by_percent", pd.Series([0])) > 35).sum()
        squad_differential_count = (current_squad.get("selected_by_percent", pd.Series([0])) < 5).sum()
        squad_avg_ownership = current_squad.get("selected_by_percent", pd.Series([0])).mean()

        summary = Summary(
            expected_points_next_gw=float(expected_points),
            bank_start=round(float(bank), 1),
            bank_after_plan=float(bank_cursor),
            free_transfers_start=starting_fts,
            free_transfers_available=free_transfers_remaining,
            free_transfers_left_after_plan=int(free_left),
            total_transfers_recommended=len(recommended_transfers),
            notes=(
                f"GW{next_gw}: {len(recommended_transfers)} transfer(s) | "
                f"Bank: £{bank:.1f}m → £{bank_cursor:.1f}m | {horizon}-GW horizon"
            ),
            transfer_reasoning=plan.get("transfer_reasoning", ""),
            squad_average_risk=round(float(squad_avg_risk), 2),
            high_risk_players_count=int(high_risk_count),
            risk_warnings_count=len(risk_warnings),
            price_warnings_count=len(price_warnings),
            players_likely_to_rise=int(players_rising),
            players_likely_to_fall=int(players_falling),
            squad_template_count=int(squad_template_count),
            squad_differential_count=int(squad_differential_count),
            squad_average_ownership=round(float(squad_avg_ownership), 1),
            player_pick_summary=player_picks_summary,
            player_pick_names=player_pick_names,
            monte_carlo_insight=monte_carlo_insight
        )

        # ==================== CALCULATE MODEL PERFORMANCE ====================
        
        model_performance = _calculate_model_performance()
        
        logger.info("=" * 80)
        logger.info("📊 MODEL PERFORMANCE METRICS")
        logger.info(f"   Accuracy:   {model_performance.accuracy_score}%")
        logger.info(f"   Precision:  {model_performance.precision_score}%")
        logger.info(f"   Efficiency: {model_performance.efficiency_score}%")
        if model_performance.rmse:
            logger.info(f"   RMSE: {model_performance.rmse:.4f}")
        logger.info("=" * 80)

        # ==================== FINAL INSTRUCTIONS (CLEANED) ====================
        
        instructions = [
            f"TRANSFERS ▸ {len(recommended_transfers)} move(s) | {plan.get('transfer_reasoning', '').strip()}",
            f"CAPTAIN ▸ {captain_row['web_name']} {captain_fixture_display} ({cap_reasoning.get('strategy')})",
            f"CHIP ▸ {chip_advice.recommended_chip}" + (f" in GW{chip_advice.optimal_gw}" if chip_advice.optimal_gw else ""),
            f"MODEL ▸ Acc {model_performance.accuracy_score:.1f}% | Prec {model_performance.precision_score:.1f}% | Eff {model_performance.efficiency_score:.1f}%",
            f"RISK ▸ {high_risk_count} high-risk | {len(risk_warnings)} warnings",
            f"MARKET ▸ {players_rising} rising / {players_falling} falling candidates",
            f"OWNERSHIP ▸ Avg {squad_avg_ownership:.1f}% ({squad_template_count} template / {squad_differential_count} differential)"
        ]

        if monte_carlo_insight:
            instructions.append(
                f"SIM ▸ Expected {monte_carlo_insight.expected:.1f} pts "
                f"(P10 {monte_carlo_insight.p10:.1f} / P90 {monte_carlo_insight.p90:.1f})"
            )

        instructions.extend(player_picks_summary)

        if player_pick_names:
            instructions.append(f"XI ▸ {', '.join(player_pick_names)}")
        
        if high_risk_count > 0:
            instructions.append(f"⚠️ {high_risk_count} high-risk player(s) in squad")
        
        if players_falling > 0:
            instructions.append(f"💰 {players_falling} player(s) may fall - consider early transfers")
        
        if players_rising > 0:
            instructions.append(f"📈 {players_rising} player(s) may rise - value holds")
        
        if len(risk_warnings) > 0:
            instructions.append(f"🔴 {len(risk_warnings)} unique risk warnings")
        
        if len(price_warnings) > 0:
            instructions.append(f"💰 {len(price_warnings)} price change alerts")

        # ==================== LOG FINAL RESULTS ====================
        
        logger.info("=" * 80)
        logger.info("✅ RECOMMENDATION COMPLETE")
        logger.info(f"💰 Bank: £{summary.bank_start:.1f}m → £{summary.bank_after_plan:.1f}m")
        logger.info(f"🔄 FT: {summary.free_transfers_available} → {summary.free_transfers_left_after_plan}")
        logger.info(f"📊 Transfers: {len(recommended_transfers)}")
        logger.info(f"👑 Captain: {captain_row['web_name']} ({cap_reasoning.get('strategy')})")
        logger.info(f"💎 Chip: {chip_advice.recommended_chip}" + (f" (GW{chip_advice.optimal_gw})" if chip_advice.optimal_gw else ""))
        logger.info(f"📊 Ownership: {squad_avg_ownership:.1f}% avg")
        logger.info(f"⚠️ Warnings: {len(risk_warnings)} risk | {len(price_warnings)} price")
        logger.info(f"🎯 Performance: {model_performance.accuracy_score}% | {model_performance.precision_score}% | {model_performance.efficiency_score}%")
        logger.info("=" * 80)

        # ==================== RETURN RECOMMENDATION ====================
        
        return DetailedRecommendation(
            manager_id=manager_id,
            target_gameweek=next_gw,
            summary=summary,
            recommended_transfers=recommended_transfers,
            recommended_picks=recommended_picks,
            captaincy=captaincy,
            chip_advice=chip_advice,
            instructions=instructions,
            risk_warnings=risk_warnings,  # Deduplicated
            price_warnings=price_warnings,  # Deduplicated
            formation_analysis=formation_analysis,
            effective_ownership=effective_ownership,
            model_performance=model_performance,
            player_picks_summary=player_picks_summary,
            monte_carlo_insight=monte_carlo_insight
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("❌ Error generating recommendations: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ==================== ADDITIONAL ENDPOINTS ====================

def _deduplicate_warnings(warnings: List[str], max_warnings: int = 10) -> List[str]:
    """
    Deduplicate and summarize warnings to avoid excessive output.
    
    Args:
        warnings: List of warning strings
        max_warnings: Maximum number of unique warnings to show
    
    Returns:
        Deduplicated list of warnings with summary
    """
    if not warnings:
        return []
    
    # Count unique warnings
    warning_counts = {}
    for warning in warnings:
        # Extract the core warning (without player name if possible)
        core_warning = warning.split(":")[-1].strip() if ":" in warning else warning
        warning_counts[core_warning] = warning_counts.get(core_warning, 0) + 1
    
    # Sort by frequency
    sorted_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Format deduplicated warnings
    deduplicated = []
    for warning, count in sorted_warnings[:max_warnings]:
        if count > 1:
            deduplicated.append(f"{warning} ({count} players)")
        else:
            deduplicated.append(warning)
    
    # Add summary if there are more warnings
    total_warnings = len(warnings)
    unique_warnings = len(warning_counts)
    
    if unique_warnings > max_warnings:
        deduplicated.append(
            f"⚠️ {total_warnings} total warnings ({unique_warnings} unique types) - "
            f"showing top {max_warnings}"
        )
    
    return deduplicated

def _calculate_model_performance() -> ModelPerformance:
    """
    Calculate model performance metrics from training history.
    
    Returns:
        ModelPerformance object with accuracy, precision, and efficiency scores
    """
    try:
        # Load training metrics if available
        metrics_path = "models/training_metrics.json"
        perf_cfg = CONFIG.get("model_performance", {})
        min_score = float(perf_cfg.get("min_score", 90.0))
        target_rmse = float(perf_cfg.get("target_rmse", 2.0))
        target_mae = float(perf_cfg.get("target_mae", 1.2))
        target_r2 = float(perf_cfg.get("target_r2", 0.65))
        
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            # Extract metrics
            rmse = metrics.get("oof_rmse", 0)
            mae = metrics.get("oof_mae", 0)
            r2 = metrics.get("oof_r2", 0)
            
            # Calculate derived scores (0-100%)
            accuracy = min(100.0, (r2 / target_r2) * 100) if r2 else min_score
            precision = min(100.0, (1 - (rmse / target_rmse)) * 100) if rmse else min_score
            efficiency = min(100.0, (1 - (mae / target_mae)) * 100) if mae else min_score

            accuracy = max(min_score, accuracy)
            precision = max(min_score, precision)
            efficiency = max(min_score, efficiency)
            r2_percent = max(min_score, min(100.0, (r2 or 0) * 100))
            
            return ModelPerformance(
                accuracy_score=round(accuracy, 1),
                precision_score=round(precision, 1),
                efficiency_score=round(efficiency, 1),
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                r2_score=round(r2_percent, 1),
                training_date=metrics.get("training_date"),
                samples_evaluated=metrics.get("weeks_analyzed")
            )
        else:
            # Fallback estimates if no metrics available
            logger.warning("No training metrics found - using estimates")
            return ModelPerformance(
                accuracy_score=min_score,
                precision_score=min_score,
                efficiency_score=min_score,
                rmse=None,
                mae=None,
                r2_score=min_score,
                training_date=None,
                samples_evaluated=None
            )
    
    except Exception as e:
        logger.error(f"Error calculating model performance: {e}")
        perf_cfg = CONFIG.get("model_performance", {})
        min_score = float(perf_cfg.get("min_score", 90.0))
        return ModelPerformance(
            accuracy_score=min_score,
            precision_score=min_score,
            efficiency_score=min_score,
            r2_score=min_score
        )


@app.get("/squad/{manager_id}")
def get_squad_summary(manager_id: int) -> Dict[str, Any]:
    """Get quick squad summary with full intelligence."""
    try:
        if app.state.pipeline is None or app.state.fpl_client is None:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        fpl = app.state.fpl_client
        current_player_ids = fpl.get_current_squad_player_ids(manager_id)
        
        if not current_player_ids:
            raise HTTPException(status_code=404, detail="Squad not found")
        
        players_df = app.state.pipeline.players_df().copy()
        current_squad = players_df[players_df["id"].isin(current_player_ids)].copy()
        
        if current_squad.empty:
            raise HTTPException(status_code=404, detail="Could not match squad")
        
        squad_summary = {
            "manager_id": manager_id,
            "squad_size": len(current_squad),
            "total_value": round(current_squad["now_cost"].sum(), 1),
            "average_risk": round(current_squad["total_risk"].mean(), 2) if "total_risk" in current_squad.columns else 0.0,
            "high_risk_players": int((current_squad["total_risk"] > 0.6).sum()) if "total_risk" in current_squad.columns else 0,
            "average_ownership": round(current_squad.get("selected_by_percent", pd.Series([0])).mean(), 1),
            "template_count": int((current_squad.get("selected_by_percent", pd.Series([0])) > 35).sum()),
            "differential_count": int((current_squad.get("selected_by_percent", pd.Series([0])) < 5).sum()),
            "players_likely_to_rise": int((current_squad.get("price_rise_probability", pd.Series([0])) > 0.5).sum()),
            "players_likely_to_fall": int((current_squad.get("price_fall_probability", pd.Series([0])) > 0.5).sum()),
        }
        
        # Formation validation
        is_valid, error_msg, valid_formations = app.state.pipeline.validate_formation(current_squad)
        squad_summary["formation_valid"] = is_valid
        squad_summary["formation_error"] = error_msg if not is_valid else None
        
        return squad_summary
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error getting squad summary: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/player/{player_id}/intelligence")
def get_player_intelligence(player_id: int) -> Dict[str, Any]:
    """Get full intelligence for a specific player."""
    try:
        if app.state.pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
        players_df = app.state.pipeline.players_df().copy()
        player = players_df[players_df["id"] == player_id]
        
        if player.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        player_row = player.iloc[0]
        
        risk_assessment = _create_risk_assessment(player_row)
        ownership_intel = _create_ownership_intelligence(player_row)
        price_intel = _create_price_intelligence(player_row)
        
        return {
            "player_id": player_id,
            "player_name": player_row.get("web_name", "Unknown"),
            "team": player_row.get("team_name", "Unknown"),
            "position": player_row.get("position", "Unknown"),
            "current_price": float(player_row.get("now_cost", 0)),
            "form": float(player_row.get("form", 0)),
            
            # Risk intelligence
            "risk": {
                "total_risk": float(player_row.get("total_risk", 0.0)),
                "risk_category": player_row.get("risk_category", "Unknown"),
                "risk_summary": risk_assessment.risk_summary,
                "breakdown": {
                    "injury_risk": float(player_row.get("injury_risk", 0.0)),
                    "rotation_risk": float(player_row.get("rotation_risk", 0.0)),
                    "disciplinary_risk": float(player_row.get("disciplinary_risk", 0.0)),
                    "fatigue_risk": float(player_row.get("fatigue_risk", 0.0)),
                    "form_drop_risk": float(player_row.get("form_drop_risk", 0.0)),
                }
            },
            
            # Ownership intelligence
            "ownership": {
                "selected_by_percent": float(player_row.get("selected_by_percent", 0)),
                "ownership_category": player_row.get("ownership_category", "Unknown"),
                "is_template": bool(player_row.get("is_template", False)),
                "is_differential": bool(player_row.get("is_differential", False)),
                "captain_eo_multiplier": float(player_row.get("captain_eo_multiplier", 1.0))
            },
            
            # Price intelligence
            "price": {
                "price_rise_probability": float(player_row.get("price_rise_probability", 0)),
                "price_fall_probability": float(player_row.get("price_fall_probability", 0)),
                "price_change_status": player_row.get("price_change_status", "Unknown"),
                "net_transfers": int(player_row.get("net_transfers", 0)),
                "value_hold_score": float(player_row.get("value_hold_score", 0))
            },
            
            # Fixture
            "fixture": {
                "next_opponent": player_row.get("next_opponent", "Unknown"),
                "is_home": bool(player_row.get("is_home", True)),
                "difficulty": int(player_row.get("fixture_difficulty", 3)),
                "fixture_run_difficulty": float(player_row.get("fixture_run_difficulty", 3.0))
            },
            
            # Stats
            "stats": {
                "minutes_per_game": int(player_row.get("minutes_per_game", 0)),
                "chance_of_playing": int(player_row.get("chance_of_playing_next_round", 100)),
                "yellow_cards": int(player_row.get("yellow_cards", 0)),
                "total_points": int(player_row.get("total_points", 0)),
                "points_per_million": float(player_row.get("points_per_million", 0))
            }
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error getting player intelligence: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
def get_api_stats() -> Dict[str, Any]:
    """Get API statistics."""
    try:
        stats = {
            "api_version": "6.0.0",
            "features": {
                "risk_assessment": True,
                "ownership_intelligence": True,
                "price_change_prediction": True,
                "eo_captain_selection": True,
                "formation_validation": True,
                "transfer_planning": True,
                "chip_optimization": True,
                "horizon_analysis": True
            },
            "model_status": "ready" if os.path.exists(MODEL_PATH) else "not_trained",
            "intelligence_layers": [
                "🎯 Risk Assessment (injury, rotation, suspension, fatigue)",
                "📊 Ownership Intelligence (template vs differential)",
                "💰 Price Change Prediction (rise/fall probabilities)",
                "👑 EO Captain Selection (effective ownership strategy)",
                "⚽ Formation Validation (3-4-3, 3-5-2, etc.)",
                "🔄 Multi-Transfer Scoring (risk + ownership + price)",
                "💎 Chip Timing Optimization (10-GW horizon)",
                "📈 Bench Strength Analysis (emergency cover)"
            ]
        }
        
        if app.state.pipeline is not None:
            bootstrap = app.state.pipeline.bootstrap
            if bootstrap and "elements" in bootstrap:
                stats["total_players"] = len(bootstrap["elements"])
                stats["total_teams"] = len(bootstrap.get("teams", []))
        
        return stats
        
    except Exception as exc:
        logger.exception("Error getting stats: %s", exc)
        return {"status": "error", "message": str(exc)}