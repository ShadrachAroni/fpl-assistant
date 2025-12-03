"""
Test script to directly test recommendations and analyze results
"""
import os
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client.fpl_client import FPLClient
from app.data.pipeline import DataPipeline
from app.models.predictor import Predictor
from app.planner.simulator import TransferSimulator
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test")

# Suppress noisy logs
for noisy_logger in ["httpx", "urllib3", "asyncio"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

def test_recommendations(manager_id: int):
    """Test recommendations for a manager ID."""
    logger.info("=" * 80)
    logger.info(f"Testing recommendations for Manager {manager_id}")
    logger.info("=" * 80)
    
    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize clients
    fpl_client = FPLClient()
    bootstrap = fpl_client.bootstrap()
    
    if not bootstrap:
        logger.error("Failed to fetch bootstrap")
        return None
    
    # Get current gameweek
    import pandas as pd
    events = pd.DataFrame(bootstrap.get("events", []))
    current_gw = int(events[events["is_current"] == True]["id"].iloc[0]) if not events[events["is_current"] == True].empty else int(events["id"].max())
    next_gw = int(events[events["is_next"] == True]["id"].iloc[0]) if not events[events["is_next"] == True].empty else current_gw + 1
    
    logger.info(f"Current GW: {current_gw}, Next GW: {next_gw}")
    
    # Initialize pipeline
    pipeline = DataPipeline(config, fpl_client=fpl_client)
    pipeline.fetch_bootstrap()
    
    # Get financial data
    bank_info = fpl_client.manager_bank_and_free_transfers(manager_id)
    bank = float(bank_info["bank"])
    free_transfers = int(bank_info["free_transfers"])
    logger.info(f"Bank: £{bank:.1f}m, Free Transfers: {free_transfers}")
    
    # Get current squad
    current_player_ids = fpl_client.get_current_squad_player_ids(manager_id)
    logger.info(f"Current squad: {len(current_player_ids)} players")
    
    # Load player data
    players_df = pipeline.players_df().copy()
    POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    players_df["position"] = players_df["element_type"].map(POS_MAP).fillna("UNK")
    
    current_squad = players_df[players_df["id"].isin(current_player_ids)].copy()
    logger.info(f"Matched {len(current_squad)} players in squad")
    
    # Load model
    model_path = "models/lightgbm_model.joblib"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    predictor = Predictor(model_path)
    
    # Ensure features exist
    for col in getattr(predictor, "feats", []):
        if col not in players_df.columns:
            players_df[col] = 0
    
    # Generate predictions
    horizon = int(config["simulation"].get("planning_horizon", 5))
    logger.info(f"Generating predictions for {horizon} gameweeks...")
    
    def assign_predictions(df, label):
        try:
            preds = predictor.predict(df)
            for i in range(horizon):
                df[f"pred_gw{next_gw + i}"] = preds
            logger.info(f"✅ {label}: predictions assigned")
        except Exception as e:
            logger.error(f"❌ {label}: {e}")
            for i in range(horizon):
                df[f"pred_gw{next_gw + i}"] = 0.0
    
    assign_predictions(current_squad, "Current squad")
    assign_predictions(players_df, "All players")
    
    # Get additional data
    fixtures_df = pd.DataFrame(fpl_client.fixtures() or [])
    selling_prices = fpl_client.get_squad_with_selling_prices(manager_id)
    blank_double_gws = fpl_client.get_blank_and_double_gameweeks()
    transfers_this_gw = fpl_client.get_transfers_this_gw(manager_id)
    manager_chips = fpl_client.get_manager_chips_used(manager_id)
    manager_perf = fpl_client.get_manager_recent_performance(manager_id)
    
    # Initialize simulator
    planner = TransferSimulator(
        current_squad=current_squad,
        all_players=players_df,
        predictor=predictor,
        config=config,
        upcoming_fixtures=fixtures_df,
        next_gw=next_gw,
        bank=bank,
        free_transfers=free_transfers,
        selling_prices=selling_prices,
        transfers_this_gw=transfers_this_gw,
        manager_chips_used=manager_chips,
        manager_performance=manager_perf,
        blank_double_gws=blank_double_gws,
    )
    
    # Generate plan
    logger.info("Running transfer simulator...")
    plan = planner.plan_for_horizon()
    
    # Analyze results
    logger.info("=" * 80)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 80)
    
    first_leg = plan.get("plan", [{}])[0] if plan.get("plan") else {}
    transfers = first_leg.get("transfers", [])
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Expected points next GW: {first_leg.get('expected_gw_points', 0):.1f}")
    logger.info(f"   Transfers recommended: {len(transfers)}")
    logger.info(f"   Bank after plan: £{bank - sum(t.get('cost_diff', 0) for t in transfers):.1f}m")
    
    logger.info(f"\n🔄 Transfers:")
    for i, t in enumerate(transfers[:5], 1):
        logger.info(f"   {i}. {t.get('out_name')} → {t.get('in_name')}")
        logger.info(f"      Gain: {t.get('predicted_gain', 0):.2f} pts | Risk: {t.get('in_total_risk', 0):.2f}")
    
    # Captain selection
    captain_row, vice_row, cap_reasoning = planner.select_captain_with_eo_intelligence(current_squad, next_gw)
    logger.info(f"\n👑 Captain:")
    logger.info(f"   {captain_row['web_name']} ({cap_reasoning.get('strategy', 'N/A')})")
    logger.info(f"   Reasoning: {cap_reasoning.get('reasoning', 'N/A')[:100]}")
    
    # Chip advice
    chip_rec = plan.get("chip_recommendation", {})
    logger.info(f"\n💎 Chip:")
    logger.info(f"   {chip_rec.get('chip', 'No chip recommended')}")
    if chip_rec.get('optimal_gw'):
        logger.info(f"   Optimal GW: {chip_rec.get('optimal_gw')}")
    
    # Risk analysis
    squad_avg_risk = current_squad["total_risk"].mean() if "total_risk" in current_squad.columns else 0.0
    high_risk_count = (current_squad["total_risk"] > 0.6).sum() if "total_risk" in current_squad.columns else 0
    logger.info(f"\n⚠️ Risk Analysis:")
    logger.info(f"   Squad average risk: {squad_avg_risk:.2f}")
    logger.info(f"   High-risk players: {high_risk_count}")
    
    # Ownership analysis
    if "selected_by_percent" in current_squad.columns:
        avg_ownership = current_squad["selected_by_percent"].mean()
        template_count = (current_squad["selected_by_percent"] > 35).sum()
        diff_count = (current_squad["selected_by_percent"] < 5).sum()
        logger.info(f"\n📊 Ownership:")
        logger.info(f"   Average: {avg_ownership:.1f}%")
        logger.info(f"   Template (>35%): {template_count}")
        logger.info(f"   Differential (<5%): {diff_count}")
    
    # Price analysis
    if "price_rise_probability" in current_squad.columns:
        rising = (current_squad["price_rise_probability"] > 0.5).sum()
        falling = (current_squad["price_fall_probability"] > 0.5).sum()
        logger.info(f"\n💰 Price Changes:")
        logger.info(f"   Likely rising: {rising}")
        logger.info(f"   Likely falling: {falling}")
    
    # Check for missing features
    logger.info(f"\n🔍 Feature Analysis:")
    required_features = getattr(predictor, "feats", [])
    missing_in_squad = [f for f in required_features if f not in current_squad.columns]
    missing_in_all = [f for f in required_features if f not in players_df.columns]
    
    if missing_in_squad:
        logger.warning(f"   Missing in squad: {len(missing_in_squad)} features")
    if missing_in_all:
        logger.warning(f"   Missing in all players: {len(missing_in_all)} features")
    
    logger.info(f"   Total features used: {len(required_features)}")
    
    # Model performance check
    metrics_path = "models/training_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        logger.info(f"\n📈 Model Performance:")
        logger.info(f"   RMSE: {metrics.get('oof_rmse', 0):.4f}")
        logger.info(f"   MAE: {metrics.get('oof_mae', 0):.4f}")
        logger.info(f"   R²: {metrics.get('oof_r2', 0):.4f}")
    
    logger.info("=" * 80)
    
    return {
        "plan": plan,
        "current_squad": current_squad,
        "players_df": players_df,
        "predictor": predictor,
        "metrics": metrics if os.path.exists(metrics_path) else None
    }

if __name__ == "__main__":
    manager_id = 3184916
    results = test_recommendations(manager_id)
    
    if results:
        logger.info("\n✅ Test completed successfully!")
    else:
        logger.error("\n❌ Test failed!")

