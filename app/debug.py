"""
FPL Assistant - Complete Diagnostic Suite
Tests all major components with detailed output

FIXED VERSION: Includes Chip Advice and Captain Selection tests
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client.fpl_client import FPLClient
from app.data.pipeline import DataPipeline
from app.models.predictor import Predictor
from app.planner.simulator import TransferSimulator
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("fpl_diagnostic")

# Suppress noisy loggers
for noisy_logger in ["httpx", "urllib3", "asyncio"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class FPLDiagnostic:
    """Comprehensive diagnostic suite for FPL Assistant."""
    
    def __init__(self, manager_id: int = None):
        self.manager_id = manager_id
        self.fpl_client = None
        self.pipeline = None
        self.config = None
        self.issues = []
        self.results = {}
        
    def load_config(self):
        """Load configuration."""
        config_path = PROJECT_ROOT / "config.yaml"
        
        if not config_path.exists():
            self.issues.append("❌ Config file not found")
            return False
        
        with open(config_path, "r") as fh:
            self.config = yaml.safe_load(fh)
        
        return True
    
    def test_bootstrap(self):
        """Test 1: Bootstrap Data Loading."""
        logger.info("=" * 80)
        logger.info("TEST 1: Bootstrap Data Loading")
        logger.info("=" * 80)
        
        try:
            self.fpl_client = FPLClient()
            self.pipeline = DataPipeline(self.config, fpl_client=self.fpl_client)
            
            bootstrap = self.pipeline.fetch_bootstrap()
            
            if not bootstrap:
                self.issues.append("❌ [Bootstrap] Failed to fetch data")
                self.results["bootstrap"] = False
                return False
            
            players = bootstrap.get("elements", [])
            teams = bootstrap.get("teams", [])
            events = bootstrap.get("events", [])
            
            logger.info(f"✅ Bootstrap: {len(players)} players")
            logger.info(f"✅ Teams map: {len(self.pipeline.teams_map)} teams")
            logger.info(f"✅ Elements: {len(players)}")
            logger.info(f"✅ Teams: {len(teams)}")
            logger.info(f"✅ Events: {len(events)}")
            
            self.results["bootstrap"] = True
            return True
            
        except Exception as e:
            self.issues.append(f"❌ [Bootstrap] {str(e)}")
            self.results["bootstrap"] = False
            return False
    
    def test_risk_indicators(self):
        """Test 2: Risk Indicators."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 2: Risk Indicators")
        logger.info("=" * 80)
        
        try:
            players_df = self.pipeline.players_df()
            
            if players_df.empty:
                self.issues.append("❌ [Risk] Players DataFrame is empty")
                self.results["risk"] = False
                return False
            
            # Check for risk columns
            required_risk_cols = [
                "total_risk",
                "injury_risk", 
                "rotation_risk",
                "disciplinary_risk",
                "fatigue_risk",
                "form_drop_risk",
                "risk_category",
                "minutes_per_game",
                "chance_of_playing_next_round",
                "yellow_cards"
            ]
            
            missing_cols = [col for col in required_risk_cols if col not in players_df.columns]
            
            if missing_cols:
                self.issues.append(f"❌ [Risk] Missing columns: {missing_cols}")
                self.results["risk"] = False
                return False
            
            logger.info(f"✅ All {len(required_risk_cols)} risk columns present")
            
            # Display risk analysis for top owned players
            logger.info("")
            logger.info("📊 Risk Analysis (Top 10 by ownership):")
            logger.info("-" * 80)
            
            top_owned = players_df.nlargest(10, "selected_by_percent")
            
            for _, player in top_owned.iterrows():
                name = player.get("web_name", "Unknown")
                risk_cat = player.get("risk_category", "Unknown")
                total_risk = player.get("total_risk", 0)
                injury_risk = player.get("injury_risk", 0)
                rotation_risk = player.get("rotation_risk", 0)
                disciplinary_risk = player.get("disciplinary_risk", 0)
                fatigue_risk = player.get("fatigue_risk", 0)
                form_drop_risk = player.get("form_drop_risk", 0)
                chance_playing = player.get("chance_of_playing_next_round", 100)
                minutes_pg = int(player.get("minutes_per_game", 0))
                yellows = int(player.get("yellow_cards", 0))
                form = player.get("form", 0)
                
                logger.info(f"\n{name} ({risk_cat}):")
                logger.info(f"  Total Risk: {total_risk:.3f}")
                logger.info(f"  - Injury: {injury_risk:.3f} (chance: {chance_playing:.1f}%)")
                logger.info(f"  - Rotation: {rotation_risk:.3f} (mins/game: {minutes_pg})")
                logger.info(f"  - Disciplinary: {disciplinary_risk:.3f} (yellows: {yellows})")
                logger.info(f"  - Fatigue: {fatigue_risk:.3f}")
                logger.info(f"  - Form Drop: {form_drop_risk:.3f} (form: {form})")
            
            # Risk statistics
            logger.info("")
            logger.info("=" * 80)
            logger.info("📈 Risk Statistics:")
            avg_risk = players_df["total_risk"].mean()
            high_risk = (players_df["total_risk"] > 0.6).sum()
            medium_risk = ((players_df["total_risk"] >= 0.3) & (players_df["total_risk"] <= 0.6)).sum()
            low_risk = (players_df["total_risk"] < 0.3).sum()
            injured = (players_df["chance_of_playing_next_round"] < 100).sum()
            rotation_prone = (players_df["rotation_risk"] > 0.5).sum()
            suspension_risk = (players_df["disciplinary_risk"] > 0.6).sum()
            
            logger.info(f"  Average Risk: {avg_risk:.3f}")
            logger.info(f"  High Risk (>0.6): {high_risk} players")
            logger.info(f"  Medium Risk (0.3-0.6): {medium_risk} players")
            logger.info(f"  Low Risk (<0.3): {low_risk} players")
            logger.info(f"  Players with Injuries: {injured}")
            logger.info(f"  Rotation Prone: {rotation_prone}")
            logger.info(f"  Suspension Risk: {suspension_risk}")
            
            self.results["risk"] = True
            return True
            
        except Exception as e:
            self.issues.append(f"❌ [Risk] {str(e)}")
            self.results["risk"] = False
            return False
    
    def test_ownership_and_price(self):
        """Test 3: Ownership & Price Intelligence."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 3: Ownership & Price Intelligence")
        logger.info("=" * 80)
        
        try:
            players_df = self.pipeline.players_df()
            
            # Check ownership columns
            ownership_cols = [
                "selected_by_percent",
                "is_template",
                "is_differential",
                "ownership_category",
                "captain_eo_multiplier"
            ]
            
            missing_ownership = [col for col in ownership_cols if col not in players_df.columns]
            
            if missing_ownership:
                self.issues.append(f"❌ [Ownership] Missing: {missing_ownership}")
                self.results["ownership"] = False
                return False
            
            logger.info(f"✅ All {len(ownership_cols)} ownership columns present")
            
            # Check price columns
            price_cols = [
                "price_rise_probability",
                "price_fall_probability",
                "price_change_status",
                "net_transfers",
                "value_hold_score"
            ]
            
            missing_price = [col for col in price_cols if col not in players_df.columns]
            
            if missing_price:
                self.issues.append(f"❌ [Price] Missing: {missing_price}")
                self.results["ownership"] = False
                return False
            
            logger.info(f"✅ All {len(price_cols)} price columns present")
            
            # Ownership distribution
            templates = players_df["is_template"].sum()
            differentials = players_df["is_differential"].sum()
            
            logger.info("")
            logger.info("📊 Ownership Distribution:")
            logger.info(f"  Templates (>35%): {templates}")
            logger.info(f"  Differentials (<5%): {differentials}")
            
            # Price changes
            rising = (players_df["price_rise_probability"] > 0.5).sum()
            falling = (players_df["price_fall_probability"] > 0.5).sum()
            
            logger.info("")
            logger.info("💰 Price Changes:")
            logger.info(f"  Likely to Rise (>50%): {rising}")
            logger.info(f"  Likely to Fall (>50%): {falling}")
            
            # Top templates
            logger.info("")
            logger.info("🔴 Top 5 Templates:")
            top_templates = players_df.nlargest(5, "selected_by_percent")
            for _, p in top_templates.iterrows():
                logger.info(f"  {p['web_name']}: {p['selected_by_percent']:.1f}% owned")
            
            # Top differentials
            logger.info("")
            logger.info("💎 Top 5 Differentials (by points):")
            differentials_df = players_df[players_df["is_differential"] == True]
            top_diffs = differentials_df.nlargest(5, "total_points")
            for _, p in top_diffs.iterrows():
                logger.info(f"  {p['web_name']}: {p['selected_by_percent']:.1f}% owned, {int(p['total_points'])} pts")
            
            self.results["ownership"] = True
            return True
            
        except Exception as e:
            self.issues.append(f"❌ [Ownership] {str(e)}")
            self.results["ownership"] = False
            return False
    
    def test_chip_advice(self):
        """Test 4: Chip Advice Generation."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 4: Chip Advice")
        logger.info("=" * 80)
        
        if not self.manager_id:
            logger.info("⚠️  No manager ID provided - skipping chip advice test")
            logger.info("   Run with: python -m app.debug --manager-id YOUR_ID")
            self.results["chip"] = None
            return None
        
        try:
            # Get manager data
            manager_data = self.fpl_client.manager(self.manager_id)
            
            if not manager_data:
                self.issues.append(f"❌ [Chip] Could not fetch manager {self.manager_id}")
                self.results["chip"] = False
                return False
            
            # Get chips used
            chips_used = self.fpl_client.get_manager_chips_used(self.manager_id)
            
            logger.info("💎 Chips Status:")
            logger.info(f"  Wildcard: {'✅ Used' if chips_used.get('wildcard') else '❌ Available'}")
            logger.info(f"  Triple Captain: {'✅ Used' if chips_used.get('triple_captain') else '❌ Available'}")
            logger.info(f"  Bench Boost: {'✅ Used' if chips_used.get('bench_boost') else '❌ Available'}")
            logger.info(f"  Free Hit: {'✅ Used' if chips_used.get('free_hit') else '❌ Available'}")
            
            # Check for DGWs/BGWs
            fixtures_df = pd.DataFrame(self.fpl_client.fixtures() or [])
            
            if not fixtures_df.empty:
                next_gw = self.fpl_client.next_gw() or self.fpl_client.current_gw()
                
                # Check next 5 GWs for DGWs
                dgw_found = []
                for i in range(5):
                    gw = next_gw + i
                    gw_fixtures = fixtures_df[fixtures_df["event"] == gw]
                    
                    if not gw_fixtures.empty:
                        team_fixture_count = {}
                        for _, row in gw_fixtures.iterrows():
                            home = int(row.get("team_h", 0))
                            away = int(row.get("team_a", 0))
                            team_fixture_count[home] = team_fixture_count.get(home, 0) + 1
                            team_fixture_count[away] = team_fixture_count.get(away, 0) + 1
                        
                        dgw_teams = [t for t, c in team_fixture_count.items() if c >= 2]
                        
                        if dgw_teams:
                            dgw_found.append((gw, len(dgw_teams)))
                
                if dgw_found:
                    logger.info("")
                    logger.info("🔥 Double Gameweeks Detected:")
                    for gw, count in dgw_found:
                        logger.info(f"  GW{gw}: {count} teams with DGW")
                else:
                    logger.info("")
                    logger.info("📅 No Double Gameweeks in next 5 GWs")
            
            logger.info("")
            logger.info("✅ Chip analysis complete")
            
            self.results["chip"] = True
            return True
            
        except Exception as e:
            self.issues.append(f"❌ [Chip] {str(e)}")
            self.results["chip"] = False
            return False
    
    def test_captain_selection(self):
        """Test 5: Captain Selection."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 5: Captain Selection")
        logger.info("=" * 80)
        
        if not self.manager_id:
            logger.info("⚠️  No manager ID provided - skipping captain test")
            logger.info("   Run with: python -m app.debug --manager-id YOUR_ID")
            self.results["captain"] = None
            return None
        
        try:
            # Check if model exists
            model_path = "models/lightgbm_model.joblib"
            
            if not os.path.exists(model_path):
                logger.info("⚠️  Model not trained yet - using form-based captain selection")
                predictor = None
            else:
                logger.info("✅ Loading trained model...")
                predictor = Predictor(model_path)
            
            # Get current squad
            current_player_ids = self.fpl_client.get_current_squad_player_ids(self.manager_id)
            
            if not current_player_ids:
                self.issues.append(f"❌ [Captain] Could not load squad for manager {self.manager_id}")
                self.results["captain"] = False
                return False
            
            players_df = self.pipeline.players_df()
            current_squad = players_df[players_df["id"].isin(current_player_ids)].copy()
            
            if current_squad.empty:
                self.issues.append(f"❌ [Captain] Could not match squad players")
                self.results["captain"] = False
                return False
            
            logger.info(f"✅ Squad loaded: {len(current_squad)} players")
            
            # Add predictions if model available
            next_gw = self.fpl_client.next_gw() or self.fpl_client.current_gw()
            
            if predictor:
                try:
                    preds = predictor.predict(current_squad)
                    current_squad[f"pred_gw{next_gw}"] = preds
                    logger.info(f"✅ Predictions generated for GW{next_gw}")
                except Exception as e:
                    logger.info(f"⚠️  Prediction failed: {e}")
                    current_squad[f"pred_gw{next_gw}"] = current_squad.get("form", 5.0)
            else:
                current_squad[f"pred_gw{next_gw}"] = current_squad.get("form", 5.0)
            
            # Use TransferSimulator for captain selection
            planner = TransferSimulator(
                current_squad=current_squad,
                all_players=players_df,
                predictor=predictor,
                config=self.config,
                upcoming_fixtures=pd.DataFrame(self.fpl_client.fixtures() or []),
                next_gw=next_gw,
                bank=10.0,
                free_transfers=1
            )
            
            captain_row, vice_row, reasoning = planner.select_captain_with_eo_intelligence(
                current_squad,
                next_gw
            )
            
            logger.info("")
            logger.info("👑 Captain Recommendation:")
            logger.info(f"  Captain: {captain_row['web_name']}")
            logger.info(f"    - Ownership: {captain_row.get('selected_by_percent', 0):.1f}%")
            logger.info(f"    - Predicted: {captain_row.get(f'pred_gw{next_gw}', 0):.1f} pts")
            logger.info(f"    - Strategy: {reasoning.get('strategy', 'N/A')}")
            
            logger.info(f"  Vice: {vice_row['web_name']}")
            logger.info(f"    - Predicted: {vice_row.get(f'pred_gw{next_gw}', 0):.1f} pts")
            
            if reasoning.get("alternatives"):
                logger.info("")
                logger.info("  Alternatives:")
                for alt in reasoning["alternatives"][:3]:
                    logger.info(f"    - {alt['name']}: {alt['predicted_pts']:.1f} pts ({alt['ownership']:.1f}% owned)")
            
            logger.info("")
            logger.info("✅ Captain selection complete")
            
            self.results["captain"] = True
            return True
            
        except Exception as e:
            self.issues.append(f"❌ [Captain] {str(e)}")
            import traceback
            traceback.print_exc()
            self.results["captain"] = False
            return False
    
    def test_data_quality(self):
        """Test 6: Data Quality Checks."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 6: Data Quality Checks")
        logger.info("=" * 80)
        
        try:
            players_df = self.pipeline.players_df()
            
            # Check for null values in critical columns
            critical_cols = [
                "web_name", "team", "now_cost", "total_points",
                "selected_by_percent", "total_risk"
            ]
            
            null_counts = {}
            for col in critical_cols:
                if col in players_df.columns:
                    null_count = players_df[col].isnull().sum()
                    if null_count > 0:
                        null_counts[col] = null_count
            
            if null_counts:
                logger.info(f"⚠️  Null values found:")
                for col, count in null_counts.items():
                    logger.info(f"  {col}: {count} nulls")
            else:
                logger.info("✅ Null value check complete")
            
            # Price range check
            min_price = players_df["now_cost"].min()
            max_price = players_df["now_cost"].max()
            logger.info(f"✅ Price range: £{min_price:.1f}m - £{max_price:.1f}m")
            
            # Ownership check
            avg_ownership = players_df["selected_by_percent"].mean()
            logger.info(f"✅ Average ownership: {avg_ownership:.1f}%")
            
            self.results["data_quality"] = True
            return True
            
        except Exception as e:
            self.issues.append(f"❌ [Data Quality] {str(e)}")
            self.results["data_quality"] = False
            return False
    
    def print_summary(self):
        """Print diagnostic summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📋 DIAGNOSTIC SUMMARY")
        logger.info("=" * 80)
        
        if self.issues:
            logger.error(f"\n❌ Found {len(self.issues)} issues:")
            for issue in self.issues:
                logger.error(f"  {issue}")
        else:
            logger.info("\n✅ No critical issues found!")
        
        logger.info("")
        logger.info("📊 Results:")
        for test_name, result in self.results.items():
            if result is True:
                status = "✅"
            elif result is False:
                status = "❌"
            else:
                status = "⚠️ "
            
            logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    def run_all(self):
        """Run all diagnostic tests."""
        logger.info("\n" + "🔍" * 40)
        logger.info("FPL ASSISTANT - FULL DIAGNOSTIC SUITE")
        logger.info("🔍" * 40 + "\n")
        
        # Load config
        if not self.load_config():
            logger.error("❌ Failed to load config")
            return
        
        # Run tests
        self.test_bootstrap()
        self.test_risk_indicators()
        self.test_ownership_and_price()
        self.test_chip_advice()
        self.test_captain_selection()
        self.test_data_quality()
        
        # Print summary
        self.print_summary()
        
        # Return success status
        return len(self.issues) == 0


def main():
    """Main diagnostic entry point."""
    parser = argparse.ArgumentParser(
        description="FPL Assistant Diagnostic Suite"
    )
    parser.add_argument(
        "--manager-id",
        type=int,
        help="Manager ID for chip/captain tests"
    )
    parser.add_argument(
        "--test",
        choices=["all", "risk", "ownership", "chip", "captain", "data"],
        default="all",
        help="Run specific test"
    )
    
    args = parser.parse_args()
    
    diagnostic = FPLDiagnostic(manager_id=args.manager_id)
    
    # Load config first
    if not diagnostic.load_config():
        logger.error("❌ Cannot proceed without config")
        sys.exit(1)
    
    # Initialize FPL client and pipeline
    diagnostic.test_bootstrap()
    
    if not diagnostic.results.get("bootstrap"):
        logger.error("❌ Cannot proceed without bootstrap data")
        sys.exit(1)
    
    # Run specific test or all tests
    if args.test == "all":
        success = diagnostic.run_all()
        sys.exit(0 if success else 1)
    elif args.test == "risk":
        diagnostic.test_risk_indicators()
    elif args.test == "ownership":
        diagnostic.test_ownership_and_price()
    elif args.test == "chip":
        diagnostic.test_chip_advice()
    elif args.test == "captain":
        diagnostic.test_captain_selection()
    elif args.test == "data":
        diagnostic.test_data_quality()
    
    diagnostic.print_summary()
    sys.exit(0 if not diagnostic.issues else 1)


if __name__ == "__main__":
    main()