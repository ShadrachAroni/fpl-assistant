"""
Data Pipeline - ENHANCED WITH PRICE INTELLIGENCE + OWNERSHIP TRACKING

FIXED VERSION: Handles missing 'appearances' column from FPL API

NEW FEATURES:
✅ Price change prediction (transfers_in/out tracking)
✅ Template vs differential identification
✅ Effective ownership (EO) calculations
✅ Formation validation
✅ Bench strength assessment
✅ Emergency cover tracking
✅ Ownership-weighted captaincy recommendations
✅ FIXED: Appearances column handling

PRODUCTION READY v6.1
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Import historical data integrator
try:
    from app.data.historical_integrator import HistoricalDataIntegrator
    HISTORICAL_AVAILABLE = True
except ImportError:
    HISTORICAL_AVAILABLE = False
    logging.warning("Historical data integrator not available - using API only")

logger = logging.getLogger("data_pipeline")
logger.setLevel(logging.INFO)


class DataPipeline:
    """
    Enhanced pipeline with price intelligence and ownership tracking.
    """
    
    # Valid FPL formations (DEF-MID-FWD, GK always 1)
    VALID_FORMATIONS = [
        (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), 
        (4, 5, 1), (5, 3, 2), (5, 4, 1)
    ]
    
    # Ownership thresholds
    TEMPLATE_THRESHOLD = 35.0  # >35% ownership = template
    DIFFERENTIAL_THRESHOLD = 5.0  # <5% ownership = differential
    PREMIUM_THRESHOLD = 10.0  # £10m+ = premium
    
    def __init__(self, config: Dict[str, Any], fpl_client=None, understat_client=None, use_historical_data: bool = True,  historical_cache_dir: str = "data/cache/historical"):
        """
        Initialize pipeline with configuration and clients.
        
        Args:
            config: Configuration dictionary
            fpl_client: FPL API client
            understat_client: Understat client (optional)
            use_historical_data: Whether to use external historical data
            historical_cache_dir: Directory for historical data cache
        """
        self.config = config
        self.fpl = fpl_client
        self.under = understat_client
        self.bootstrap = None
        self.global_analytics_df = None
        self.teams_map = {}
        
        # Enhanced configuration
        sim_cfg = config.get("simulation", {})
        self.horizon = int(sim_cfg.get("planning_horizon", 5))
        self.chip_horizon = int(sim_cfg.get("chip_analysis_horizon", 10))
        
        # Price change thresholds
        self.price_rise_threshold = 100.0  # Net transfers in per 100k managers
        self.price_fall_threshold = -100.0
        
        # Rotation prone managers
        self.rotation_prone_managers = {
            "Man City": ["Guardiola"],
            "Chelsea": ["Pochettino"],
            "Liverpool": ["Klopp"],
        }
        
        # Historical data integration
        self.use_historical_data = use_historical_data and HISTORICAL_AVAILABLE
        self.historical_integrator = None
        
        if self.use_historical_data:
            try:
                self.historical_integrator = HistoricalDataIntegrator(
                    cache_dir=historical_cache_dir,
                    current_season="2024-25"
                )
                logger.info("✅ Historical data integration enabled")
            except Exception as e:
                logger.warning(f"⚠️ Historical data integration failed: {e}")
                self.use_historical_data = False
        else:
            logger.info("ℹ️ Using FPL API only (no external historical data)")
    
    def fetch_bootstrap(self) -> Dict[str, Any]:
        """Fetches and caches static FPL bootstrap data."""
        if self.fpl is None:
            logger.error("FPLClient not initialized")
            return {}
        
        try:
            self.bootstrap = self.fpl.bootstrap()
            if self.bootstrap and "elements" in self.bootstrap:
                logger.info(f"✅ Bootstrap: {len(self.bootstrap.get('elements', []))} players")
                
                # Build teams map
                teams = self.bootstrap.get("teams", [])
                for team in teams:
                    team_id = team.get("id")
                    if team_id:
                        self.teams_map[team_id] = {
                            "name": team.get("name", "Unknown"),
                            "short_name": team.get("short_name", "UNK"),
                            "strength": team.get("strength", 3),
                            "strength_overall_home": team.get("strength_overall_home", 1000),
                            "strength_overall_away": team.get("strength_overall_away", 1000),
                            "strength_defence_home": team.get("strength_defence_home", 1000),
                            "strength_defence_away": team.get("strength_defence_away", 1000),
                            "strength_attack_home": team.get("strength_attack_home", 1000),
                            "strength_attack_away": team.get("strength_attack_away", 1000),
                        }
                
                logger.info(f"✅ Teams map: {len(self.teams_map)} teams")
            else:
                logger.error("Bootstrap returned empty data")
            return self.bootstrap
        except Exception as e:
            logger.error(f"Bootstrap fetch failed: {e}")
            return {}
    
    def get_team_name(self, team_id: int, short: bool = False) -> str:
        """Get team name by ID."""
        team_info = self.teams_map.get(team_id, {})
        if short:
            return team_info.get("short_name", "UNK")
        return team_info.get("name", "Unknown")
    
    def players_df(self) -> pd.DataFrame:
        """Build enriched player DataFrame with FULL intelligence."""
        if self.bootstrap is None:
            logger.debug("Fetching bootstrap...")
            self.fetch_bootstrap()
        
        if not self.bootstrap or "elements" not in self.bootstrap:
            logger.error("Bootstrap data missing")
            return pd.DataFrame()
        
        elements = self.bootstrap.get("elements", [])
        if not elements:
            logger.error("No player elements")
            return pd.DataFrame()
        
        teams = {t["id"]: t["name"] for t in self.bootstrap.get("teams", [])}
        df = pd.DataFrame(elements)
        
        if df.empty:
            logger.error("Empty DataFrame")
            return df
        
        # Basic enrichment
        df["team_name"] = df["team"].map(teams)
        df["form"] = pd.to_numeric(df.get("form", 7.5), errors="coerce").fillna(7.5)
        
        # Price conversion
        df["now_cost"] = pd.to_numeric(df.get("now_cost", 50), errors="coerce").fillna(50) / 10.0
        df["now_cost"] = df["now_cost"].round(1)
        
        # Position mapping
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        df["position"] = df.get("element_type", 0).map(pos_map).fillna("UNK")
        
        # === NEW: OWNERSHIP & TEMPLATE TRACKING ===
        df = self._enrich_with_ownership_intelligence(df)
        
        # === NEW: PRICE CHANGE PREDICTION ===
        df = self._enrich_with_price_intelligence(df)
        
        # Advanced metrics
        df = self._enrich_with_advanced_metrics(df)
        
        # Fixture difficulty for horizon WITH OPPONENT INFO
        df = self._enrich_with_fixture_horizon(df)
        
        # COMPREHENSIVE RISK INDICATORS - FIXED
        df = self._enrich_with_comprehensive_risk_indicators(df)
        
        # === NEW: FORMATION COMPATIBILITY ===
        df = self._add_formation_metadata(df)
        
        logger.debug(f"Built enhanced players_df: {len(df)} players with {len(df.columns)} features")
        
        return df
    
    def _enrich_with_ownership_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ownership intelligence:
        - Template identification (>35% owned)
        - Differential identification (<5% owned)
        - Effective ownership calculations
        - Captain EO predictions
        """
        try:
            # Ownership percentage
            df["selected_by_percent"] = pd.to_numeric(
                df.get("selected_by_percent", 0), 
                errors="coerce"
            ).fillna(0)
            
            # Template vs Differential classification
            df["is_template"] = df["selected_by_percent"] > self.TEMPLATE_THRESHOLD
            df["is_differential"] = df["selected_by_percent"] < self.DIFFERENTIAL_THRESHOLD
            df["is_premium"] = df["now_cost"] >= self.PREMIUM_THRESHOLD
            
            # Ownership category
            df["ownership_category"] = df["selected_by_percent"].apply(
                lambda x: "🔴 Essential (>50%)" if x > 50
                else "🟠 Template (35-50%)" if x > 35
                else "🟡 Popular (15-35%)" if x > 15
                else "🟢 Standard (5-15%)" if x > 5
                else "💎 Differential (<5%)"
            )
            
            # Captain effective ownership (EO) estimation
            # High ownership players get captained more often
            df["captain_eo_multiplier"] = df["selected_by_percent"].apply(
                lambda x: 2.5 if x > 50  # Ultra-template captain (Salah, Haaland)
                else 2.0 if x > 35
                else 1.5 if x > 15
                else 1.2 if x > 5
                else 1.0
            )
            
            # Template player must-have score
            df["template_priority"] = (
                (df["selected_by_percent"] / 100.0) * 
                df.get("form", 7.5) * 
                (df["now_cost"] / 10.0)
            )
            
            logger.info(f"📊 Ownership intelligence added:")
            logger.info(f"   Templates (>35%): {df['is_template'].sum()}")
            logger.info(f"   Differentials (<5%): {df['is_differential'].sum()}")
            logger.info(f"   Premiums (£10m+): {df['is_premium'].sum()}")
            
        except Exception as e:
            logger.warning(f"Ownership intelligence failed: {e}")
            df["is_template"] = False
            df["is_differential"] = False
            df["is_premium"] = False
            df["ownership_category"] = "🟡 Standard"
            df["captain_eo_multiplier"] = 1.0
            df["template_priority"] = 0.0
        
        return df
    
    def _enrich_with_price_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price change prediction intelligence:
        - Track transfers in/out
        - Predict price rises/falls
        - Calculate opportunity cost of waiting
        - Identify value holds
        """
        try:
            # Transfers in/out
            df["transfers_in_event"] = pd.to_numeric(
                df.get("transfers_in_event", 0), 
                errors="coerce"
            ).fillna(0)
            
            df["transfers_out_event"] = pd.to_numeric(
                df.get("transfers_out_event", 0), 
                errors="coerce"
            ).fillna(0)
            
            # Net transfers
            df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]
            
            # Cost change tracking
            df["cost_change_event"] = pd.to_numeric(
                df.get("cost_change_event", 0), 
                errors="coerce"
            ).fillna(0)
            
            df["cost_change_start"] = pd.to_numeric(
                df.get("cost_change_start", 0), 
                errors="coerce"
            ).fillna(0)
            
            # Price change probability (simplified heuristic)
            # Actual FPL algorithm is secret, but net transfers is main factor
            df["price_rise_probability"] = df["net_transfers"].apply(
                lambda x: min(1.0, max(0.0, (x / self.price_rise_threshold))) if x > 0 else 0.0
            )
            
            df["price_fall_probability"] = df["net_transfers"].apply(
                lambda x: min(1.0, max(0.0, (abs(x) / abs(self.price_fall_threshold)))) if x < 0 else 0.0
            )
            
            # Price change category
            df["price_change_status"] = df.apply(
                lambda row: "📈 Rising (>80%)" if row["price_rise_probability"] > 0.8
                else "⬆️ Likely Rise (>50%)" if row["price_rise_probability"] > 0.5
                else "📉 Falling (>80%)" if row["price_fall_probability"] > 0.8
                else "⬇️ Likely Fall (>50%)" if row["price_fall_probability"] > 0.5
                else "➡️ Stable",
                axis=1
            )
            
            # Value hold score (rising players worth holding)
            df["value_hold_score"] = (
                df["price_rise_probability"] * 
                (df["now_cost"] / 10.0) * 
                df.get("form", 7.5)
            )
            
            # Opportunity cost (waiting to transfer out a falling player)
            df["opportunity_cost"] = df["price_fall_probability"] * 0.1  # Max £0.1m loss
            
            rising_count = (df["price_rise_probability"] > 0.5).sum()
            falling_count = (df["price_fall_probability"] > 0.5).sum()
            
            logger.info(f"💰 Price intelligence added:")
            logger.info(f"   Likely rising: {rising_count}")
            logger.info(f"   Likely falling: {falling_count}")
            
        except Exception as e:
            logger.warning(f"Price intelligence failed: {e}")
            df["net_transfers"] = 0
            df["price_rise_probability"] = 0.0
            df["price_fall_probability"] = 0.0
            df["price_change_status"] = "➡️ Stable"
            df["value_hold_score"] = 0.0
            df["opportunity_cost"] = 0.0
        
        return df
    
    def _add_formation_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add formation compatibility metadata."""
        try:
            # Mark premium positions (high-value starters)
            df["is_premium_position"] = (
                (df["position"].isin(["MID", "FWD"])) & 
                (df["now_cost"] >= 8.0)
            )
            
            # Mark budget enablers (cheap starters)
            df["is_budget_enabler"] = (
                (df["now_cost"] <= 4.5) & 
                (df["minutes"] > 500)
            )
            
            # Mark bench fodder (non-playing cheap players)
            df["is_bench_fodder"] = (
                (df["now_cost"] <= 4.5) & 
                (df["minutes"] < 200)
            )
            
            logger.debug("Formation metadata added")
            
        except Exception as e:
            logger.debug(f"Formation metadata failed: {e}")
        
        return df
    
    def _enrich_with_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced metrics (ICT, xG, xA, etc)."""
        try:
            # ICT Index
            df["ict_index"] = pd.to_numeric(df.get("ict_index", 0), errors="coerce").fillna(0)
            df["influence"] = pd.to_numeric(df.get("influence", 0), errors="coerce").fillna(0)
            df["creativity"] = pd.to_numeric(df.get("creativity", 0), errors="coerce").fillna(0)
            df["threat"] = pd.to_numeric(df.get("threat", 0), errors="coerce").fillna(0)
            
            # Expected metrics
            df["expected_goals"] = pd.to_numeric(df.get("expected_goals", 0), errors="coerce").fillna(0)
            df["expected_assists"] = pd.to_numeric(df.get("expected_assists", 0), errors="coerce").fillna(0)
            df["expected_goal_involvements"] = pd.to_numeric(df.get("expected_goal_involvements", 0), errors="coerce").fillna(0)
            df["expected_goals_conceded"] = pd.to_numeric(df.get("expected_goals_conceded", 0), errors="coerce").fillna(0)
            
            # Minutes and reliability - FIXED
            df["minutes"] = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0)
            df["starts"] = pd.to_numeric(df.get("starts", 0), errors="coerce").fillna(0)
            
            # Calculate appearances if not present (starts + substitute appearances)
            if "appearances" not in df.columns:
                # Use starts as a proxy, or infer from minutes played
                df["appearances"] = df["starts"].copy()
                # If player has minutes but no recorded starts, they appeared as sub
                df.loc[(df["minutes"] > 0) & (df["starts"] == 0), "appearances"] = 1
                # Ensure at least 1 to avoid division by zero
                df["appearances"] = df["appearances"].replace(0, 1)
            else:
                df["appearances"] = pd.to_numeric(df.get("appearances", 1), errors="coerce").fillna(1)
            
            df["reliability"] = (df["starts"] / df["appearances"].replace(0, 1)).fillna(0)
            
            # Clean sheets
            df["clean_sheets"] = pd.to_numeric(df.get("clean_sheets", 0), errors="coerce").fillna(0)
            df["goals_conceded"] = pd.to_numeric(df.get("goals_conceded", 0), errors="coerce").fillna(0)
            
            # Bonus
            df["bonus"] = pd.to_numeric(df.get("bonus", 0), errors="coerce").fillna(0)
            df["bps"] = pd.to_numeric(df.get("bps", 0), errors="coerce").fillna(0)
            
            # Points per million
            df["total_points"] = pd.to_numeric(df.get("total_points", 0), errors="coerce").fillna(0)
            df["points_per_million"] = (df["total_points"] / df["now_cost"].replace(0, 1)).fillna(0)
            
            # Disciplinary
            df["yellow_cards"] = pd.to_numeric(df.get("yellow_cards", 0), errors="coerce").fillna(0)
            df["red_cards"] = pd.to_numeric(df.get("red_cards", 0), errors="coerce").fillna(0)
            
            # Attacking
            df["goals_scored"] = pd.to_numeric(df.get("goals_scored", 0), errors="coerce").fillna(0)
            df["assists"] = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0)
            
            # Penalties
            df["penalties_order"] = pd.to_numeric(df.get("penalties_order"), errors="coerce")
            df["penalties_missed"] = pd.to_numeric(df.get("penalties_missed", 0), errors="coerce").fillna(0)
            df["penalties_saved"] = pd.to_numeric(df.get("penalties_saved", 0), errors="coerce").fillna(0)
            
            # Saves
            df["saves"] = pd.to_numeric(df.get("saves", 0), errors="coerce").fillna(0)
            
            logger.debug("Advanced metrics added")
            
        except Exception as e:
            logger.debug(f"Advanced metrics enrichment failed: {e}")
        
        return df
    
    def _enrich_with_fixture_horizon(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fixture difficulty AND opponent information for next N gameweeks."""
        try:
            if self.fpl is None:
                df["fixture_difficulty"] = 3
                df["fixture_run_difficulty"] = 3.0
                df["next_opponent"] = "Unknown"
                df["next_opponent_short"] = "UNK"
                df["is_home"] = True
                return df
            
            fixtures = pd.DataFrame(self.fpl.fixtures() or [])
            events = pd.DataFrame(self.bootstrap.get("events", []))
            
            if events.empty or fixtures.empty:
                df["fixture_difficulty"] = 3
                df["fixture_run_difficulty"] = 3.0
                df["next_opponent"] = "Unknown"
                df["next_opponent_short"] = "UNK"
                df["is_home"] = True
                return df
            
            # Get next gameweek
            next_gw_rows = events[events.get("is_next", False) == True]
            if next_gw_rows.empty:
                next_gw_rows = events[events.get("is_current", False) == True]
            if next_gw_rows.empty:
                next_gw_rows = events.nlargest(1, "id")
            
            if next_gw_rows.empty:
                df["fixture_difficulty"] = 3
                df["fixture_run_difficulty"] = 3.0
                df["next_opponent"] = "Unknown"
                df["next_opponent_short"] = "UNK"
                df["is_home"] = True
                return df
            
            next_gw = int(next_gw_rows.iloc[0]["id"])
            
            # Build fixture map
            team_fixture_run = {}
            team_next_opponent = {}
            team_next_is_home = {}
            team_fixture_count = {}
            
            for i in range(min(self.horizon, 7)):
                gw = next_gw + i
                gw_fixtures = fixtures[fixtures.get("event") == gw]
                
                for _, row in gw_fixtures.iterrows():
                    try:
                        home = int(row.get("team_h", 0))
                        away = int(row.get("team_a", 0))
                        h_diff = int(row.get("team_h_difficulty", 3))
                        a_diff = int(row.get("team_a_difficulty", 3))
                        
                        if home not in team_fixture_run:
                            team_fixture_run[home] = []
                            team_fixture_count[home] = {}
                        if away not in team_fixture_run:
                            team_fixture_run[away] = []
                            team_fixture_count[away] = {}
                        
                        team_fixture_run[home].append(h_diff)
                        team_fixture_run[away].append(a_diff)
                        
                        team_fixture_count[home][gw] = team_fixture_count[home].get(gw, 0) + 1
                        team_fixture_count[away][gw] = team_fixture_count[away].get(gw, 0) + 1
                        
                        if i == 0:
                            team_next_opponent[home] = away
                            team_next_opponent[away] = home
                            team_next_is_home[home] = True
                            team_next_is_home[away] = False
                            
                    except:
                        continue
            
            # Next GW fixture difficulty
            next_fix = fixtures[fixtures.get("event") == next_gw]
            team_next_diff = {}
            
            for _, row in next_fix.iterrows():
                try:
                    home = int(row.get("team_h", 0))
                    away = int(row.get("team_a", 0))
                    h_diff = int(row.get("team_h_difficulty", 3))
                    a_diff = int(row.get("team_a_difficulty", 3))
                    team_next_diff[home] = h_diff
                    team_next_diff[away] = a_diff
                except:
                    continue
            
            df["fixture_difficulty"] = df["team"].map(team_next_diff).fillna(3).astype(int)
            
            # Average fixture run
            team_avg_diff = {
                team: np.mean(difficulties) if difficulties else 3.0
                for team, difficulties in team_fixture_run.items()
            }
            
            df["fixture_run_difficulty"] = df["team"].map(team_avg_diff).fillna(3.0)
            
            # Opponent info
            df["next_opponent_id"] = df["team"].map(team_next_opponent).fillna(0).astype(int)
            df["next_opponent"] = df["next_opponent_id"].apply(
                lambda x: self.get_team_name(x, short=False) if x > 0 else "No fixture"
            )
            df["next_opponent_short"] = df["next_opponent_id"].apply(
                lambda x: self.get_team_name(x, short=True) if x > 0 else "—"
            )
            df["is_home"] = df["team"].map(team_next_is_home).fillna(True)
            
            # DGW detection
            df["has_dgw_next"] = df["team"].apply(
                lambda t: team_fixture_count.get(t, {}).get(next_gw, 0) >= 2
            )
            
            logger.debug(f"Fixture horizon added for {len(team_fixture_run)} teams")
            
        except Exception as e:
            logger.debug(f"Fixture horizon enrichment failed: {e}")
            df["fixture_difficulty"] = 3
            df["fixture_run_difficulty"] = 3.0
            df["next_opponent"] = "Unknown"
            df["next_opponent_short"] = "UNK"
            df["is_home"] = True
            df["has_dgw_next"] = False
        
        return df
    
    def _enrich_with_comprehensive_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add COMPREHENSIVE risk indicators - FIXED VERSION."""
        try:
            # Injury/availability
            df["chance_of_playing_next_round"] = pd.to_numeric(
                df.get("chance_of_playing_next_round", 100), 
                errors="coerce"
            ).fillna(100)
            
            df["injury_risk"] = (100 - df["chance_of_playing_next_round"]) / 100.0
            df["has_news"] = df.get("news", "").fillna("").str.len() > 0
            df["news_severity"] = df["has_news"].astype(int) * 0.3
            
            # Minutes volatility - FIXED
            df["minutes_per_game"] = (
                df["minutes"] / df["appearances"].replace(0, 1)
            ).fillna(0).clip(0, 90)  # Cap at 90 minutes max
            
            df["rotation_risk_base"] = df["minutes_per_game"].apply(
                lambda x: 0.0 if x >= 70 else (70 - x) / 70.0 if x > 0 else 1.0
            )
            
            df["minutes_volatility"] = (1 - df["reliability"]) * 0.5
            
            df["manager_rotation_risk"] = df["team_name"].apply(
                lambda team: 0.3 if team in self.rotation_prone_managers else 0.0
            )
            
            df["rotation_risk"] = (
                df["rotation_risk_base"] * 0.5 +
                df["minutes_volatility"] * 0.3 +
                df["manager_rotation_risk"] * 0.2
            ).clip(0, 1)
            
            # Disciplinary
            df["suspension_risk"] = df["yellow_cards"].apply(
                lambda yc: 0.0 if yc < 3 
                else 0.3 if yc == 3 
                else 0.6 if yc == 4 
                else 0.9 if yc >= 5 
                else 0.0
            )
            
            df["red_card_risk"] = (df["red_cards"] > 0).astype(float) * 0.5
            
            df["disciplinary_risk"] = (
                df["suspension_risk"] * 0.7 +
                df["red_card_risk"] * 0.3
            ).clip(0, 1)
            
            # Fatigue
            df["fatigue_risk"] = df["minutes"].apply(
                lambda m: 0.0 if m < 1000
                else 0.2 if m < 1500
                else 0.4 if m < 2000
                else 0.6 if m < 2500
                else 0.8
            )
            
            df["fatigue_risk"] = df.apply(
                lambda row: row["fatigue_risk"] * 1.3 if row.get("has_dgw_next", False) else row["fatigue_risk"],
                axis=1
            ).clip(0, 1)
            
            # Form drop
            form_numeric = pd.to_numeric(df.get("form", 7.5), errors="coerce").fillna(7.5)
            
            df["form_drop_risk"] = form_numeric.apply(
                lambda x: 1.0 if x < 3.0 
                else 0.7 if x < 4.0
                else 0.5 if x < 5.0 
                else 0.2 if x < 6.0
                else 0.0
            )
            
            # Defensive fragility
            df["opponent_attack_strength"] = df.apply(
                lambda row: self._get_opponent_attack_strength(
                    row.get("next_opponent_id", 0),
                    row.get("is_home", True)
                ),
                axis=1
            )
            
            df["defensive_fragility_risk"] = df.apply(
                lambda row: self._calculate_defensive_fragility(row),
                axis=1
            )
            
            # Penalty status
            df["is_penalty_taker"] = (df["penalties_order"] == 1).fillna(False)
            df["penalty_risk"] = df.apply(
                lambda row: 0.0 if row.get("is_penalty_taker", False)
                else 0.2 if pd.notnull(row.get("penalties_order")) and row.get("penalties_order") == 2
                else 0.0,
                axis=1
            )
            
            # TOTAL RISK
            df["total_risk"] = (
                df["injury_risk"] * 0.25 +
                df["news_severity"] * 0.10 +
                df["rotation_risk"] * 0.20 +
                df["disciplinary_risk"] * 0.15 +
                df["fatigue_risk"] * 0.10 +
                df["form_drop_risk"] * 0.15 +
                df["defensive_fragility_risk"] * 0.03 +
                df["penalty_risk"] * 0.02
            ).clip(0, 1)
            
            df["risk_category"] = df["total_risk"].apply(
                lambda r: "🟢 Low" if r < 0.3
                else "🟡 Medium" if r < 0.6
                else "🔴 High"
            )
            
            logger.info(f"✅ Risk indicators added")
            
        except Exception as e:
            logger.warning(f"Risk indicators failed: {e}")
            # Ensure all risk columns exist with default values
            risk_columns = {
                "injury_risk": 0.0,
                "rotation_risk": 0.0,
                "disciplinary_risk": 0.0,
                "fatigue_risk": 0.0,
                "form_drop_risk": 0.0,
                "total_risk": 0.0,
                "risk_category": "🟡 Medium",
                "minutes_per_game": 0,
                "chance_of_playing_next_round": 100
            }
            
            for col, default_val in risk_columns.items():
                if col not in df.columns:
                    df[col] = default_val
        
        return df
    
    def _get_opponent_attack_strength(self, opponent_id: int, is_home: bool) -> float:
        """Get opponent's attack strength."""
        if opponent_id == 0 or opponent_id not in self.teams_map:
            return 1000.0
        
        team_info = self.teams_map[opponent_id]
        
        if is_home:
            return float(team_info.get("strength_attack_away", 1000))
        else:
            return float(team_info.get("strength_attack_home", 1000))
    
    def _calculate_defensive_fragility(self, row: pd.Series) -> float:
        """Calculate defensive fragility risk for DEF/GK."""
        position = row.get("position", "MID")
        
        if position not in ["DEF", "GK"]:
            return 0.0
        
        xgc = float(row.get("expected_goals_conceded", 0))
        goals_conceded = int(row.get("goals_conceded", 0))
        
        if xgc > 1.5 or goals_conceded > 2:
            return 0.6
        elif xgc > 1.0 or goals_conceded > 1:
            return 0.4
        elif xgc > 0.5:
            return 0.2
        
        return 0.0
    
    def validate_formation(self, squad: pd.DataFrame) -> Tuple[bool, str, List[Tuple[int, int, int]]]:
        """
        Validate if squad can form valid FPL formations.
        
        Returns:
            (is_valid, error_message, valid_formations)
        """
        if len(squad) != 15:
            return False, f"Squad has {len(squad)} players (need 15)", []
        
        # Count by position
        gk_count = len(squad[squad["position"] == "GK"])
        def_count = len(squad[squad["position"] == "DEF"])
        mid_count = len(squad[squad["position"] == "MID"])
        fwd_count = len(squad[squad["position"] == "FWD"])
        
        # Check position requirements
        if gk_count != 2:
            return False, f"Need 2 GK (have {gk_count})", []
        if def_count < 3 or def_count > 5:
            return False, f"Need 3-5 DEF (have {def_count})", []
        if mid_count < 3 or mid_count > 5:
            return False, f"Need 3-5 MID (have {mid_count})", []
        if fwd_count < 1 or fwd_count > 3:
            return False, f"Need 1-3 FWD (have {fwd_count})", []
        
        # Check valid formations (11 players: 1 GK + 10 outfield)
        valid_formations = []
        for formation in self.VALID_FORMATIONS:
            req_def, req_mid, req_fwd = formation
            if req_def <= def_count and req_mid <= mid_count and req_fwd <= fwd_count:
                valid_formations.append(formation)
        
        if not valid_formations:
            return False, f"No valid formation possible with {def_count}-{mid_count}-{fwd_count}", []
        
        return True, "Valid", valid_formations
    
    def assess_bench_strength(self, squad: pd.DataFrame, next_gw: int) -> Dict[str, Any]:
        """
        Assess bench quality and emergency cover.
        
        Returns dict with bench analysis.
        """
        if len(squad) != 15:
            return {"error": "Squad must have 15 players"}
        
        # Sort by predicted points (assuming pred_gw{next_gw} exists)
        pred_col = f"pred_gw{next_gw}"
        if pred_col not in squad.columns:
            # Use form as fallback
            squad = squad.copy()
            squad["_temp_pred"] = squad.get("form", 0)
            pred_col = "_temp_pred"
        
        sorted_squad = squad.sort_values(pred_col, ascending=False)
        
        # Best 11
        starting_xi = sorted_squad.head(11)
        bench = sorted_squad.tail(4)
        
        # Bench analysis
        bench_total_pred = bench[pred_col].sum()
        bench_avg_pred = bench[pred_col].mean()
        
        # Check for playing bench (>1pt expected)
        playing_bench = bench[bench[pred_col] > 1.0]
        
        # Emergency cover (check positions)
        has_gk_cover = len(bench[bench["position"] == "GK"]) > 0
        has_def_cover = len(bench[bench["position"] == "DEF"]) > 0
        
        # Bench fodder detection (non-playing cheap players)
        bench_fodder = bench[
            (bench["now_cost"] <= 4.5) & 
            (bench.get("minutes", 0) < 200)
        ]
        
        return {
            "bench_strength": bench_avg_pred,
            "bench_total_points": bench_total_pred,
            "playing_bench_count": len(playing_bench),
            "has_gk_cover": has_gk_cover,
            "has_def_cover": has_def_cover,
            "bench_fodder_count": len(bench_fodder),
            "bench_players": bench[["web_name", "position", "now_cost", pred_col]].to_dict("records")
        }
    
    def calculate_effective_ownership(
        self, 
        squad: pd.DataFrame, 
        captain_id: int,
        vice_id: int
    ) -> Dict[str, float]:
        """
        Calculate effective ownership (EO) for squad.
        
        EO accounts for:
        - Player ownership %
        - Captain multiplier (2x or 3x with chip)
        - Template player differential
        """
        squad = squad.copy()
        
        # Base EO is ownership %
        squad["eo"] = squad["selected_by_percent"]
        
        # Captain gets 2x EO
        squad.loc[squad["id"] == captain_id, "eo"] *= 2
        
        # Calculate squad EO
        total_eo = squad["eo"].sum()
        avg_eo = squad["eo"].mean()
        
        # Template count (high ownership)
        template_count = (squad["selected_by_percent"] > 35).sum()
        
        # Differential count (low ownership)
        differential_count = (squad["selected_by_percent"] < 5).sum()
        
        return {
            "total_eo": total_eo,
            "average_eo": avg_eo,
            "template_count": template_count,
            "differential_count": differential_count,
            "captain_eo": squad[squad["id"] == captain_id]["eo"].iloc[0] if not squad[squad["id"] == captain_id].empty else 0
        }
        
    # ==================== ENHANCED TRAINING DATA METHODS ====================
    
    def build_train_features(
        self, 
        season: str = "2024", 
        rolling_window: int = 4,
        use_multi_season: bool = True,
        seasons: List[str] = None
    ) -> pd.DataFrame:
        """
        Build training features with HISTORICAL DATA INTEGRATION.
        
        This method now uses three strategies in priority order:
        1. External historical data (Vaastav's dataset) - BEST
        2. FPL API enriched with historical ownership
        3. FPL API with synthetic ownership (fallback)
        
        Args:
            season: Current season (e.g., "2024")
            rolling_window: Rolling window for features
            use_multi_season: Whether to use multiple seasons
            seasons: Specific seasons to include
        
        Returns:
            DataFrame with comprehensive training features
        """
        logger.info("=" * 80)
        logger.info("🔧 BUILDING TRAINING FEATURES WITH HISTORICAL DATA")
        logger.info("=" * 80)
        
        # STRATEGY 1: Use external historical data (PREFERRED)
        if self.use_historical_data and self.historical_integrator:
            logger.info("📊 Strategy 1: Using external historical data (Vaastav)")
            
            try:
                if use_multi_season:
                    # Load multiple seasons for better training
                    if seasons is None:
                        seasons = ["2022-23", "2023-24", "2024-25"]
                    
                    logger.info(f"   Loading seasons: {seasons}")
                    df = self.historical_integrator.get_multi_season_data(
                        seasons=seasons,
                        min_gameweeks=5,
                        include_current_season=True
                    )
                else:
                    # Single season
                    df = self.historical_integrator.get_merged_gameweek_data(
                        season=f"{season}-{int(season[-2:])+1}"
                    )
                
                if not df.empty:
                    # Prepare for training
                    df = self.historical_integrator.prepare_training_features(df)
                    
                    # Validate data quality
                    metrics = self.historical_integrator.validate_data_quality(df)
                    
                    logger.info(f"✅ Historical data loaded: {len(df)} records")
                    logger.info(f"   Unique players: {metrics['unique_players']:,}")
                    logger.info(f"   Gameweeks: {metrics['gameweeks_covered']}")
                    logger.info(f"   Has ownership: {metrics['has_ownership']}")
                    
                    if metrics["warnings"]:
                        logger.warning(f"   ⚠️ {len(metrics['warnings'])} warnings")
                    
                    # Enrich with additional features
                    df = self._enrich_training_features(df)
                    
                    return df
                else:
                    logger.warning("   ⚠️ Historical data empty - falling back to Strategy 2")
            
            except Exception as e:
                logger.warning(f"   ⚠️ Historical data failed: {e}")
                logger.info("   Falling back to Strategy 2")
        
        # STRATEGY 2: FPL API enriched with historical ownership
        logger.info("📊 Strategy 2: Using FPL API + historical enrichment")
        
        try:
            df = self._build_from_fpl_api(rolling_window)
            
            if not df.empty and self.use_historical_data and self.historical_integrator:
                # Enrich with historical ownership
                logger.info("   Enriching with historical ownership...")
                df = self.historical_integrator.enrich_current_season_data(df)
            
            if not df.empty:
                logger.info(f"✅ FPL API data loaded: {len(df)} records")
                return df
            else:
                logger.warning("   ⚠️ FPL API data empty - falling back to Strategy 3")
        
        except Exception as e:
            logger.warning(f"   ⚠️ FPL API failed: {e}")
            logger.info("   Falling back to Strategy 3")
        
        # STRATEGY 3: Synthetic fallback
        logger.info("📊 Strategy 3: Using synthetic data (fallback)")
        
        if self.bootstrap is None:
            self.fetch_bootstrap()
        
        if not self.bootstrap or "elements" not in self.bootstrap:
            logger.error("❌ Cannot build features - no data sources available")
            return pd.DataFrame()
        
        elements = self.bootstrap.get("elements", [])
        df = self._build_synthetic_training_data(elements)
        
        if not df.empty:
            logger.warning("⚠️ Using synthetic data - limited accuracy expected")
            logger.info(f"   Generated {len(df)} synthetic records")
        
        return df
    
    def _build_from_fpl_api(self, rolling_window: int = 4) -> pd.DataFrame:
        """
        Build training data from FPL API (element_summary).
        
        This is Strategy 2 - uses current season data from API.
        """
        if self.bootstrap is None:
            self.fetch_bootstrap()
        
        if self.fpl is None or not self.bootstrap or "elements" not in self.bootstrap:
            return pd.DataFrame()
        
        elements = self.bootstrap.get("elements", [])
        
        # Build player lookup with current data
        player_current_data = {}
        for player in elements:
            pid = player.get("id")
            if pid:
                player_current_data[pid] = {
                    "selected_by_percent": float(player.get("selected_by_percent", 0)),
                    "now_cost": float(player.get("now_cost", 50)) / 10.0,
                    "team": player.get("team", 0),
                    "element_type": player.get("element_type", 0),
                    "team_name": self.get_team_name(player.get("team", 0))
                }
        
        logger.info(f"📊 Fetching data for {len(elements)} players from FPL API...")
        
        all_histories = []
        success_count = 0
        
        for idx, player in enumerate(elements):
            pid = player.get("id")
            if not pid:
                continue
            
            try:
                summary = self.fpl.element_summary(pid)
                history_data = summary.get("history", [])
                
                if not history_data:
                    continue
                
                history_df = pd.DataFrame(history_data)
                
                if "event" not in history_df.columns:
                    if "round" in history_df.columns:
                        history_df["event"] = history_df["round"]
                    else:
                        continue
                
                history_df["player_id"] = pid
                history_df["web_name"] = player.get("web_name", "Unknown")
                
                # Add current player data
                if pid in player_current_data:
                    for key, value in player_current_data[pid].items():
                        history_df[key] = value
                
                all_histories.append(history_df)
                success_count += 1
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"   Progress: {idx + 1}/{len(elements)} ({success_count} successful)")
            
            except Exception as e:
                logger.debug(f"Failed for player {pid}: {e}")
                continue
        
        if not all_histories:
            logger.error("❌ No player histories retrieved")
            return pd.DataFrame()
        
        logger.info(f"✅ Retrieved histories for {success_count} players")
        
        # Combine all histories
        df = pd.concat(all_histories, ignore_index=True, sort=False)
        df = df.sort_values(["player_id", "event"])
        
        # Add gameweek column
        df["gameweek"] = df["event"]
        
        # Enrich with training features
        df = self._enrich_training_features(df)
        
        # Create target variable
        df["target_next_points"] = df.groupby("player_id")["total_points"].shift(-1)
        df.dropna(subset=["target_next_points"], inplace=True)
        
        return df
    
    def _enrich_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add critical features expected by validation and training.
        
        This ensures all necessary columns exist regardless of source.
        """
        try:
            # Ensure numeric types
            numeric_cols = [
                "minutes", "goals_scored", "assists", "clean_sheets",
                "goals_conceded", "bonus", "bps", "influence", "creativity",
                "threat", "ict_index", "value", "selected_by_percent", "total_points"
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
            # Add form if not present
            if "form" not in df.columns:
                df["form"] = df.groupby("player_id")["total_points"].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
            else:
                df["form"] = pd.to_numeric(df.get("form", 7.5), errors="coerce").fillna(7.5)
            
            # Add fixture difficulty
            if "opponent_difficulty" not in df.columns:
                if "difficulty" in df.columns:
                    df["opponent_difficulty"] = df["difficulty"]
                else:
                    df["opponent_difficulty"] = 3
            
            # Add is_home
            if "is_home" not in df.columns:
                if "was_home" in df.columns:
                    df["is_home"] = df["was_home"]
                else:
                    df["is_home"] = True
            
            # Ensure selected_by_percent exists
            if "selected_by_percent" not in df.columns:
                logger.warning("⚠️ No ownership data - using default 0")
                df["selected_by_percent"] = 0
            
            logger.debug("✅ Training features enriched")
            
        except Exception as e:
            logger.warning(f"⚠️ Training feature enrichment failed: {e}")
        
        return df
    
    def _build_synthetic_training_data(self, elements: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build synthetic training data from bootstrap (fallback)."""
        logger.info("🔧 Building synthetic training data...")
        
        synthetic_records = []
        
        for player in elements:
            pid = player.get("id")
            if not pid:
                continue
            
            total_points = float(player.get("total_points", 0))
            
            # Use starts if available, otherwise infer from minutes
            starts = int(player.get("starts", 0))
            minutes = int(player.get("minutes", 0))
            
            if starts == 0 and minutes > 0:
                # Estimate appearances from minutes (assume 90 mins per full game)
                appearances = max(1, int(minutes / 45))  # Count if played 45+ mins
            else:
                appearances = max(1, starts)
            
            if appearances == 0 or total_points == 0:
                continue
            
            avg_points = total_points / appearances
            
            for gw_offset in range(5):
                variance = np.random.normal(0, avg_points * 0.3)
                gw_points = max(0, avg_points + variance)
                
                record = {
                    "player_id": pid,
                    "web_name": player.get("web_name", "Unknown"),
                    "event": gw_offset + 1,
                    "total_points": gw_points,
                    "minutes": float(minutes) / max(1, appearances),
                    "goals_scored": float(player.get("goals_scored", 0)) / max(1, appearances),
                    "assists": float(player.get("assists", 0)) / max(1, appearances),
                    "clean_sheets": float(player.get("clean_sheets", 0)) / max(1, appearances),
                    "bonus": float(player.get("bonus", 0)) / max(1, appearances),
                    "bps": float(player.get("bps", 0)) / max(1, appearances),
                    "influence": float(player.get("influence", 0)),
                    "creativity": float(player.get("creativity", 0)),
                    "threat": float(player.get("threat", 0)),
                    "ict_index": float(player.get("ict_index", 0)),
                    "expected_goals": float(player.get("expected_goals", 0)) / max(1, appearances),
                    "expected_assists": float(player.get("expected_assists", 0)) / max(1, appearances),
                    "now_cost": float(player.get("now_cost", 50)) / 10.0,
                    "selected_by_percent": float(player.get("selected_by_percent", 0)),
                }
                
                synthetic_records.append(record)
        
        if not synthetic_records:
            logger.error("❌ Could not generate synthetic data")
            return pd.DataFrame()
        
        df = pd.DataFrame(synthetic_records)
        df = df.sort_values(["player_id", "event"])
        df["target_next_points"] = df.groupby("player_id")["total_points"].shift(-1)
        df.dropna(subset=["target_next_points"], inplace=True)
        df["predicted_points_next_gw"] = df["total_points"]
        
        logger.info(f"✅ Synthetic data: {len(df)} records")
        logger.warning("⚠️ Using synthetic data - model performance may be limited")
        
        return df