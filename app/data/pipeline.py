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

import json
import logging
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:  # pragma: no cover
    from app.api_client.sportmonks_client import SportmonksClient
    from app.api_client.pl_injury_client import PremierLeagueInjuryClient
    from app.api_client.bbc_client import BBCLineupClient
    from app.api_client.news_client import GNewsClient, NewsAPIClient, RSSClient

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
    
    def __init__(
        self,
        config: Dict[str, Any],
        fpl_client=None,
        understat_client=None,
        sportmonks_client: Optional["SportmonksClient"] = None,
        pl_injury_client: Optional["PremierLeagueInjuryClient"] = None,
        bbc_client: Optional["BBCLineupClient"] = None,
        gnews_client: Optional["GNewsClient"] = None,
        newsapi_client: Optional["NewsAPIClient"] = None,
        rss_client: Optional["RSSClient"] = None,
        use_historical_data: bool = True,
        historical_cache_dir: str = "data/cache/historical"
    ):
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
        self.sportmonks = sportmonks_client
        self.pl_injuries = pl_injury_client
        self.bbc_client = bbc_client
        self.gnews_client = gnews_client
        self.newsapi_client = newsapi_client
        self.rss_client = rss_client
        self.bootstrap = None
        self.global_analytics_df = None
        self.teams_map = {}
        
        # Enhanced configuration
        sim_cfg = config.get("simulation", {})
        self.horizon = int(sim_cfg.get("planning_horizon", 5))
        self.chip_horizon = int(sim_cfg.get("chip_analysis_horizon", 10))
        
        # Training + weighting configs
        training_cfg = config.get("training", {})
        self.bayesian_confidence = float(training_cfg.get("bayesian_form_confidence", 0.7))

        self.multi_objective_weights = sim_cfg.get("multi_objective_weights", {
            "points": 0.35,
            "safety": 0.20,
            "value": 0.15,
            "differential": 0.15,
            "fixtures": 0.15
        })

        # Market intelligence + contextual intel
        self.market_cfg = config.get("market_intelligence", {})
        self.intel_cfg = config.get("contextual_intel", {})
        self.risk_cfg = config.get("risk_assessment", {})
        self.rotation_monitor_cfg = self.risk_cfg.get("rotation_monitor", {})
        self.rotation_prone_managers = self._load_rotation_prone_metadata()

        # Price change thresholds
        base_threshold = float(self.market_cfg.get("price_rise_base_threshold", 100000))
        self.price_rise_threshold = base_threshold  # transfers in baseline
        self.price_fall_threshold = -base_threshold
        
        self.team_momentum_map: Dict[int, float] = {}
        self.manager_change_map = self.intel_cfg.get("manager_changes", {})
        self.european_lookup = self.intel_cfg.get("european_competitions", {})
        self.congested_teams = self._build_congested_team_lookup(self.european_lookup)
        self.current_gw: Optional[int] = None

        # External injury intelligence (Sportmonks)
        injury_cfg = config.get("injury_intel", {})
        self.external_injuries_enabled = bool(injury_cfg.get("external_sources_enabled", False)) and (
            self.sportmonks is not None or pl_injury_client is not None
        )
        self.external_injury_refresh_minutes = int(injury_cfg.get("refresh_minutes", 60))
        self.external_fallback_chance = int(injury_cfg.get("fallback_chance_percent", 25))
        self.external_high_threshold = float(injury_cfg.get("high_severity_threshold", 0.85))
        self.external_medium_threshold = float(injury_cfg.get("medium_severity_threshold", 0.6))
        self.external_default_severity = float(injury_cfg.get("default_severity", 0.75))
        self.priority_competitions = injury_cfg.get("priority_competitions", [])

        sportmonks_cfg = config.get("sources", {}).get("sportmonks", {})
        self.external_injury_leagues = sportmonks_cfg.get("league_ids", [])
        self.external_include_player_details = sportmonks_cfg.get("include_player_details", True)
        self._external_injury_cache: List[Dict[str, Any]] = []
        self._external_injury_last_fetch: Optional[datetime] = None
        self._pl_injury_cache: List[Dict[str, Any]] = []
        self._pl_injury_last_fetch: Optional[datetime] = None

        lineup_cfg = config.get("lineup_intel", {})
        self.lineup_enabled = bool(lineup_cfg.get("enabled", False)) and self.bbc_client is not None
        self.lineup_refresh_minutes = int(lineup_cfg.get("refresh_minutes", 20))
        self.lineup_pre_kickoff_window = int(lineup_cfg.get("pre_kickoff_window_minutes", 180))
        self._bbc_lineup_cache: Dict[int, Dict[str, Any]] = {}
        self._bbc_event_map: Dict[int, str] = {}

        understat_cfg = config.get("understat", {})
        self.understat_season = understat_cfg.get("season", "2024")
        self.understat_refresh_hours = float(understat_cfg.get("refresh_hours", 6))
        self._understat_cache: Dict[str, Dict[str, Any]] = {}
        self._understat_last_fetch: Optional[datetime] = None

        news_cfg = config.get("news_intel", {})
        self.news_enabled = bool(news_cfg)
        self.news_poll_minutes = int(news_cfg.get("poll_minutes", 10))
        self.news_keywords = [kw.lower() for kw in news_cfg.get("keywords", [])]
        self._news_cache: List[Dict[str, Any]] = []
        self._news_last_fetch: Optional[datetime] = None
        
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

                events = self.bootstrap.get("events", [])
                if events:
                    current_event = next((e for e in events if e.get("is_current")), None)
                    if current_event:
                        self.current_gw = current_event.get("id")
                
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
                            "recent_form": team.get("form", ""),
                        }
                
                logger.info(f"✅ Teams map: {len(self.teams_map)} teams")
                self.team_momentum_map = self._build_team_momentum_map()
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

        # Age + contextual metadata
        df["date_of_birth"] = pd.to_datetime(df.get("date_of_birth"), errors="coerce")
        df["age"] = df["date_of_birth"].apply(self._calculate_age_years)

        # Core scoring helpers
        df["total_points"] = pd.to_numeric(df.get("total_points", 0), errors="coerce").fillna(0)
        df["points_per_game_raw"] = pd.to_numeric(df.get("points_per_game", 0), errors="coerce").fillna(0)
        df["points_per_million"] = (
            df["total_points"] / df["now_cost"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).round(3)

        df["selected_by_percent"] = pd.to_numeric(
            df.get("selected_by_percent", 0),
            errors="coerce"
        ).fillna(0)

        df["differential_score"] = (
            (1 - (df["selected_by_percent"] / 100.0)) * df["form"]
        ).round(3)
        df["team_momentum"] = df["team"].map(self.team_momentum_map).fillna(1.0)
        df["manager_change_factor"] = df["team_name"].apply(self._calculate_manager_change_impact)
        
        # === NEW: OWNERSHIP & TEMPLATE TRACKING ===
        df = self._enrich_with_ownership_intelligence(df)
        
        # === NEW: PRICE CHANGE PREDICTION ===
        df = self._enrich_with_price_intelligence(df)
        
        next_gw_id = self._get_next_gameweek_id()

        # Advanced metrics
        df = self._enrich_with_advanced_metrics(df)

        # Understat expected metrics
        df = self._enrich_with_understat_metrics(df)

        # External injury intelligence (cross-league/international)
        df = self._apply_external_injuries(df)

        # News signals
        df = self._apply_news_signals(df, next_gw_id)
        
        # Fixture difficulty for horizon WITH OPPONENT INFO
        df = self._enrich_with_fixture_horizon(df)

        # BBC lineups (expected/confirmed)
        df = self._enrich_with_bbc_lineups(df, next_gw_id)
        
        # COMPREHENSIVE RISK INDICATORS - FIXED
        df = self._enrich_with_comprehensive_risk_indicators(df)

        # Bayesian form + volatility estimates for simulations
        df = self._apply_bayesian_form(df)
        df["points_std_dev"] = self._estimate_points_std_dev(df)
        
        # === NEW: FORMATION COMPATIBILITY ===
        df = self._add_formation_metadata(df)

        # Model-ready derived signals
        df = self._apply_model_signal_features(df)
        
        logger.debug(f"Built enhanced players_df: {len(df)} players with {len(df.columns)} features")
        
        return df

    @staticmethod
    def _defragment_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a shallow copy to defragment the underlying block manager.
        Helps avoid pandas PerformanceWarning when adding many columns.
        """
        return df.copy()

    def _load_rotation_prone_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load rotation-prone metadata from the monitor cache if available.
        Falls back to static config list otherwise.
        """
        monitor_cfg = self.rotation_monitor_cfg or {}
        cache_path = Path(monitor_cfg.get("cache_path", "data/rotation_watch.json"))
        fallback_teams = self.risk_cfg.get("rotation_prone_teams", [])
        metadata: Dict[str, Dict[str, Any]] = {}

        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)

                for entry in payload.get("teams", []):
                    if entry.get("is_rotation_prone"):
                        team_name = entry.get("team_name")
                        if team_name:
                            metadata[team_name] = entry
            except Exception as exc:
                logger.warning(f"⚠️ Rotation monitor cache load failed: {exc}")

        if not metadata and fallback_teams:
            fallback_score = float(monitor_cfg.get("fallback_rotation_score", 0.45))
            metadata = {
                team: {
                    "team_name": team,
                    "rotation_score": fallback_score,
                    "source": "config_fallback",
                }
                for team in fallback_teams
            }

        return metadata

    def _rotation_risk_for_team(self, team_name: str) -> float:
        """
        Map a team name to the rotation risk multiplier contributed by
        manager-level tinkering.
        """
        entry = self.rotation_prone_managers.get(team_name)
        if not entry:
            return 0.0

        score = float(
            entry.get(
                "rotation_score",
                self.rotation_monitor_cfg.get("fallback_rotation_score", 0.4),
            )
        )
        return max(0.0, min(1.0, score))
    
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
            
            ownership_pct = df.get("selected_by_percent", 0).fillna(0).clip(lower=1.0)
            dynamic_threshold = self.price_rise_threshold * (ownership_pct / 100.0).clip(lower=0.25)
            rise_multiplier = float(self.market_cfg.get("threshold_multiplier", 1.2))
            likely_multiplier = float(self.market_cfg.get("likely_multiplier", 0.8))

            rise_ratio = df["transfers_in_event"] / dynamic_threshold.replace(0, np.nan)
            fall_ratio = df["transfers_out_event"] / dynamic_threshold.replace(0, np.nan)

            def _prob_from_ratio(ratio: float, high: float, mid: float) -> float:
                if pd.isna(ratio) or ratio <= 0:
                    return 0.0
                if ratio >= high:
                    return 1.0
                if ratio >= mid:
                    return 0.75
                return max(0.0, min(0.75, ratio / mid * 0.75))

            df["price_rise_probability"] = rise_ratio.apply(
                lambda r: _prob_from_ratio(r, rise_multiplier, likely_multiplier)
            )

            df["price_fall_probability"] = fall_ratio.apply(
                lambda r: _prob_from_ratio(r, rise_multiplier, likely_multiplier)
            )
            
            # Price change category
            df["price_change_status"] = df.apply(
                lambda row: "🚀 Rising tonight" if row["price_rise_probability"] >= 1.0
                else "📈 Likely rising" if row["price_rise_probability"] >= 0.75
                else "⚠️ Falling tonight" if row["price_fall_probability"] >= 1.0
                else "📉 Likely falling" if row["price_fall_probability"] >= 0.75
                else "➡️ Stable",
                axis=1
            )

            df["template_penalty"] = df["selected_by_percent"].apply(
                lambda ownership: self._calculate_template_penalty(ownership, False)
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

    def _enrich_with_understat_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Understat expected metrics to each player."""
        if self.under is None:
            for col in ["understat_xg", "understat_xa", "understat_npxg", "understat_xg_chain", "understat_xg_buildup"]:
                df[col] = 0.0
            return df

        stats_map = self._load_understat_player_stats()
        if not stats_map:
            for col in ["understat_xg", "understat_xa", "understat_npxg", "understat_xg_chain", "understat_xg_buildup"]:
                df[col] = 0.0
            return df

        df["understat_xg"] = 0.0
        df["understat_xa"] = 0.0
        df["understat_npxg"] = 0.0
        df["understat_xg_chain"] = 0.0
        df["understat_xg_buildup"] = 0.0

        for idx, row in df.iterrows():
            entry = self._match_understat_entry(row, stats_map)
            if not entry:
                continue

            df.at[idx, "understat_xg"] = float(entry.get("xG", 0.0))
            df.at[idx, "understat_xa"] = float(entry.get("xA", 0.0))
            df.at[idx, "understat_npxg"] = float(entry.get("npxG", entry.get("npxg", 0.0)))
            df.at[idx, "understat_xg_chain"] = float(entry.get("xGChain", 0.0))
            df.at[idx, "understat_xg_buildup"] = float(entry.get("xGBuildup", 0.0))

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

    def _apply_model_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features used by the prediction model.
        Ensures consistency between training data and inference DataFrames.
        """
        try:
            df = self._defragment_df(df)
            minutes = pd.to_numeric(df.get("minutes", 0), errors="coerce").fillna(0)
            minutes_safe = minutes.replace(0, np.nan)

            goals = pd.to_numeric(df.get("goals_scored", 0), errors="coerce").fillna(0)
            assists = pd.to_numeric(df.get("assists", 0), errors="coerce").fillna(0)
            clean_sheets = pd.to_numeric(df.get("clean_sheets", 0), errors="coerce").fillna(0)
            saves = pd.to_numeric(df.get("saves", 0), errors="coerce").fillna(0)
            expected_goal_involvements = pd.to_numeric(
                df.get("expected_goal_involvements", 0), errors="coerce"
            ).fillna(0)
            expected_goals_conceded = pd.to_numeric(
                df.get("expected_goals_conceded", 0), errors="coerce"
            ).fillna(0)

            fixture_diff = pd.to_numeric(
                df.get("fixture_difficulty", df.get("opponent_difficulty", 3)),
                errors="coerce"
            ).fillna(3)

            team_momentum = pd.to_numeric(df.get("team_momentum", 1.0), errors="coerce").fillna(1.0)
            form = pd.to_numeric(df.get("form", 7.5), errors="coerce").fillna(7.5)
            total_points = pd.to_numeric(df.get("total_points", 0), errors="coerce").fillna(0)

            per90 = lambda series: ((series / minutes_safe) * 90).replace([np.inf, -np.inf], 0).fillna(0)

            df["goal_contrib_per90"] = per90(goals + assists)
            df["xgi_per90"] = per90(expected_goal_involvements)
            df["threat_per90"] = per90(pd.to_numeric(df.get("threat", 0), errors="coerce").fillna(0))
            df["ict_per90"] = per90(pd.to_numeric(df.get("ict_index", 0), errors="coerce").fillna(0))
            df["value_per90"] = per90(total_points)

            df["fixture_adjusted_form"] = (5 - fixture_diff) * form
            df["momentum_adjusted_form"] = team_momentum * form

            is_home = df.get("is_home")
            if is_home is None:
                home_factor = pd.Series(1.0, index=df.index)
            else:
                home_series = df["is_home"].fillna(True).astype(bool)
                home_factor = np.where(home_series, 1.05, 0.95)
            df["home_advantage_factor"] = home_factor

            defensive_mask = df.get("position", "").isin(["DEF", "GK"]) if "position" in df.columns else pd.Series(False, index=df.index)
            defensive_actions = per90(clean_sheets + saves)
            df["defensive_actions_per90"] = np.where(defensive_mask, defensive_actions, 0.0)

            df["defensive_xg_delta"] = (expected_goals_conceded - df.get("goals_conceded", 0)).clip(-5, 5)
            df["attacking_fixture_delta"] = df["goal_contrib_per90"] * (5 - fixture_diff)

        except Exception as exc:
            logger.debug(f"Model signal feature enrichment failed: {exc}")
        finally:
            # Guarantee no NaNs in newly created columns
            model_cols = [
                "goal_contrib_per90",
                "xgi_per90",
                "threat_per90",
                "ict_per90",
                "value_per90",
                "fixture_adjusted_form",
                "momentum_adjusted_form",
                "home_advantage_factor",
                "defensive_actions_per90",
                "defensive_xg_delta",
                "attacking_fixture_delta",
            ]
            for col in model_cols:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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

            df["future_fixture_quality"] = df["fixture_run_difficulty"].apply(
                lambda x: max(0.0, 5 - float(x)) if pd.notnull(x) else 0.0
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
            df = self._defragment_df(df)
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
            
            df["manager_rotation_risk"] = df["team_name"].apply(self._rotation_risk_for_team)

            df["congestion_risk"] = df["team_name"].apply(
                lambda name: self._calculate_congestion_risk(name, self.current_gw)
            )

            df["age_risk"] = df.apply(
                lambda row: self._calculate_age_risk(row.get("age"), row.get("position")),
                axis=1
            )
            
            df["rotation_risk"] = (
                df["rotation_risk_base"] * 0.35 +
                df["minutes_volatility"] * 0.25 +
                df["manager_rotation_risk"] * 0.2 +
                df["congestion_risk"] * 0.1 +
                df["manager_change_factor"] * 0.1
            ).clip(0, 1)
            
            # Disciplinary
            current_gw = self.current_gw or 0
            df["disciplinary_risk"] = df["yellow_cards"].fillna(0).astype(int).apply(
                lambda yc: self._calculate_discipline_risk(yc, current_gw)
            )
            df["disciplinary_risk"] = np.maximum(
                df["disciplinary_risk"],
                (df.get("red_cards", 0) > 0).astype(float) * 0.3
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

            df["penalty_bonus"] = df.apply(self._get_penalty_bonus, axis=1)
            
            # TOTAL RISK
            df["total_risk"] = (
                df["injury_risk"] * 0.25 +
                df["news_severity"] * 0.10 +
                df["rotation_risk"] * 0.20 +
                df["disciplinary_risk"] * 0.15 +
                df["fatigue_risk"] * 0.10 +
                df["form_drop_risk"] * 0.15 +
                df["defensive_fragility_risk"] * 0.03 +
                df["penalty_risk"] * 0.02 +
                df["congestion_risk"] * 0.05 +
                df["age_risk"] * 0.05
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

    # ==================== EXTERNAL INJURY HELPERS ====================

    def _apply_external_injuries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-league injury intel (e.g., international breaks)."""
        if not self.external_injuries_enabled:
            return df

        injuries = self._get_external_injury_data()
        if not injuries:
            return df

        if "chance_of_playing_next_round" not in df.columns:
            df["chance_of_playing_next_round"] = 100

        df["chance_of_playing_next_round"] = pd.to_numeric(
            df.get("chance_of_playing_next_round", 100),
            errors="coerce"
        ).fillna(100).astype(int)

        if "news" not in df.columns:
            df["news"] = ""
        else:
            df["news"] = df["news"].fillna("")

        for column, default in [
            ("external_injury_flag", False),
            ("external_injury_note", ""),
            ("external_injury_source", ""),
            ("external_injury_updated", ""),
            ("external_expected_return", "")
        ]:
            if column not in df.columns:
                df[column] = default

        name_index = self._build_player_name_index(df)
        if not name_index:
            return df

        updates = 0

        for injury in injuries:
            player_name = injury.get("player_name")
            if not player_name:
                continue

            name_key = self._normalize_name(player_name)
            matches = name_index.get(name_key)

            if not matches and " " in player_name:
                last_name = player_name.split()[-1]
                matches = name_index.get(self._normalize_name(last_name))

            if not matches:
                continue

            severity = injury.get("severity", self.external_default_severity)
            note = injury.get("details") or injury.get("status") or ""
            updated = injury.get("updated_at", "")
            expected_return = injury.get("return_date", "")
            source = injury.get("source", "public")

            for idx in matches:
                current_chance = int(df.at[idx, "chance_of_playing_next_round"])

                severity_chance = max(0, int((1 - severity) * 100))
                if severity >= self.external_medium_threshold:
                    severity_chance = min(severity_chance, self.external_fallback_chance)

                df.at[idx, "chance_of_playing_next_round"] = min(current_chance, severity_chance)
                df.at[idx, "external_injury_flag"] = True
                df.at[idx, "external_injury_note"] = note
                df.at[idx, "external_injury_source"] = source
                df.at[idx, "external_injury_updated"] = updated
                df.at[idx, "external_expected_return"] = expected_return

                existing_news = df.at[idx, "news"]
                addition = f"[EXT] {note}"
                df.at[idx, "news"] = addition if not existing_news else f"{existing_news} | {addition}"

                updates += 1

        if updates:
            logger.info(f"🩺 External injury updates applied to {updates} player(s)")

        return df

    def _apply_news_signals(self, df: pd.DataFrame, next_gw: Optional[int]) -> pd.DataFrame:
        if not self.news_enabled:
            return df

        articles = self._fetch_news_articles()
        if not articles:
            return df

        if "news_flag" not in df.columns:
            df["news_flag"] = False
        if "news_sources" not in df.columns:
            df["news_sources"] = ""
        if "news_last_headline" not in df.columns:
            df["news_last_headline"] = ""

        name_index = self._build_player_name_index(df)
        if not name_index:
            return df

        updates = 0
        for article in articles:
            headline = article.get("title") or ""
            description = article.get("description") or ""
            combined = f"{headline} {description}".lower()

            if not any(keyword in combined for keyword in self.news_keywords):
                continue

            matched = False
            for name, indices in name_index.items():
                if name and name in combined:
                    for idx in indices:
                        df.at[idx, "news_flag"] = True
                        df.at[idx, "news_sources"] = article.get("source") or article.get("url") or "news"
                        df.at[idx, "news_last_headline"] = headline
                        updates += 1
                    matched = True
            if matched:
                continue

        if updates:
            logger.info(f"📰 News signals applied to {updates} player(s)")

        return df

    def _fetch_news_articles(self) -> List[Dict[str, Any]]:
        if (
            self._news_cache
            and self._news_last_fetch
            and datetime.utcnow() - self._news_last_fetch < timedelta(minutes=self.news_poll_minutes)
        ):
            return self._news_cache

        articles: List[Dict[str, Any]] = []

        if self.gnews_client:
            articles.extend(self.gnews_client.fetch_articles("football premier league injury"))

        if self.newsapi_client:
            articles.extend(self.newsapi_client.fetch_articles())

        if self.rss_client:
            articles.extend(self.rss_client.fetch_articles())

        self._news_cache = articles
        self._news_last_fetch = datetime.utcnow()
        return articles

    def _enrich_with_bbc_lineups(self, df: pd.DataFrame, next_gw: Optional[int]) -> pd.DataFrame:
        """Attach BBC lineup signals for fixtures close to deadline."""
        if not self.lineup_enabled or not next_gw or self.fpl is None or self.bbc_client is None:
            return df

        fixtures = self.fpl.fixtures() or []
        if not fixtures:
            return df

        if "bbc_lineup_status" not in df.columns:
            df["bbc_lineup_status"] = "unknown"
        else:
            df["bbc_lineup_status"] = df["bbc_lineup_status"].fillna("unknown")

        if "bbc_lineup_source" not in df.columns:
            df["bbc_lineup_source"] = ""
        else:
            df["bbc_lineup_source"] = df["bbc_lineup_source"].fillna("")

        if "bbc_lineup_updated" not in df.columns:
            df["bbc_lineup_updated"] = ""
        else:
            df["bbc_lineup_updated"] = df["bbc_lineup_updated"].fillna("")

        now_utc = datetime.now(timezone.utc)
        team_lookup = self._build_team_player_lookup(df)

        for fixture in fixtures:
            if fixture.get("event") != next_gw:
                continue

            kickoff = fixture.get("kickoff_time")
            if not kickoff:
                continue

            kickoff_dt = pd.to_datetime(kickoff, utc=True).to_pydatetime()
            minutes_until = (kickoff_dt - now_utc).total_seconds() / 60
            if minutes_until > self.lineup_pre_kickoff_window or minutes_until < -180:
                continue

            lineup = self._get_bbc_lineup_for_fixture(fixture, kickoff_dt)
            if not lineup:
                continue

            self._apply_lineup_status(df, fixture, lineup, team_lookup)

        return df

    def _get_bbc_lineup_for_fixture(self, fixture: Dict[str, Any], kickoff_dt: datetime) -> Optional[Dict[str, Any]]:
        fixture_id = fixture.get("id")
        if fixture_id is None:
            return None

        cache_entry = self._bbc_lineup_cache.get(fixture_id)
        if cache_entry:
            fetched = cache_entry.get("fetched")
            if fetched and datetime.utcnow() - fetched < timedelta(minutes=self.lineup_refresh_minutes):
                return cache_entry.get("data")

        event_id = self._bbc_event_map.get(fixture_id)
        if not event_id:
            home_name = self.get_team_name(fixture.get("team_h"), short=False)
            away_name = self.get_team_name(fixture.get("team_a"), short=False)
            event_id = self.bbc_client.find_event_id(kickoff_dt, home_name, away_name)
            if not event_id:
                return None
            self._bbc_event_map[fixture_id] = event_id

        lineup = self.bbc_client.get_match_lineups(event_id)
        if lineup:
            self._bbc_lineup_cache[fixture_id] = {"data": lineup, "fetched": datetime.utcnow()}
        return lineup

    def _apply_lineup_status(
        self,
        df: pd.DataFrame,
        fixture: Dict[str, Any],
        lineup: Dict[str, Any],
        team_lookup: Dict[int, Dict[str, List[int]]],
    ) -> None:
        home_team = fixture.get("team_h")
        away_team = fixture.get("team_a")

        if home_team:
            self._update_team_lineup(
                df,
                home_team,
                lineup.get("home"),
                team_lookup,
                lineup.get("updated_at"),
            )

        if away_team:
            self._update_team_lineup(
                df,
                away_team,
                lineup.get("away"),
                team_lookup,
                lineup.get("updated_at"),
            )

    def _update_team_lineup(
        self,
        df: pd.DataFrame,
        team_id: int,
        lineup_info: Optional[Dict[str, Any]],
        team_lookup: Dict[int, Dict[str, List[int]]],
        updated_at: Optional[str],
    ) -> None:

        if not lineup_info:
            return

        name_map = team_lookup.get(team_id, {})

        def assign_status(players: List[Dict[str, Any]], status: str) -> None:
            for player in players or []:
                norm = self._normalize_name(player.get("name"))
                for idx in name_map.get(norm, []):
                    df.at[idx, "bbc_lineup_status"] = status
                    df.at[idx, "bbc_lineup_source"] = "BBC"
                    df.at[idx, "bbc_lineup_updated"] = updated_at or ""

        assign_status(lineup_info.get("starters"), "starter")
        assign_status(lineup_info.get("bench"), "bench")

    def _build_team_player_lookup(self, df: pd.DataFrame) -> Dict[int, Dict[str, List[int]]]:
        lookup: Dict[int, Dict[str, List[int]]] = {}

        for idx, row in df.iterrows():
            team_id = int(row.get("team", 0))
            names = {
                row.get("web_name"),
                row.get("second_name"),
                row.get("first_name"),
                f"{row.get('first_name', '')} {row.get('second_name', '')}".strip(),
                row.get("known_as"),
            }

            for name in filter(None, names):
                norm = self._normalize_name(name)
                if not norm:
                    continue
                lookup.setdefault(team_id, {}).setdefault(norm, []).append(idx)

        return lookup

    def _get_external_injury_data(self) -> List[Dict[str, Any]]:
        injuries: List[Dict[str, Any]] = []

        if self.pl_injuries is not None:
            injuries.extend(self._normalize_pl_injuries(self._get_pl_injury_data()))

        if self.external_injuries_enabled and self.sportmonks is not None:
            injuries.extend(self._normalize_sportmonks_injuries(self._get_sportmonks_injury_data()))

        return injuries

    def _get_pl_injury_data(self) -> List[Dict[str, Any]]:
        if self.pl_injuries is None:
            return []

        if (
            self._pl_injury_cache
            and self._pl_injury_last_fetch
            and datetime.utcnow() - self._pl_injury_last_fetch < timedelta(minutes=self.external_injury_refresh_minutes)
        ):
            return self._pl_injury_cache

        try:
            data = self.pl_injuries.list_injuries()
            self._pl_injury_cache = data or []
            self._pl_injury_last_fetch = datetime.utcnow()
        except Exception as exc:
            logger.warning(f"⚠️ Premier League injury fetch failed: {exc}")
            self._pl_injury_cache = []

        return self._pl_injury_cache

    def _get_sportmonks_injury_data(self) -> List[Dict[str, Any]]:
        if self.sportmonks is None:
            return []

        if (
            self._external_injury_cache
            and self._external_injury_last_fetch
            and datetime.utcnow() - self._external_injury_last_fetch < timedelta(minutes=self.external_injury_refresh_minutes)
        ):
            return self._external_injury_cache

        try:
            injuries = self.sportmonks.list_injuries(
                league_ids=self.external_injury_leagues,
                include_player_details=self.external_include_player_details,
            )
            self._external_injury_cache = injuries or []
            self._external_injury_last_fetch = datetime.utcnow()
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning(f"⚠️ External injury fetch failed: {exc}")
            self._external_injury_cache = []

        return self._external_injury_cache

    def _normalize_pl_injuries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for entry in entries or []:
            name_block = entry.get("name") or {}
            player_name = " ".join(filter(None, [name_block.get("first"), name_block.get("last")])).strip()
            if not player_name:
                player_name = entry.get("name") or ""

            injuries = entry.get("injuries") or entry.get("injury") or []
            team_name = (
                entry.get("currentTeam", {})
                .get("club", {})
                .get("name")
                or entry.get("team", {})
                .get("name")
            )

            for issue in injuries:
                details = issue.get("info") or issue.get("detail") or issue.get("description")
                status = issue.get("status") or issue.get("type", {}).get("name")
                normalized.append(
                    {
                        "player_name": player_name,
                        "team": team_name,
                        "details": details,
                        "status": status,
                        "updated_at": issue.get("startDate") or issue.get("updatedDate"),
                        "return_date": issue.get("expectedReturnDate") or issue.get("returnDate"),
                        "source": "premier-league",
                        "severity": self._infer_external_severity({"details": details, "status": status}),
                    }
                )

        return normalized

    def _normalize_sportmonks_injuries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for injury in entries or []:
            player_block = injury.get("player") or {}
            player_data = player_block.get("data", player_block)
            player_name = (
                player_data.get("fullname")
                or player_data.get("short_name")
                or player_data.get("name")
            )

            if not player_name:
                continue

            details = injury.get("details") or injury.get("reason")
            status = injury.get("status") or injury.get("short_description")

            normalized.append(
                {
                    "player_name": player_name,
                    "team": player_data.get("team_name"),
                    "details": details,
                    "status": status,
                    "updated_at": injury.get("updated_at") or injury.get("start_date"),
                    "return_date": injury.get("return_date") or injury.get("end_date"),
                    "source": "sportmonks",
                    "severity": self._infer_external_severity(injury),
                }
            )

        return normalized

    def _build_player_name_index(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        index: Dict[str, List[int]] = {}

        for idx, row in df.iterrows():
            names = {
                row.get("web_name"),
                row.get("first_name"),
                row.get("second_name"),
                f"{row.get('first_name', '')} {row.get('second_name', '')}".strip(),
                row.get("known_as")
            }

            for name in filter(None, names):
                norm = self._normalize_name(name)
                if not norm:
                    continue
                index.setdefault(norm, []).append(idx)

        return index

    @staticmethod
    def _normalize_name(value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", str(value))
        normalized = normalized.encode("ascii", "ignore").decode("utf-8")
        return re.sub(r"[^a-z0-9]", "", normalized.lower())

    @staticmethod
    def _extract_injury_player_name(injury: Dict[str, Any]) -> str:
        if not injury:
            return ""

        if "player_name" in injury:
            return injury["player_name"]

        player_block = injury.get("player")
        if isinstance(player_block, dict):
            data = player_block.get("data", player_block)
            return data.get("fullname") or data.get("display_name") or data.get("name") or ""

        return ""

    def _infer_external_severity(self, injury: Dict[str, Any]) -> float:
        text_blobs = [
            injury.get("details"),
            injury.get("status"),
            injury.get("type", {}).get("data", {}).get("name") if isinstance(injury.get("type"), dict) else None,
        ]
        text = " ".join(filter(None, text_blobs)).lower()

        if any(keyword in text for keyword in ["acl", "rupture", "fracture", "ligament"]):
            return 0.95
        if any(keyword in text for keyword in ["hamstring", "groin", "ankle", "thigh"]):
            return max(self.external_high_threshold, 0.85)
        if any(keyword in text for keyword in ["knock", "minor", "rest"]):
            return 0.45
        if any(keyword in text for keyword in ["illness", "fatigue"]):
            return 0.4

        return self.external_default_severity

    @staticmethod
    def _format_external_injury_note(injury: Dict[str, Any]) -> str:
        parts = []

        injury_type = injury.get("type")
        if isinstance(injury_type, dict):
            injury_type = injury_type.get("data", {}).get("name", "")

        if injury_type:
            parts.append(injury_type)

        if injury.get("details"):
            parts.append(injury["details"])

        status = injury.get("status") or injury.get("short_description")
        if status:
            parts.append(status)

        competition = injury.get("league_name") or injury.get("competition")
        if competition:
            parts.append(f"({competition})")

        return " ".join(parts).strip()

    # ==================== CONTEXT / RISK HELPERS ====================

    def _build_team_momentum_map(self) -> Dict[int, float]:
        """Translate team form strings into momentum multipliers."""
        momentum_map = {}
        points_lookup = {"W": 3, "D": 1, "L": 0}

        for team_id, info in self.teams_map.items():
            form_str = (info.get("recent_form") or "")[:6]
            if not form_str:
                momentum_map[team_id] = 1.0
                continue

            total_points = sum(points_lookup.get(ch.upper(), 0) for ch in form_str)

            if total_points >= 14:
                momentum_map[team_id] = 1.2
            elif total_points <= 4:
                momentum_map[team_id] = 0.8
            else:
                momentum_map[team_id] = 1.0

        return momentum_map

    def _build_congested_team_lookup(self, competitions: Dict[str, List[str]]) -> set:
        """Build set of teams participating in congested schedules."""
        congested = set()
        for teams in competitions.values():
            if teams:
                congested.update(teams)
        return congested

    def _calculate_age_years(self, dob: pd.Timestamp) -> Optional[float]:
        """Calculate age in years from date of birth."""
        if pd.isna(dob):
            return None
        today = datetime.utcnow()
        years = (today - dob).days / 365.25
        return round(years, 1)

    def _calculate_manager_change_impact(self, team_name: str) -> float:
        """Estimate additional rotation risk immediately after a manager change."""
        gws_since_change = self.manager_change_map.get(team_name, None)
        if gws_since_change is None:
            return 0.0
        if gws_since_change == 0:
            return 0.5
        if 1 <= gws_since_change <= 3:
            return 0.3
        if 4 <= gws_since_change <= 6:
            return 0.1
        return 0.0

    def _calculate_congestion_risk(self, team_name: str, gw: Optional[int] = None) -> float:
        """
        Estimate rotation risk from fixture congestion (European competitions, cups).
        """
        if not team_name:
            return 0.1
        if team_name in self.congested_teams:
            return 0.4
        return 0.1

    def _calculate_discipline_risk(self, yellow_cards: int, gw: int) -> float:
        """Risk of suspension based on yellow cards and season checkpoint."""
        if gw == 0:
            return 0.0
        if yellow_cards >= 9 and gw < 32:
            return 0.9
        if yellow_cards >= 4 and gw < 19:
            return 0.8
        if yellow_cards >= 5:
            return 0.5
        if yellow_cards >= 3:
            return 0.3
        return 0.0

    def _calculate_age_risk(self, age: Optional[float], position: str) -> float:
        """Age-related decline risk."""
        if age is None:
            return 0.0
        if position == "FWD" and age > 32:
            return 0.5
        if position == "MID" and age > 33:
            return 0.4
        if position == "DEF" and age > 34:
            return 0.3
        if position == "GK" and age > 36:
            return 0.25
        return 0.0

    def _get_penalty_bonus(self, row: pd.Series) -> float:
        """Expected bonus from being a first-choice penalty taker."""
        is_pen_taker = (row.get("penalties_order") == 1)
        team_penalties_per_gw = float(row.get("team_penalties_per_gw", 0.18))
        if not is_pen_taker:
            return 0.0
        return team_penalties_per_gw * 0.85 * 5

    def _apply_bayesian_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Blend season-long and recent form using Bayesian update."""
        df["bayesian_form"] = (
            (1 - self.bayesian_confidence) * df["points_per_game_raw"] +
            self.bayesian_confidence * df["form"]
        ).round(3)
        return df

    def _estimate_points_std_dev(self, df: pd.DataFrame) -> pd.Series:
        """Estimate per-player standard deviation for Monte Carlo simulation."""
        base = (df["form"].fillna(5.0) / 6.0).clip(lower=0.5)
        risk_multiplier = 1 + df.get("total_risk", 0).fillna(0)
        return (base * risk_multiplier).round(3)

    def _calculate_template_penalty(self, ownership: float, is_captaincy: bool) -> float:
        """Penalty for highly owned players (reduced rank upside)."""
        if is_captaincy:
            eo = ownership * 0.02
            if eo > 1.0:
                return self.market_cfg.get("template_penalty_captain", 0.5)
        if ownership > 50:
            return self.market_cfg.get("template_penalty_player", 0.3)
        return 0.0

    def _load_understat_player_stats(self) -> Dict[str, Dict[str, Any]]:
        if self.under is None:
            return {}

        if (
            self._understat_cache
            and self._understat_last_fetch
            and datetime.utcnow() - self._understat_last_fetch < timedelta(hours=self.understat_refresh_hours)
        ):
            return self._understat_cache

        stats = self.under.get_all_players_stats(season=self.understat_season) or []
        cache: Dict[str, Dict[str, Any]] = {}
        for entry in stats:
            name = entry.get("player_name") or entry.get("title")
            if not name:
                continue
            norm = self._normalize_name(name)
            if norm:
                cache[norm] = entry

        self._understat_cache = cache
        self._understat_last_fetch = datetime.utcnow()
        return cache

    def _match_understat_entry(self, row: pd.Series, stats_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [
            row.get("web_name"),
            row.get("second_name"),
            row.get("first_name"),
            f"{row.get('first_name', '')} {row.get('second_name', '')}".strip(),
            row.get("known_as"),
        ]

        for name in filter(None, candidates):
            norm = self._normalize_name(name)
            if norm in stats_map:
                return stats_map[norm]

        return None

    def _get_next_gameweek_id(self) -> Optional[int]:
        events = self.bootstrap.get("events", []) if self.bootstrap else []
        for event in events:
            if event.get("is_next"):
                return event.get("id")
        current = next((e for e in events if e.get("is_current")), None)
        if current:
            return current.get("id")
        return None
        
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
                    
                    if len(metrics.get("warnings", [])) > 0:
                        logger.warning(f"   ⚠️ {len(metrics['warnings'])} warnings")
                    
                    # Enrich with additional features
                    df = self._enrich_training_features(df)
                    
                    return df
                else:
                    logger.warning("   ⚠️ Historical data empty - falling back to Strategy 2")
            
            except Exception as e:
                logger.warning(f"   ⚠️ Historical data failed: {e}")
                import traceback
                traceback.print_exc()  # ← ADD THIS LINE
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
                
                # ✅ FIX: Ensure 'event' column exists with proper fallback
                if "event" not in history_df.columns:
                    if "round" in history_df.columns:
                        history_df["event"] = history_df["round"]
                    elif "GW" in history_df.columns:
                        history_df["event"] = history_df["GW"]
                    else:
                        # Create sequential gameweek numbers if no event column
                        history_df["event"] = range(1, len(history_df) + 1)
                
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
        
        # ✅ FIX: Ensure gameweek column exists (required for historical enrichment)
        if "gameweek" not in df.columns:
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

            if "event" not in df.columns:
                if "gameweek" in df.columns:
                    df["event"] = df["gameweek"]
                elif "round" in df.columns:
                    df["event"] = df["round"]
                else:
                    logger.warning("⚠️ No event/gameweek column found, creating sequential")
                    df["event"] = range(1, len(df) + 1)
            
            # Ensure selected_by_percent exists
            if "selected_by_percent" not in df.columns:
                logger.warning("⚠️ No ownership data - using default 0")
                df["selected_by_percent"] = 0

            if "fixture_difficulty" not in df.columns:
                df["fixture_difficulty"] = pd.to_numeric(
                    df.get("opponent_difficulty", 3), errors="coerce"
                ).fillna(3)

            if "team_momentum" not in df.columns and "team" in df.columns:
                df["team_momentum"] = df["team"].map(self.team_momentum_map).fillna(1.0)

            if "is_home" in df.columns:
                df["is_home"] = df["is_home"].fillna(True).astype(bool)
            elif "was_home" in df.columns:
                df["is_home"] = df["was_home"].fillna(True).astype(bool)
            else:
                df["is_home"] = True

            df = self._apply_model_signal_features(df)
            
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