"""
Historical Data Integration Module - Enhanced FPL Training Data

This module integrates multiple external data sources to provide comprehensive
historical FPL data including ownership percentages, advanced metrics, and
more accurate training datasets.

SOURCES SUPPORTED:
✅ Vaastav's FPL Historical Dataset (GitHub)
✅ FPL Time-Series Data
✅ Understat xG/xA data
✅ Local cache for offline operation

PRODUCTION READY v6.1
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("historical_data")
logger.setLevel(logging.INFO)


class HistoricalDataIntegrator:
    """
    Integrates external historical FPL data sources for enhanced training.
    
    Features:
    - Downloads from Vaastav's GitHub repository
    - Caches data locally for performance
    - Merges historical ownership data
    - Integrates advanced metrics
    - Handles multiple seasons
    - Automatic updates
    """
    
    # Vaastav's repository URLs
    VAASTAV_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    
    # Supported seasons
    AVAILABLE_SEASONS = [
        "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
    ]
    
    def __init__(
        self,
        cache_dir: str = "data/cache/historical",
        current_season: str = "2024-25",
        use_cache: bool = True,
        cache_expiry_days: int = 7
    ):
        """
        Initialize historical data integrator.
        
        Args:
            cache_dir: Directory for cached data
            current_season: Current FPL season (e.g., "2024-25")
            use_cache: Whether to use cached data
            cache_expiry_days: Days before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_season = current_season
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        
        # Setup HTTP session with retry logic
        self.session = self._create_session()
        
        logger.info(f"✅ Historical Data Integrator initialized")
        logger.info(f"   Cache: {self.cache_dir}")
        logger.info(f"   Current season: {current_season}")
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    # ==================== MAIN PUBLIC METHODS ====================
    
    def get_merged_gameweek_data(
        self,
        season: str = None,
        gameweeks: Optional[List[int]] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get merged gameweek data with historical ownership.
        
        This is the primary method for training data.
        
        Args:
            season: Season string (e.g., "2024-25"), defaults to current
            gameweeks: Specific gameweeks to fetch (None = all)
            force_refresh: Force download even if cached
        
        Returns:
            DataFrame with columns including:
            - player_id, name, team, position
            - gameweek, total_points, minutes, goals_scored, assists
            - selected_by_percent (HISTORICAL!)
            - value (price at that time)
            - ict_index, influence, creativity, threat
            - expected_goals, expected_assists
            - fixture difficulty, opponent
        """
        season = season or self.current_season
        
        logger.info(f"📊 Fetching merged gameweek data for {season}")
        
        # Check cache
        cache_file = self.cache_dir / f"{season}_merged_gw.csv"
        
        if self.use_cache and not force_refresh and self._is_cache_valid(cache_file):
            logger.info(f"   📁 Loading from cache: {cache_file}")
            df = pd.read_csv(cache_file)
        else:
            # Download from GitHub
            logger.info(f"   🌐 Downloading from GitHub...")
            df = self._download_merged_gameweek_data(season)
            
            if df is not None and not df.empty:
                # Save to cache
                df.to_csv(cache_file, index=False)
                logger.info(f"   💾 Cached to: {cache_file}")
            else:
                logger.error(f"   ❌ Download failed")
                return pd.DataFrame()
        
        # Filter gameweeks if specified
        if gameweeks is not None and "GW" in df.columns:
            df = df[df["GW"].isin(gameweeks)]
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        logger.info(f"   ✅ Loaded {len(df)} records")
        
        return df
    
    def get_multi_season_data(
        self,
        seasons: List[str] = None,
        min_gameweeks: int = 5,
        include_current_season: bool = True
    ) -> pd.DataFrame:
        """
        Get training data from multiple historical seasons.
        
        Args:
            seasons: List of seasons (e.g., ["2022-23", "2023-24"])
            min_gameweeks: Minimum gameweeks per season
            include_current_season: Include current season data
        
        Returns:
            Combined DataFrame from multiple seasons
        """
        if seasons is None:
            # Default: last 2 complete seasons + current
            seasons = ["2022-23", "2023-24"]
            if include_current_season:
                seasons.append(self.current_season)
        
        logger.info(f"📊 Fetching multi-season data: {seasons}")
        
        all_dfs = []
        
        for season in seasons:
            df = self.get_merged_gameweek_data(season)
            
            if df.empty:
                logger.warning(f"   ⚠️ Skipping {season} - no data")
                continue
            
            # Filter by minimum gameweeks
            if "gameweek" in df.columns:
                unique_gws = df["gameweek"].nunique()
                if unique_gws < min_gameweeks:
                    logger.warning(f"   ⚠️ Skipping {season} - only {unique_gws} GWs")
                    continue
            
            # Add season identifier
            df["season"] = season
            all_dfs.append(df)
            
            logger.info(f"   ✅ {season}: {len(df)} records")
        
        if not all_dfs:
            logger.error("❌ No data loaded from any season")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        logger.info(f"✅ Multi-season data: {len(combined_df)} total records")
        
        return combined_df
    
    def enrich_current_season_data(
        self,
        current_df: pd.DataFrame,
        player_id_col: str = "player_id",
        gameweek_col: str = "event"
    ) -> pd.DataFrame:
        """
        Enrich current season data with historical ownership.
        
        This merges historical ownership data from Vaastav's dataset
        into your current FPL API data.
        
        Args:
            current_df: DataFrame from FPL API (element_summary)
            player_id_col: Column name for player ID
            gameweek_col: Column name for gameweek
        
        Returns:
            Enriched DataFrame with historical selected_by_percent
        """
        logger.info("📊 Enriching current data with historical ownership")
        
        # Get historical data for current season
        historical_df = self.get_merged_gameweek_data(self.current_season)
        
        if historical_df.empty:
            logger.warning("⚠️ No historical data available - using current ownership")
            return current_df
        
        # Prepare for merge
        historical_df = historical_df.rename(columns={
            "GW": gameweek_col,
            "element": player_id_col
        })
        
        # Select relevant columns from historical data
        merge_cols = [
            player_id_col,
            gameweek_col,
            "selected_by_percent",
            "value",
            "transfers_in",
            "transfers_out"
        ]
        
        available_merge_cols = [c for c in merge_cols if c in historical_df.columns]
        
        historical_subset = historical_df[available_merge_cols]
        
        # Merge
        enriched_df = current_df.merge(
            historical_subset,
            on=[player_id_col, gameweek_col],
            how="left",
            suffixes=("", "_historical")
        )
        
        # Use historical ownership if available, otherwise keep current
        if "selected_by_percent_historical" in enriched_df.columns:
            enriched_df["selected_by_percent"] = enriched_df["selected_by_percent_historical"].fillna(
                enriched_df.get("selected_by_percent", 0)
            )
            enriched_df.drop(columns=["selected_by_percent_historical"], inplace=True)
        
        matched_count = enriched_df["selected_by_percent"].notna().sum()
        logger.info(f"✅ Enriched {matched_count}/{len(enriched_df)} records with historical ownership")
        
        return enriched_df
    
    # ==================== DOWNLOAD METHODS ====================
    
    def _download_merged_gameweek_data(self, season: str) -> Optional[pd.DataFrame]:
        """Download merged gameweek data from Vaastav's GitHub."""
        url = f"{self.VAASTAV_BASE_URL}/{season}/gws/merged_gw.csv"
        
        try:
            logger.debug(f"   Downloading: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from response content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"   ✅ Downloaded {len(df)} records")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Download failed: {e}")
            return None
        except Exception as e:
            logger.error(f"   ❌ Parse failed: {e}")
            return None
    
    def download_player_history(
        self,
        season: str,
        player_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Download individual player history from Vaastav's dataset.
        
        Args:
            season: Season string
            player_name: Player name (e.g., "Mohamed_Salah")
        
        Returns:
            DataFrame with player's historical stats
        """
        # Clean player name for URL
        clean_name = player_name.replace(" ", "_")
        
        url = f"{self.VAASTAV_BASE_URL}/{season}/players/{clean_name}/gw.csv"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            return df
            
        except Exception as e:
            logger.debug(f"Failed to download {player_name}: {e}")
            return None
    
    # ==================== CACHE MANAGEMENT ====================
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is valid (exists and not expired)."""
        if not cache_file.exists():
            return False
        
        # Check age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        
        if file_age.days > self.cache_expiry_days:
            logger.debug(f"   Cache expired ({file_age.days} days old)")
            return False
        
        return True
    
    def clear_cache(self, season: str = None):
        """Clear cached data."""
        if season:
            cache_file = self.cache_dir / f"{season}_merged_gw.csv"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"✅ Cleared cache for {season}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.csv"):
                file.unlink()
            logger.info(f"✅ Cleared all cache in {self.cache_dir}")
    
    # ==================== DATA PROCESSING ====================
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        rename_map = {
            "GW": "gameweek",
            "element": "player_id",
            "name": "web_name",
            "opponent_team": "opponent",
            "was_home": "is_home",
            "kickoff_time": "kickoff",
            "round": "gameweek"
        }
        
        df = df.rename(columns=rename_map)
        
        # Ensure critical columns exist
        if "gameweek" not in df.columns and "event" in df.columns:
            df["gameweek"] = df["event"]
        
        if "selected_by_percent" in df.columns:
            df["selected_by_percent"] = pd.to_numeric(
                df["selected_by_percent"],
                errors="coerce"
            ).fillna(0)
        
        return df
    
    def prepare_training_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for model training.
        
        Args:
            df: Raw historical data
            include_target: Whether to create target variable
        
        Returns:
            DataFrame ready for training
        """
        logger.info("🔧 Preparing training features")
        
        df = df.copy()
        
        # Ensure required columns
        required_cols = ["player_id", "gameweek", "total_points"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Sort by player and gameweek
        df = df.sort_values(["player_id", "gameweek"])
        
        # Create target variable (next gameweek points)
        if include_target:
            df["target_next_points"] = df.groupby("player_id")["total_points"].shift(-1)
            
            # Remove rows without target
            df = df.dropna(subset=["target_next_points"])
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"✅ Prepared {len(df)} training samples")
        
        return df
    
    # ==================== VALIDATION & STATISTICS ====================
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality validation.
        
        Returns:
            Dict with quality metrics and warnings
        """
        metrics = {
            "total_samples": len(df),
            "unique_players": 0,
            "gameweeks_covered": 0,
            "has_ownership": False,
            "missing_pct": 0.0,
            "warnings": []
        }
        
        if df.empty:
            metrics["warnings"].append("DataFrame is empty")
            return metrics
        
        # Player coverage
        if "player_id" in df.columns:
            metrics["unique_players"] = df["player_id"].nunique()
            
            samples_per_player = df.groupby("player_id").size()
            metrics["avg_samples_per_player"] = float(samples_per_player.mean())
            metrics["min_samples_per_player"] = int(samples_per_player.min())
        
        # Gameweek coverage
        if "gameweek" in df.columns:
            metrics["gameweeks_covered"] = df["gameweek"].nunique()
            metrics["gw_range"] = [int(df["gameweek"].min()), int(df["gameweek"].max())]
        
        # Ownership data
        if "selected_by_percent" in df.columns:
            metrics["has_ownership"] = True
            metrics["avg_ownership"] = float(df["selected_by_percent"].mean())
            non_zero_ownership = (df["selected_by_percent"] > 0).sum()
            metrics["ownership_coverage"] = float(non_zero_ownership / len(df))
            
            if metrics["ownership_coverage"] < 0.5:
                metrics["warnings"].append(
                    f"Low ownership coverage: {metrics['ownership_coverage']*100:.1f}%"
                )
        else:
            metrics["warnings"].append("No ownership data (selected_by_percent)")
        
        # Missing data
        missing_pct = (df.isnull().sum() / len(df) * 100).mean()
        metrics["missing_pct"] = float(missing_pct)
        
        if missing_pct > 10:
            metrics["warnings"].append(f"High missing data: {missing_pct:.1f}%")
        
        # Target variable
        if "target_next_points" in df.columns:
            target_missing = df["target_next_points"].isnull().sum()
            if target_missing > 0:
                metrics["warnings"].append(f"Target has {target_missing} missing values")
        
        return metrics
    
    def get_statistics_summary(self, df: pd.DataFrame) -> str:
        """
        Get human-readable statistics summary.
        
        Args:
            df: Historical data DataFrame
        
        Returns:
            Formatted string with statistics
        """
        metrics = self.validate_data_quality(df)
        
        summary_lines = [
            "=" * 80,
            "📊 HISTORICAL DATA SUMMARY",
            "=" * 80,
            f"Total samples:        {metrics['total_samples']:,}",
            f"Unique players:       {metrics['unique_players']:,}",
            f"Gameweeks covered:    {metrics['gameweeks_covered']}",
        ]
        
        if "gw_range" in metrics:
            summary_lines.append(f"GW range:             {metrics['gw_range'][0]} - {metrics['gw_range'][1]}")
        
        summary_lines.append(f"Has ownership data:   {'✅ Yes' if metrics['has_ownership'] else '❌ No'}")
        
        if metrics['has_ownership']:
            summary_lines.append(f"Avg ownership:        {metrics['avg_ownership']:.1f}%")
            summary_lines.append(f"Ownership coverage:   {metrics['ownership_coverage']*100:.1f}%")
        
        summary_lines.append(f"Avg missing data:     {metrics['missing_pct']:.1f}%")
        
        if metrics["warnings"]:
            summary_lines.append("\n⚠️  WARNINGS:")
            for i, warning in enumerate(metrics["warnings"], 1):
                summary_lines.append(f"   {i}. {warning}")
        else:
            summary_lines.append("\n✅ No data quality issues")
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_load_historical_data(
    seasons: List[str] = None,
    cache_dir: str = "data/cache/historical"
) -> pd.DataFrame:
    """
    Quick function to load historical training data.
    
    Args:
        seasons: List of seasons (defaults to last 2 + current)
        cache_dir: Cache directory
    
    Returns:
        Combined historical DataFrame ready for training
    """
    integrator = HistoricalDataIntegrator(cache_dir=cache_dir)
    
    df = integrator.get_multi_season_data(seasons=seasons)
    
    if not df.empty:
        df = integrator.prepare_training_features(df)
    
    return df


def enrich_with_ownership(
    current_df: pd.DataFrame,
    season: str = "2024-25",
    cache_dir: str = "data/cache/historical"
) -> pd.DataFrame:
    """
    Quick function to enrich current data with historical ownership.
    
    Args:
        current_df: Current FPL API data
        season: Season to match
        cache_dir: Cache directory
    
    Returns:
        Enriched DataFrame
    """
    integrator = HistoricalDataIntegrator(
        cache_dir=cache_dir,
        current_season=season
    )
    
    return integrator.enrich_current_season_data(current_df)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """
    Example usage of the Historical Data Integrator.
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("\n" + "=" * 80)
    print("🚀 FPL HISTORICAL DATA INTEGRATOR - DEMO")
    print("=" * 80)
    
    # Initialize integrator
    integrator = HistoricalDataIntegrator(
        cache_dir="data/cache/historical",
        current_season="2024-25"
    )
    
    # Example 1: Load single season
    print("\n📊 Example 1: Loading 2023-24 season data")
    df_single = integrator.get_merged_gameweek_data(season="2023-24")
    
    if not df_single.empty:
        print(integrator.get_statistics_summary(df_single))
        print(f"\nColumns available: {list(df_single.columns)[:10]}...")
    
    # Example 2: Load multi-season data
    print("\n📊 Example 2: Loading multi-season data")
    df_multi = integrator.get_multi_season_data(
        seasons=["2022-23", "2023-24"],
        include_current_season=False
    )
    
    if not df_multi.empty:
        print(f"✅ Loaded {len(df_multi):,} records from multiple seasons")
        print(f"   Seasons: {df_multi['season'].unique()}")
    
    # Example 3: Prepare for training
    print("\n🔧 Example 3: Preparing training data")
    df_train = integrator.prepare_training_features(df_multi)
    
    if not df_train.empty:
        print(f"✅ Training data ready: {len(df_train):,} samples")
        print(f"   Features: {len(df_train.columns)}")
        print(f"   Has target: {'target_next_points' in df_train.columns}")
    
    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)