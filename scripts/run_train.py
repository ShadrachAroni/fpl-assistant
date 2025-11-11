"""
Training Script - Trains LightGBM model for FPL point predictions
Run with: python -m scripts.run_train

ENHANCED VERSION: Comprehensive data validation and quality checks
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client.fpl_client import FPLClient
from app.data.pipeline import DataPipeline
from app.models.trainer import train_lightgbm
import yaml

# ===== Configure Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("training_script")

# Suppress noisy HTTP logs
for noisy_logger in ["httpx", "urllib3", "asyncio", "fpl_assistant", "uvicorn.access"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def validate_data_quality(df):
    """
    Comprehensive data quality validation.
    
    Returns:
        tuple: (is_valid: bool, warnings: list, metrics: dict)
    """
    warnings = []
    metrics = {}
    is_valid = True
    
    # Basic metrics
    metrics['total_samples'] = len(df)
    metrics['unique_players'] = df['player_id'].nunique() if 'player_id' in df.columns else 0
    metrics['num_features'] = len(df.columns)
    
    # Gameweek coverage
    if 'gameweek' in df.columns:
        metrics['gameweeks_covered'] = df['gameweek'].nunique()
        metrics['min_gameweek'] = df['gameweek'].min()
        metrics['max_gameweek'] = df['gameweek'].max()
    else:
        warnings.append("No 'gameweek' column found")
    
    # Samples per player
    if 'player_id' in df.columns:
        samples_per_player = df.groupby('player_id').size()
        metrics['avg_samples_per_player'] = samples_per_player.mean()
        metrics['min_samples_per_player'] = samples_per_player.min()
        metrics['max_samples_per_player'] = samples_per_player.max()
        
        if metrics['min_samples_per_player'] < 3:
            warnings.append(f"Some players have <3 samples (min: {metrics['min_samples_per_player']})")
    
    # Check for critical features
    critical_features = [
        'form', 'minutes', 'total_points', 'selected_by_percent',
        'opponent_difficulty', 'is_home'
    ]
    missing_critical = [f for f in critical_features if f not in df.columns]
    if missing_critical:
        warnings.append(f"Missing critical features: {', '.join(missing_critical)}")
        metrics['missing_critical_features'] = missing_critical
    
    # Check target variable
    if 'total_points' in df.columns:
        metrics['points_mean'] = df['total_points'].mean()
        metrics['points_std'] = df['total_points'].std()
        metrics['points_min'] = df['total_points'].min()
        metrics['points_max'] = df['total_points'].max()
        
        if df['total_points'].isnull().any():
            warnings.append("Target variable 'total_points' has missing values")
    else:
        warnings.append("Target variable 'total_points' not found")
        is_valid = False
    
    # Missing value analysis
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 20]
    
    if len(high_missing) > 0:
        warnings.append(f"{len(high_missing)} features have >20% missing values")
        metrics['high_missing_features'] = high_missing.to_dict()
    
    metrics['avg_missing_pct'] = missing_pct.mean()
    
    # Check for sufficient data
    if metrics['total_samples'] < 2000:
        warnings.append(f"Low sample count: {metrics['total_samples']} (recommended: 2000+)")
        is_valid = False
    
    if metrics['unique_players'] < 200:
        warnings.append(f"Low player count: {metrics['unique_players']} (recommended: 200+)")
        is_valid = False
    
    if 'gameweeks_covered' in metrics and metrics['gameweeks_covered'] < 5:
        warnings.append(f"Limited history: {metrics['gameweeks_covered']} GWs (recommended: 5+)")
    
    return is_valid, warnings, metrics


def log_data_quality_report(metrics, warnings):
    """Pretty print data quality report."""
    logger.info("\n" + "=" * 80)
    logger.info("📊 DATA QUALITY REPORT")
    logger.info("=" * 80)
    
    # Dataset size
    logger.info("\n📈 Dataset Size:")
    logger.info(f"   Total samples:        {metrics['total_samples']:,}")
    logger.info(f"   Unique players:       {metrics['unique_players']:,}")
    logger.info(f"   Features:             {metrics['num_features']}")
    
    # Temporal coverage
    if 'gameweeks_covered' in metrics:
        logger.info(f"\n📅 Temporal Coverage:")
        logger.info(f"   Gameweeks covered:    {metrics['gameweeks_covered']}")
        logger.info(f"   GW range:             {metrics['min_gameweek']} - {metrics['max_gameweek']}")
    
    # Per-player statistics
    if 'avg_samples_per_player' in metrics:
        logger.info(f"\n👤 Per-Player Statistics:")
        logger.info(f"   Avg samples/player:   {metrics['avg_samples_per_player']:.1f}")
        logger.info(f"   Min samples/player:   {metrics['min_samples_per_player']}")
        logger.info(f"   Max samples/player:   {metrics['max_samples_per_player']}")
    
    # Target variable
    if 'points_mean' in metrics:
        logger.info(f"\n🎯 Target Variable (total_points):")
        logger.info(f"   Mean:                 {metrics['points_mean']:.2f}")
        logger.info(f"   Std Dev:              {metrics['points_std']:.2f}")
        logger.info(f"   Range:                [{metrics['points_min']:.0f}, {metrics['points_max']:.0f}]")
    
    # Data quality
    logger.info(f"\n✨ Data Quality:")
    logger.info(f"   Avg missing %:        {metrics['avg_missing_pct']:.2f}%")
    
    if 'missing_critical_features' in metrics:
        logger.info(f"   Missing critical:     {', '.join(metrics['missing_critical_features'])}")
    
    # Warnings
    if warnings:
        logger.info("\n⚠️  WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            logger.info(f"   {i}. {warning}")
    else:
        logger.info("\n✅ No data quality issues detected")
    
    logger.info("=" * 80)


def determine_cv_splits(n_samples, n_players):
    """Determine appropriate number of CV splits based on data size."""
    if n_samples < 500:
        return 3
    elif n_samples < 1500:
        return 4
    else:
        return 5


def main():
    """Main training pipeline with comprehensive validation."""
    logger.info("=" * 80)
    logger.info("🚀 FPL ASSISTANT - MODEL TRAINING")
    logger.info("=" * 80)

    # ===== LOAD CONFIG =====
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        logger.error(f"❌ Config not found at {config_path}")
        return

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    logger.info(f"✅ Loaded config from {config_path}")

    # ===== INITIALIZE FPL CLIENT =====
    logger.info("\n📡 STEP 1: Initializing FPL Client...")

    try:
        fpl_client = FPLClient()
        logger.info("✅ FPLClient initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize FPLClient: {e}")
        return

    # ===== FETCH DATA =====
    logger.info("\n📡 STEP 2: Fetching bootstrap data...")

    try:
        pipeline = DataPipeline(config, fpl_client=fpl_client)
        logger.info("✅ Pipeline initialized")

        bootstrap = pipeline.fetch_bootstrap()

        if not bootstrap:
            logger.error("❌ Failed to fetch bootstrap")
            return

        players_count = len(bootstrap.get("elements", []))
        teams_count = len(bootstrap.get("teams", []))
        logger.info(f"✅ Bootstrap loaded: {players_count} players, {teams_count} teams")

    except Exception as e:
        logger.error(f"❌ Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return

    # ===== BUILD TRAINING FEATURES =====
    logger.info("\n🔧 STEP 3: Building training features...")
    logger.info("   This may take a few minutes...")

    try:
        rolling_window = config.get("training", {}).get("rolling_window_gw", 4)
        logger.info(f"   Using rolling window: {rolling_window} gameweeks")
        
        df = pipeline.build_train_features(rolling_window=rolling_window)

        if df.empty:
            logger.error("❌ No training data could be generated")
            logger.error("   Possible causes:")
            logger.error("   - FPL API element_summary endpoint unavailable")
            logger.error("   - Network connectivity issues")
            logger.error("   - Insufficient historical data")
            logger.error("   - Authentication required but not provided")
            return

        logger.info(f"✅ Training data generated: {len(df):,} records")

    except Exception as e:
        logger.error(f"❌ Failed to build features: {e}")
        import traceback
        traceback.print_exc()
        return

    # ===== VALIDATE DATA QUALITY =====
    logger.info("\n🔍 STEP 4: Validating data quality...")

    try:
        is_valid, warnings, metrics = validate_data_quality(df)
        log_data_quality_report(metrics, warnings)

        if not is_valid:
            logger.error("\n❌ DATA VALIDATION FAILED")
            logger.error("   Training cannot proceed with insufficient data")
            logger.error("   Recommendations:")
            logger.error("   - Verify FPL API access and credentials")
            logger.error("   - Ensure sufficient gameweek history exists")
            logger.error("   - Check network connectivity")
            logger.error("   - Review pipeline logs for errors")
            return

        if warnings:
            logger.warning("\n⚠️  Data quality warnings detected")
            logger.warning("   Model may have reduced accuracy")
            logger.warning("   Consider addressing warnings before production use")
            
            # Ask for confirmation if warnings exist
            response = input("\n   Continue training anyway? (y/n): ")
            if response.lower() != 'y':
                logger.info("   Training cancelled by user")
                return

    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ===== TRAIN MODEL =====
    logger.info("\n🤖 STEP 5: Training LightGBM model...")
    logger.info("   This will take several minutes...")

    try:
        # Determine optimal CV splits
        n_splits_config = config.get("training", {}).get("n_splits_cv", 5)
        n_splits = determine_cv_splits(len(df), metrics['unique_players'])
        
        if n_splits != n_splits_config:
            logger.info(f"   Adjusted CV splits: {n_splits_config} → {n_splits} (based on data size)")
        else:
            logger.info(f"   Cross-validation splits: {n_splits}")
        
        result = train_lightgbm(df, n_splits=n_splits)

        # ===== TRAINING RESULTS =====
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 Model Performance:")
        logger.info(f"   Out-of-Fold RMSE:     {result.get('oof_rmse', 0):.4f}")
        logger.info(f"   Out-of-Fold MAE:      {result.get('oof_mae', 0):.4f}")
        logger.info(f"   Out-of-Fold R²:       {result.get('oof_r2', 0):.4f}")
        
        logger.info(f"\n📈 Model Details:")
        logger.info(f"   Features used:        {result.get('n_features', 0)}")
        logger.info(f"   Training samples:     {metrics['total_samples']:,}")
        logger.info(f"   CV folds:             {n_splits}")
        
        # Feature importance summary
        if result.get('feature_importance'):
            logger.info(f"\n🔝 Top 5 Features:")
            for i, feat in enumerate(result['feature_importance'][:5], 1):
                logger.info(f"   {i}. {feat['feature']}: {feat['importance']:.1f}")
            
            # Check for risk features
            risk_features = [f for f in ['total_risk', 'injury_risk', 'rotation_risk', 'disciplinary_risk']
                           if any(f in feat['feature'] for feat in result['feature_importance'])]
            if risk_features:
                logger.info(f"   ✅ Risk features included: {', '.join(risk_features)}")
        
        logger.info(f"\n💾 Output:")
        logger.info(f"   Model saved to:       models/lightgbm_model.joblib")
        
        # Performance assessment
        oof_rmse = result.get('oof_rmse', 999)
        if oof_rmse < 2.0:
            logger.info(f"\n🎉 Excellent model performance (RMSE < 2.0)")
        elif oof_rmse < 3.0:
            logger.info(f"\n👍 Good model performance (RMSE < 3.0)")
        elif oof_rmse < 4.0:
            logger.info(f"\n⚠️  Acceptable performance (RMSE < 4.0) - consider more data/features")
        else:
            logger.warning(f"\n⚠️  Below target performance (RMSE > 4.0)")
            logger.warning(f"   Consider:")
            logger.warning(f"   - Gathering more training data")
            logger.warning(f"   - Adding more predictive features")
            logger.warning(f"   - Tuning hyperparameters")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 MODEL TRAINING PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info("   You can now run the API server with:")
        logger.info("   python -m uvicorn app.main:app --reload")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)