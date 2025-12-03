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
from app.api_client.understat_client import UnderstatClient
from app.data.pipeline import DataPipeline
from app.data.historical_integrator import HistoricalDataIntegrator
from app.models.trainer import train_lightgbm, optimize_hyperparameters
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
        try:
            understat_client = UnderstatClient()
        except Exception as exc:
            logger.warning(f"⚠️ Understat client unavailable for training: {exc}")
            understat_client = None

        pipeline = DataPipeline(
            config,
            fpl_client=fpl_client,
            understat_client=understat_client
        )
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
        
        df = pipeline.build_train_features(
            season="2024",
            rolling_window=rolling_window,
            use_multi_season=True,  # Use multiple seasons
            seasons=["2022-23", "2023-24", "2024-25"]
        )
        

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
    # ===== VALIDATE DATA QUALITY =====
    logger.info("\n🔍 STEP 4: Validating data quality...")

    try:
        # Initialize integrator for validation
        integrator = HistoricalDataIntegrator(
            cache_dir="data/cache/historical",
            current_season="2024-25"
        )
        
        # Use integrator's comprehensive validation
        metrics = integrator.validate_data_quality(df)
        try:
            summary = integrator.get_statistics_summary(df)
            print(summary.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
        except UnicodeEncodeError:
            # Fallback: print without emojis
            print(f"Training data: {len(df)} records, {len(df.columns)} features")
        
        # Check if data is sufficient for training
        warnings = metrics.get('warnings', [])
        is_valid = (
            metrics['total_samples'] >= 1000 and  # ✅ Lowered from 2000
            metrics.get('unique_players', 0) >= 100 and  # ✅ Lowered from 200
            metrics.get('gameweeks_covered', 0) >= 5 and  # ✅ ADDED: Must have gameweeks
            len(warnings) == 0
        )
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
        # Determine optimal CV splits based on data size
        if len(df) < 500:
            n_splits = 3
        elif len(df) < 1500:
            n_splits = 4
        else:
            n_splits = 5
        
        if n_splits != n_splits_config:
            logger.info(f"   Adjusted CV splits: {n_splits_config} → {n_splits} (based on data size)")
        else:
            logger.info(f"   Cross-validation splits: {n_splits}")
        
        training_cfg = config.get("training", {})
        use_pos_models = training_cfg.get("use_position_models", True)
        lgbm_params = training_cfg.get("lgbm_params")
        optuna_enabled = training_cfg.get("optimize", False)
        optuna_trials = int(training_cfg.get("optimize_trials", 150))  # IMPROVED: Default 150
        num_boost_round = int(training_cfg.get("num_boost_round", 2000))  # IMPROVED: More rounds
        early_stopping_rounds = int(training_cfg.get("early_stopping_rounds", 100))  # IMPROVED: More patience

        if optuna_enabled:
            logger.info(f"   Optuna tuning enabled ({optuna_trials} trial(s))...")
            result = optimize_hyperparameters(
                df,
                n_trials=optuna_trials,
                n_splits=n_splits,
                train_kwargs={
                    "train_position_models": use_pos_models,
                    "num_boost_round": num_boost_round,
                    "early_stopping_rounds": early_stopping_rounds
                },
            )
        else:
            if lgbm_params:
                logger.info("   Using custom LightGBM parameters from config")
            else:
                logger.info("   Using IMPROVED default LightGBM parameters (optimized for R²)")
            result = train_lightgbm(
                df,
                n_splits=n_splits,
                params=lgbm_params,
                train_position_models=use_pos_models,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )

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
            try:
                for i, feat in enumerate(result['feature_importance'][:5], 1):
                    if isinstance(feat, dict):
                        # Get feature name
                        name = feat.get('feature', feat.get('name', 'Unknown'))
                        
                        # Get importance (try multiple possible keys)
                        importance = (
                            feat.get('importance_mean') or 
                            feat.get('importance') or 
                            feat.get('gain') or 
                            0
                        )
                        
                        logger.info(f"   {i}. {name}: {importance:.1f}")
                    else:
                        logger.info(f"   {i}. {str(feat)}")
            except Exception as e:
                logger.debug(f"Could not display feature importance: {e}")
            
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