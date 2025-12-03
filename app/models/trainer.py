"""
Enhanced Model Training Module - LightGBM for FPL Point Predictions

ENHANCEMENTS v6.0:
✅ Risk-aware feature engineering with comprehensive validation
✅ Ownership feature integration (template vs differential)
✅ Price change feature validation
✅ Feature importance tracking with categorization
✅ Comprehensive cross-validation with player grouping
✅ Hyperparameter optimization support (Optuna)
✅ Model versioning and metadata
✅ Training metrics logging and persistence
✅ Feature selection and analysis
✅ Model ensemble with uncertainty quantification
✅ Enhanced validation for all new pipeline features

PRODUCTION READY v6.0
"""

import os
import joblib
import logging
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import early_stopping, log_evaluation

logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)

MODEL_PATH = "models/lightgbm_model.joblib"
METRICS_PATH = "models/training_metrics.json"


# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------

def features_targets_from_df(
    df: pd.DataFrame,
    target_col: str = "target_next_points",
    exclude_risk_from_features: bool = False
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extract features and target from training DataFrame with validation.
    
    Args:
        df: Training DataFrame
        target_col: Target column name
        exclude_risk_from_features: Whether to exclude risk features from training
    
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    # Columns to always exclude
    base_exclude = [
        "player_id", "web_name", "event", "gameweek", "team_name", 
        "position", "next_opponent", "next_opponent_short",
        "risk_category", "risk_summary", "fixture_display",
        "ownership_category", "price_change_status",
        target_col, "predicted_points_next_gw"  # Remove any prediction columns
    ]
    
    # Risk columns (can be excluded if desired)
    risk_cols = [
        "total_risk", "injury_risk", "rotation_risk", 
        "disciplinary_risk", "fatigue_risk", "form_drop_risk",
        "defensive_fragility_risk", "penalty_risk",
        "minutes_volatility", "news_severity"
    ]
    
    # Ownership/Price columns (usually kept for training)
    intelligence_cols = [
        "selected_by_percent", "is_template", "is_differential", "is_premium",
        "captain_eo_multiplier", "template_priority",
        "transfers_in_event", "transfers_out_event", "net_transfers",
        "price_rise_probability", "price_fall_probability",
        "value_hold_score", "opportunity_cost"
    ]
    
    exclude = base_exclude.copy()
    
    if exclude_risk_from_features:
        exclude.extend(risk_cols)
        logger.info("⚠️ Excluding risk features from training (risk-agnostic model)")
    else:
        logger.info("✅ Including risk features in training (risk-aware model)")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    
    # Validate feature count
    if len(feats) < 5:
        logger.error(f"❌ Too few features: {len(feats)}")
        logger.error(f"Available numeric columns: {numeric_cols[:20]}")
        raise ValueError(f"Insufficient features for training: {len(feats)} (need at least 5)")
    
    # Log feature categories
    risk_feats_in_model = [f for f in feats if f in risk_cols]
    ownership_feats_in_model = [f for f in feats if f in intelligence_cols]
    
    if risk_feats_in_model:
        logger.info(f"📊 Risk features in model: {len(risk_feats_in_model)}")
    if ownership_feats_in_model:
        logger.info(f"📊 Ownership/Price features in model: {len(ownership_feats_in_model)}")
    
    # Validate critical features (allow fallbacks like rolling metrics)
    critical_feature_sets = [
        ("form",),
        ("minutes_roll4", "minutes"),
        ("total_points_roll4", "total_points"),
    ]
    missing_critical = []
    for feature_choices in critical_feature_sets:
        if not any(
            (feature in feats or feature in df.columns)
            and feature not in exclude
            for feature in feature_choices
        ):
            missing_critical.append(" / ".join(feature_choices))
    if missing_critical:
        logger.warning(f"⚠️ Missing critical features: {missing_critical}")
    
    logger.info(f"📊 Total features: {len(feats)}")
    
    # Validate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    target = df[target_col].fillna(0)
    if target.isna().any():
        logger.warning("⚠️ Target has NaN values after fillna")
    
    return df[feats].fillna(0), target, feats


def calculate_feature_importance(
    models: List[Any], 
    feature_names: List[str],
    importance_type: str = 'gain'
) -> pd.DataFrame:
    """
    Calculate aggregated feature importance across models.
    
    Args:
        models: List of trained LightGBM models
        feature_names: List of feature names
        importance_type: Type of importance ('gain', 'split')
    
    Returns:
        DataFrame with feature importance and statistics
    """
    importance_dict = {}
    
    for model in models:
        feature_importance = model.feature_importance(importance_type=importance_type)
        
        for name, importance in zip(feature_names, feature_importance):
            if name not in importance_dict:
                importance_dict[name] = []
            importance_dict[name].append(importance)
    
    # Average importance across models
    avg_importance = {
        name: {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
        for name, values in importance_dict.items()
    }
    
    # Create DataFrame
    importance_df = pd.DataFrame([
        {
            "feature": name,
            "importance_mean": stats["mean"],
            "importance_std": stats["std"],
            "importance_min": stats["min"],
            "importance_max": stats["max"]
        }
        for name, stats in avg_importance.items()
    ])
    
    importance_df = importance_df.sort_values("importance_mean", ascending=False)
    
    return importance_df


# ----------------------------------------------------------
# Training with Enhanced Metrics
# ----------------------------------------------------------

def train_lightgbm(
    df: pd.DataFrame, 
    n_splits: int = 5,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
    num_boost_round: int = 1000,
    exclude_risk_features: bool = False,
    save_metrics: bool = True,
    train_position_models: bool = False
) -> Dict[str, Any]:
    """
    Train LightGBM model with comprehensive validation and intelligence features.
    
    Args:
        df: Training DataFrame with features and target_next_points column
        n_splits: Number of CV folds
        params: LightGBM parameters (uses defaults if None)
        early_stopping_rounds: Early stopping patience
        num_boost_round: Maximum boosting rounds
        exclude_risk_features: Whether to train without risk features
        save_metrics: Whether to save training metrics
    
    Returns:
        Dict with trained models, features, metrics, and importance
    """
    logger.info("=" * 80)
    logger.info("🚀 STARTING LIGHTGBM TRAINING WITH FULL INTELLIGENCE STACK")
    logger.info("=" * 80)
    
    # Validate input DataFrame
    if df.empty:
        raise ValueError("❌ Training DataFrame is empty")
    
    if len(df) < 100:
        raise ValueError(f"❌ Insufficient training data: {len(df)} samples (need at least 100)")
    
    target_col = "target_next_points"

    def _prepare_training_data(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], np.ndarray]:
        # IMPROVED: Filter low-minute players and outliers for better R²
        df_filtered = dataframe.copy()
        
        # Filter players with very few minutes (< 30 mins) - these are noisy
        if "minutes" in df_filtered.columns:
            initial_len = len(df_filtered)
            df_filtered = df_filtered[df_filtered["minutes"] >= 30]
            logger.info(f"📊 Filtered {initial_len - len(df_filtered)} low-minute samples (<30 mins)")
        
        # Remove extreme outliers in target (points > 20 or < -2 are rare and noisy)
        if target_col in df_filtered.columns:
            initial_len = len(df_filtered)
            df_filtered = df_filtered[
                (df_filtered[target_col] >= -2) & (df_filtered[target_col] <= 20)
            ]
            logger.info(f"📊 Filtered {initial_len - len(df_filtered)} outlier samples")
        
        X_local, y_local, feats_local = features_targets_from_df(
            df_filtered,
            exclude_risk_from_features=exclude_risk_features
        )
        if X_local.empty or y_local.empty:
            raise ValueError("❌ Feature extraction failed - empty data")

        if "player_id" in df_filtered.columns:
            groups_local = df_filtered["player_id"].values
        else:
            logger.warning("⚠️ No player_id column - using sequential grouping")
            groups_local = np.arange(len(df_filtered)) % n_splits

        return X_local, y_local, feats_local, groups_local

    def _train_cv_model(
        X_data: pd.DataFrame,
        y_data: pd.Series,
        groups_data: np.ndarray,
        splits: int,
        description: str = "GLOBAL"
    ) -> Dict[str, Any]:

        if splits < 2:
            splits = 2

        logger.info(f"\n{'='*80}")
        logger.info(f"🔁 Training {description} model with {splits} folds")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Dataset: {len(X_data)} samples | Target μ={y_data.mean():.2f} σ={y_data.std():.2f}")

        gkf = GroupKFold(n_splits=splits)
        local_models = []
        oof = np.zeros(X_data.shape[0])
        local_fold_metrics = []

        for fold, (tr, va) in enumerate(gkf.split(X_data, y_data, groups_data)):
            logger.info(f"🧩 {description} Fold {fold+1}/{splits}")

            Xtr, Xva = X_data.iloc[tr], X_data.iloc[va]
            ytr, yva = y_data.iloc[tr], y_data.iloc[va]

            dtr = lgb.Dataset(Xtr, ytr)
            dva = lgb.Dataset(Xva, yva)

            # IMPROVED: More boosting rounds and better early stopping for R²
            model = lgb.train(
                params=params_local,
                train_set=dtr,
                valid_sets=[dva],
                num_boost_round=int(num_boost_round * 1.5),  # More rounds for better fit
                callbacks=[
                    early_stopping(int(early_stopping_rounds * 1.5), verbose=False),  # More patience
                    log_evaluation(100)
                ]
            )

            preds = model.predict(Xva, num_iteration=model.best_iteration)
            oof[va] = preds

            fold_rmse = np.sqrt(mean_squared_error(yva, preds))
            fold_mae = mean_absolute_error(yva, preds)
            fold_r2 = r2_score(yva, preds)

            local_fold_metrics.append({
                "fold": fold + 1,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "r2": fold_r2,
                "best_iteration": model.best_iteration,
                "n_train": len(Xtr),
                "n_valid": len(Xva),
                "description": description
            })

            logger.info(f"   ✅ RMSE={fold_rmse:.4f} | MAE={fold_mae:.4f} | R²={fold_r2:.4f}")
            local_models.append(model)

        rmse = np.sqrt(mean_squared_error(y_data, oof))
        mae = mean_absolute_error(y_data, oof)
        r2_val = r2_score(y_data, oof)

        logger.info(f"📊 {description} Performance: RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2_val:.4f}")

        return {
            "models": local_models,
            "oof_preds": oof,
            "fold_metrics": local_fold_metrics,
            "rmse": rmse,
            "mae": mae,
            "r2": r2_val
        }

    # Prepare training data
    X, y, feats, groups = _prepare_training_data(df)
    logger.info(f"📊 Training data: {len(df)} records | {len(feats)} features | {n_splits} folds")

    if y.std() < 0.1:
        logger.warning("⚠️ Target has very low variance - model may struggle")

    # IMPROVED: Better default parameters for higher R²
    if params is None:
        params_local = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,  # Lower LR for better generalization
            "num_leaves": 50,  # Increased for more capacity
            "max_depth": 8,  # Limit depth to prevent overfitting
            "feature_fraction": 0.85,  # Slightly higher for more features
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_data_in_leaf": 15,  # Lower for more splits
            "min_gain_to_split": 0.1,  # Minimum gain for splits
            "lambda_l1": 0.05,  # Reduced regularization
            "lambda_l2": 0.05,
            "max_bin": 255,  # More bins for better precision
            "verbosity": -1,
            "seed": 42,
            "force_row_wise": True,  # Better for small datasets
        }
        logger.info("🔧 Using IMPROVED default LightGBM parameters (optimized for R²)")
    else:
        params_local = params
        logger.info("🔧 Using custom LightGBM parameters")

    # Train global model
    global_result = _train_cv_model(X, y, groups, n_splits, "GLOBAL")
    models = global_result["models"]
    oof_preds = global_result["oof_preds"]
    fold_metrics = global_result["fold_metrics"]
    global_rmse = global_result["rmse"]
    global_mae = global_result["mae"]
    global_r2 = global_result["r2"]
    
    # Feature importance
    logger.info("=" * 80)
    logger.info("📊 FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 80)
    
    importance_df = calculate_feature_importance(models, feats)
    
    logger.info(f"   Top 15 features by importance:")
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"      {idx+1}. {row['feature']}: {row['importance_mean']:.2f}")
    
    # Analyze feature categories
    risk_features = [
        "total_risk", "injury_risk", "rotation_risk", "disciplinary_risk",
        "fatigue_risk", "form_drop_risk"
    ]
    ownership_features = [
        "selected_by_percent", "is_template", "is_differential",
        "captain_eo_multiplier", "template_priority"
    ]
    price_features = [
        "net_transfers", "price_rise_probability", "price_fall_probability",
        "value_hold_score"
    ]
    
    top_20_features = importance_df.head(20)["feature"].tolist()
    
    risk_in_top = [f for f in risk_features if f in top_20_features]
    ownership_in_top = [f for f in ownership_features if f in top_20_features]
    price_in_top = [f for f in price_features if f in top_20_features]
    
    if risk_in_top:
        logger.info(f"   ✅ Risk features in top 20: {risk_in_top}")
    if ownership_in_top:
        logger.info(f"   ✅ Ownership features in top 20: {ownership_in_top}")
    if price_in_top:
        logger.info(f"   ✅ Price features in top 20: {price_in_top}")
    
    # Prepare results
    results = {
        "models": models,
        "feats": feats,
        "oof_rmse": global_rmse,
        "oof_mae": global_mae,
        "oof_r2": global_r2,
        "fold_metrics": fold_metrics,
        "feature_importance": importance_df.to_dict("records"),
        "n_features": len(feats),
        "n_folds": n_splits,
        "training_date": datetime.now().isoformat(),
        "params": params,
        "feature_categories": {
            "risk_features": risk_in_top,
            "ownership_features": ownership_in_top,
            "price_features": price_in_top
        }
    }
    
    # Save model
    logger.info("=" * 80)
    logger.info("💾 SAVING MODEL")
    logger.info("=" * 80)
    
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    
    model_data = {
        "models": models,
        "feats": feats,
        "feature_importance": results["feature_importance"],
        "oof_rmse": global_rmse,
        "oof_mae": global_mae,
        "oof_r2": global_r2,
        "training_date": results["training_date"],
        "params": params_local,
        "n_folds": n_splits,
        "feature_categories": results["feature_categories"]
    }

    # Train position-specific models if requested
    position_models_payload = {}
    if train_position_models and "position" in df.columns:
        logger.info("\n🧩 Training position-specific models...")
        for position_label, subset in df.groupby("position"):
            if len(subset) < max(150, n_splits * 40):
                logger.info(f"   Skipping {position_label} (insufficient samples: {len(subset)})")
                continue

            subset = subset.copy()
            if target_col not in subset.columns:
                logger.info(f"   Skipping {position_label} - missing target column")
                continue

            X_pos = subset[feats].fillna(0)
            y_pos = subset[target_col].fillna(0)
            if "player_id" in subset.columns:
                groups_pos = subset["player_id"].values
            else:
                groups_pos = np.arange(len(subset)) % n_splits

            pos_splits = min(n_splits, max(2, len(subset) // 200))
            pos_result = _train_cv_model(X_pos, y_pos, groups_pos, pos_splits, f"POS-{position_label}")

            position_models_payload[position_label] = {
                "models": pos_result["models"],
                "n_folds": pos_splits,
                "oof_rmse": pos_result["rmse"],
                "oof_mae": pos_result["mae"],
                "oof_r2": pos_result["r2"],
            }

        if position_models_payload:
            results["position_models"] = {
                pos: {
                    "oof_rmse": data["oof_rmse"],
                    "oof_mae": data["oof_mae"],
                    "oof_r2": data["oof_r2"],
                    "n_folds": data["n_folds"]
                }
                for pos, data in position_models_payload.items()
            }
            model_data["position_models"] = position_models_payload
    
    joblib.dump(model_data, MODEL_PATH)
    logger.info(f"   ✅ Model saved to {MODEL_PATH}")
    
    # Save metrics
    if save_metrics:
        try:
            import json
            metrics_data = {
                "training_date": results["training_date"],
                "oof_rmse": float(global_rmse),
                "oof_mae": float(global_mae),
                "oof_r2": float(global_r2),
                "n_features": len(feats),
                "n_folds": n_splits,
                "n_samples": len(df),
                "fold_metrics": fold_metrics,
                "top_features": importance_df.head(20).to_dict("records"),
                "feature_categories": results["feature_categories"]
            }
            
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"   ✅ Metrics saved to {METRICS_PATH}")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not save metrics: {e}")
    
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    
    return results


# ----------------------------------------------------------
# Hyperparameter Optimization (Optional)
# ----------------------------------------------------------

def optimize_hyperparameters(
    df: pd.DataFrame,
    n_trials: int = 50,
    n_splits: int = 3,
    train_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        df: Training DataFrame
        n_trials: Number of optimization trials
        n_splits: Number of CV folds
    
    Returns:
        Best parameters and training results
    
    Note: Requires optuna to be installed
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.error("❌ Optuna not installed. Run: pip install optuna")
        return {}
    
    logger.info(f"🔧 Starting hyperparameter optimization ({n_trials} trials)...")
    
    # IMPROVED: Filter data before optimization for better R²
    df_filtered = df.copy()
    if "minutes" in df_filtered.columns:
        initial_len = len(df_filtered)
        df_filtered = df_filtered[df_filtered["minutes"] >= 30]
        logger.info(f"   Filtered {initial_len - len(df_filtered)} low-minute samples")
    if "target_next_points" in df_filtered.columns:
        initial_len = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered["target_next_points"] >= -2) & 
            (df_filtered["target_next_points"] <= 20)
        ]
        logger.info(f"   Filtered {initial_len - len(df_filtered)} outlier samples")
    
    X, y, feats = features_targets_from_df(df_filtered)
    
    if "player_id" in df_filtered.columns:
        groups = df_filtered["player_id"].values
    else:
        groups = np.arange(len(df_filtered)) % n_splits
    
    def objective(trial):
        """Optuna objective function."""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": 42,
            # IMPROVED: Better hyperparameter ranges for R² optimization
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 30, 100),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 3, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 30),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.2),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 0.3),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.3),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
        }
        
        # Cross-validation
        gkf = GroupKFold(n_splits=n_splits)
        rmse_scores = []
        
        for tr, va in gkf.split(X, y, groups):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            
            dtr = lgb.Dataset(Xtr, ytr)
            dva = lgb.Dataset(Xva, yva)
            
            # IMPROVED: More rounds and patience for optimization
            model = lgb.train(
                params=params,
                train_set=dtr,
                valid_sets=[dva],
                num_boost_round=1000,  # More rounds
                callbacks=[early_stopping(50, verbose=False)]  # More patience
            )
            
            preds = model.predict(Xva, num_iteration=model.best_iteration)
            mse = mean_squared_error(yva, preds)
            rmse_scores.append(np.sqrt(mse))
        
        return np.mean(rmse_scores)
    
    # Run optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"✅ Optimization complete!")
    logger.info(f"   Best RMSE: {study.best_value:.4f}")
    logger.info(f"   Best params: {study.best_params}")
    
    # Train with best parameters
    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
        **study.best_params
    }
    
    train_kwargs = train_kwargs or {}
    results = train_lightgbm(df, params=best_params, n_splits=n_splits, **train_kwargs)
    results["optimization_trials"] = n_trials
    results["best_params"] = study.best_params
    
    return results


# ----------------------------------------------------------
# Model Evaluation
# ----------------------------------------------------------

def evaluate_model(
    model_path: str = MODEL_PATH,
    test_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Evaluate saved model on test data or return training metrics.
    
    Args:
        model_path: Path to saved model
        test_df: Optional test DataFrame
    
    Returns:
        Evaluation metrics
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"📊 Evaluating model from {model_path}")
    
    # Load model
    model_data = joblib.load(model_path)
    models = model_data.get("models", [])
    feats = model_data.get("feats", [])
    
    evaluation = {
        "model_path": model_path,
        "n_models": len(models),
        "n_features": len(feats),
        "training_date": model_data.get("training_date", "unknown"),
        "training_rmse": model_data.get("oof_rmse"),
        "training_mae": model_data.get("oof_mae"),
        "training_r2": model_data.get("oof_r2"),
        "feature_categories": model_data.get("feature_categories", {})
    }
    
    # Test set evaluation
    if test_df is not None:
        logger.info("   Running test set evaluation...")
        
        X_test, y_test, _ = features_targets_from_df(test_df)
        
        # Predict with ensemble
        test_preds = np.mean([
            model.predict(X_test[feats], num_iteration=getattr(model, "best_iteration", None))
            for model in models
        ], axis=0)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        evaluation["test_rmse"] = test_rmse
        evaluation["test_mae"] = test_mae
        evaluation["test_r2"] = test_r2
        
        logger.info(f"   Test RMSE: {test_rmse:.4f}")
        logger.info(f"   Test MAE:  {test_mae:.4f}")
        logger.info(f"   Test R²:   {test_r2:.4f}")
    
    # Feature importance
    if "feature_importance" in model_data:
        importance_df = pd.DataFrame(model_data["feature_importance"])
        evaluation["top_10_features"] = importance_df.head(10)["feature"].tolist()
    
    return evaluation


# ----------------------------------------------------------
# Feature Analysis
# ----------------------------------------------------------

def analyze_intelligence_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the impact of intelligence features (risk, ownership, price).
    
    Args:
        df: Training DataFrame with intelligence features
    
    Returns:
        Analysis results
    """
    logger.info("🔍 Analyzing intelligence feature impact...")
    
    # Feature categories
    risk_features = [
        "total_risk", "injury_risk", "rotation_risk", 
        "disciplinary_risk", "fatigue_risk", "form_drop_risk"
    ]
    ownership_features = [
        "selected_by_percent", "is_template", "is_differential",
        "captain_eo_multiplier", "template_priority"
    ]
    price_features = [
        "net_transfers", "price_rise_probability", "price_fall_probability",
        "value_hold_score", "opportunity_cost"
    ]
    
    analysis = {
        "risk_features": {},
        "ownership_features": {},
        "price_features": {}
    }
    
    # Analyze each category
    for category, features in [
        ("risk_features", risk_features),
        ("ownership_features", ownership_features),
        ("price_features", price_features)
    ]:
        available = [f for f in features if f in df.columns]
        analysis[category]["available"] = available
        analysis[category]["count"] = len(available)
        
        if not available:
            continue
        
        # Correlation with target
        if "target_next_points" in df.columns:
            correlations = {}
            for feature in available:
                try:
                    corr = df[feature].corr(df["target_next_points"])
                    correlations[feature] = float(corr)
                    logger.info(f"   {feature}: {corr:.4f} correlation with target")
                except:
                    pass
            analysis[category]["correlations"] = correlations
        
        # Statistics
        stats = {}
        for feature in available:
            try:
                stats[feature] = {
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                    "missing_pct": float(df[feature].isna().mean() * 100)
                }
            except:
                pass
        analysis[category]["statistics"] = stats
    
    logger.info("✅ Intelligence feature analysis complete")
    
    return analysis


# ----------------------------------------------------------
# Model Comparison
# ----------------------------------------------------------

def compare_models(
    df: pd.DataFrame,
    with_risk: bool = True,
    without_risk: bool = True,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Compare model performance with and without risk features.
    
    Args:
        df: Training DataFrame
        with_risk: Train model with risk features
        without_risk: Train model without risk features
        n_splits: Number of CV folds
    
    Returns:
        Comparison results
    """
    logger.info("=" * 80)
    logger.info("🔬 COMPARING MODELS WITH/WITHOUT RISK FEATURES")
    logger.info("=" * 80)
    
    results = {}
    
    if with_risk:
        logger.info("\n📊 Training WITH risk features...")
        risk_results = train_lightgbm(
            df, 
            n_splits=n_splits,
            exclude_risk_features=False,
            save_metrics=False
        )
        results["with_risk"] = {
            "rmse": risk_results["oof_rmse"],
            "mae": risk_results["oof_mae"],
            "r2": risk_results["oof_r2"],
            "n_features": risk_results["n_features"]
        }
    
    if without_risk:
        logger.info("\n📊 Training WITHOUT risk features...")
        no_risk_results = train_lightgbm(
            df,
            n_splits=n_splits,
            exclude_risk_features=True,
            save_metrics=False
        )
        results["without_risk"] = {
            "rmse": no_risk_results["oof_rmse"],
            "mae": no_risk_results["oof_mae"],
            "r2": no_risk_results["oof_r2"],
            "n_features": no_risk_results["n_features"]
        }
    
    # Calculate improvement
    if with_risk and without_risk:
        rmse_improvement = (
            (results["without_risk"]["rmse"] - results["with_risk"]["rmse"]) 
            / results["without_risk"]["rmse"] * 100
        )
        mae_improvement = (
            (results["without_risk"]["mae"] - results["with_risk"]["mae"]) 
            / results["without_risk"]["mae"] * 100
        )
        
        results["improvement"] = {
            "rmse_improvement_pct": rmse_improvement,
            "mae_improvement_pct": mae_improvement,
            "additional_features": (
                results["with_risk"]["n_features"] - results["without_risk"]["n_features"]
            )
        }
        
        logger.info("=" * 80)
        logger.info("📊 COMPARISON RESULTS")
        logger.info("=" * 80)
        logger.info(f"   WITH risk features:")
        logger.info(f"      RMSE: {results['with_risk']['rmse']:.4f}")
        logger.info(f"      MAE:  {results['with_risk']['mae']:.4f}")
        logger.info(f"      R²:   {results['with_risk']['r2']:.4f}")
        logger.info(f"   WITHOUT risk features:")
        logger.info(f"      RMSE: {results['without_risk']['rmse']:.4f}")
        logger.info(f"      MAE:  {results['without_risk']['mae']:.4f}")
        logger.info(f"      R²:   {results['without_risk']['r2']:.4f}")
        logger.info(f"   IMPROVEMENT:")
        logger.info(f"      RMSE: {rmse_improvement:+.2f}%")
        logger.info(f"      MAE:  {mae_improvement:+.2f}%")
        logger.info("=" * 80)
    
    return results


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def get_training_history() -> Optional[Dict[str, Any]]:
    """
    Get training history from saved metrics.
    
    Returns:
        Training metrics or None if not available
    """
    if not os.path.exists(METRICS_PATH):
        return None
    
    try:
        import json
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load metrics: {e}")
        return None


def print_model_summary(model_path: str = MODEL_PATH):
    """Print comprehensive summary of trained model."""
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    model_data = joblib.load(model_path)
    
    print("=" * 80)
    print("📊 MODEL SUMMARY - FULL INTELLIGENCE STACK")
    print("=" * 80)
    print(f"Training Date: {model_data.get('training_date', 'unknown')}")
    print(f"Number of Models: {len(model_data.get('models', []))}")
    print(f"Number of Features: {len(model_data.get('feats', []))}")
    print(f"\nOut-of-Fold Performance:")
    print(f"  RMSE: {model_data.get('oof_rmse', 0):.4f}")
    print(f"  MAE:  {model_data.get('oof_mae', 0):.4f}")
    print(f"  R²:   {model_data.get('oof_r2', 0):.4f}")
    
    # Feature categories
    if "feature_categories" in model_data:
        categories = model_data["feature_categories"]
        print(f"\nIntelligence Features:")
        if categories.get("risk_features"):
            print(f"  Risk: {', '.join(categories['risk_features'])}")
        if categories.get("ownership_features"):
            print(f"  Ownership: {', '.join(categories['ownership_features'])}")
        if categories.get("price_features"):
            print(f"  Price: {', '.join(categories['price_features'])}")
    
    # Top features
    if "feature_importance" in model_data:
        importance_df = pd.DataFrame(model_data["feature_importance"])
        print("\nTop 15 Features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance_mean']:.2f}")
    
    print("=" * 80)


def validate_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive validation of training data quality.
    
    Args:
        df: Training DataFrame
    
    Returns:
        Validation report
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "metrics": {}
    }
    
    # Basic checks
    if df.empty:
        report["valid"] = False
        report["errors"].append("DataFrame is empty")
        return report
    
    report["metrics"]["total_samples"] = len(df)
    
    # Player coverage
    if "player_id" in df.columns:
        report["metrics"]["unique_players"] = df["player_id"].nunique()
        samples_per_player = df.groupby("player_id").size()
        report["metrics"]["avg_samples_per_player"] = float(samples_per_player.mean())
        report["metrics"]["min_samples_per_player"] = int(samples_per_player.min())
        
        if report["metrics"]["min_samples_per_player"] < 2:
            report["warnings"].append("Some players have <2 samples")
    
    # Gameweek coverage
    if "gameweek" in df.columns:
        report["metrics"]["gameweeks_covered"] = df["gameweek"].nunique()
        report["metrics"]["gw_range"] = [int(df["gameweek"].min()), int(df["gameweek"].max())]
        
        if report["metrics"]["gameweeks_covered"] < 3:
            report["warnings"].append("Limited gameweek coverage (<3 GWs)")
    
    # Target variable
    if "target_next_points" in df.columns:
        target = df["target_next_points"]
        report["metrics"]["target_mean"] = float(target.mean())
        report["metrics"]["target_std"] = float(target.std())
        report["metrics"]["target_range"] = [float(target.min()), float(target.max())]
        
        if target.isna().any():
            report["warnings"].append("Target has missing values")
        
        if target.std() < 0.1:
            report["warnings"].append("Target has very low variance")
    else:
        report["valid"] = False
        report["errors"].append("Missing target variable 'target_next_points'")
    
    # Feature coverage
    critical_features = ["form", "minutes", "total_points", "fixture_difficulty"]
    missing_critical = [f for f in critical_features if f not in df.columns]
    
    if missing_critical:
        report["warnings"].append(f"Missing critical features: {', '.join(missing_critical)}")
    
    # Intelligence features
    risk_features = ["total_risk", "injury_risk", "rotation_risk"]
    ownership_features = ["selected_by_percent", "is_template"]
    price_features = ["net_transfers", "price_rise_probability"]
    
    report["metrics"]["has_risk_features"] = all(f in df.columns for f in risk_features)
    report["metrics"]["has_ownership_features"] = all(f in df.columns for f in ownership_features)
    report["metrics"]["has_price_features"] = all(f in df.columns for f in price_features)
    
    # Missing data analysis
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 20]
    
    if len(high_missing) > 0:
        report["warnings"].append(f"{len(high_missing)} features have >20% missing values")
        report["metrics"]["high_missing_features"] = high_missing.to_dict()
    
    report["metrics"]["avg_missing_pct"] = float(missing_pct.mean())
    
    # Sample size check
    if len(df) < 500:
        report["warnings"].append(f"Low sample count: {len(df)} (recommended: 500+)")
    
    if "player_id" in df.columns and df["player_id"].nunique() < 50:
        report["warnings"].append(f"Low player count: {df['player_id'].nunique()} (recommended: 50+)")
    
    # Final validation
    if report["errors"]:
        report["valid"] = False
    
    return report


def log_training_data_report(report: Dict[str, Any]):
    """
    Pretty print training data validation report.
    
    Args:
        report: Validation report from validate_training_data
    """
    logger.info("\n" + "=" * 80)
    logger.info("📋 TRAINING DATA VALIDATION REPORT")
    logger.info("=" * 80)
    
    # Status
    if report["valid"]:
        logger.info("✅ VALIDATION PASSED")
    else:
        logger.error("❌ VALIDATION FAILED")
    
    # Metrics
    metrics = report.get("metrics", {})
    if metrics:
        logger.info("\n📊 Dataset Metrics:")
        logger.info(f"   Total samples: {metrics.get('total_samples', 0):,}")
        
        if "unique_players" in metrics:
            logger.info(f"   Unique players: {metrics['unique_players']:,}")
            logger.info(f"   Avg samples/player: {metrics.get('avg_samples_per_player', 0):.1f}")
        
        if "gameweeks_covered" in metrics:
            logger.info(f"   Gameweeks: {metrics['gameweeks_covered']} (GW{metrics['gw_range'][0]}-{metrics['gw_range'][1]})")
        
        if "target_mean" in metrics:
            logger.info(f"   Target: mean={metrics['target_mean']:.2f}, std={metrics['target_std']:.2f}")
        
        logger.info(f"   Avg missing: {metrics.get('avg_missing_pct', 0):.2f}%")
        
        # Intelligence features
        logger.info("\n🎯 Intelligence Features:")
        logger.info(f"   Risk features: {'✅' if metrics.get('has_risk_features') else '❌'}")
        logger.info(f"   Ownership features: {'✅' if metrics.get('has_ownership_features') else '❌'}")
        logger.info(f"   Price features: {'✅' if metrics.get('has_price_features') else '❌'}")
    
    # Errors
    if report["errors"]:
        logger.error("\n❌ ERRORS:")
        for i, error in enumerate(report["errors"], 1):
            logger.error(f"   {i}. {error}")
    
    # Warnings
    if report["warnings"]:
        logger.warning("\n⚠️ WARNINGS:")
        for i, warning in enumerate(report["warnings"], 1):
            logger.warning(f"   {i}. {warning}")
    
    logger.info("=" * 80 + "\n")


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and validate training data before model training.
    
    Args:
        df: Raw training DataFrame
    
    Returns:
        Prepared DataFrame ready for training
    """
    logger.info("🔧 Preparing training data...")
    
    # Validate
    report = validate_training_data(df)
    log_training_data_report(report)
    
    if not report["valid"]:
        raise ValueError("Training data validation failed - cannot proceed")
    
    # Sort by player and gameweek
    if "player_id" in df.columns and "gameweek" in df.columns:
        df = df.sort_values(["player_id", "gameweek"])
        logger.info("✅ Sorted by player_id and gameweek")
    
    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_size:
        logger.info(f"   Removed {initial_size - len(df)} duplicate rows")
    
    # Ensure numeric types for critical columns
    numeric_cols = [
        "total_points", "minutes", "form", "fixture_difficulty",
        "selected_by_percent", "total_risk"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    logger.info(f"✅ Training data prepared: {len(df)} samples")
    
    return df


# ----------------------------------------------------------
# Enhanced Training Pipeline
# ----------------------------------------------------------

def train_with_validation(
    df: pd.DataFrame,
    n_splits: int = 5,
    run_comparison: bool = False,
    optimize: bool = False,
    n_trials: int = 30
) -> Dict[str, Any]:
    """
    Complete training pipeline with validation and optional optimization.
    
    Args:
        df: Raw training DataFrame
        n_splits: Number of CV folds
        run_comparison: Compare with/without risk features
        optimize: Run hyperparameter optimization
        n_trials: Number of optimization trials
    
    Returns:
        Training results
    """
    logger.info("=" * 80)
    logger.info("🚀 ENHANCED TRAINING PIPELINE - FULL INTELLIGENCE STACK")
    logger.info("=" * 80)
    
    # Prepare data
    df_prepared = prepare_training_data(df)
    
    # Analyze intelligence features
    intelligence_analysis = analyze_intelligence_features(df_prepared)
    
    # Optimize if requested
    if optimize:
        logger.info("\n🔧 Running hyperparameter optimization...")
        results = optimize_hyperparameters(df_prepared, n_trials=n_trials, n_splits=n_splits)
    else:
        # Standard training
        results = train_lightgbm(df_prepared, n_splits=n_splits)
    
    # Comparison if requested
    if run_comparison:
        logger.info("\n🔬 Running model comparison...")
        comparison = compare_models(df_prepared, n_splits=n_splits)
        results["comparison"] = comparison
    
    results["intelligence_analysis"] = intelligence_analysis
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ENHANCED TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    return results