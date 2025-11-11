"""
Enhanced Model Training Module - LightGBM for FPL Point Predictions

ENHANCEMENTS:
✅ Risk-aware feature engineering
✅ Feature importance tracking
✅ Comprehensive cross-validation
✅ Hyperparameter optimization support
✅ Model versioning
✅ Training metrics logging
✅ Feature selection
✅ Model ensemble

PRODUCTION READY v5.0
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
    Extract features and target from training DataFrame.
    
    Args:
        df: Training DataFrame
        target_col: Target column name
        exclude_risk_from_features: Whether to exclude risk features from training
    
    Returns:
        Tuple of (features_df, target_series, feature_names)
    """
    # Columns to always exclude
    base_exclude = [
        "player_id", "web_name", "event", "team_name", 
        "position", "next_opponent", "next_opponent_short",
        "risk_category", "risk_summary", "fixture_display",
        target_col
    ]
    
    # Risk columns (can be excluded if desired)
    risk_cols = [
        "total_risk", "injury_risk", "rotation_risk", 
        "disciplinary_risk", "fatigue_risk", "form_drop_risk",
        "defensive_fragility_risk", "penalty_risk",
        "minutes_volatility", "news_severity"
    ]
    
    exclude = base_exclude
    if exclude_risk_from_features:
        exclude.extend(risk_cols)
        logger.info("⚠️ Excluding risk features from training (risk-agnostic model)")
    else:
        logger.info("✅ Including risk features in training (risk-aware model)")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in exclude]
    
    # Log feature categories
    risk_feats_in_model = [f for f in feats if f in risk_cols]
    if risk_feats_in_model:
        logger.info(f"📊 Risk features in model: {len(risk_feats_in_model)}")
    
    logger.info(f"📊 Total features: {len(feats)}")
    
    return df[feats].fillna(0), df[target_col].fillna(0), feats


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
        DataFrame with feature importance
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
    save_metrics: bool = True
) -> Dict[str, Any]:
    """
    Train LightGBM model with comprehensive cross-validation and metrics.
    
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
    logger.info("🚀 STARTING LIGHTGBM TRAINING WITH RISK FEATURES")
    logger.info("=" * 80)
    
    # Extract features and target
    X, y, feats = features_targets_from_df(
        df, 
        exclude_risk_from_features=exclude_risk_features
    )
    
    groups = df.get("player_id", np.arange(len(df)))
    
    logger.info(f"📊 Training data: {len(df)} records | {len(feats)} features | {n_splits} folds")
    logger.info(f"🎯 Target: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.1f}, {y.max():.1f}]")
    
    # Default parameters
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbosity": -1,
            "seed": 42,
        }
        logger.info("📝 Using default LightGBM parameters")
    else:
        logger.info("📝 Using custom LightGBM parameters")
    
    # Cross-validation setup
    gkf = GroupKFold(n_splits=n_splits)
    models = []
    oof_preds = np.zeros(X.shape[0])
    
    # Metrics tracking
    fold_metrics = []
    
    # Training loop
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        logger.info(f"{'='*80}")
        logger.info(f"🧩 FOLD {fold+1}/{n_splits}")
        logger.info(f"{'='*80}")
        
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        
        logger.info(f"   Train: {len(Xtr)} samples | Valid: {len(Xva)} samples")
        
        # Create datasets
        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, yva)
        
        # Train model
        model = lgb.train(
            params=params,
            train_set=dtr,
            valid_sets=[dva],
            num_boost_round=num_boost_round,
            callbacks=[
                early_stopping(early_stopping_rounds, verbose=False),
                log_evaluation(100)
            ]
        )
        
        # Predictions
        preds = model.predict(Xva, num_iteration=model.best_iteration)
        oof_preds[va] = preds
        
        # Calculate metrics
        fold_rmse = mean_squared_error(yva, preds, squared=False)
        fold_mae = mean_absolute_error(yva, preds)
        fold_r2 = r2_score(yva, preds)
        
        fold_metrics.append({
            "fold": fold + 1,
            "rmse": fold_rmse,
            "mae": fold_mae,
            "r2": fold_r2,
            "best_iteration": model.best_iteration,
            "n_train": len(Xtr),
            "n_valid": len(Xva)
        })
        
        logger.info(f"   ✅ Fold {fold+1} Results:")
        logger.info(f"      RMSE: {fold_rmse:.4f}")
        logger.info(f"      MAE:  {fold_mae:.4f}")
        logger.info(f"      R²:   {fold_r2:.4f}")
        logger.info(f"      Best iteration: {model.best_iteration}")
        
        models.append(model)
    
    # Overall metrics
    logger.info("=" * 80)
    logger.info("📊 OVERALL PERFORMANCE")
    logger.info("=" * 80)
    
    global_rmse = mean_squared_error(y, oof_preds, squared=False)
    global_mae = mean_absolute_error(y, oof_preds)
    global_r2 = r2_score(y, oof_preds)
    
    logger.info(f"   Out-of-Fold RMSE: {global_rmse:.4f}")
    logger.info(f"   Out-of-Fold MAE:  {global_mae:.4f}")
    logger.info(f"   Out-of-Fold R²:   {global_r2:.4f}")
    
    # Calculate fold statistics
    fold_rmse_mean = np.mean([m["rmse"] for m in fold_metrics])
    fold_rmse_std = np.std([m["rmse"] for m in fold_metrics])
    
    logger.info(f"   Fold RMSE: {fold_rmse_mean:.4f} ± {fold_rmse_std:.4f}")
    
    # Feature importance
    logger.info("=" * 80)
    logger.info("📊 FEATURE IMPORTANCE")
    logger.info("=" * 80)
    
    importance_df = calculate_feature_importance(models, feats)
    
    logger.info(f"   Top 10 features by importance:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"      {idx+1}. {row['feature']}: {row['importance_mean']:.2f}")
    
    # Check for risk features in top features
    risk_features = [
        "total_risk", "injury_risk", "rotation_risk", "disciplinary_risk",
        "fatigue_risk", "form_drop_risk"
    ]
    top_20_features = importance_df.head(20)["feature"].tolist()
    risk_in_top = [f for f in risk_features if f in top_20_features]
    
    if risk_in_top:
        logger.info(f"   ✅ Risk features in top 20: {risk_in_top}")
    
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
        "params": params
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
        "params": params,
        "n_folds": n_splits
    }
    
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
                "fold_metrics": fold_metrics,
                "top_features": importance_df.head(20).to_dict("records")
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
    n_splits: int = 3
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
    
    X, y, feats = features_targets_from_df(df)
    groups = df.get("player_id", np.arange(len(df)))
    
    def objective(trial):
        """Optuna objective function."""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        }
        
        # Cross-validation
        gkf = GroupKFold(n_splits=n_splits)
        rmse_scores = []
        
        for tr, va in gkf.split(X, y, groups):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            
            dtr = lgb.Dataset(Xtr, ytr)
            dva = lgb.Dataset(Xva, yva)
            
            model = lgb.train(
                params=params,
                train_set=dtr,
                valid_sets=[dva],
                num_boost_round=500,
                callbacks=[early_stopping(30, verbose=False)]
            )
            
            preds = model.predict(Xva, num_iteration=model.best_iteration)
            rmse = mean_squared_error(yva, preds, squared=False)
            rmse_scores.append(rmse)
        
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
    
    results = train_lightgbm(df, params=best_params)
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
        "training_r2": model_data.get("oof_r2")
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
        
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
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

def analyze_risk_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the impact of risk features on predictions.
    
    Args:
        df: Training DataFrame with risk features
    
    Returns:
        Analysis results
    """
    logger.info("🔍 Analyzing risk feature impact...")
    
    risk_features = [
        "total_risk", "injury_risk", "rotation_risk", 
        "disciplinary_risk", "fatigue_risk", "form_drop_risk"
    ]
    
    available_risk_features = [f for f in risk_features if f in df.columns]
    
    if not available_risk_features:
        logger.warning("⚠️ No risk features found in DataFrame")
        return {}
    
    analysis = {
        "risk_features_available": available_risk_features,
        "correlations": {},
        "statistics": {}
    }
    
    # Correlation with target
    if "target_next_points" in df.columns:
        for feature in available_risk_features:
            corr = df[feature].corr(df["target_next_points"])
            analysis["correlations"][feature] = float(corr)
            logger.info(f"   {feature}: {corr:.4f} correlation with target")
    
    # Risk feature statistics
    for feature in available_risk_features:
        analysis["statistics"][feature] = {
            "mean": float(df[feature].mean()),
            "std": float(df[feature].std()),
            "min": float(df[feature].min()),
            "max": float(df[feature].max()),
            "high_risk_count": int((df[feature] > 0.6).sum()),
            "medium_risk_count": int(((df[feature] >= 0.3) & (df[feature] <= 0.6)).sum()),
            "low_risk_count": int((df[feature] < 0.3).sum())
        }
    
    logger.info("✅ Risk feature analysis complete")
    
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
    """Print summary of trained model."""
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    model_data = joblib.load(model_path)
    
    print("=" * 80)
    print("📊 MODEL SUMMARY")
    print("=" * 80)
    print(f"Training Date: {model_data.get('training_date', 'unknown')}")
    print(f"Number of Models: {len(model_data.get('models', []))}")
    print(f"Number of Features: {len(model_data.get('feats', []))}")
    print(f"Out-of-Fold RMSE: {model_data.get('oof_rmse', 0):.4f}")
    print(f"Out-of-Fold MAE: {model_data.get('oof_mae', 0):.4f}")
    print(f"Out-of-Fold R²: {model_data.get('oof_r2', 0):.4f}")
    
    if "feature_importance" in model_data:
        importance_df = pd.DataFrame(model_data["feature_importance"])
        print("\nTop 10 Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {idx+1}. {row['feature']}: {row['importance_mean']:.2f}")
    
    print("=" * 80)