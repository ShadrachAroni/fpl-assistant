"""
Enhanced Predictor Module - FPL Point Predictions with Risk Adjustment

ENHANCEMENTS:
✅ Risk-adjusted predictions
✅ Fixture-weighted predictions
✅ Horizon decay for distant gameweeks
✅ Form weighting
✅ Confidence intervals
✅ Multi-model ensemble
✅ Feature importance tracking
✅ Prediction explanations

PRODUCTION READY v5.0
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
logger = logging.getLogger("predictor")
logger.setLevel(logging.INFO)

MODEL_PATH = "models/lightgbm_model.joblib"


class Predictor:
    """
    Enhanced predictor with risk-adjusted predictions and comprehensive analysis.
    
    Features:
    - Multi-model ensemble
    - Risk-adjusted predictions
    - Fixture weighting
    - Horizon decay
    - Form weighting
    - Confidence intervals
    - Feature importance
    """

    def __init__(self, model_path: str = MODEL_PATH, enable_risk_adjustment: bool = True):
        """
        Initialize predictor with trained models.
        
        Args:
            model_path: Path to saved model file
            enable_risk_adjustment: Whether to apply risk adjustments to predictions
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

        logger.info(f"📦 Loading trained LightGBM model(s) from '{model_path}'...")
        data = joblib.load(model_path)

        if isinstance(data, dict):
            self.models = data.get("models", [])
            self.feats = data.get("feats", [])
            self.feature_importance = data.get("feature_importance", {})
        else:
            # Legacy format
            self.models = [data]
            self.feats = getattr(data, "feature_name_", [])
            self.feature_importance = {}

        if not self.models:
            raise ValueError("❌ No valid LightGBM models found in the saved file.")

        self.enable_risk_adjustment = enable_risk_adjustment
        self.n_models = len(self.models)

        logger.info(
            f"✅ Loaded {self.n_models} model(s). Feature count: {len(self.feats)} | "
            f"Risk adjustment: {'ON' if enable_risk_adjustment else 'OFF'}"
        )

    # ----------------------------------------------------------
    # Core Prediction Methods
    # ----------------------------------------------------------

    def predict(
        self, 
        df: pd.DataFrame, 
        clip_min: float = 0.0,
        apply_risk_adjustment: bool = True,
        apply_form_weighting: bool = True
    ) -> np.ndarray:
        """
        Predict expected FPL points for each player.

        Args:
            df: Player features DataFrame
            clip_min: Minimum value to clip predictions
            apply_risk_adjustment: Whether to apply risk adjustments
            apply_form_weighting: Whether to apply form weighting

        Returns:
            Array of predicted points
        """
        if df.empty:
            logger.warning("⚠️ Received empty DataFrame for prediction.")
            return np.array([])

        # Auto-detect features if not set
        if not self.feats:
            self.feats = self._auto_detect_features(df)

        # Prepare feature matrix
        X = self._prepare_features(df)

        # Get base predictions from ensemble
        base_preds = self._predict_ensemble(X)

        # Apply enhancements
        enhanced_preds = base_preds.copy()

        # 1. Form weighting
        if apply_form_weighting and "form" in df.columns:
            enhanced_preds = self._apply_form_weighting(enhanced_preds, df)

        # 2. Risk adjustment
        if apply_risk_adjustment and self.enable_risk_adjustment:
            enhanced_preds = self._apply_risk_adjustment(enhanced_preds, df)

        # 3. Clip negative predictions
        enhanced_preds = np.maximum(enhanced_preds, clip_min)

        logger.debug(
            f"✅ Generated predictions for {len(enhanced_preds)} players | "
            f"Range: [{enhanced_preds.min():.2f}, {enhanced_preds.max():.2f}]"
        )

        return enhanced_preds

    def predict_with_confidence(
        self, 
        df: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.
        
        Args:
            df: Player features
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if df.empty:
            return np.array([]), np.array([]), np.array([])

        X = self._prepare_features(df)

        # Get predictions from each model
        all_preds = np.array([
            model.predict(X, num_iteration=getattr(model, "best_iteration", None))
            for model in self.models
        ])

        # Calculate mean and std
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin = z_score * std_preds

        lower_bound = np.maximum(mean_preds - margin, 0)
        upper_bound = mean_preds + margin

        logger.debug(f"✅ Predictions with {confidence_level*100:.0f}% confidence intervals")

        return mean_preds, lower_bound, upper_bound

    # ----------------------------------------------------------
    # Horizon Planning
    # ----------------------------------------------------------

    def predict_horizon(
        self, 
        df: pd.DataFrame, 
        gw_list: List[int],
        decay_factor: float = 0.03,
        apply_fixture_weighting: bool = True
    ) -> pd.DataFrame:
        """
        Predict multiple gameweeks with horizon decay.

        Args:
            df: Player features
            gw_list: List of gameweek numbers to predict
            decay_factor: Decay per GW (uncertainty increases with distance)
            apply_fixture_weighting: Whether to weight by fixture difficulty

        Returns:
            DataFrame with pred_gw{n} columns
        """
        result_df = df.copy()
        
        # Get base predictions
        base_preds = self.predict(result_df)

        for idx, gw in enumerate(gw_list):
            # Apply horizon decay (predictions less reliable for distant GWs)
            horizon_decay = (1 - decay_factor) ** idx
            
            # Base prediction with decay
            gw_preds = base_preds * horizon_decay

            # Apply fixture weighting if available
            if apply_fixture_weighting and "fixture_difficulty" in result_df.columns:
                gw_preds = self._apply_fixture_weighting(gw_preds, result_df, gw)

            result_df[f"pred_gw{gw}"] = gw_preds

        logger.info(f"✅ Generated horizon predictions for {len(gw_list)} gameweeks")

        return result_df

    def predict_horizon_with_confidence(
        self,
        df: pd.DataFrame,
        gw_list: List[int],
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Predict horizon with confidence intervals for each GW.
        
        Args:
            df: Player features
            gw_list: Gameweek numbers
            confidence_level: Confidence level
        
        Returns:
            DataFrame with pred_gw{n}, pred_gw{n}_lower, pred_gw{n}_upper
        """
        result_df = df.copy()

        for idx, gw in enumerate(gw_list):
            preds, lower, upper = self.predict_with_confidence(result_df, confidence_level)
            
            # Apply horizon decay
            decay = (1 - 0.03) ** idx
            
            result_df[f"pred_gw{gw}"] = preds * decay
            result_df[f"pred_gw{gw}_lower"] = lower * decay
            result_df[f"pred_gw{gw}_upper"] = upper * decay

        return result_df

    # ----------------------------------------------------------
    # Risk Adjustment
    # ----------------------------------------------------------

    def _apply_risk_adjustment(self, predictions: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Apply comprehensive risk adjustment to predictions.
        
        Risk factors:
        - Total risk (max 30% penalty)
        - Injury risk (extra 20% if >50%)
        - Rotation risk (extra 15% if >70%)
        - Disciplinary risk (extra 10% if >60%)
        - Fatigue risk (extra 8% if >60%)
        """
        adjusted = predictions.copy()

        # Base risk penalty (max 30%)
        if "total_risk" in df.columns:
            total_risk = df["total_risk"].fillna(0).values
            risk_penalty = 1 - (total_risk * 0.30)
            adjusted = adjusted * risk_penalty
            
            logger.debug(f"Applied base risk adjustment (avg penalty: {(1 - risk_penalty.mean())*100:.1f}%)")

        # Critical risk penalties
        if "injury_risk" in df.columns:
            injury_mask = df["injury_risk"].fillna(0) > 0.5
            if injury_mask.any():
                adjusted[injury_mask] *= 0.80  # -20% for high injury risk
                logger.debug(f"Applied injury penalty to {injury_mask.sum()} players")

        if "rotation_risk" in df.columns:
            rotation_mask = df["rotation_risk"].fillna(0) > 0.7
            if rotation_mask.any():
                adjusted[rotation_mask] *= 0.85  # -15% for rotation prone
                logger.debug(f"Applied rotation penalty to {rotation_mask.sum()} players")

        if "disciplinary_risk" in df.columns:
            disciplinary_mask = df["disciplinary_risk"].fillna(0) > 0.6
            if disciplinary_mask.any():
                adjusted[disciplinary_mask] *= 0.90  # -10% for suspension risk
                logger.debug(f"Applied disciplinary penalty to {disciplinary_mask.sum()} players")

        if "fatigue_risk" in df.columns:
            fatigue_mask = df["fatigue_risk"].fillna(0) > 0.6
            if fatigue_mask.any():
                adjusted[fatigue_mask] *= 0.92  # -8% for high fatigue
                logger.debug(f"Applied fatigue penalty to {fatigue_mask.sum()} players")

        return adjusted

    def _apply_form_weighting(self, predictions: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Apply form-based weighting to predictions.
        
        Recent form (last 3 GWs) can boost or penalize predictions:
        - Form >8.0: +10% boost
        - Form <4.0: -10% penalty
        """
        if "form" not in df.columns:
            return predictions

        form = df["form"].fillna(7.5).values
        
        # Form multiplier (range: 0.9 to 1.1)
        form_multiplier = 1 + (form - 7.5) * 0.02
        form_multiplier = np.clip(form_multiplier, 0.9, 1.1)

        adjusted = predictions * form_multiplier

        logger.debug(
            f"Applied form weighting (avg multiplier: {form_multiplier.mean():.3f})"
        )

        return adjusted

    def _apply_fixture_weighting(
        self, 
        predictions: np.ndarray, 
        df: pd.DataFrame,
        gw: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply fixture difficulty weighting.
        
        - Easy fixtures (≤2): +20% boost
        - Hard fixtures (≥4): -15% penalty
        """
        if "fixture_difficulty" not in df.columns:
            return predictions

        difficulty = df["fixture_difficulty"].fillna(3).values

        # Fixture multiplier
        fixture_multiplier = np.ones_like(predictions)
        fixture_multiplier[difficulty <= 2] = 1.20  # Easy fixtures
        fixture_multiplier[difficulty >= 4] = 0.85  # Hard fixtures

        adjusted = predictions * fixture_multiplier

        logger.debug(
            f"Applied fixture weighting for GW{gw or 'current'} "
            f"(easy: {(difficulty <= 2).sum()}, hard: {(difficulty >= 4).sum()})"
        )

        return adjusted

    # ----------------------------------------------------------
    # Feature Preparation
    # ----------------------------------------------------------

    def _auto_detect_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect numeric features, excluding metadata columns."""
        exclude = [
            "id", "player_id", "web_name", "team_name", "element_type", 
            "position", "next_opponent", "next_opponent_short", "risk_category",
            "team", "fixture_display", "risk_summary"
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c not in exclude and not c.startswith("pred_gw")]
        
        logger.debug(f"Auto-detected {len(features)} features")
        return features

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix for prediction.
        
        - Fills missing features with 0
        - Handles NaN values
        - Ensures correct feature order
        """
        # Fill missing features
        missing_features = [f for f in self.feats if f not in df.columns]
        if missing_features:
            logger.debug(f"⚠️ Missing {len(missing_features)} features, filling with 0")
            for col in missing_features:
                df[col] = 0

        # Extract feature matrix
        X = df[self.feats].fillna(0)

        return X

    def _predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble of models and average results.
        """
        all_preds = []

        for model in self.models:
            preds = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
            all_preds.append(preds)

        # Average predictions across models
        ensemble_preds = np.mean(all_preds, axis=0)

        return ensemble_preds

    # ----------------------------------------------------------
    # Analysis & Explanation
    # ----------------------------------------------------------

    def explain_prediction(
        self, 
        df: pd.DataFrame, 
        player_idx: int = 0,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Explain prediction for a specific player.
        
        Args:
            df: Player features
            player_idx: Index of player to explain
            top_n: Number of top features to show
        
        Returns:
            Dictionary with prediction breakdown
        """
        if player_idx >= len(df):
            raise IndexError(f"Player index {player_idx} out of range")

        player = df.iloc[player_idx]
        X = self._prepare_features(df.iloc[[player_idx]])

        # Get base prediction
        base_pred = self._predict_ensemble(X)[0]

        # Get feature values
        feature_values = {}
        for feat in self.feats[:top_n]:
            if feat in player:
                feature_values[feat] = float(player[feat])

        # Calculate risk adjustment
        risk_adjustment = 1.0
        if "total_risk" in player:
            risk_adjustment = 1 - (float(player["total_risk"]) * 0.30)

        # Form adjustment
        form_adjustment = 1.0
        if "form" in player:
            form = float(player["form"])
            form_adjustment = 1 + (form - 7.5) * 0.02

        # Final prediction
        final_pred = base_pred * risk_adjustment * form_adjustment

        explanation = {
            "player_name": player.get("web_name", "Unknown"),
            "base_prediction": round(base_pred, 2),
            "risk_adjustment": round(risk_adjustment, 3),
            "form_adjustment": round(form_adjustment, 3),
            "final_prediction": round(final_pred, 2),
            "top_features": feature_values,
            "risk_breakdown": {
                "total_risk": float(player.get("total_risk", 0)),
                "injury_risk": float(player.get("injury_risk", 0)),
                "rotation_risk": float(player.get("rotation_risk", 0)),
                "disciplinary_risk": float(player.get("disciplinary_risk", 0)),
            } if "total_risk" in player else None
        }

        return explanation

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.models:
            return pd.DataFrame()

        # Aggregate feature importance across models
        importance_dict = {}

        for model in self.models:
            feature_importance = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()

            for name, importance in zip(feature_names, feature_importance):
                if name not in importance_dict:
                    importance_dict[name] = []
                importance_dict[name].append(importance)

        # Average importance across models
        avg_importance = {
            name: np.mean(values) 
            for name, values in importance_dict.items()
        }

        # Create DataFrame
        importance_df = pd.DataFrame([
            {"feature": name, "importance": importance}
            for name, importance in avg_importance.items()
        ])

        importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

        return importance_df

    # ----------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------

    def feature_summary(self) -> List[str]:
        """Get list of features used by the model."""
        return self.feats

    def model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "n_models": self.n_models,
            "n_features": len(self.feats),
            "risk_adjustment_enabled": self.enable_risk_adjustment,
            "model_path": MODEL_PATH,
            "features": self.feats[:10] + ["..."] if len(self.feats) > 10 else self.feats
        }

    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that DataFrame has required features.
        
        Returns:
            Dictionary with validation results
        """
        missing = [f for f in self.feats if f not in df.columns]
        extra = [c for c in df.columns if c in self.feats and c not in df.columns]

        return {
            "valid": len(missing) == 0,
            "missing_features": missing,
            "n_missing": len(missing),
            "has_risk_features": all(
                risk_col in df.columns 
                for risk_col in ["total_risk", "injury_risk", "rotation_risk"]
            ),
            "total_features": len(self.feats),
            "coverage": 1 - (len(missing) / len(self.feats)) if self.feats else 0
        }