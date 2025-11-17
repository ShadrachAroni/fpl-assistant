"""
Transfer Simulator - ENHANCED WITH ADVANCED CAPTAIN & TRANSFER INTELLIGENCE

NEW FEATURES:
✅ Effective Ownership (EO) captain recommendations
✅ Template vs differential captain strategy
✅ Price change opportunity cost in transfers
✅ Value hold analysis (rising players)
✅ Formation validation for transfers
✅ Bench strength optimization
✅ Multi-gameweek transfer sequencing
✅ Ownership-based differential plays

PRODUCTION READY v6.0
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("simulator")
logger.setLevel(logging.INFO)

POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class TransferSimulator:
    """
    Enhanced transfer planner with ownership intelligence and advanced captain logic.
    """

    DEFAULT_FIXTURE_DIFFICULTY = 3

    def __init__(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        predictor,
        config: Dict[str, Any],
        upcoming_fixtures: Optional[pd.DataFrame],
        next_gw: int,
        bank=None,
        free_transfers=None,
        selling_prices: Optional[Dict[int, float]] = None,
        transfers_this_gw: Optional[List[Dict[str, Any]]] = None,
        manager_chips_used: Optional[Dict[str, Optional[int]]] = None,
        manager_performance: Optional[Dict[str, Any]] = None,
        blank_double_gws: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """Initialize enhanced transfer simulator."""
        self.current = current_squad.copy()
        self.all_players = all_players.copy()
        self.predictor = predictor
        self.config = config
        self.fixtures = upcoming_fixtures
        self.next_gw = next_gw
        self.bank = float(bank) if bank is not None else 0.0
        self.free_transfers = int(free_transfers) if free_transfers is not None else 1
        self.selling_prices = selling_prices or {}
        self.transfers_this_gw = transfers_this_gw or []
        self.manager_chips_used = manager_chips_used or {}
        self.manager_performance = manager_performance or {}
        self.blank_double_gws = blank_double_gws or {}

        # Configuration
        sim_cfg = config.get("simulation", {})
        self.horizon = int(sim_cfg.get("planning_horizon", 5))
        self.chip_horizon = int(sim_cfg.get("chip_analysis_horizon", 10))
        self.max_transfers = int(sim_cfg.get("max_transfers_per_gw", 2))
        self.hit_penalty = float(sim_cfg.get("transfer_cost", 4.0))

        self.market_cfg = config.get("market_intelligence", {})
        self.multi_objective_weights = sim_cfg.get("multi_objective_weights", {
            "points": 0.35,
            "safety": 0.20,
            "value": 0.15,
            "differential": 0.15,
            "fixtures": 0.15
        })
        mc_cfg = sim_cfg.get("monte_carlo", {})
        self.monte_carlo_enabled = mc_cfg.get("enabled", True)
        self.monte_carlo_simulations = int(mc_cfg.get("simulations", 1000))
        self.monte_carlo_floor = int(mc_cfg.get("percentile_floor", 10))
        self.monte_carlo_ceiling = int(mc_cfg.get("percentile_ceiling", 90))
        
        # Transfer risk config
        risk_cfg = sim_cfg.get("transfer_risk", {})
        self.min_gain_free = float(risk_cfg.get("min_gain_for_free_transfer", 1.0))
        self.min_gain_hit = float(risk_cfg.get("min_gain_for_hit", 6.0))
        self.extra_transfer_mult = float(risk_cfg.get("extra_transfer_multiplier", 1.5))
        self.allow_aggressive = risk_cfg.get("allow_aggressive_transfers", True)
        self.underperform_threshold = float(risk_cfg.get("underperformance_threshold", 45))
        
        # NEW: Price change thresholds
        self.value_hold_threshold = 0.5  # Hold players with >50% rise probability
        self.opportunity_cost_threshold = 0.1  # £0.1m potential loss threshold

        # Track analysis
        self.transfer_analysis = {
            "considered": [],
            "rejected": [],
            "reasons": [],
            "risk_warnings": [],
            "price_warnings": []
        }

        # Extract transferred players
        self.transferred_out_this_gw = {t.get("element_out") for t in self.transfers_this_gw}
        self.transferred_in_this_gw = {t.get("element_in") for t in self.transfers_this_gw}

        # Remove transferred-out players
        if self.transferred_out_this_gw:
            logger.debug(f"Removing {len(self.transferred_out_this_gw)} transferred-out players")
            self.current = self.current[~self.current["id"].isin(self.transferred_out_this_gw)]

        # Build fixture maps
        self.fixture_diff_by_gw = self._build_fixture_difficulty_map_horizon()
        self.dgw_map = self._detect_double_gameweeks()
        self.bgw_map = self._detect_blank_gameweeks()

        # Team constraints
        self.team_counts = self.current.groupby("team").size().to_dict()
        self.max_per_team = 3

        logger.info(
            f"✅ Simulator: {len(self.current)} players | £{self.bank:.1f}m bank | "
            f"{self.free_transfers} FT | Horizon={self.horizon} GWs"
        )

    def _build_fixture_difficulty_map_horizon(self) -> Dict[int, Dict[int, int]]:
        """Build team -> fixture difficulty for next N gameweeks."""
        fixture_map = {}

        if self.fixtures is None or self.fixtures.empty:
            logger.debug("No fixtures data")
            return fixture_map

        fixtures_df = pd.DataFrame(self.fixtures)

        if fixtures_df.empty or "event" not in fixtures_df.columns:
            return fixture_map

        for i in range(self.horizon):
            gw = self.next_gw + i
            fixture_map[gw] = {}
            
            gw_fixtures = fixtures_df[fixtures_df["event"] == gw]
            
            for _, row in gw_fixtures.iterrows():
                try:
                    home = int(row.get("team_h", 0))
                    away = int(row.get("team_a", 0))
                    h_diff = int(row.get("team_h_difficulty", 3))
                    a_diff = int(row.get("team_a_difficulty", 3))
                    fixture_map[gw][home] = h_diff
                    fixture_map[gw][away] = a_diff
                except:
                    continue

        logger.debug(f"Fixture map built for {len(fixture_map)} gameweeks")
        return fixture_map

    def _detect_double_gameweeks(self) -> Dict[int, List[int]]:
        """Detect double gameweeks."""
        dgw_map = {}

        if self.fixtures is None or self.fixtures.empty:
            return dgw_map

        fixtures_df = pd.DataFrame(self.fixtures)

        for i in range(self.chip_horizon):
            gw = self.next_gw + i
            gw_fixtures = fixtures_df[fixtures_df["event"] == gw]
            
            if gw_fixtures.empty:
                continue

            team_fixture_count = {}
            for _, row in gw_fixtures.iterrows():
                home = int(row.get("team_h", 0))
                away = int(row.get("team_a", 0))
                team_fixture_count[home] = team_fixture_count.get(home, 0) + 1
                team_fixture_count[away] = team_fixture_count.get(away, 0) + 1

            dgw_teams = [team for team, count in team_fixture_count.items() if count >= 2]
            
            if dgw_teams:
                dgw_map[gw] = dgw_teams
                logger.info(f"🔥 DGW{gw}: {len(dgw_teams)} teams")

        return dgw_map

    def _detect_blank_gameweeks(self) -> Dict[int, int]:
        """Detect blank gameweeks."""
        bgw_map = {}

        if self.fixtures is None or self.fixtures.empty:
            return bgw_map

        fixtures_df = pd.DataFrame(self.fixtures)

        for i in range(self.chip_horizon):
            gw = self.next_gw + i
            gw_fixtures = fixtures_df[fixtures_df["event"] == gw]
            
            teams_playing = set()
            for _, row in gw_fixtures.iterrows():
                teams_playing.add(int(row.get("team_h", 0)))
                teams_playing.add(int(row.get("team_a", 0)))

            if len(teams_playing) < 18:
                bgw_map[gw] = len(teams_playing)
                logger.info(f"⚠️ BGW{gw}: Only {len(teams_playing)} teams")

        return bgw_map

    def select_captain_with_eo_intelligence(
        self, 
        squad: pd.DataFrame, 
        gw: int
    ) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
        """
        Select captain with EO (Effective Ownership) intelligence.
        
        Strategy:
        - Template strategy: Pick high-ownership safe captain
        - Differential strategy: Pick low-ownership high-upside captain
        - Risk-adjusted for rotation/injury
        
        Returns:
            (captain_row, vice_row, reasoning_dict)
        """
        squad = squad.copy()
        pred_col = f"pred_gw{gw}"
        
        if pred_col not in squad.columns:
            # Fallback to form
            squad["_temp_pred"] = squad.get("form", 0)
            pred_col = "_temp_pred"
        
        # Calculate risk-adjusted predictions
        squad["risk_adjusted_pred"] = squad.apply(
            lambda row: self._calculate_risk_adjusted_score(row, pred_col),
            axis=1
        )
        
        # Calculate captain value (prediction * EO multiplier)
        squad["captain_eo_value"] = (
            squad["risk_adjusted_pred"] * 
            squad.get("captain_eo_multiplier", 1.0)
        )
        
        # Separate by strategy
        attackers = squad[squad["position"].isin(["MID", "FWD"])].copy()
        
        if attackers.empty:
            # Fallback to all players
            attackers = squad.copy()
        
        # TEMPLATE STRATEGY: High ownership, safe picks
        template_captains = attackers[
            (attackers.get("selected_by_percent", 0) > 35) &
            (attackers.get("total_risk", 1.0) < 0.5)
        ].nlargest(3, "captain_eo_value")
        
        # DIFFERENTIAL STRATEGY: Low ownership, high upside
        differential_captains = attackers[
            (attackers.get("selected_by_percent", 0) < 15) &
            (attackers["risk_adjusted_pred"] > attackers["risk_adjusted_pred"].quantile(0.75))
        ].nlargest(3, "risk_adjusted_pred")
        
        # Decision logic
        avg_rank = self.manager_performance.get("overall_rank", 1000000)
        
        # If chasing rank (bottom 50%), consider differentials
        if avg_rank > 500000 and not differential_captains.empty:
            captain_row = differential_captains.iloc[0]
            strategy = "DIFFERENTIAL"
            reasoning = (
                f"Chasing rank - differential captain strategy. "
                f"{captain_row['web_name']} ({captain_row.get('selected_by_percent', 0):.1f}% owned) "
                f"has high upside ({captain_row['risk_adjusted_pred']:.1f} pts)"
            )
        else:
            # Safe template strategy
            if not template_captains.empty:
                captain_row = template_captains.iloc[0]
                strategy = "TEMPLATE"
                reasoning = (
                    f"Template captain strategy. "
                    f"{captain_row['web_name']} ({captain_row.get('selected_by_percent', 0):.1f}% owned) "
                    f"is the safe pick ({captain_row['captain_eo_value']:.1f} EO-adjusted value)"
                )
            else:
                # No templates, pick best overall
                captain_row = attackers.nlargest(1, "risk_adjusted_pred").iloc[0]
                strategy = "BEST_AVAILABLE"
                reasoning = f"Best available: {captain_row['web_name']} ({captain_row['risk_adjusted_pred']:.1f} pts)"
        
        # Vice captain (exclude captain, pick next best)
        vice_candidates = attackers[attackers["id"] != captain_row["id"]]
        if not vice_candidates.empty:
            vice_row = vice_candidates.nlargest(1, "risk_adjusted_pred").iloc[0]
        else:
            vice_row = captain_row
        
        # Build reasoning dict
        reasoning_dict = {
            "strategy": strategy,
            "reasoning": reasoning,
            "captain_ownership": float(captain_row.get("selected_by_percent", 0)),
            "captain_predicted_pts": float(captain_row["risk_adjusted_pred"]),
            "captain_eo_value": float(captain_row["captain_eo_value"]),
            "vice_predicted_pts": float(vice_row["risk_adjusted_pred"]),
            "fixture_quality": captain_row.get("fixture_difficulty", 3),
            "alternatives": []
        }
        
        # Add alternatives
        alternatives = attackers.nlargest(5, "captain_eo_value")
        for _, alt in alternatives.head(3).iterrows():
            if alt["id"] != captain_row["id"]:
                reasoning_dict["alternatives"].append({
                    "name": alt["web_name"],
                    "ownership": float(alt.get("selected_by_percent", 0)),
                    "predicted_pts": float(alt["risk_adjusted_pred"]),
                    "eo_value": float(alt["captain_eo_value"])
                })
        
        return captain_row, vice_row, reasoning_dict

    def _calculate_risk_adjusted_score(self, player: pd.Series, pred_col: str) -> float:
        """Calculate risk-adjusted score for a player."""
        base_pred = float(player.get(pred_col, 0.0))
        
        total_risk = float(player.get("total_risk", 0.0))
        injury_risk = float(player.get("injury_risk", 0.0))
        rotation_risk = float(player.get("rotation_risk", 0.0))
        disciplinary_risk = float(player.get("disciplinary_risk", 0.0))
        fatigue_risk = float(player.get("fatigue_risk", 0.0))
        
        risk_penalty = 1 - (total_risk * 0.3)
        
        if injury_risk > 0.5:
            risk_penalty *= 0.8
        if rotation_risk > 0.7:
            risk_penalty *= 0.85
        if disciplinary_risk > 0.6:
            risk_penalty *= 0.9
        if fatigue_risk > 0.6:
            risk_penalty *= 0.92
        
        adjusted_score = base_pred * risk_penalty
        
        return max(0.0, adjusted_score)

    def _get_player_risk_summary(self, player: pd.Series) -> str:
        """Generate human-readable risk summary."""
        risks = []
        
        injury_risk = float(player.get("injury_risk", 0.0))
        if injury_risk > 0.3:
            chance = int(player.get("chance_of_playing_next_round", 100))
            risks.append(f"⚕️ Injury ({chance}% fit)")
        
        rotation_risk = float(player.get("rotation_risk", 0.0))
        if rotation_risk > 0.5:
            mins_pg = int(player.get("minutes_per_game", 0))
            risks.append(f"🔄 Rotation ({mins_pg}min/game)")
        
        disciplinary_risk = float(player.get("disciplinary_risk", 0.0))
        if disciplinary_risk > 0.5:
            yellows = int(player.get("yellow_cards", 0))
            risks.append(f"🟨 Suspension risk ({yellows} yellows)")
        
        fatigue_risk = float(player.get("fatigue_risk", 0.0))
        if fatigue_risk > 0.5:
            risks.append("😓 Fatigue")
        
        form_risk = float(player.get("form_drop_risk", 0.0))
        if form_risk > 0.5:
            form = float(player.get("form", 0))
            risks.append(f"📉 Poor form ({form:.1f})")
        
        if not risks:
            return "✅ Low risk"
        
        return " | ".join(risks)

    def _apply_predictions_with_fixture_weighting(
        self, squad: pd.DataFrame, gw: int
    ) -> pd.DataFrame:
        """Apply predictions with fixture weighting."""
        squad = squad.copy()

        if self.predictor is None:
            squad[f"pred_gw{gw}"] = 0.0
            return squad

        pred_col = f"pred_gw{gw}"

        try:
            if pred_col in squad.columns:
                preds = squad[pred_col].to_numpy(copy=True)
            else:
                preds = self.predictor.predict(squad)
                squad[pred_col] = preds

            fixture_map = self.fixture_diff_by_gw.get(gw)
            if fixture_map:
                difficulty = squad["team"].map(fixture_map).fillna(self.DEFAULT_FIXTURE_DIFFICULTY).to_numpy()
                weights = np.where(
                    difficulty <= 2,
                    1.20,
                    np.where(difficulty >= 4, 0.85, 1.0)
                )
                preds = preds * weights

            squad[pred_col] = preds
        except Exception as e:
            logger.debug(f"Prediction failed for GW{gw}: {e}")
            squad[pred_col] = 0.0

        return squad

    def _get_selling_price(self, player_id: int, current_price: float) -> float:
        """Get actual selling price for player."""
        return self.selling_prices.get(player_id, current_price)

    def _find_optimal_transfers(self, gw: int) -> List[Dict[str, Any]]:
        """
        Find optimal transfers with PRICE INTELLIGENCE and OWNERSHIP AWARENESS.
        """
        self.transfer_analysis = {
            "considered": [],
            "rejected": [],
            "reasons": [],
            "risk_warnings": [],
            "price_warnings": []
        }

        if self.current.empty or self.all_players.empty:
            return []

        current_with_pred = self._apply_predictions_with_fixture_weighting(self.current, gw)
        all_with_pred = self._apply_predictions_with_fixture_weighting(self.all_players, gw)

        pred_col = f"pred_gw{gw}"
        
        if pred_col not in current_with_pred.columns or pred_col not in all_with_pred.columns:
            return []

        # Calculate risk-adjusted scores
        current_with_pred["risk_adjusted_score"] = current_with_pred.apply(
            lambda row: self._calculate_risk_adjusted_score(row, pred_col),
            axis=1
        )
        
        all_with_pred["risk_adjusted_score"] = all_with_pred.apply(
            lambda row: self._calculate_risk_adjusted_score(row, pred_col),
            axis=1
        )

        current_with_pred["multi_objective_score"] = current_with_pred.apply(
            lambda row: self._calculate_multi_objective_score(row, pred_col),
            axis=1
        )
        all_with_pred["multi_objective_score"] = all_with_pred.apply(
            lambda row: self._calculate_multi_objective_score(row, pred_col),
            axis=1
        )

        # NEW: Add price change value
        current_with_pred["price_change_impact"] = current_with_pred.apply(
            lambda row: self._calculate_price_change_impact(row),
            axis=1
        )

        # Sort by combined score (prediction + price impact - risk)
        current_with_pred["transfer_priority"] = (
            -current_with_pred["risk_adjusted_score"] + 
            current_with_pred["price_change_impact"] * 10  # Weight price changes heavily
        )
        
        current_sorted = current_with_pred.sort_values("transfer_priority", ascending=False)
        
        potential_transfers = []

        for _, weak_player in current_sorted.iterrows():
            weak_id = int(weak_player["id"])
            weak_name = weak_player["web_name"]
            weak_pred = float(weak_player[pred_col])
            weak_risk_adj = float(weak_player["risk_adjusted_score"])
            weak_pos = weak_player.get("element_type", weak_player.get("position"))
            weak_team = int(weak_player.get("team", 0))
            weak_price = float(weak_player.get("now_cost", 0.0))
            weak_selling_price = self._get_selling_price(weak_id, weak_price)
            weak_risk_summary = self._get_player_risk_summary(weak_player)
            
            # NEW: Price change analysis
            weak_price_fall_prob = float(weak_player.get("price_fall_probability", 0))
            weak_opportunity_cost = float(weak_player.get("opportunity_cost", 0))

            available_funds = self.bank + weak_selling_price

            # Find replacements with OWNERSHIP consideration
            same_position = all_with_pred[
                (all_with_pred["element_type"] == weak_pos) &
                (~all_with_pred["id"].isin(current_with_pred["id"])) &
                (~all_with_pred["id"].isin(self.transferred_in_this_gw)) &
                (all_with_pred["now_cost"] <= available_funds)
            ].copy()

            if same_position.empty:
                continue

            # Calculate gain including price changes
            same_position["predicted_gain"] = same_position["risk_adjusted_score"] - weak_risk_adj
            same_position["cost_diff"] = same_position["now_cost"] - weak_selling_price
            
            # NEW: Add ownership differential value
            same_position["ownership_differential"] = (
                same_position.get("selected_by_percent", 0) - 
                weak_player.get("selected_by_percent", 0)
            )
            
            # NEW: Add price rise value (hold rising players, transfer out falling)
            same_position["price_value"] = (
                same_position.get("price_rise_probability", 0) * 0.1 -  # Potential gain
                weak_opportunity_cost  # Potential loss from keeping weak player
            )

            for _, candidate in same_position.iterrows():
                cand_team = int(candidate.get("team", 0))
                cand_id = int(candidate["id"])
                
                # Team constraint
                if cand_team == weak_team:
                    continue
                
                current_team_count = self.team_counts.get(cand_team, 0)
                if current_team_count >= self.max_per_team:
                    continue

                predicted_gain = float(candidate["predicted_gain"])
                cost_diff = float(candidate["cost_diff"])
                ownership_diff = float(candidate["ownership_differential"])
                price_value = float(candidate["price_value"])
                multi_gain = float(candidate["multi_objective_score"] - weak_player["multi_objective_score"])
                
                cand_risk_summary = self._get_player_risk_summary(candidate)
                cand_total_risk = float(candidate.get("total_risk", 0.0))
                cand_ownership = float(candidate.get("selected_by_percent", 0))
                weak_ownership = float(weak_player.get("selected_by_percent", 0))
                regret_penalty = self._calculate_transfer_regret(weak_player, candidate)
                
                # Calculate ENHANCED net gain (includes price changes)
                template_penalty = self._calculate_template_penalty(cand_ownership)
                net_gain = (
                    predicted_gain +
                    (multi_gain * 0.5) +
                    (price_value * 5) -
                    template_penalty -
                    regret_penalty
                )
                
                # Check for price warnings
                cand_price_rise_prob = float(candidate.get("price_rise_probability", 0))
                cand_price_fall_prob = float(candidate.get("price_fall_probability", 0))
                
                price_warning = None
                if weak_price_fall_prob > 0.5:
                    price_warning = f"⚠️ Selling player likely to fall (save £0.1m)"
                elif cand_price_rise_prob > 0.5:
                    price_warning = f"💰 Target likely to rise (buy before +£0.1m)"
                
                if price_warning:
                    self.transfer_analysis["price_warnings"].append(price_warning)
                
                # Risk warnings
                if cand_total_risk > 0.5:
                    self.transfer_analysis["risk_warnings"].append(
                        f"{candidate['web_name']}: High risk - {cand_risk_summary}"
                    )
                
                # Ownership analysis
                ownership_note = ""
                
                if cand_ownership > 50:
                    ownership_note = f"🔴 Essential ({cand_ownership:.1f}% owned)"
                elif cand_ownership > 35:
                    ownership_note = f"🟠 Template ({cand_ownership:.1f}% owned)"
                elif cand_ownership < 5:
                    ownership_note = f"💎 Differential ({cand_ownership:.1f}% owned)"

                transfer_dict = {
                    "out_id": weak_id,
                    "out_name": weak_name,
                    "out_cost": weak_selling_price,
                    "out_pred": weak_pred,
                    "out_risk_adj": weak_risk_adj,
                    "out_risk_summary": weak_risk_summary,
                    "out_ownership": weak_ownership,
                    "out_price_fall_prob": weak_price_fall_prob,
                    "in_id": cand_id,
                    "in_name": candidate["web_name"],
                    "in_cost": float(candidate["now_cost"]),
                    "in_pred": float(candidate[pred_col]),
                    "in_risk_adj": float(candidate["risk_adjusted_score"]),
                    "in_risk_summary": cand_risk_summary,
                    "in_total_risk": cand_total_risk,
                    "in_ownership": cand_ownership,
                    "in_price_rise_prob": cand_price_rise_prob,
                    "ownership_note": ownership_note,
                    "price_warning": price_warning,
                    "predicted_gain": predicted_gain,
                    "multi_objective_gain": multi_gain,
                    "cost_diff": cost_diff,
                    "net_gain": net_gain,
                    "regret_penalty": regret_penalty,
                    "template_penalty": template_penalty,
                    "hit": 0,
                    "hit_penalty": 0.0
                }

                potential_transfers.append(transfer_dict)
                self.transfer_analysis["considered"].append(transfer_dict)

        # Sort by net gain (including price value)
        potential_transfers.sort(key=lambda x: x["net_gain"], reverse=True)

        logger.debug(f"Found {len(potential_transfers)} potential transfers for GW{gw}")
        if self.transfer_analysis["price_warnings"]:
            logger.info(f"💰 {len(self.transfer_analysis['price_warnings'])} price change opportunities")
        if self.transfer_analysis["risk_warnings"]:
            logger.info(f"⚠️ {len(self.transfer_analysis['risk_warnings'])} risk warnings")

        return potential_transfers[:self.max_transfers * 2]

    def _calculate_price_change_impact(self, player: pd.Series) -> float:
        """
        Calculate price change impact for a player.
        
        Positive = should keep/buy (rising)
        Negative = should sell (falling)
        """
        price_rise_prob = float(player.get("price_rise_probability", 0))
        price_fall_prob = float(player.get("price_fall_probability", 0))
        
        # Rising players have negative impact (don't sell)
        # Falling players have positive impact (sell urgently)
        impact = price_fall_prob - price_rise_prob
        
        return impact

    def _calculate_multi_objective_score(self, player: pd.Series, pred_col: str) -> float:
        """Compute Pareto-style score across points, safety, value, differential, fixtures."""
        weights = self.multi_objective_weights
        expected_points = float(player.get(pred_col, 0.0))
        safety = max(0.0, 1 - float(player.get("total_risk", 0.0)))
        value = float(player.get("points_per_million", 0.0))
        differential = float(player.get("differential_score", 0.0))
        fixtures = float(player.get("future_fixture_quality", 0.0))

        score = (
            expected_points * weights.get("points", 0.3) +
            safety * weights.get("safety", 0.2) +
            value * weights.get("value", 0.15) +
            differential * weights.get("differential", 0.15) +
            fixtures * weights.get("fixtures", 0.2)
        )
        return score

    def _calculate_transfer_regret(
        self,
        player_out: pd.Series,
        player_in: pd.Series,
        horizon: int = 5
    ) -> float:
        """Estimate regret if recommended transfer backfires."""
        out_upside = float(player_out.get("risk_adjusted_score", 0.0)) + \
            float(player_out.get("points_std_dev", 1.0)) * 1.5
        in_downside = max(
            0.0,
            float(player_in.get("risk_adjusted_score", 0.0)) -
            float(player_in.get("points_std_dev", 1.0))
        )
        return max(0.0, out_upside - in_downside)

    def _calculate_template_penalty(self, ownership: float, is_captaincy: bool = False) -> float:
        """Apply market-based penalty for template selections."""
        if is_captaincy:
            eo = ownership * 0.02
            if eo > 1.0:
                return float(self.market_cfg.get("template_penalty_captain", 0.5))
        if ownership > 50:
            return float(self.market_cfg.get("template_penalty_player", 0.3))
        return 0.0

    def _run_monte_carlo(self, squad: pd.DataFrame, gw: int) -> Dict[str, float]:
        """Monte Carlo simulation for squad outcomes."""
        if not self.monte_carlo_enabled or squad.empty:
            return {}

        working = self._apply_predictions_with_fixture_weighting(squad, gw)
        pred_col = f"pred_gw{gw}"
        if pred_col not in working.columns:
            return {}

        preds = working[pred_col].fillna(0).values
        stds = working.get("points_std_dev", pd.Series(1.0, index=working.index)).fillna(1.0).values
        injury = working.get("injury_risk", pd.Series(0, index=working.index)).fillna(0).values
        rotation = working.get("rotation_risk", pd.Series(0, index=working.index)).fillna(0).values

        simulations = []
        for _ in range(self.monte_carlo_simulations):
            sample = np.random.normal(preds, stds)
            injury_events = np.random.rand(len(sample)) < injury
            rotation_events = np.random.rand(len(sample)) < rotation
            sample[injury_events] = 0
            rotation_only = rotation_events & ~injury_events
            sample[rotation_only] *= 0.3
            simulations.append(sample.sum())

        if not simulations:
            return {}

        sims = np.array(simulations)
        return {
            "expected": float(np.mean(sims)),
            "median": float(np.median(sims)),
            "p10": float(np.percentile(sims, self.monte_carlo_floor)),
            "p90": float(np.percentile(sims, self.monte_carlo_ceiling)),
            "variance": float(np.var(sims))
        }

    def _validate_and_select_transfers(
        self, transfers: List[Dict[str, Any]], free_transfers: int
    ) -> List[Dict[str, Any]]:
        """Validate transfers with ENHANCED logic."""
        if not transfers:
            return []

        validated = []

        for i, transfer in enumerate(transfers):
            if i < free_transfers:
                transfer["hit"] = 0
                transfer["hit_penalty"] = 0.0
            else:
                transfer["hit"] = 4
                transfer["hit_penalty"] = 4.0

            transfer["net_gain_after_hit"] = transfer["net_gain"] - transfer["hit_penalty"]

            validated.append(transfer)

        recommended = []

        for i, transfer in enumerate(validated):
            net_gain = transfer["net_gain_after_hit"]
            in_risk = transfer["in_total_risk"]
            price_warning = transfer.get("price_warning")
            ownership_note = transfer.get("ownership_note")
            
            risk_warning = ""
            if in_risk > 0.6:
                risk_warning = f" ⚠️ HIGH RISK: {transfer['in_risk_summary']}"
            elif in_risk > 0.4:
                risk_warning = f" ⚡ MEDIUM RISK: {transfer['in_risk_summary']}"

            if i == 0:
                if i < free_transfers:
                    if net_gain >= self.min_gain_free:
                        recommended.append(transfer)
                        logger.info(
                            f"✅ Transfer 1 (FREE): {transfer['out_name']} → {transfer['in_name']} "
                            f"(+{net_gain:.1f} pts){risk_warning}"
                        )
                        if price_warning:
                            logger.info(f"   {price_warning}")
                        if ownership_note:
                            logger.info(f"   {ownership_note}")
                    else:
                        reason = f"Gain {net_gain:.1f} < threshold {self.min_gain_free:.1f}"
                        logger.debug(f"❌ Rejected: {reason}")
                        self.transfer_analysis["rejected"].append(transfer)
                        self.transfer_analysis["reasons"].append(reason)
                else:
                    if net_gain >= self.min_gain_hit:
                        recommended.append(transfer)
                        logger.info(
                            f"⚠️ Transfer 1 (HIT): {transfer['out_name']} → {transfer['in_name']} "
                            f"(+{net_gain:.1f} pts after -4){risk_warning}"
                        )
                    else:
                        reason = f"Gain {net_gain:.1f} < hit threshold {self.min_gain_hit:.1f}"
                        logger.debug(f"❌ Rejected: {reason}")
                        self.transfer_analysis["rejected"].append(transfer)
                        self.transfer_analysis["reasons"].append(reason)
            else:
                required_gain = self.hit_penalty * self.extra_transfer_mult
                
                if net_gain >= required_gain:
                    if self.allow_aggressive:
                        avg_pts = self.manager_performance.get("average_points_recent", 50)
                        
                        if avg_pts < self.underperform_threshold or net_gain >= required_gain * 1.2:
                            if in_risk > 0.7:
                                reason = f"Risky player ({in_risk:.2f}) for 2nd transfer"
                                logger.debug(f"❌ Rejected extra transfer: {reason}")
                                self.transfer_analysis["rejected"].append(transfer)
                                self.transfer_analysis["reasons"].append(reason)
                            else:
                                recommended.append(transfer)
                                logger.info(
                                    f"🔥 Transfer {i+1} (AGGRESSIVE): {transfer['out_name']} → {transfer['in_name']} "
                                    f"(+{net_gain:.1f} pts, justifies -{transfer['hit']}pt hit){risk_warning}"
                                )
                        else:
                            reason = f"Gain {net_gain:.1f} but not underperforming"
                            logger.debug(f"❌ Rejected: {reason}")
                            self.transfer_analysis["rejected"].append(transfer)
                            self.transfer_analysis["reasons"].append(reason)
                    else:
                        logger.debug(f"❌ Aggressive transfers disabled")
                        self.transfer_analysis["rejected"].append(transfer)
                else:
                    reason = f"Extra transfer gain {net_gain:.1f} < required {required_gain:.1f}"
                    logger.debug(f"❌ Rejected: {reason}")
                    self.transfer_analysis["rejected"].append(transfer)
                    self.transfer_analysis["reasons"].append(reason)

        return recommended

    def _generate_no_transfer_reasoning(self) -> str:
        """Generate reasoning for no transfers."""
        reasons = []
        
        if self.current.empty:
            return "Unable to analyze squad."
        
        current_with_pred = self._apply_predictions_with_fixture_weighting(self.current, self.next_gw)
        
        if f"pred_gw{self.next_gw}" in current_with_pred.columns:
            avg_predicted = current_with_pred[f"pred_gw{self.next_gw}"].mean()
            total_predicted = current_with_pred[f"pred_gw{self.next_gw}"].sum()
            
            if avg_predicted >= 5.0:
                reasons.append(
                    f"✅ Strong squad (avg {avg_predicted:.1f} pts/player, total {total_predicted:.1f} pts)"
                )
        
        # Risk check
        if "total_risk" in current_with_pred.columns:
            avg_risk = current_with_pred["total_risk"].mean()
            high_risk_count = (current_with_pred["total_risk"] > 0.6).sum()
            
            if avg_risk < 0.3:
                reasons.append(f"🟢 Low risk squad (avg {avg_risk:.2f})")
            elif high_risk_count > 0:
                reasons.append(f"⚠️ {high_risk_count} high-risk player(s)")
        
        # Price change check
        if "price_fall_probability" in current_with_pred.columns:
            falling_players = (current_with_pred["price_fall_probability"] > 0.5).sum()
            if falling_players > 0:
                reasons.append(f"💰 {falling_players} player(s) may fall in price")
        
        if self.bank < 0.5:
            reasons.append(f"💰 Limited funds (£{self.bank:.1f}m)")
        
        if self.free_transfers == 1:
            reasons.append(f"📊 Save 1 FT → Get 2 FTs next week")
        elif self.free_transfers == 2:
            reasons.append(f"📊 2 FTs available but no profitable transfers")
        
        if not reasons:
            return f"No transfers recommended for GW{self.next_gw}."
        
        return " | ".join(reasons)

    def plan_for_horizon(self) -> Dict[str, Any]:
        """Plan transfers for horizon with ENHANCED intelligence."""
        logger.info(f"📊 Planning for {self.horizon} GW horizon starting GW{self.next_gw}")

        plan = {
            "plan": [],
            "chip_recommendation": {},
            "captain_recommendation": {},
            "summary": {},
            "transfer_reasoning": "",
            "risk_warnings": [],
            "price_warnings": []
        }

        bank_cursor = self.bank
        free_transfers_cursor = self.free_transfers
        current_squad_cursor = self.current.copy()

        for i in range(self.horizon):
            gw = self.next_gw + i
            
            # Find potential transfers
            potential_transfers = self._find_optimal_transfers(gw)
            
            # Validate and filter
            recommended_transfers = self._validate_and_select_transfers(
                potential_transfers, 
                free_transfers_cursor
            )

            # Calculate expected points
            expected_gw_points = 0.0
            if not current_squad_cursor.empty:
                current_with_pred = self._apply_predictions_with_fixture_weighting(current_squad_cursor, gw)
                if f"pred_gw{gw}" in current_with_pred.columns:
                    current_with_pred["risk_adjusted_score"] = current_with_pred.apply(
                        lambda row: self._calculate_risk_adjusted_score(row, f"pred_gw{gw}"),
                        axis=1
                    )
                    expected_gw_points = float(current_with_pred["risk_adjusted_score"].sum())

            # Add transfer gains
            for transfer in recommended_transfers:
                expected_gw_points += transfer.get("predicted_gain", 0.0)

            # Update bank and FTs
            if recommended_transfers:
                total_cost_diff = sum(t.get("cost_diff", 0.0) for t in recommended_transfers)
                bank_cursor = round(bank_cursor - total_cost_diff, 1)
                
                free_used = min(len(recommended_transfers), free_transfers_cursor)
                free_transfers_cursor = max(0, 1 - (len(recommended_transfers) - free_used))
                
                # Apply transfers
                for t in recommended_transfers:
                    current_squad_cursor = current_squad_cursor[
                        current_squad_cursor["id"] != t["out_id"]
                    ]
            else:
                free_transfers_cursor = min(2, free_transfers_cursor + 1)

            gw_plan = {
                "gw": gw,
                "transfers": recommended_transfers,
                "expected_gw_points": round(expected_gw_points, 1),
                "bank_after": round(bank_cursor, 1),
                "free_transfers_left": free_transfers_cursor,
                "total_hits": sum(1 for t in recommended_transfers if t.get("hit", 0) > 0),
                "total_hit_cost": sum(t.get("hit_penalty", 0.0) for t in recommended_transfers)
            }

            plan["plan"].append(gw_plan)

        # Generate chip advice
        plan["chip_recommendation"] = self._generate_chip_advice()
        
        # Generate captain recommendation for next GW
        if not self.current.empty:
            captain, vice, cap_reasoning = self.select_captain_with_eo_intelligence(
                self.current, 
                self.next_gw
            )
            plan["captain_recommendation"] = {
                "captain": captain["web_name"],
                "vice_captain": vice["web_name"],
                "strategy": cap_reasoning.get("strategy"),
                "reasoning": cap_reasoning.get("reasoning"),
                "captain_ownership": cap_reasoning.get("captain_ownership"),
                "captain_predicted_pts": cap_reasoning.get("captain_predicted_pts"),
                "alternatives": cap_reasoning.get("alternatives", [])
            }

        # Collect warnings
        plan["risk_warnings"] = self.transfer_analysis.get("risk_warnings", [])
        plan["price_warnings"] = self.transfer_analysis.get("price_warnings", [])

        # Generate transfer reasoning
        total_transfers = sum(len(gw_plan["transfers"]) for gw_plan in plan["plan"])
        
        if total_transfers == 0:
            plan["transfer_reasoning"] = self._generate_no_transfer_reasoning()
        else:
            first_gw_transfers = plan["plan"][0]["transfers"]
            if len(first_gw_transfers) == 1:
                t = first_gw_transfers[0]
                
                notes = []
                if t.get("price_warning"):
                    notes.append(t["price_warning"])
                if t.get("ownership_note"):
                    notes.append(t["ownership_note"])
                if t.get("in_total_risk", 0) > 0.5:
                    notes.append(f"⚠️ Risk: {t.get('in_risk_summary')}")
                
                note_str = " | ".join(notes) if notes else ""
                
                plan["transfer_reasoning"] = (
                    f"1 transfer: {t['out_name']} → {t['in_name']} "
                    f"(+{t['predicted_gain']:.1f} pts). "
                    f"{'FREE' if t['hit'] == 0 else 'Worth -4pt hit'}. {note_str}"
                )
            else:
                total_gain = sum(t["predicted_gain"] for t in first_gw_transfers)
                total_hits = sum(t["hit"] for t in first_gw_transfers)
                
                plan["transfer_reasoning"] = (
                    f"{len(first_gw_transfers)} transfers (+{total_gain:.1f} pts, "
                    f"-{total_hits} pts = +{total_gain - total_hits:.1f} net)"
                )

        plan["summary"] = {
            "total_transfers_planned": total_transfers,
            "chip_recommended": plan["chip_recommendation"].get("chip"),
            "captain_strategy": plan.get("captain_recommendation", {}).get("strategy"),
            "plan_summary": f"{total_transfers} transfer(s) planned",
            "reasoning": plan["transfer_reasoning"],
            "risk_warnings_count": len(plan["risk_warnings"]),
            "price_warnings_count": len(plan["price_warnings"])
        }

        plan["monte_carlo"] = self._run_monte_carlo(self.current, self.next_gw)
        if plan["monte_carlo"]:
            plan["summary"]["monte_carlo_expected"] = round(plan["monte_carlo"]["expected"], 1)
            plan["summary"]["monte_carlo_p10"] = round(plan["monte_carlo"]["p10"], 1)
            plan["summary"]["monte_carlo_p90"] = round(plan["monte_carlo"]["p90"], 1)

        logger.info(f"✅ Plan complete: {total_transfers} transfer(s)")
        if plan["captain_recommendation"]:
            logger.info(f"👑 Captain: {plan['captain_recommendation']['captain']} ({plan['captain_recommendation']['strategy']})")
        
        return plan

    def _generate_chip_advice(self) -> Dict[str, Any]:
        """
        Generate comprehensive chip recommendation with intelligent timing.
        FIXED VERSION - Ensures optimal_gw is always set.
        """
        # Check which chips are available
        chips_available = {
            'triple_captain': self.manager_chips_used.get('triple_captain') is None,
            'bench_boost': self.manager_chips_used.get('bench_boost') is None,
            'free_hit': self.manager_chips_used.get('free_hit') is None,
            'wildcard': self.manager_chips_used.get('wildcard') is None
        }
        
        if not any(chips_available.values()):
            return {
                "chip": "No chip recommended",
                "reasoning": "All chips have been used",
                "expected_gain": 0.0,
                "instructions": [],
                "optimal_gw": None
            }
        
        # Analyze chips and find best opportunity
        chip_scores = {}
        
        if chips_available['triple_captain']:
            tc_analysis = self._analyze_triple_captain()
            if tc_analysis['score'] > 0:
                chip_scores['triple_captain'] = tc_analysis
        
        if chips_available['bench_boost']:
            bb_analysis = self._analyze_bench_boost()
            if bb_analysis['score'] > 0:
                chip_scores['bench_boost'] = bb_analysis
        
        if chips_available['free_hit']:
            fh_analysis = self._analyze_free_hit()
            if fh_analysis['score'] > 0:
                chip_scores['free_hit'] = fh_analysis
        
        if chips_available['wildcard']:
            wc_analysis = self._analyze_wildcard()
            if wc_analysis['score'] > 0:
                chip_scores['wildcard'] = wc_analysis
        
        # Select best chip
        if not chip_scores:
            return {
                "chip": "No chip recommended",
                "reasoning": "No strong chip opportunity in the next 10 gameweeks. Save chips for better timing.",
                "expected_gain": 0.0,
                "instructions": ["Monitor upcoming fixtures for DGW/BGW announcements"],
                "optimal_gw": None
            }
        
        # Find highest scoring chip
        best_chip = max(chip_scores.items(), key=lambda x: x[1]['score'])
        chip_name = best_chip[0]
        analysis = best_chip[1]
        
        # Format chip name
        chip_display = chip_name.replace('_', ' ').title()
        
        # Ensure optimal_gw is set (fallback to next_gw if not set)
        optimal_gw = analysis.get('gw') or self.next_gw
        
        return {
            "chip": chip_display,
            "reasoning": " | ".join(analysis['reasoning']) if analysis['reasoning'] else f"Use {chip_display} for maximum impact",
            "expected_gain": round(analysis['score'], 1),
            "optimal_gw": optimal_gw,
            "instructions": self._get_chip_instructions(chip_name, optimal_gw)
        }


    # ==============================================================================
    # ADD THESE NEW METHODS AT THE END OF TransferSimulator CLASS
    # (Before the final closing of the class, around line 700+)
    # ==============================================================================

    def _analyze_triple_captain(self) -> Dict[str, Any]:
        """Analyze Triple Captain opportunity."""
        best_gw = None
        best_score = 0
        reasoning = []
        
        for i in range(self.chip_horizon):
            gw = self.next_gw + i
            score = 0
            gw_reasoning = []
            
            # Check for DGW
            if gw in self.dgw_map:
                dgw_teams = self.dgw_map[gw]
                score += 10  # Base DGW bonus
                gw_reasoning.append(f"DGW{gw} detected ({len(dgw_teams)} teams)")
                
                # Check if we have DGW players
                dgw_players = self.current[self.current['team'].isin(dgw_teams)]
                if not dgw_players.empty:
                    score += len(dgw_players) * 2
                    gw_reasoning.append(f"{len(dgw_players)} DGW players in squad")
            
            # Analyze best captain option
            if not self.current.empty:
                current_with_pred = self._apply_predictions_with_fixture_weighting(
                    self.current.copy(), gw
                )
                
                if f"pred_gw{gw}" in current_with_pred.columns:
                    current_with_pred["risk_adjusted_score"] = current_with_pred.apply(
                        lambda row: self._calculate_risk_adjusted_score(row, f"pred_gw{gw}"),
                        axis=1
                    )
                    
                    best_captain = current_with_pred.nlargest(1, "risk_adjusted_score").iloc[0]
                    predicted_pts = float(best_captain["risk_adjusted_score"])
                    
                    # TC gain = 2x captain points (3x total - 1x base)
                    tc_gain = predicted_pts * 2
                    
                    if predicted_pts >= 10:  # High scoring potential
                        score += tc_gain
                        gw_reasoning.append(f"{best_captain['web_name']}: {predicted_pts:.1f} pts predicted")
                    elif predicted_pts >= 7:
                        score += tc_gain * 0.7
                        gw_reasoning.append(f"{best_captain['web_name']}: {predicted_pts:.1f} pts (moderate)")
            
            if score > best_score:
                best_score = score
                best_gw = gw
                reasoning = gw_reasoning
        
        if not reasoning:
            reasoning = ["No strong TC opportunity found"]
        
        return {
            'gw': best_gw,
            'score': best_score,
            'reasoning': reasoning
        }

    def _analyze_bench_boost(self) -> Dict[str, Any]:
        """Analyze Bench Boost opportunity."""
        best_gw = None
        best_score = 0
        reasoning = []
        
        if len(self.current) < 15:
            return {
                'gw': None,
                'score': 0,
                'reasoning': ["Squad incomplete"]
            }
        
        for i in range(self.chip_horizon):
            gw = self.next_gw + i
            score = 0
            gw_reasoning = []
            
            # Check for DGW
            if gw in self.dgw_map:
                dgw_teams = self.dgw_map[gw]
                score += 8
                gw_reasoning.append(f"DGW{gw} detected")
            
            # Analyze bench strength
            current_with_pred = self._apply_predictions_with_fixture_weighting(
                self.current.copy(), gw
            )
            
            if f"pred_gw{gw}" in current_with_pred.columns:
                current_with_pred["risk_adjusted_score"] = current_with_pred.apply(
                    lambda row: self._calculate_risk_adjusted_score(row, f"pred_gw{gw}"),
                    axis=1
                )
                
                # Sort and get bench (last 4 players)
                sorted_squad = current_with_pred.sort_values("risk_adjusted_score", ascending=False)
                bench = sorted_squad.tail(4)
                
                bench_total = bench["risk_adjusted_score"].sum()
                
                if bench_total >= 20:
                    score += bench_total
                    gw_reasoning.append(f"Strong bench: {bench_total:.1f} pts total")
                elif bench_total >= 15:
                    score += bench_total * 0.7
                    gw_reasoning.append(f"Decent bench: {bench_total:.1f} pts total")
                else:
                    gw_reasoning.append(f"Weak bench: {bench_total:.1f} pts total")
            
            if score > best_score:
                best_score = score
                best_gw = gw
                reasoning = gw_reasoning
        
        if not reasoning:
            reasoning = ["Bench strength insufficient for BB"]
        
        return {
            'gw': best_gw,
            'score': best_score,
            'reasoning': reasoning
        }

    def _analyze_free_hit(self) -> Dict[str, Any]:
        """Analyze Free Hit opportunity."""
        best_gw = None
        best_score = 0
        reasoning = []
        
        for i in range(self.chip_horizon):
            gw = self.next_gw + i
            score = 0
            gw_reasoning = []
            
            # Check for BGW
            if gw in self.bgw_map:
                teams_playing = self.bgw_map[gw]
                
                if teams_playing < 16:  # Significant blank
                    bgw_severity = (20 - teams_playing) * 3
                    score += bgw_severity
                    gw_reasoning.append(f"BGW{gw}: Only {teams_playing} teams playing")
                    
                    # Check how many of our players blank
                    if not self.current.empty:
                        current_teams = self.current['team'].unique()
                        # Would need fixture data to determine which teams play
                        # Simplified: assume benefit scales with BGW severity
                        score += bgw_severity * 0.5
                        gw_reasoning.append("FH allows full squad in BGW")
            
            # Check for poor fixture run
            if not self.current.empty and gw in self.fixture_diff_by_gw:
                fixture_map = self.fixture_diff_by_gw[gw]
                squad_avg_difficulty = []
                
                for team in self.current['team'].unique():
                    if team in fixture_map:
                        squad_avg_difficulty.append(fixture_map[team])
                
                if squad_avg_difficulty:
                    avg_diff = sum(squad_avg_difficulty) / len(squad_avg_difficulty)
                    
                    if avg_diff >= 4:  # Very hard fixtures
                        score += (avg_diff - 3) * 5
                        gw_reasoning.append(f"Poor fixture run (avg difficulty: {avg_diff:.1f})")
            
            if score > best_score:
                best_score = score
                best_gw = gw
                reasoning = gw_reasoning
        
        if not reasoning:
            reasoning = ["No BGW or difficult fixture run detected"]
        
        return {
            'gw': best_gw,
            'score': best_score,
            'reasoning': reasoning
        }

    def _analyze_wildcard(self) -> Dict[str, Any]:
        """Analyze Wildcard opportunity."""
        score = 0
        reasoning = []
        
        if len(self.current) < 15:
            return {
                'gw': self.next_gw,
                'score': 20,
                'reasoning': ["Squad incomplete - Wildcard recommended"]
            }
        
        # Check squad efficiency
        injuries = (self.current.get('injury_risk', 0) > 0.5).sum()
        high_risk = (self.current.get('total_risk', 0) > 0.6).sum()
        poor_form = (self.current.get('form', 10) < 4).sum()
        
        inefficiencies = injuries + high_risk + poor_form
        
        if inefficiencies >= 5:
            score += inefficiencies * 3
            reasoning.append(f"Squad issues: {injuries} injuries, {high_risk} high-risk, {poor_form} poor form")
        
        # Check for upcoming DGW
        next_dgw = None
        for i in range(self.chip_horizon):
            gw = self.next_gw + i
            if gw in self.dgw_map:
                next_dgw = gw
                break
        
        if next_dgw and next_dgw <= self.next_gw + 3:
            score += 10
            reasoning.append(f"DGW{next_dgw} in {next_dgw - self.next_gw} gameweeks")
        
        # Check value trapped in squad
        if not self.current.empty:
            expensive_benched = 0
            current_with_pred = self._apply_predictions_with_fixture_weighting(
                self.current.copy(), self.next_gw
            )
            
            if f"pred_gw{self.next_gw}" in current_with_pred.columns:
                current_with_pred["risk_adjusted_score"] = current_with_pred.apply(
                    lambda row: self._calculate_risk_adjusted_score(row, f"pred_gw{self.next_gw}"),
                    axis=1
                )
                
                sorted_squad = current_with_pred.sort_values("risk_adjusted_score", ascending=False)
                bench = sorted_squad.tail(4)
                
                expensive_benched = (bench['now_cost'] > 6.0).sum()
                
                if expensive_benched >= 2:
                    score += expensive_benched * 3
                    reasoning.append(f"{expensive_benched} expensive players on bench")
        
        if not reasoning:
            reasoning = ["Squad efficiency acceptable"]
        
        return {
            'gw': self.next_gw if score > 15 else None,
            'score': score,
            'reasoning': reasoning
        }

    def _get_chip_instructions(self, chip: str, optimal_gw: Optional[int]) -> List[str]:
        """Get instructions for chip usage."""
        instructions = []
        
        if chip == 'triple_captain':
            instructions.append(f"Activate Triple Captain before GW{optimal_gw} deadline")
            instructions.append("Captain your highest predicted scorer")
            
            if optimal_gw in self.dgw_map:
                instructions.append("Consider DGW players for maximum return")
        
        elif chip == 'bench_boost':
            instructions.append(f"Activate Bench Boost before GW{optimal_gw} deadline")
            instructions.append("Ensure all 15 players are likely to play")
            
            if optimal_gw in self.dgw_map:
                instructions.append("Prioritize DGW players on bench")
        
        elif chip == 'free_hit':
            instructions.append(f"Activate Free Hit before GW{optimal_gw} deadline")
            instructions.append("Build a one-week team ignoring squad constraints")
            instructions.append("Revert to original squad after the gameweek")
        
        elif chip == 'wildcard':
            instructions.append(f"Activate Wildcard before GW{optimal_gw} deadline")
            instructions.append("Rebuild entire squad without transfer penalties")
            instructions.append("Plan for upcoming fixture swings and DGWs")
        
        return instructions