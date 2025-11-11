"""
Enhanced Pydantic schemas with ownership, price intelligence, and formation data.

NEW FIELDS:
✅ Ownership categories (template/differential)
✅ Price change probabilities
✅ Effective ownership (EO) calculations
✅ Captain strategy reasoning
✅ Formation validation
✅ Bench strength metrics

PRODUCTION READY v6.0
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment for a player."""
    total_risk: float
    risk_category: str
    injury_risk: Optional[float] = None
    rotation_risk: Optional[float] = None
    disciplinary_risk: Optional[float] = None
    fatigue_risk: Optional[float] = None
    form_drop_risk: Optional[float] = None
    risk_summary: Optional[str] = None


class OwnershipIntelligence(BaseModel):
    """Ownership and template analysis."""
    selected_by_percent: float
    ownership_category: str  # "Essential", "Template", "Popular", "Standard", "Differential"
    is_template: bool  # >35% owned
    is_differential: bool  # <5% owned
    captain_eo_multiplier: float  # Effective ownership multiplier for captaincy
    template_priority: Optional[float] = None  # Must-have score


class PriceIntelligence(BaseModel):
    """Price change prediction and value analysis."""
    price_rise_probability: float  # 0.0 to 1.0
    price_fall_probability: float  # 0.0 to 1.0
    price_change_status: str  # "Rising", "Falling", "Stable"
    net_transfers: int  # Transfers in - transfers out
    value_hold_score: Optional[float] = None  # Score for holding rising players
    opportunity_cost: Optional[float] = None  # Cost of keeping falling players


class PlayerPick(BaseModel):
    """Enhanced player details with ownership and price intelligence."""
    player_name: str
    position: str
    team_name: str
    predicted_points_next_gw: float
    risk_adjusted_points: Optional[float] = None
    fixture_difficulty: Optional[int] = None
    form: Optional[float] = None
    expected_gain: Optional[float] = None
    reasoning: Optional[str] = None
    
    # Fixture information
    next_opponent: Optional[str] = None
    next_opponent_short: Optional[str] = None
    is_home: Optional[bool] = None
    fixture_display: Optional[str] = None
    
    # Risk information
    risk_assessment: Optional[RiskAssessment] = None
    minutes_per_game: Optional[int] = None
    yellow_cards: Optional[int] = None
    chance_of_playing: Optional[int] = None
    
    # NEW: Ownership information
    ownership_intel: Optional[OwnershipIntelligence] = None
    
    # NEW: Price information
    price_intel: Optional[PriceIntelligence] = None
    current_price: Optional[float] = None


class TransferItem(BaseModel):
    """Enhanced transfer recommendation with ownership and price intelligence."""
    transfer: str
    details: str
    
    # Fixture comparison
    out_opponent: Optional[str] = None
    in_opponent: Optional[str] = None
    fixture_upgrade: Optional[str] = None
    
    # Risk comparison
    out_risk_summary: Optional[str] = None
    in_risk_summary: Optional[str] = None
    in_total_risk: Optional[float] = None
    risk_warning: Optional[str] = None
    
    # NEW: Ownership comparison
    out_ownership: Optional[float] = None  # % owned
    in_ownership: Optional[float] = None   # % owned
    ownership_note: Optional[str] = None   # "Template", "Differential", etc.
    
    # NEW: Price change intelligence
    out_price_fall_prob: Optional[float] = None
    in_price_rise_prob: Optional[float] = None
    price_warning: Optional[str] = None  # Price change urgency


class CaptaincyRecommendation(BaseModel):
    """Enhanced captain recommendation with EO intelligence."""
    captain: str
    vice_captain: str
    reasoning: str
    
    # Fixture info
    captain_opponent: Optional[str] = None
    captain_fixture_display: Optional[str] = None
    
    # Risk info
    captain_risk_summary: Optional[str] = None
    captain_risk_category: Optional[str] = None
    
    # NEW: Ownership and EO intelligence
    captain_strategy: Optional[str] = None  # "TEMPLATE", "DIFFERENTIAL", "BEST_AVAILABLE"
    captain_ownership: Optional[float] = None  # % owned
    captain_predicted_pts: Optional[float] = None
    captain_eo_value: Optional[float] = None  # EO-adjusted value
    
    # NEW: Alternative captains
    alternatives: Optional[List[Dict[str, Any]]] = None


class FormationAnalysis(BaseModel):
    """Formation validation and bench strength analysis."""
    is_valid_formation: bool
    valid_formations: List[str]  # E.g., ["3-4-3", "3-5-2"]
    position_breakdown: Dict[str, int]  # {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    
    # Bench analysis
    bench_strength: Optional[float] = None  # Average predicted points
    playing_bench_count: Optional[int] = None  # Players expected to play
    has_gk_cover: Optional[bool] = None
    has_def_cover: Optional[bool] = None
    bench_fodder_count: Optional[int] = None


class EffectiveOwnership(BaseModel):
    """Effective ownership calculations for squad."""
    total_eo: float  # Sum of EO across squad
    average_eo: float  # Average EO per player
    template_count: int  # High ownership players (>35%)
    differential_count: int  # Low ownership players (<5%)
    captain_eo: float  # Captain's effective ownership


class ChipAdvice(BaseModel):
    """Chip usage suggestion with fixture weighting and timing logic."""
    recommended_chip: Optional[str] = None
    reasoning: Optional[str] = None
    expected_gain: Optional[float] = None
    instructions: Optional[List[str]] = None
    optimal_gw: Optional[int] = None


class Summary(BaseModel):
    """Enhanced summary with ownership and price intelligence."""
    expected_points_next_gw: float
    bank_start: float
    bank_after_plan: float
    free_transfers_start: int
    free_transfers_available: int
    free_transfers_left_after_plan: int
    total_transfers_recommended: int
    transfer_reasoning: Optional[str] = None
    notes: str
    
    # Risk summary
    squad_average_risk: Optional[float] = None
    high_risk_players_count: Optional[int] = None
    risk_warnings_count: Optional[int] = None
    
    # NEW: Price change summary
    price_warnings_count: Optional[int] = None
    players_likely_to_rise: Optional[int] = None
    players_likely_to_fall: Optional[int] = None
    
    # NEW: Ownership summary
    squad_template_count: Optional[int] = None
    squad_differential_count: Optional[int] = None
    squad_average_ownership: Optional[float] = None

class ModelPerformance(BaseModel):
    """Model performance metrics."""
    accuracy_score: float  # Overall prediction accuracy
    precision_score: float  # Precision of high-scoring predictions
    efficiency_score: float  # Transfer recommendation efficiency
    rmse: Optional[float] = None  # Root mean squared error
    mae: Optional[float] = None  # Mean absolute error
    r2_score: Optional[float] = None  # R² score
    training_date: Optional[str] = None
    samples_evaluated: Optional[int] = None


class DetailedRecommendation(BaseModel):
    """Full recommendation response with comprehensive intelligence."""
    manager_id: int
    target_gameweek: int
    summary: Summary
    recommended_transfers: List[TransferItem]
    recommended_picks: List[PlayerPick]
    captaincy: CaptaincyRecommendation
    chip_advice: ChipAdvice
    instructions: List[str]
    risk_warnings: Optional[List[str]] = None
    price_warnings: Optional[List[str]] = None
    formation_analysis: Optional[FormationAnalysis] = None
    effective_ownership: Optional[EffectiveOwnership] = None
    model_performance: Optional[ModelPerformance] = None  # NEW
