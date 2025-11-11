"""
Model weight configuration and weighting schemes - ENHANCED VERSION

Determines how different features are weighted in transfer recommendations
with full risk assessment and multi-transfer validation.
"""

# === Feature Weights for Transfer Scoring ===

# Prediction confidence weight: how much to trust the ML model vs market trends
MODEL_WEIGHT_DEFAULT = 0.70

# Form weight: recent performance weighting
FORM_WEIGHT = 0.40

# Fixture weight: upcoming opponent difficulty (enhanced with horizon)
FIXTURE_WEIGHT = 0.35  # Increased from 0.30

# Trend weight: global player selection trends
TREND_WEIGHT = 0.30

# Cost efficiency weight
COST_EFFICIENCY_WEIGHT = 0.05

# === Retention Penalty Weights ===

# How much to penalize selling a player with good form
FORM_RETENTION_MULTIPLIER = 0.4

# How much to penalize selling expensive players
VALUE_RETENTION_MULTIPLIER = 0.3

# How much to penalize selling players with easy fixtures
FIXTURE_RETENTION_MULTIPLIER = 0.25  # Increased from 0.20

# === Transfer Evaluation Weights (ENHANCED) ===

# Minimum gain threshold to recommend FREE transfer
MIN_GAIN_THRESHOLD = 1.0

# Minimum gain threshold if using a paid transfer (4pt hit)
MIN_PAID_TRANSFER_THRESHOLD = 6.0  # Increased from 4.5 for safety

# Penalty for recommending duplicate transfers in same GW
DUPLICATE_TRANSFER_PENALTY = 2.0

# === Multi-Transfer Risk Assessment ===

# Extra transfer multiplier - need to justify the -4pt hit
EXTRA_TRANSFER_MULTIPLIER = 1.5  # Need 1.5x the hit cost (6 pts minimum)

# Aggressive multiplier for critical situations
AGGRESSIVE_TRANSFER_MULTIPLIER = 1.2  # Lower threshold if underperforming

# Maximum transfers to consider per gameweek
MAX_TRANSFERS_PER_GW = 2

# === Chip Weighting (ENHANCED) ===

# Triple Captain multiplier
TRIPLE_CAPTAIN_MULTIPLIER = 3.0

# Triple Captain DGW bonus
TRIPLE_CAPTAIN_DGW_BONUS = 1.5

# Minimum predicted points for TC activation
MIN_TC_PREDICTED_POINTS = 12.0

# Bench Boost expected value
BENCH_BOOST_MULTIPLIER = 1.0

# Minimum bench total for BB
MIN_BENCH_BOOST_POINTS = 20.0

# Free Hit restructure value
FREE_HIT_RESTRUCTURE_VALUE = 15.0

# Free Hit blank gameweek priority multiplier
FREE_HIT_BGW_MULTIPLIER = 2.0

# Wildcard restructure threshold
WILDCARD_MIN_EFFICIENCY = 0.75

# === Underperformance Detection (ENHANCED) ===

# Average points threshold to consider manager underperforming
UNDERPERFORMANCE_THRESHOLD = 45

# Rank percentile to trigger aggressive transfers
AGGRESSIVE_TRANSFER_PERCENTILE = 25

# Minimum consecutive poor weeks to trigger aggressive mode
CONSECUTIVE_POOR_WEEKS = 3

# === Dynamic Transfer Logic (ENHANCED) ===

# Enable dynamic 2nd transfer for ABSOLUTELY NECESSARY scenarios
ALLOW_DYNAMIC_TRANSFERS = True

# Gain multiplier required to justify 2nd transfer (4pt hit)
EXTRA_TRANSFER_GAIN_MULTIPLIER = 1.5  # Need 1.5x the hit cost

# Performance trend weight for extra transfer decision
UNDERPERFORMANCE_WEIGHT = 0.6

# Critical situation threshold (injuries, suspensions, etc)
CRITICAL_SITUATION_THRESHOLD = 3

# === Horizon Analysis Weights ===

# Planning horizon in gameweeks
PLANNING_HORIZON = 5

# Chip analysis horizon in gameweeks
CHIP_ANALYSIS_HORIZON = 10

# Future gameweek discount factor (reduce weight for distant GWs)
FUTURE_GW_DISCOUNT = 0.90  # Each GW ahead is worth 90% of previous

# === Fixture Difficulty Thresholds ===

# Easy fixture threshold
EASY_FIXTURE_THRESHOLD = 2

# Hard fixture threshold
HARD_FIXTURE_THRESHOLD = 4

# Fixture run length to analyze
FIXTURE_RUN_LENGTH = 5

# === Double/Blank Gameweek Weights ===

# DGW player value multiplier
DGW_MULTIPLIER = 1.8

# BGW player penalty multiplier
BGW_PENALTY = 0.3

# Minimum DGW players to avoid Free Hit
MIN_DGW_PLAYERS_WITHOUT_FH = 8

# === Squad Structure Validation ===

# Maximum players per team
MAX_PLAYERS_PER_TEAM = 3

# Required positions (GK-DEF-MID-FWD)
REQUIRED_POSITIONS = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3
}

# Minimum starting XI quality threshold
MIN_STARTING_XI_AVG_POINTS = 5.0

# === Risk Assessment Thresholds ===

# Variance threshold for risky players
HIGH_VARIANCE_THRESHOLD = 3.0

# Injury risk penalty
INJURY_RISK_PENALTY = 0.5

# Rotation risk penalty
ROTATION_RISK_PENALTY = 0.3

# Form drop threshold (consecutive low scores)
FORM_DROP_THRESHOLD = 2

# === Transfer Timing Weights ===

# Bonus for early transfers (before deadline)
EARLY_TRANSFER_BONUS = 0.1

# Penalty for late transfers (after team news)
LATE_TRANSFER_PENALTY = 0.2

# Deadline day penalty (higher risk)
DEADLINE_DAY_PENALTY = 0.3

# === Chip Timing Optimization ===

# Optimal GW for Triple Captain (relative to current)
OPTIMAL_TC_GW_RANGE = (0, 7)  # Within next 7 gameweeks

# Optimal GW for Bench Boost (relative to current)
OPTIMAL_BB_GW_RANGE = (0, 7)

# Optimal GW for Free Hit (relative to current)
OPTIMAL_FH_GW_RANGE = (0, 7)

# Optimal GW for Wildcard (immediate decision)
OPTIMAL_WC_IMMEDIATE = True

# === Advanced Metrics ===

# xG weight (if available)
XG_WEIGHT = 0.25

# xA weight (if available)
XA_WEIGHT = 0.20

# ICT Index weight
ICT_WEIGHT = 0.15

# Minutes played reliability weight
MINUTES_WEIGHT = 0.30

# Clean sheet probability weight (defenders/keepers)
CLEAN_SHEET_WEIGHT = 0.40

# === Logging and Debug ===

# Enable detailed transfer analysis logging
DETAILED_LOGGING = True

# Log rejected transfers for debugging
LOG_REJECTED_TRANSFERS = True

# Log chip opportunity analysis
LOG_CHIP_ANALYSIS = True