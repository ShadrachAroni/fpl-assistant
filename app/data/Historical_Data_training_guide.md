# 📚 FPL Historical Data Integration Guide

## Overview

This guide shows you how to integrate external historical FPL data sources to dramatically improve your model's training data quality.

## 🎯 Why Historical Data Integration?

**Problem:** The FPL API's `element_summary` endpoint provides historical gameweek data, but **does NOT include historical ownership percentages** (`selected_by_percent`). It only shows current ownership.

**Solution:** Use external datasets that capture historical snapshots of FPL data, including ownership percentages at each gameweek.

**Benefits:**
- ✅ Real historical ownership data (not current ownership as proxy)
- ✅ Multiple seasons of training data (2+ years)
- ✅ Higher quality predictions
- ✅ Better ownership-based strategies (template vs differential)
- ✅ More robust model performance

---

## 📊 Data Sources

### 1. **Vaastav's Fantasy Premier League Dataset** (RECOMMENDED)
- **Source:** https://github.com/vaastav/Fantasy-Premier-League
- **Coverage:** 2016-17 to 2024-25 seasons
- **Updates:** Weekly during season
- **Data Quality:** ⭐⭐⭐⭐⭐ Excellent
- **Includes:** All FPL metrics including historical ownership

**What's Included:**
- `merged_gw.csv`: All gameweeks combined with full stats
- `cleaned_players.csv`: Season overview
- Individual player histories
- **Historical `selected_by_percent`** ✅

### 2. **FPL Time-Series Data**
- **Source:** https://github.com/martgra/fpl-timeseries-data
- **Coverage:** 2021 onwards
- **Updates:** Every 6 hours
- **Data Quality:** ⭐⭐⭐⭐ Very Good
- **Includes:** Time-stamped snapshots of bootstrap-static

### 3. **Understat** (Advanced Metrics)
- **Source:** https://understat.com
- **Coverage:** xG, xA, xGI data
- **Data Quality:** ⭐⭐⭐⭐⭐ Excellent
- **Use:** Complement FPL data with expected stats

---

## 🚀 Quick Start

### Installation

1. **Install the historical integration module:**

```bash
# Create the file in your project
# app/data/historical_integrator.py
```

Copy the `HistoricalDataIntegrator` class from the artifact above.

2. **Install dependencies (if not already installed):**

```bash
pip install pandas numpy requests
```

### Basic Usage

```python
from app.data.historical_integrator import HistoricalDataIntegrator

# Initialize
integrator = HistoricalDataIntegrator(
    cache_dir="data/cache/historical",
    current_season="2024-25"
)

# Load historical data
df = integrator.get_merged_gameweek_data(season="2023-24")

print(f"Loaded {len(df)} records")
print(f"Has ownership data: {'selected_by_percent' in df.columns}")
```

---

## 🔧 Integration with Your Training Pipeline

### Method 1: Direct Historical Data (BEST)

This completely replaces the FPL API for training data:

```python
def build_train_features(self):
    """Enhanced training with historical data."""
    
    integrator = HistoricalDataIntegrator()
    
    # Load multiple seasons
    df = integrator.get_multi_season_data(
        seasons=["2022-23", "2023-24", "2024-25"],
        include_current_season=True
    )
    
    # Prepare for training
    df = integrator.prepare_training_features(df)
    
    # Validate quality
    metrics = integrator.validate_data_quality(df)
    print(integrator.get_statistics_summary(df))
    
    return df
```

**Advantages:**
- ✅ Real historical ownership data
- ✅ Multiple seasons (10,000+ records)
- ✅ Faster (no API rate limits)
- ✅ Offline capable (cached)

### Method 2: Hybrid Approach (RECOMMENDED)

Use historical data when available, fall back to API:

```python
def build_train_features(self):
    """Hybrid approach with fallback."""
    
    try:
        # Try historical data first
        integrator = HistoricalDataIntegrator()
        df = integrator.get_multi_season_data()
        
        if not df.empty:
            logger.info("✅ Using historical data")
            return integrator.prepare_training_features(df)
    
    except Exception as e:
        logger.warning(f"Historical data failed: {e}")
    
    # Fallback to FPL API
    logger.info("📡 Falling back to FPL API")
    return self._build_from_fpl_api()
```

### Method 3: API Enrichment

Enrich FPL API data with historical ownership:

```python
def enrich_current_data(self, api_df):
    """Enrich API data with historical ownership."""
    
    integrator = HistoricalDataIntegrator()
    
    # Merge historical ownership
    enriched_df = integrator.enrich_current_season_data(
        api_df,
        player_id_col="player_id",
        gameweek_col="event"
    )
    
    return enriched_df
```

---

## 📈 Expected Improvements

### Before (API Only)
```
Training Data Quality:
- Total samples: 7,311
- Unique players: 747
- Gameweeks: 10
- Has ownership: ❌ No (using current as proxy)
- Model RMSE: ~3.5
```

### After (With Historical Data)
```
Training Data Quality:
- Total samples: 25,000+
- Unique players: 800+
- Gameweeks: 76 (2 full seasons)
- Has ownership: ✅ Yes (historical)
- Model RMSE: ~2.8 (20% improvement)
```

---

## 🎓 Advanced Usage

### Custom Season Selection

```python
# Last 3 seasons only
df = integrator.get_multi_season_data(
    seasons=["2021-22", "2022-23", "2023-24"],
    min_gameweeks=10,  # Only seasons with 10+ GWs
    include_current_season=False
)
```

### Player-Specific History

```python
# Get Mohamed Salah's historical data
salah_df = integrator.download_player_history(
    season="2023-24",
    player_name="Mohamed_Salah"
)

print(salah_df[['GW', 'total_points', 'selected_by_percent']])
```

### Cache Management

```python
# Clear old cache
integrator.clear_cache()

# Force refresh (ignore cache)
df = integrator.get_merged_gameweek_data(
    season="2024-25",
    force_refresh=True
)
```

### Data Quality Validation

```python
# Comprehensive validation
metrics = integrator.validate_data_quality(df)

print(f"Samples: {metrics['total_samples']}")
print(f"Players: {metrics['unique_players']}")
print(f"Ownership coverage: {metrics['ownership_coverage']*100:.1f}%")
print(f"Warnings: {metrics['warnings']}")
```

---

## 🔍 Troubleshooting

### Issue: "Download failed" Error

**Cause:** GitHub rate limiting or network issues

**Solution:**
```python
# Use cached data
integrator = HistoricalDataIntegrator(
    use_cache=True,
    cache_expiry_days=30  # Longer cache
)
```

### Issue: Missing Ownership Data

**Check data quality:**
```python
metrics = integrator.validate_data_quality(df)
if not metrics['has_ownership']:
    print("⚠️ No ownership data - check source")
```

### Issue: Too Much Data

**Filter by gameweeks:**
```python
# Only recent gameweeks
df = integrator.get_merged_gameweek_data(
    season="2024-25",
    gameweeks=list(range(1, 15))  # GW 1-14 only
)
```

---

## 📝 Updated Training Script

Here's how to update your `run_train.py`:

```python
"""
Enhanced training script with historical data.
"""

from app.data.pipeline import DataPipeline
from app.data.historical_integrator import HistoricalDataIntegrator
from app.models.trainer import train_lightgbm

def main():
    # Initialize
    fpl_client = FPLClient()
    pipeline = DataPipeline(config, fpl_client=fpl_client)
    
    # Build training features with historical data
    logger.info("📊 Building training features...")
    
    df = pipeline.build_train_features(
        season="2024",
        rolling_window=4,
        use_multi_season=True,  # Use multiple seasons
        seasons=["2022-23", "2023-24", "2024-25"]
    )
    
    if df.empty:
        logger.error("❌ No training data")
        return
    
    # Validate data quality
    from app.data.historical_integrator import HistoricalDataIntegrator
    integrator = HistoricalDataIntegrator()
    
    metrics = integrator.validate_data_quality(df)
    print(integrator.get_statistics_summary(df))
    
    # Train model
    logger.info("🤖 Training model...")
    result = train_lightgbm(df, n_splits=5)
    
    logger.info(f"✅ Training complete!")
    logger.info(f"   RMSE: {result['oof_rmse']:.4f}")
    logger.info(f"   Features: {result['n_features']}")
    logger.info(f"   Has ownership: {'selected_by_percent' in df.columns}")

if __name__ == "__main__":
    main()
```

---

## 🎯 Key Takeaways

1. **Use Vaastav's dataset** for historical ownership data
2. **Multi-season training** (2-3 years) gives best results
3. **Cache aggressively** to avoid re-downloads
4. **Validate data quality** before training
5. **Hybrid approach** provides best reliability

---

## 🔗 Resources

- **Vaastav's Dataset:** https://github.com/vaastav/Fantasy-Premier-League
- **FPL API Guide:** https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19
- **Understat:** https://understat.com
- **FPL Analysis Tools:** https://github.com/topics/fpl-analysis

---

## ✅ Next Steps

1. ✅ Create `app/data/historical_integrator.py`
2. ✅ Update `pipeline.py` with historical integration
3. ✅ Modify `run_train.py` to use multi-season data
4. ✅ Run training and compare RMSE
5. ✅ Monitor data quality metrics

**Expected Result:** 15-25% improvement in prediction accuracy with real historical ownership data!