# Quick Fix Script - Run this FIRST before training
# Save as: clear_cache_and_test.py

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache_cleaner")

# Clear cache
cache_dir = Path("data/cache/historical")
if cache_dir.exists():
    logger.info(f"🗑️ Removing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
    logger.info("✅ Cache cleared")
else:
    logger.info("ℹ️ Cache directory doesn't exist")

# Recreate empty cache directory
cache_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"📁 Created fresh cache directory: {cache_dir}")

# Test download
logger.info("\n🧪 Testing data download...")
from app.data.historical_integrator import HistoricalDataIntegrator

integrator = HistoricalDataIntegrator(
    cache_dir="data/cache/historical",
    current_season="2024-25"
)

# Test single season first
logger.info("\n📥 Testing single season download (2023-24)...")
df = integrator.get_merged_gameweek_data(season="2023-24", force_refresh=True)

if not df.empty:
    logger.info(f"✅ Single season test: {len(df)} records")
    logger.info(f"   Index unique: {df.index.is_unique}")
    logger.info(f"   Columns: {len(df.columns)}")
    logger.info(f"   Sample columns: {list(df.columns)[:10]}")
else:
    logger.error("❌ Single season test failed")
    exit(1)

# Test multi-season
logger.info("\n📥 Testing multi-season download...")
df_multi = integrator.get_multi_season_data(
    seasons=["2022-23", "2023-24"],
    include_current_season=False
)

if not df_multi.empty:
    logger.info(f"✅ Multi-season test: {len(df_multi)} records")
    logger.info(f"   Index unique: {df_multi.index.is_unique}")
    logger.info(f"   Seasons: {df_multi['season'].unique()}")
else:
    logger.error("❌ Multi-season test failed")
    exit(1)

logger.info("\n" + "="*60)
logger.info("✅ ALL TESTS PASSED!")
logger.info("="*60)
logger.info("\nYou can now run training:")
logger.info("python -m scripts.run_train")