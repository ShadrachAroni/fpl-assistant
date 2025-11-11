# app/api_client/fbref_client.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from typing import Dict, Any, List
logger = logging.getLogger("fbref_client")
logger.setLevel(logging.INFO)

FBREF_BASE = "https://fbref.com"

def fetch_player_stats_fbref(player_slug: str, season_slug: str = "2024-2025"):
    """
    Given a player slug (as used on fbref), fetch per-match and per-season stats.
    This is a basic scraper and must be used carefully (fbref terms).
    """
    url = f"{FBREF_BASE}/en/players/{player_slug}/matchlogs/{season_slug}/summary/{player_slug}-match-log"
    r = requests.get(url, headers={"User-Agent":"fpl-assistant/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # Many fbref tables are encoded in commented HTML; robust scraping needed
    tables = soup.find_all("table")
    if not tables:
        return {}
    frames = pd.read_html(str(tables[0]))
    return frames[0].to_dict(orient="records")
