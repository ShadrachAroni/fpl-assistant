# Wrapper for Understat data (xG, shots, etc.)
# app/api_client/understat_client.py
from understat import Understat
import asyncio
import nest_asyncio
import logging
from typing import List, Dict

nest_asyncio.apply()  # to allow nested loops in Jupyter-like environments
logger = logging.getLogger("understat_client")
logger.setLevel(logging.INFO)

class UnderstatClient:
    def __init__(self):
        # Understat uses an async API internally
        self.loop = asyncio.get_event_loop()
        self.understat = Understat(self.loop)

    def get_player_shot_data(self, player_name: str, season: str = "2024") -> List[Dict]:
        """
        Returns shot-by-shot data for a given player name and season.
        Player lookup by name; fuzzy matches may be required for duplicates.
        """
        data = self.loop.run_until_complete(self.understat.get_players_stats(season=season))
        # get matching player
        matches = [p for p in data if p.get("title","").lower() == player_name.lower()]
        if not matches:
            # try substring
            matches = [p for p in data if player_name.lower() in p.get("title","").lower()]
        if not matches:
            return []
        player = matches[0]
        player_id = player['id']
        shots = self.loop.run_until_complete(self.understat.get_player_shots(player_id, season=season))
        return shots

    def get_team_xg(self, team_name: str, season: str = "2024"):
        # Could be implemented via get_team_stats
        return self.loop.run_until_complete(self.understat.get_team_stats(team=team_name, season=season))
