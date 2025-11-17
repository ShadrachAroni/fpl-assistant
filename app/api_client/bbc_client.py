"""
BBC Sport scraper utilities for line-up and team news collection.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("bbc_client")
logger.setLevel(logging.INFO)


class BBCLineupClient:
    SCORES_URL = "https://www.bbc.co.uk/sport/football/scores-fixtures/{date}"
    MATCH_DATA_URL = "https://push.api.bbci.co.uk/beta/sport/football/match-data/{event_id}"

    def __init__(self, timeout: float = 15.0) -> None:
        self.session = httpx.Client(timeout=timeout)

    # ---------- Fixtures / event lookup ----------

    def find_event_id(self, date: datetime, home_team: str, away_team: str) -> Optional[str]:
        """
        Locate the BBC event id for a specific fixture by scraping the scores page.
        """
        date_str = date.strftime("%Y-%m-%d")
        url = self.SCORES_URL.format(date=date_str)

        try:
            resp = self.session.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f"⚠️ BBC scores fetch failed for {date_str}: {exc}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        fixtures = soup.select("[data-event-id]")

        if not fixtures:
            logger.debug(f"No BBC fixtures found for {date_str}")
            return None

        home_key = self._normalize_team(home_team)
        away_key = self._normalize_team(away_team)

        for node in fixtures:
            event_id = node.attrs.get("data-event-id")
            home_label = node.select_one(".sp-c-fixture__team--home .sp-c-fixture__team-name")
            away_label = node.select_one(".sp-c-fixture__team--away .sp-c-fixture__team-name")

            if not home_label or not away_label:
                continue

            if (
                self._normalize_team(home_label.get_text(strip=True)) == home_key
                and self._normalize_team(away_label.get_text(strip=True)) == away_key
            ):
                return event_id

        return None

    # ---------- Line-up data ----------

    def get_match_lineups(self, event_id: str) -> Dict[str, Any]:
        """
        Pull line-up data from the BBC match-data endpoint.
        Returns a dict with home/away formations and starter/bench lists.
        """
        try:
            resp = self.session.get(self.MATCH_DATA_URL.format(event_id=event_id))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f"⚠️ BBC lineup fetch failed for event {event_id}: {exc}")
            return {}

        data = resp.json()
        match = data.get("match") or {}
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}

        parsed = {
            "event_id": event_id,
            "home": self._parse_team_lineup(home),
            "away": self._parse_team_lineup(away),
            "updated_at": match.get("updatedAt") or data.get("lastUpdated"),
        }

        return parsed

    # ---------- Helpers ----------

    @staticmethod
    def _parse_team_lineup(team_block: Dict[str, Any]) -> Dict[str, Any]:
        players: List[Dict[str, Any]] = []
        starters: List[Dict[str, Any]] = []
        bench: List[Dict[str, Any]] = []

        squad = team_block.get("players") or team_block.get("squad") or []

        for player in squad:
            info = {
                "name": player.get("fullName") or player.get("name") or player.get("shortName"),
                "position": player.get("position"),
                "role": player.get("role") or player.get("status"),
            }
            players.append(info)

            role = (player.get("role") or "").lower()
            status = (player.get("status") or "").lower()

            if role in {"starting", "starter"} or status in {"confirmed", "starting"}:
                starters.append(info)
            elif role in {"substitute", "bench"} or status == "sub":
                bench.append(info)

        return {
            "team_name": team_block.get("name"),
            "formation": team_block.get("formation"),
            "players": players,
            "starters": starters,
            "bench": bench,
        }

    @staticmethod
    def _normalize_team(value: str) -> str:
        return (
            value.lower()
            .replace("fc", "")
            .replace("afc", "")
            .replace("the ", "")
            .replace("women", "")
            .strip()
        )

