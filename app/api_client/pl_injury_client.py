"""
Premier League injury hub scraper using the public PulseLive football API.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger("pl_injury_client")
logger.setLevel(logging.INFO)


class PremierLeagueInjuryClient:
    """
    Fetches publicly available injury/suspension information from the
    Premier League's PulseLive API. No authentication is required, but
    the endpoint expects specific headers mimicking the website.
    """

    BASE_URL = "https://footballapi.pulselive.com/football/players"
    DEFAULT_HEADERS = {
        "Origin": "https://www.premierleague.com",
        "Referer": "https://www.premierleague.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    def __init__(
        self,
        comp_seasons: Optional[List[int]] = None,
        page_size: int = 40,
        timeout: float = 15.0,
    ) -> None:
        self.comp_seasons = comp_seasons or [719]  # 2024/25 default
        self.page_size = page_size
        self.session = httpx.Client(timeout=timeout)

    def list_injuries(self) -> List[Dict[str, Any]]:
        """
        Returns all players currently marked with an injury/suspension flag.
        """
        results: List[Dict[str, Any]] = []
        page = 0

        while True:
            params = {
                "page": page,
                "pageSize": self.page_size,
                "injuries": "true",
                "type": "player",
                "altIds": "true",
                "compSeasons": ",".join(map(str, self.comp_seasons)),
                "detail": "2",
            }

            try:
                resp = self.session.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.DEFAULT_HEADERS,
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(f"⚠️ Premier League injury fetch failed: {exc}")
                break

            payload = resp.json()
            content = payload.get("content") or []
            if not content:
                break

            results.extend(content)

            page_info = payload.get("pageInfo") or {}
            total_pages = page_info.get("numPages") or page_info.get("pages") or 1
            page += 1

            if page >= total_pages:
                break

        logger.info(f"🩺 Premier League injury feed returned {len(results)} players")
        return results

