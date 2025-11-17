"""
Sportmonks API client - focused on pulling cross-league injury intelligence.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("sportmonks_client")
logger.setLevel(logging.INFO)


class SportmonksClient:
    """Lightweight Sportmonks client used for injury intelligence."""

    def __init__(
        self,
        base_url: str,
        api_token: Optional[str] = None,
        injuries_endpoint: str = "v3/football/injuries",
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        token = api_token or os.getenv("SPORTMONKS_API_TOKEN")
        if token:
            token = token.strip()
            if token in {"${SPORTMONKS_API_TOKEN}", "YOUR_API_TOKEN", "YOUR_TOKEN_HERE"}:
                token = ""
        self.api_token = token
        self.injuries_endpoint = injuries_endpoint.lstrip("/")
        self.session = httpx.Client(timeout=timeout)

        if not self.api_token:
            logger.warning("⚠️ SPORTMONKS_API_TOKEN not provided - injury feed disabled")

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_token:
            raise ValueError("Sportmonks API token not configured")

        params = params.copy() if params else {}
        params.setdefault("api_token", self.api_token)

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def list_injuries(
        self,
        league_ids: Optional[List[int]] = None,
        season_ids: Optional[List[int]] = None,
        include_player_details: bool = True,
        per_page: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch latest injuries with optional league filters."""
        if not self.api_token:
            logger.debug("Sportmonks API token not configured; skipping injury fetch")
            return []
        params: Dict[str, Any] = {
            "per_page": per_page,
            "page": 1,
        }

        if include_player_details:
            params["include"] = "player;team;type"

        if league_ids:
            params["filters[league_id]"] = ",".join(map(str, league_ids))

        if season_ids:
            params["filters[season_id]"] = ",".join(map(str, season_ids))

        try:
            payload = self._request(self.injuries_endpoint, params=params)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning(f"⚠️ Sportmonks injuries request failed: {exc}")
            return []

        if isinstance(payload, dict):
            return payload.get("data") or payload.get("response") or []

        if isinstance(payload, list):
            return payload

        return []

