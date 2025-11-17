"""
Rotation monitor to track managers/teams that frequently rotate their
starting XI. Uses official FPL element history data so it updates as
soon as a new gameweek finishes.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.api_client.fpl_client import FPLClient

logger = logging.getLogger("rotation_monitor")
logger.setLevel(logging.INFO)


class RotationMonitor:
    """
    Periodically builds a rotation report:
      - Pulls starter information for the last N gameweeks
      - Scores volatility for every Premier League club
      - Generates a JSON cache consumed by the main data pipeline
    """

    def __init__(
        self,
        fpl_client: Optional[FPLClient],
        config: Dict[str, Any],
        window_gws: Optional[int] = None,
        starter_minutes: Optional[int] = None,
        prone_threshold: Optional[float] = None,
    ) -> None:
        self.fpl = fpl_client or FPLClient()

        risk_cfg = (config or {}).get("risk_assessment", {})
        rotation_cfg = risk_cfg.get("rotation_monitor", {})

        self.window_gws = window_gws or int(rotation_cfg.get("window_gameweeks", 5))
        self.starter_minutes = starter_minutes or int(rotation_cfg.get("starter_minutes", 55))
        self.prone_threshold = prone_threshold or float(rotation_cfg.get("prone_threshold", 0.4))
        self.cache_path = Path(rotation_cfg.get("cache_path", "data/rotation_watch.json"))

        cache_dir = rotation_cfg.get("summary_cache_dir", "data/cache/player_history")
        self.summary_cache_dir = Path(cache_dir)
        self.summary_cache_dir.mkdir(parents=True, exist_ok=True)

        cache_hours = float(rotation_cfg.get("summary_cache_hours", 6))
        self.summary_cache_ttl = timedelta(hours=cache_hours)

        ctx_cfg = (config or {}).get("contextual_intel", {})
        self.manager_directory = ctx_cfg.get("manager_directory", {})

    # ------------------------------------------------------------------ #
    # Public orchestration helpers
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """Build report and persist it to disk."""
        report = self.build_rotation_report()
        self.persist(report)
        self._log_report_summary(report)
        return report

    def build_rotation_report(self) -> Dict[str, Any]:
        """Compute rotation volatility for every team."""
        bootstrap = self.fpl.bootstrap()
        elements = bootstrap.get("elements", [])
        teams_raw = bootstrap.get("teams", [])

        teams_map = {team["id"]: team.get("name", "Unknown") for team in teams_raw if "id" in team}

        last_gw = self.fpl.last_completed_gw() or self.fpl.current_gw()
        if not last_gw:
            raise RuntimeError("Unable to determine last completed gameweek")

        start_gw = max(1, last_gw - self.window_gws + 1)
        window_gws = list(range(start_gw, last_gw + 1))
        window_set = set(window_gws)

        lineups = self._collect_lineups(elements, window_set)
        team_stats = self._calculate_team_rotation(lineups, teams_map, window_gws)

        prone_teams = [entry["team_name"] for entry in team_stats if entry.get("is_rotation_prone")]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "last_completed_gw": last_gw,
            "window_gameweeks": window_gws,
            "starter_minutes": self.starter_minutes,
            "prone_threshold": self.prone_threshold,
            "rotation_prone_teams": prone_teams,
            "teams": sorted(team_stats, key=lambda x: x.get("rotation_score", 0), reverse=True),
        }

    def persist(self, report: Dict[str, Any]) -> None:
        """Write report JSON to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info(
            "💾 Rotation monitor cached %d teams → %s",
            len(report.get("teams", [])),
            self.cache_path,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _collect_lineups(
        self,
        players: Iterable[Dict[str, Any]],
        window_gws: Set[int],
    ) -> Dict[int, Dict[int, Set[int]]]:
        """
        Build a mapping of team_id -> {gw -> starter_ids}.
        Starter defined as minutes >= starter_minutes for that GW.
        """
        team_lineups: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

        for player in players:
            player_id = player.get("id")
            team_id = player.get("team")
            total_minutes = int(player.get("minutes", 0) or 0)

            if not player_id or not team_id or total_minutes == 0:
                continue

            history = self._load_player_history(player_id)
            if not history:
                continue

            for entry in history:
                gw = entry.get("round")
                if gw is None:
                    continue

                try:
                    gw_int = int(gw)
                except (TypeError, ValueError):
                    continue

                if gw_int not in window_gws:
                    continue

                minutes = int(entry.get("minutes", 0) or 0)
                if minutes < self.starter_minutes:
                    continue

                team_lineups[team_id][gw_int].add(player_id)

        return team_lineups

    def _calculate_team_rotation(
        self,
        lineups: Dict[int, Dict[int, Set[int]]],
        teams_map: Dict[int, str],
        window_gws: List[int],
    ) -> List[Dict[str, Any]]:
        """Calculate volatility metrics for each team."""
        team_stats: List[Dict[str, Any]] = []

        for team_id, team_name in teams_map.items():
            gw_lineups = lineups.get(team_id, {})
            ordered_gws = [gw for gw in window_gws if gw in gw_lineups]

            if len(ordered_gws) < 2:
                continue

            overlaps: List[float] = []
            change_counts: List[float] = []
            unique_starters: Set[int] = set()
            prev_lineup: Optional[Set[int]] = None

            for gw in ordered_gws:
                lineup = gw_lineups.get(gw, set())
                unique_starters.update(lineup)

                if prev_lineup is not None and lineup:
                    overlap = len(prev_lineup & lineup)
                    denom = max(min(len(prev_lineup), len(lineup), 11), 1)
                    overlap_ratio = overlap / denom
                    overlaps.append(overlap_ratio)

                    baseline = max(len(prev_lineup), len(lineup), 11)
                    change_counts.append(max(0, baseline - overlap))

                prev_lineup = lineup

            if not overlaps:
                continue

            avg_overlap = sum(overlaps) / len(overlaps)
            rotation_score = max(0.0, min(1.0, 1.0 - avg_overlap))

            avg_changes = sum(change_counts) / len(change_counts) if change_counts else 0.0

            team_stats.append(
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "manager": self.manager_directory.get(team_name),
                    "window_gws": ordered_gws,
                    "avg_overlap_ratio": round(avg_overlap, 3),
                    "avg_changes": round(avg_changes, 2),
                    "unique_starters": len(unique_starters),
                    "rotation_score": round(rotation_score, 3),
                    "is_rotation_prone": rotation_score >= self.prone_threshold,
                }
            )

        return team_stats

    def _load_player_history(self, player_id: int) -> List[Dict[str, Any]]:
        """Fetch player element-summary with lightweight caching."""
        cache_file = self.summary_cache_dir / f"{player_id}.json"

        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                fetched_at = payload.get("fetched_at")
                if fetched_at:
                    fetched_dt = datetime.fromisoformat(fetched_at)
                    if datetime.now(timezone.utc) - fetched_dt < self.summary_cache_ttl:
                        return payload.get("history", [])
            except Exception as exc:  # pragma: no cover - cache fallback
                logger.debug("Rotation cache load failed for %s: %s", player_id, exc)

        summary = self.fpl.element_summary(player_id) or {}
        history = summary.get("history", [])

        try:
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "player_id": player_id,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "history": history,
                    },
                    fh,
                )
        except Exception as exc:  # pragma: no cover - cache fallback
            logger.debug("Rotation cache write failed for %s: %s", player_id, exc)

        return history

    @staticmethod
    def _log_report_summary(report: Dict[str, Any]) -> None:
        prone = report.get("rotation_prone_teams", [])
        logger.info(
            "📊 Rotation monitor: %d prone teams flagged across GW%s",
            len(prone),
            "-".join(map(str, report.get("window_gameweeks", []))) if report.get("window_gameweeks") else "?",
        )
        if prone:
            logger.info("   → %s", ", ".join(prone))

