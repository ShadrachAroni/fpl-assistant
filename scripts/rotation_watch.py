"""
CLI helper to keep the rotation monitor fresh.

Example usage:
    python -m scripts.rotation_watch --interval-minutes 180
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from app.analytics import RotationMonitor
from app.api_client.fpl_client import FPLClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rotation_watch")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor rotation-prone managers")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--window", type=int, help="Override window length (gameweeks)")
    parser.add_argument("--starter-minutes", type=int, help="Minutes threshold for starters")
    parser.add_argument("--threshold", type=float, help="Rotation score threshold to flag teams")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=0,
        help="Run continuously every N minutes (0 = single run)",
    )
    return parser.parse_args()


def run_once(config_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(config_path)
    fpl_client = FPLClient()
    monitor = RotationMonitor(
        fpl_client=fpl_client,
        config=config,
        window_gws=args.window,
        starter_minutes=args.starter_minutes,
        prone_threshold=args.threshold,
    )
    return monitor.run()


def main() -> int:
    args = parse_args()
    report = run_once(args.config, args)
    logger.info(
        "🚨 Rotation summary: %d teams flagged → %s",
        len(report.get("rotation_prone_teams", [])),
        report.get("rotation_prone_teams", []),
    )

    interval = max(0, int(args.interval_minutes or 0))
    if interval == 0:
        return 0

    logger.info("⏱️ Continuing to refresh every %d minutes...", interval)
    try:
        while True:
            time.sleep(interval * 60)
            report = run_once(args.config, args)
            logger.info(
                "✅ Rotation refresh @ %s → %d teams",
                report.get("generated_at"),
                len(report.get("rotation_prone_teams", [])),
            )
    except KeyboardInterrupt:
        logger.info("🛑 Rotation monitor stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())

