"""
FPL API Client - FREE TRANSFERS FIX + ELEMENT SUMMARY FIX

The issue: The API's current_gw() returns the NEXT gameweek if the current one is live.
This causes transfers to be counted against the wrong gameweek.

Solution: Use the gameweek that the deadline hasn't passed for yet.
"""

import os
from typing import Dict, Any, Optional, List
import httpx
import logging
import pandas as pd
from datetime import datetime, timezone

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("fpl_client")
logger.setLevel(logging.INFO)

FPL_BASE = os.getenv("FPL_BASE_URL", "https://fantasy.premierleague.com/api")
FPL_USERNAME = os.getenv("FPL_USERNAME", "")
FPL_PASSWORD = os.getenv("FPL_PASSWORD", "")
FPL_SESSION_COOKIE = os.getenv("FPL_SESSION_COOKIE", "")


class FPLClient:
    """FPL Client - Complete + LIVE Bypass + FREE TRANSFERS FIX"""

    def __init__(
        self,
        base_url: str = FPL_BASE,
        username: str = FPL_USERNAME,
        password: str = FPL_PASSWORD,
        session_cookie: Optional[str] = FPL_SESSION_COOKIE,
    ):
        self.base = base_url.rstrip("/")
        self.session = httpx.Client(timeout=30.0)
        self.username = username
        self.password = password
        self.session_cookie = session_cookie
        self._bootstrap_cache: Optional[Dict[str, Any]] = None

        if not self.session_cookie and self.username and self.password:
            logger.info("🔐 Attempting auto-login...")
            self._authenticate()
        elif self.session_cookie:
            logger.debug("✅ Using session cookie")
        else:
            logger.warning("⚠️ No credentials")

    def _authenticate(self) -> bool:
        """Authenticate with FPL."""
        try:
            logger.info(f"🔒 Logging in...")
            login_url = f"{self.base}/entry/login/"
            payload = {
                "login": self.username,
                "password": self.password,
                "redirect_uri": "https://fantasy.premierleague.com/",
                "app": "plfpl-web",
            }
            headers = {"User-Agent": "fpl-assistant/3.0"}
            response = self.session.post(login_url, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if "entry" in data:
                    logger.info(f"✅ Login successful!")
                    return True
        except Exception as e:
            logger.error(f"❌ Auth error: {e}")
        return False

    def _get(self, path: str, params: Dict[str, Any] = None, no_cache: bool = True) -> Dict[str, Any]:
        """Make GET request with cache busting."""
        url = f"{self.base}/{path.lstrip('/')}"
        headers = {"User-Agent": "fpl-assistant/3.0"}
        
        if no_cache:
            headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            headers["Pragma"] = "no-cache"
            headers["Expires"] = "0"
        
        cookies = {}
        if self.session_cookie:
            cookies["pl_profile"] = self.session_cookie

        try:
            resp = self.session.get(url, params=params, headers=headers, cookies=cookies)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.debug(f"HTTP {e.response.status_code}: {url}")
            return {}
        except Exception as e:
            logger.error(f"Failed: {url} - {e}")
            return {}

    # ================== BOOTSTRAP & GAMEWEEK ==================

    def bootstrap(self) -> Dict[str, Any]:
        """Fetch bootstrap."""
        if self._bootstrap_cache is None:
            logger.debug("Fetching bootstrap...")
            self._bootstrap_cache = self._get("bootstrap-static/") or {}
        return self._bootstrap_cache

    def get_gameweeks(self) -> List[Dict[str, Any]]:
        """Get gameweeks."""
        return self.bootstrap().get("events", [])

    def current_gw(self) -> Optional[int]:
        """
        Get current GW - the one that is actually live/active now.
        
        NOTE: This returns the GW that is currently being played,
        NOT the next GW that you're planning for.
        """
        events = self.get_gameweeks()
        for e in events:
            if e.get("is_current"):
                return e["id"]
        return None

    def next_gw(self) -> Optional[int]:
        """Get next GW - the one you can make transfers for."""
        events = self.get_gameweeks()
        for e in events:
            if e.get("is_next"):
                return e["id"]
        return None
    
    def planning_gw(self) -> Optional[int]:
        """
        Get the GW you're currently planning for.
        
        This is the GW where:
        - You can still make transfers
        - The deadline hasn't passed yet
        - Your transfers will be applied
        
        This is usually is_next=True, but if current GW hasn't started,
        it's the current GW.
        """
        events = self.get_gameweeks()
        
        # First check if there's a "next" gameweek
        for e in events:
            if e.get("is_next"):
                return e["id"]
        
        # If no "next", then current is still open for planning
        for e in events:
            if e.get("is_current"):
                return e["id"]
        
        return None

    def last_completed_gw(self) -> Optional[int]:
        """Get last completed GW."""
        events = self.get_gameweeks()
        completed = [e for e in events if e.get("finished")]
        if completed:
            return max(e["id"] for e in completed if "id" in e)
        return None

    # ================== MANAGER DATA ==================

    def manager(self, manager_id: int) -> Dict[str, Any]:
        """Get manager entry data."""
        return self._get(f"entry/{manager_id}/", no_cache=True)

    def manager_history(self, manager_id: int) -> Dict[str, Any]:
        """Get manager history."""
        return self._get(f"entry/{manager_id}/history/", no_cache=True) or {}

    def manager_transfers(self, manager_id: int) -> List[Dict[str, Any]]:
        """Get manager transfers."""
        transfers = self._get(f"entry/{manager_id}/transfers/", no_cache=True)
        return transfers if isinstance(transfers, list) else []

    def manager_picks(self, manager_id: int, gw: Optional[int] = None) -> Dict[str, Any]:
        """Get manager picks for GW."""
        if gw is None:
            gw = self.current_gw()  # Use current GW for picks

        if gw is None:
            logger.error("❌ No current GW")
            return {}

        logger.debug(f"Fetching picks for GW{gw}")
        picks = self._get(f"entry/{manager_id}/event/{gw}/picks/", no_cache=True)
        
        if not picks:
            logger.warn(f"⚠️ No picks for GW{gw}")
        
        return picks
    
    def get_latest_squad_picks(self, manager_id: int) -> Dict[str, Any]:
        """
        Get the most recent squad picks that exist.
        
        This is needed because picks for future GWs don't exist yet.
        We need the last confirmed squad as the base.
        """
        # Try current GW first
        current_gw = self.current_gw()
        if current_gw:
            picks = self.manager_picks(manager_id, gw=current_gw)
            if picks and picks.get("picks"):
                logger.debug(f"Using GW{current_gw} picks as base")
                return picks
        
        # If current doesn't work, try last completed
        last_gw = self.last_completed_gw()
        if last_gw:
            picks = self.manager_picks(manager_id, gw=last_gw)
            if picks and picks.get("picks"):
                logger.debug(f"Using GW{last_gw} picks as base")
                return picks
        
        logger.error("❌ No picks available")
        return {}

    def element_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get player summary including history.
        
        This endpoint provides per-gameweek history for a player.
        """
        return self._get(f"element-summary/{player_id}/", no_cache=False)

    # ================== REQUIRED BY MAIN.PY ==================

    def get_current_squad_player_ids(self, manager_id: int) -> List[int]:
        """Get current squad player IDs - uses LIVE reconstruction."""
        return self.get_live_squad_with_transfers(manager_id)

    def get_live_squad_with_transfers(self, manager_id: int) -> List[int]:
        """
        Get LIVE squad with transfer reconstruction.
        
        Key insight: For future GWs, picks don't exist yet. We need to:
        1. Get the latest confirmed squad (current or last completed GW)
        2. Apply any pending transfers for the planning GW
        """
        planning_gw = self.planning_gw()

        if planning_gw is None:
            logger.error("❌ No planning GW")
            return []

        logger.info(f"🔍 Getting LIVE squad for GW{planning_gw}...")

        # Get the latest available squad picks
        picks = self.get_latest_squad_picks(manager_id)
        team = picks.get("picks", [])

        if not team:
            logger.error(f"❌ No picks available")
            return []

        # Start with base squad
        squad_ids = [p["element"] for p in team if "element" in p]
        logger.debug(f"📋 Base squad: {len(squad_ids)} players")

        # Get all transfers for planning GW
        all_transfers = self.manager_transfers(manager_id)
        planning_gw_transfers = [t for t in all_transfers if t.get("event") == planning_gw]

        if planning_gw_transfers:
            logger.info(f"🔄 Applying {len(planning_gw_transfers)} transfer(s) from GW{planning_gw}")
            
            for transfer in planning_gw_transfers:
                player_out = transfer.get("element_out")
                player_in = transfer.get("element_in")
                
                if player_out in squad_ids:
                    squad_ids.remove(player_out)
                    logger.debug(f"   OUT: Player {player_out}")
                
                if player_in not in squad_ids:
                    squad_ids.append(player_in)
                    logger.debug(f"   IN:  Player {player_in}")
        else:
            logger.info(f"✅ No transfers made yet for GW{planning_gw}")

        logger.info(f"✅ LIVE squad: {len(squad_ids)} players (with transfers applied)")
        return squad_ids

    def get_manager_chips_used(self, manager_id: int) -> Dict[str, Optional[int]]:
        """
        Get chips used - FIXED to read active_chip field.
        """
        chips_used = {
            'triple_captain': None,
            'bench_boost': None,
            'free_hit': None,
            'wildcard': None
        }

        history_data = self.manager_history(manager_id)

        if not isinstance(history_data, dict):
            logger.warning("⚠️ No history data")
            return chips_used

        current_season = history_data.get("current", [])
        chips_data = history_data.get("chips", [])
        
        # METHOD 1: Check chips array (most reliable)
        if chips_data:
            logger.info(f"📊 Found {len(chips_data)} chip entries in history")
            for chip_entry in chips_data:
                chip_name = chip_entry.get("name", "").lower()
                event = chip_entry.get("event")
                
                logger.debug(f"  Chip '{chip_name}' used in GW{event}")
                
                if "wildcard" in chip_name:
                    chips_used['wildcard'] = event
                elif "bench" in chip_name or "bboost" in chip_name:
                    chips_used['bench_boost'] = event
                elif "free" in chip_name or "freehit" in chip_name:
                    chips_used['free_hit'] = event
                elif "triple" in chip_name or "3xc" in chip_name:
                    chips_used['triple_captain'] = event
        
        # METHOD 2: Fallback to current season data
        if not any(chips_used.values()):
            if not isinstance(current_season, list) or not current_season:
                logger.debug("No current season data for fallback")
            else:
                logger.info(f"📊 Checking {len(current_season)} GW entries as fallback...")

                for entry in current_season:
                    event = entry.get("event")
                    chip = entry.get("active_chip") or entry.get("chip")

                    logger.debug(f"  GW{event}: active_chip={repr(entry.get('active_chip'))}, chip={repr(entry.get('chip'))}")

                    if chip is not None and str(chip).strip():
                        chip_str = str(chip).lower().strip()
                        
                        logger.info(f"✅ Found chip '{chip}' used in GW{event}")
                        
                        if chip_str == "wildcard":
                            chips_used['wildcard'] = event
                        elif chip_str == "bboost":
                            chips_used['bench_boost'] = event
                        elif chip_str == "freehit":
                            chips_used['free_hit'] = event
                        elif chip_str == "3xc":
                            chips_used['triple_captain'] = event

        logger.info(f"💎 Final chips: {chips_used}")
        return chips_used

    def manager_bank_and_free_transfers(self, manager_id: int) -> Dict[str, Any]:
        """
        Get bank and free transfers - FIXED VERSION.
        
        Key fixes:
        1. Uses planning_gw() to target the right gameweek
        2. Gets FT from last COMPLETED GW, not current
        3. Handles case where planning GW picks don't exist yet
        """
        planning_gw = self.planning_gw()
        current_gw = self.current_gw()

        if planning_gw is None:
            return {
                "bank": 0.0,
                "free_transfers": 1,
                "transfers_made_this_gw": 0,
                "starting_free_transfers": 1,
            }

        logger.info(f"💰 Getting bank/FT for planning GW{planning_gw} (current GW{current_gw})")

        # Get LIVE entry data
        entry = self.manager(manager_id)
        
        if not entry:
            logger.error("❌ No entry data")
            return {
                "bank": 0.0,
                "free_transfers": 1,
                "transfers_made_this_gw": 0,
                "starting_free_transfers": 1,
            }

        # Get LIVE bank - use last_deadline_bank as it's more reliable
        bank_raw = entry.get("last_deadline_bank", entry.get("bank", 0))
        if bank_raw is None:
            bank_raw = 0
        live_bank = float(bank_raw) / 10.0

        logger.info(f"💰 LIVE Bank: £{live_bank:.1f}m")

        # Count transfers made for PLANNING GW
        all_transfers = self.manager_transfers(manager_id)
        planning_gw_transfers = [t for t in all_transfers if t.get("event") == planning_gw]
        event_transfers = len(planning_gw_transfers)
        
        # Calculate transfer cost (4 points per hit)
        # If we have picks data for planning GW, use it; otherwise calculate from transfers
        picks = self.manager_picks(manager_id, gw=planning_gw)
        if picks and picks.get("entry_history"):
            entry_hist = picks.get("entry_history", {})
            event_transfers = int(entry_hist.get("event_transfers", event_transfers))
            event_transfers_cost = int(entry_hist.get("event_transfers_cost", 0))
        else:
            # Planning GW picks don't exist yet - count from transfers
            event_transfers_cost = 0
            logger.debug(f"No picks for GW{planning_gw} yet, using transfer count: {event_transfers}")

        logger.info(f"💰 GW{planning_gw}: Transfers={event_transfers}, Cost={event_transfers_cost}pts")

        # Calculate starting FT from LAST COMPLETED GW
        last_completed = self.last_completed_gw()
        starting_ft = 1

        if last_completed and last_completed > 0:
            try:
                picks_prev = self.manager_picks(manager_id, gw=last_completed)
                entry_hist_prev = picks_prev.get("entry_history", {}) or {}
                prev_transfers = int(entry_hist_prev.get("event_transfers", 0))
                
                # You get 2 FT if you made 0 transfers last completed GW, else 1 FT
                starting_ft = 2 if prev_transfers == 0 else 1
                logger.info(f"📊 GW{last_completed} (completed): {prev_transfers} transfers → Start GW{planning_gw} with {starting_ft} FT")
            except Exception as e:
                logger.debug(f"Could not fetch GW{last_completed}: {e}")
                starting_ft = 1
        
        # Calculate remaining FT
        if event_transfers == 0:
            remaining_ft = starting_ft
            logger.info(f"✅ No transfers made yet for GW{planning_gw} → {remaining_ft} FT available")
        elif event_transfers_cost == 0:
            # Used free transfers only
            remaining_ft = max(0, starting_ft - event_transfers)
            logger.info(f"📊 Used {event_transfers} FT → {remaining_ft} remaining")
        else:
            # Took hits
            hits = event_transfers_cost // 4
            ft_used = event_transfers - hits
            remaining_ft = max(0, starting_ft - ft_used)
            logger.info(f"📊 {event_transfers} transfers ({ft_used} FT + {hits} hits) → {remaining_ft} FT remaining")

        logger.info(f"📊 FT Summary: Start={starting_ft} | Made={event_transfers} | Remaining={remaining_ft}")

        return {
            "bank": round(live_bank, 1),
            "free_transfers": remaining_ft,
            "transfers_made_this_gw": event_transfers,
            "starting_free_transfers": starting_ft,
        }

    def get_squad_with_selling_prices(self, manager_id: int) -> Dict[int, float]:
        """Get selling prices - uses latest available picks."""
        picks = self.get_latest_squad_picks(manager_id)
        picks_list = picks.get("picks", [])

        selling_prices = {}

        for pick in picks_list:
            player_id = pick.get("element")
            selling_price = pick.get("selling_price")

            if selling_price is not None and player_id:
                try:
                    sp = float(selling_price) / 10.0
                    selling_prices[player_id] = sp
                except (ValueError, TypeError):
                    pass

        return selling_prices

    def get_transfers_this_gw(self, manager_id: int) -> List[Dict[str, Any]]:
        """Get transfers this GW (planning GW)."""
        planning_gw = self.planning_gw()

        if planning_gw is None:
            return []

        transfers = self.manager_transfers(manager_id)
        gw_transfers = [t for t in transfers if t.get("event") == planning_gw]

        logger.info(f"📦 GW{planning_gw}: {len(gw_transfers)} transfer(s)")
        return gw_transfers

    def get_manager_recent_performance(self, manager_id: int, weeks: int = 5) -> Dict[str, Any]:
        """Get recent performance."""
        history_data = self.manager_history(manager_id)
        current_season = history_data.get("current", [])

        if not current_season:
            return {"weeks_analyzed": 0, "average_points": 0}

        recent = sorted(current_season, key=lambda x: x.get("event", 0), reverse=True)[:weeks]
        total_points = sum(entry.get("points", 0) for entry in recent)
        avg_points = total_points / len(recent) if recent else 0

        return {
            "weeks_analyzed": len(recent),
            "average_points_recent": round(avg_points, 2),
            "total_points_recent": total_points,
        }

    def fixtures(self) -> List[Dict[str, Any]]:
        """Get fixtures."""
        fixtures = self._get("fixtures/")
        return fixtures if isinstance(fixtures, list) else []

    def get_blank_and_double_gameweeks(self) -> Dict[int, Dict[str, Any]]:
        """Get blank/double GWs."""
        fixtures = self.fixtures()

        if not fixtures:
            return {}

        fixtures_df = pd.DataFrame(fixtures)

        if fixtures_df.empty or "event" not in fixtures_df.columns:
            return {}

        blank_and_double = {}

        for event in fixtures_df["event"].unique():
            if pd.isna(event):
                continue

            event = int(event)
            event_fixtures = fixtures_df[fixtures_df["event"] == event]

            if event_fixtures.empty:
                blank_and_double[event] = {"type": "blank", "teams_playing": 0}
            else:
                teams_playing = len(
                    set(
                        list(event_fixtures["team_h"].dropna().unique())
                        + list(event_fixtures["team_a"].dropna().unique())
                    )
                )

                if teams_playing < 20:
                    blank_and_double[event] = {"type": "blank", "teams_playing": teams_playing}

        return blank_and_double

    # ================== LIVE BYPASS METHODS ==================

    def get_live_manager_data(self, manager_id: int) -> Dict[str, Any]:
        """Get LIVE manager data."""
        logger.info(f"💾 Getting LIVE manager data for {manager_id}")
        entry = self.manager(manager_id)
        logger.info(f"✅ Entry data retrieved")
        return entry

    def get_live_bank(self, manager_id: int) -> float:
        """Get LIVE bank balance."""
        entry = self.get_live_manager_data(manager_id)
        bank_raw = entry.get("last_deadline_bank", entry.get("bank", 0))
        if bank_raw is None:
            bank_raw = 0
        live_bank = float(bank_raw) / 10.0
        logger.info(f"💰 LIVE Bank: £{live_bank:.1f}m")
        return live_bank

    def get_all_transfers(self, manager_id: int) -> List[Dict[str, Any]]:
        """Get ALL transfers."""
        logger.info(f"🔍 Getting LIVE transfers...")
        transfers = self.manager_transfers(manager_id)
        logger.info(f"✅ Retrieved {len(transfers)} transfers")
        return transfers

    def get_live_chip_history(self, manager_id: int) -> Dict[str, Optional[int]]:
        """Get LIVE chip history."""
        logger.info(f"💎 Getting LIVE chip history...")
        return self.get_manager_chips_used(manager_id)

    def get_live_squad_current_gw(self, manager_id: int) -> List[int]:
        """Get LIVE squad with transfers applied."""
        return self.get_live_squad_with_transfers(manager_id)