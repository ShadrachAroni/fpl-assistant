import pandas as pd
import numpy as np
import logging
from app.api_client.fpl_client import FPLClient

logger = logging.getLogger("global_trends")
logger.setLevel(logging.INFO)


class GlobalFPLAnalytics:
    """
    Fetches and processes global FPL data for market sentiment analysis.
    Provides 'trend_score' and 'selected_by_percent' for each player,
    fully compatible with DataPipeline.players_df().
    """

    def __init__(self):
        self.fpl = FPLClient()

    def get_market_trends(self) -> pd.DataFrame:
        """
        Retrieves FPL bootstrap data, normalizes values, and computes a
        global trend score reflecting player popularity and transfer activity.

        Returns:
            DataFrame with columns ['id', 'trend_score', 'selected_by_percent'].
        """
        try:
            # ---------------- Fetch and Build Base Data ----------------
            bootstrap = self.fpl.bootstrap()
            elements = bootstrap.get("elements", [])
            if not elements:
                logger.warning("⚠️ No elements found in FPL bootstrap.")
                return pd.DataFrame(columns=["id", "trend_score", "selected_by_percent"])

            df = pd.DataFrame(elements)

            # ✅ Ensure essential columns exist
            for col in ["id", "selected_by_percent", "transfers_in_event", "transfers_out_event", "form", "now_cost"]:
                if col not in df.columns:
                    df[col] = 0

            # ---------------- Normalize and Clean Data ----------------
            df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)

            df["selected_by_percent"] = (
                df["selected_by_percent"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("%", "", regex=False)
                .replace("", "0")
            )
            df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)

            for col in ["transfers_in_event", "transfers_out_event", "form", "now_cost"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # ---------------- Compute Trend Score ----------------
            df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]
            df["trend_score"] = (
                0.6 * np.log1p(df["net_transfers"].clip(lower=0)) +
                0.4 * df["selected_by_percent"]
            ).fillna(0)

            # ---------------- Clean and Return ----------------
            result = df[["id", "trend_score", "selected_by_percent"]].copy()

            # Force correct dtypes to match pipeline before merge
            result["id"] = result["id"].astype(int)
            result["trend_score"] = result["trend_score"].astype(float)
            result["selected_by_percent"] = result["selected_by_percent"].astype(float)

            logger.info("✅ Trend metrics computed successfully (%d players).", len(result))
            #logger.info("Sample trend data:\n%s", result.head())

            return result

        except Exception as e:
            logger.warning("⚠️ Error computing market trends: %s", e)
            # Return DataFrame with expected columns to avoid merge issues
            return pd.DataFrame({
                "id": [],
                "trend_score": [],
                "selected_by_percent": []
            })
