import numpy as np
import pandas as pd

from app.api_client.fpl_client import FPLClient

from .rotation_monitor import RotationMonitor

__all__ = ["GlobalFPLAnalytics", "RotationMonitor"]


class GlobalFPLAnalytics:
    """
    Fetches and processes global FPL data for market sentiment analysis.
    Integrates crowd wisdom via ownership, transfer trends, and popularity metrics.
    """

    def __init__(self):
        self.fpl = FPLClient()

    def get_market_trends(self) -> pd.DataFrame:
        """
        Fetch global FPL bootstrap data and compute trend metrics.
        Returns a DataFrame with ownership and trend-based scores.
        """
        try:
            elements = self.fpl.bootstrap()["elements"]
            df = pd.DataFrame(elements)[[
                "id",
                "web_name",
                "selected_by_percent",
                "transfers_in_event",
                "transfers_out_event",
                "form",
                "now_cost"
            ]]

            df["selected_by_percent"] = df["selected_by_percent"].astype(float)
            df["transfers_in_event"] = df["transfers_in_event"].astype(float)
            df["transfers_out_event"] = df["transfers_out_event"].astype(float)

            df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]

            # Trend score — balances ownership & short-term momentum
            df["trend_score"] = (
                0.6 * np.log1p(df["net_transfers"].clip(lower=0)) +
                0.4 * df["selected_by_percent"]
            )

            df["trend_score"] = df["trend_score"].fillna(0)
            return df[["id", "trend_score", "selected_by_percent"]]

        except Exception as e:
            print(f"⚠️ Error computing market trends: {e}")
            return pd.DataFrame(columns=["id", "trend_score", "selected_by_percent"])
