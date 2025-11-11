"""
Utility functions for model handling, data processing, and feature engineering.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Any


def ensure_dir(path: str) -> None:
    """
    Ensure that the directory for a given file path exists.
    Creates it recursively if it doesn't exist.

    Args:
        path (str): Full file path or directory path.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def rolling_features(player_history: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Compute rolling averages for all numeric columns in a player's
    gameweek history DataFrame (as returned by the FPL API).

    Args:
        player_history (pd.DataFrame): Player's per-gameweek historical data.
        window (int): Rolling window size in gameweeks. Default is 4.

    Returns:
        pd.DataFrame: DataFrame with original columns + rolling average columns.
    """
    # FPL API uses 'round' for gameweek — alias it to 'event' for consistency
    if "event" not in player_history.columns and "round" in player_history.columns:
        player_history = player_history.rename(columns={"round": "event"})

    if "event" not in player_history.columns:
        raise KeyError("Expected column 'event' or 'round' not found in player history data.")

    # Sort and index by gameweek
    df = player_history.sort_values("event").set_index("event")

    # Compute rolling means for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    rolled = df[numeric_cols].rolling(window=window, min_periods=1).mean()
    rolled = rolled.add_suffix(f"_roll{window}")

    # Combine and restore 'event' as a column
    return pd.concat([df, rolled], axis=1).reset_index()


def save_model(obj: Any, path: str) -> None:
    """
    Save a model or Python object to disk using joblib.

    Args:
        obj (Any): Object to save.
        path (str): Path (including filename) to save the object to.
    """
    ensure_dir(path)
    joblib.dump(obj, path)


def load_model(path: str) -> Any:
    """
    Load a previously saved model or Python object from disk.

    Args:
        path (str): Path to the saved object.

    Returns:
        Any: Loaded Python object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)
