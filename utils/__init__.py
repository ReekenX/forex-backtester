"""
Utils package for Forex Trading Strategy Analysis

This package contains modules for backtesting and analyzing forex trading strategies.
"""

import pandas as pd
from typing import List, Tuple

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# Risk-Reward Ratio configurations with breakeven win rates
RRR_CONFIGS: List[Tuple[int, float]] = [
    (1, 50.0),  # 1:1 RRR - need 50% to break even
    (2, 33.3),  # 1:2 RRR - need 33.3% to break even
    (3, 25.0),  # 1:3 RRR - need 25% to break even
]

# ============================================================================
# WIN CONDITION FILTERS
# ============================================================================


def _win_condition_normal(df: pd.DataFrame, ratio: int, sl_column: str) -> pd.DataFrame:
    """Standard win condition: TP > ratio * SL."""
    return df[(df["TP"] > ratio * df[sl_column]) & (df["SL"] != df["Pullback"])]


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================


def load_and_clean_data(filepath: str = "../data/eurusd.csv") -> pd.DataFrame:
    """
    Load EUR/USD data from CSV and clean NaN values.

    Args:
        filepath: Path to the CSV file containing trading data

    Returns:
        Cleaned dataframe with trading data
    """
    df = pd.read_csv(filepath)

    # Define columns that should have NaN replaced with 0
    columns_to_fillna = [
        "SL",
        "TP",
        "Hour",
        "Hours Until News",
        "Extra",
    ]

    # Fill NaN values to prevent calculation errors
    for col in columns_to_fillna:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# This file makes utils a proper Python package
