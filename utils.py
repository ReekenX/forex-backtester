"""
Forex Trading Strategy Analysis Utilities

This module provides comprehensive tools for backtesting and analyzing forex trading strategies.
It includes functions for data loading, strategy evaluation, and performance visualization.

Main components:
- Data loading and cleaning
- Strategy definition and evaluation
- Risk-Reward Ratio (RRR) analysis
- Entry timing analysis
- Performance visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# Risk-Reward Ratio configurations with breakeven win rates
RRR_CONFIGS: List[Tuple[int, float]] = [
    (1, 50.0),  # 1:1 RRR - need 50% to break even
    (2, 33.3),  # 1:2 RRR - need 33.3% to break even
    (3, 25.0),  # 1:3 RRR - need 25% to break even
]

# Entry method mapping for display
ENTRY_TYPE_NAMES: Dict[str, str] = {
    "SL": "1M CC",
}

# Table styling defaults
DEFAULT_COLUMN_WIDTH = "250px"
DEFAULT_HIGHLIGHT_COLOR = "green"

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================


def load_and_clean_data(filepath: str = "./eurusd.csv") -> pd.DataFrame:
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


# ============================================================================
# STRATEGY DEFINITION AND MANAGEMENT
# ============================================================================


class Strategy:
    """
    Encapsulates a trading strategy with its filter logic and metadata.

    Attributes:
        name: Strategy identifier
        filter_func: Function that filters trades based on strategy rules
        description: Human-readable description of the strategy
    """

    def __init__(self, name: str, filter_func: Callable, description: str = ""):
        """
        Initialize a trading strategy.

        Args:
            name: Strategy name
            filter_func: Lambda or function that takes df and returns filtered df
            description: Optional description of the strategy
        """
        self.name = name
        self.filter_func = filter_func
        self.description = description

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the strategy filter to a dataframe of trades."""
        return self.filter_func(df)


# ============================================================================
# HELPER FUNCTIONS FOR CALCULATIONS
# ============================================================================


def _calculate_wins_and_losses(
    df: pd.DataFrame, ratio: int, sl_column: str, win_condition_func: Callable
) -> Tuple[int, int]:
    """
    Calculate wins and losses for a given RRR ratio.

    Args:
        df: DataFrame with trade data
        ratio: Risk-reward ratio (1, 2, or 3)
        sl_column: Stop loss column to use
        win_condition_func: Function that determines winning trades

    Returns:
        Tuple of (wins, losses)
    """
    wins = len(win_condition_func(df, ratio, sl_column))
    losses = len(df) - wins
    return wins, losses


def _format_rrr_metrics(
    total_trades: int,
    wins: int,
    losses: int,
    ratio: int,
    breakeven_rate: float,
    entry_type: str,
) -> List:
    """
    Format RRR metrics for display.

    Returns:
        List of formatted metrics [total, wins, losses, win_rate, edge, outcome, entry]
    """
    if total_trades == 0:
        return [0, 0, 0, "0.0%", "0.0%", "0R", entry_type]

    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven_rate
    outcome = (wins * ratio) - losses

    return [
        total_trades,
        wins,
        losses,
        f"{win_rate:.1f}%",
        f"{edge:.1f}%",
        f"{outcome}R",
        entry_type,
    ]


def _create_empty_rrr_summary(strategy_name: str, entry_type: str) -> pd.DataFrame:
    """Create an empty RRR summary DataFrame for strategies with no trades."""
    summary_data = {
        strategy_name: [
            "Total trades",
            "Wins",
            "Losses",
            "Win Rate",
            "Edge",
            "Outcome",
            "Entry",
        ]
    }
    for ratio, _ in RRR_CONFIGS:
        summary_data[f"1:{ratio} RRR"] = [0, 0, 0, "0.0%", "0.0%", "0R", entry_type]
    return pd.DataFrame(summary_data)


# ============================================================================
# WIN CONDITION FILTERS
# ============================================================================


def _win_condition_normal(df: pd.DataFrame, ratio: int, sl_column: str) -> pd.DataFrame:
    """Standard win condition: TP > ratio * SL."""
    return df[df["TP"] > ratio * df[sl_column]]


# ============================================================================
# RRR CALCULATION FUNCTIONS
# ============================================================================


def calculate_rrr_stats(
    data_df: pd.DataFrame, strategy_name: str, sl_column: str = "SL"
) -> pd.DataFrame:
    """
    Calculate comprehensive Risk-Reward Ratio statistics for a trading strategy.

    Args:
        data_df: Filtered DataFrame containing trades for this strategy
        strategy_name: Name of the strategy for labeling
        sl_column: Column to use for stop loss calculations

    Returns:
        Statistics table with metrics for each RRR level
    """
    total_trades = len(data_df)
    entry_str = ENTRY_TYPE_NAMES[sl_column]

    if total_trades == 0:
        return _create_empty_rrr_summary(strategy_name, entry_str)

    summary_data = {
        strategy_name: [
            "Total trades",
            "Wins",
            "Losses",
            "Win Rate",
            "Edge",
            "Outcome",
            "Entry",
        ]
    }

    for ratio, breakeven_rate in RRR_CONFIGS:
        wins, losses = _calculate_wins_and_losses(
            data_df, ratio, sl_column, _win_condition_normal
        )

        summary_data[f"1:{ratio} RRR"] = _format_rrr_metrics(
            total_trades, wins, losses, ratio, breakeven_rate, entry_str
        )

    return pd.DataFrame(summary_data)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_pullback_profitability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze how pullback size affects trade profitability.

    Creates a table showing profitability statistics for different pullback types and RRR ratios.

    Args:
        df: Trading data with Pullback, TP, and SL columns

    Returns:
        Dictionary containing the pullback analysis DataFrame
    """
    pullback_configs = [
        ("No Pullback", lambda d: d),
        ("Pullback >= 0.5 pips", lambda d: d[d["Pullback"] >= 0.5]),
        ("Pullback >= 1.0 pip", lambda d: d[d["Pullback"] >= 1.0]),
        ("Pullback 50%", lambda d: d[d['Pullback'] >= d['SL'] * 0.5]),
    ]

    pullback_rows = []

    for pullback_name, filter_func in pullback_configs:
        filtered_df = filter_func(df)

        for ratio, breakeven_rate in RRR_CONFIGS:
            total_trades = len(filtered_df)

            if total_trades > 0:
                # TODO: if pullback is used, this changes SL and TP values that are not calculated here
                profitable = filtered_df[
                    (filtered_df["SL"] != filtered_df["Pullback"])
                    & (filtered_df["Pullback"] < filtered_df["SL"])
                    & (filtered_df["TP"] >= (ratio * filtered_df["SL"]))
                ]
                wins = len(profitable)
                losses = total_trades - wins
                win_rate = wins / total_trades * 100
            else:
                wins = 0
                losses = 0
                win_rate = 0.0

            pullback_rows.append({
                'Type': pullback_name if ratio == 1 else '',
                'RRR': f'1:{ratio}',
                'Total Trades': total_trades,
                'Wins': wins,
                'Losses': losses,
                'Win %': f"{win_rate:.1f}%"
            })

    final_table = pd.DataFrame(pullback_rows)
    return {"Pullback Analysis": final_table}


def analyze_hour_profitability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze which hours produce the most profitable trades.

    Creates a table showing wins, losses, and win percentage for each trading hour.

    Args:
        df: Trading data with Hour and TP columns

    Returns:
        Dictionary containing the hour analysis DataFrame
    """
    # Extract hour if not already present
    hour_df = df.copy()
    if 'Hour' not in hour_df.columns:
        # If no Hour column, try to extract from Date or Time columns
        if 'Date' in hour_df.columns:
            hour_df['Hour'] = pd.to_datetime(hour_df['Date']).dt.hour
        else:
            # Hour column should already exist in the data
            raise ValueError("No Hour column found in data")

    # Calculate wins (TP > 0 means profitable trade)
    hour_df['Is_Win'] = hour_df['TP'] > 0

    # Remove rows with NaN or empty hours before analysis
    hour_df = hour_df.dropna(subset=['Hour'])
    hour_df = hour_df[hour_df['Hour'] != 0]

    # Check if we have any data left after filtering
    if len(hour_df) == 0:
        # Return empty table if no hour data available
        empty_df = pd.DataFrame(columns=['Hour', 'Total Trades', 'Wins', 'Losses', 'Win %'])
        return {"Hour Analysis": empty_df}

    # Group by hour and calculate statistics
    hour_stats = hour_df.groupby('Hour').agg(
        Total_Trades=('Is_Win', 'count'),
        Wins=('Is_Win', 'sum'),
        Losses=('Is_Win', lambda x: (~x).sum())
    ).reset_index()

    # Calculate win percentage
    hour_stats['Win_Percentage'] = (hour_stats['Wins'] / hour_stats['Total_Trades'] * 100).round(1)

    # Format hour column for display (convert to int first to handle float values)
    hour_stats['Hour_Display'] = hour_stats['Hour'].astype(int).apply(lambda x: f"{x:02d}h")

    # Format win percentage for display
    hour_stats['Win_Percentage_Display'] = hour_stats['Win_Percentage'].apply(lambda x: f"{x:.1f}%")

    # Prepare final table with proper column names
    final_table = hour_stats[['Hour_Display', 'Total_Trades', 'Wins', 'Losses', 'Win_Percentage_Display']].copy()
    final_table.columns = ['Hour', 'Total Trades', 'Wins', 'Losses', 'Win %']

    # Sort by hour
    final_table = final_table.sort_values(by='Hour').reset_index(drop=True)

    return {"Hour Analysis": final_table}


def analyze_weekday_profitability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze which weekdays produce the most profitable trades.

    Creates a table showing wins, losses, and win percentage for each weekday.

    Args:
        df: Trading data with Weekday and TP columns

    Returns:
        Dictionary containing the weekday analysis DataFrame
    """
    # Work with copy of data
    weekday_df = df.copy()

    # Check if Weekday column exists
    if 'Weekday' not in weekday_df.columns:
        raise ValueError("No Weekday column found in data")

    # Calculate wins (TP > 0 means profitable trade)
    weekday_df['Is_Win'] = weekday_df['TP'] > 0

    # Group by weekday and calculate statistics
    weekday_stats = weekday_df.groupby('Weekday').agg(
        Total_Trades=('Is_Win', 'count'),
        Wins=('Is_Win', 'sum'),
        Losses=('Is_Win', lambda x: (~x).sum())
    ).reset_index()

    # Calculate win percentage
    weekday_stats['Win_Percentage'] = (weekday_stats['Wins'] / weekday_stats['Total_Trades'] * 100).round(1)

    # Format win percentage for display
    weekday_stats['Win_Percentage_Display'] = weekday_stats['Win_Percentage'].apply(lambda x: f"{x:.1f}%")

    # Prepare final table with proper column names
    final_table = weekday_stats[['Weekday', 'Total_Trades', 'Wins', 'Losses', 'Win_Percentage_Display']].copy()
    final_table.columns = ['Weekday', 'Total Trades', 'Wins', 'Losses', 'Win %']

    # Define proper weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Sort by weekday order
    final_table['Weekday'] = pd.Categorical(final_table['Weekday'], categories=weekday_order, ordered=True)
    final_table = final_table.sort_values('Weekday').reset_index(drop=True)

    return {"Weekday Analysis": final_table}


def analyze_sl_distribution(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze trade distribution and profitability by stop loss size ranges.

    Creates a table showing wins, losses, and win percentage for different SL ranges.

    Args:
        df: Trading data with SL and TP columns

    Returns:
        Dictionary containing the SL distribution analysis DataFrame
    """
    # Define SL ranges for analysis
    sl_ranges = [
        ("SL < 1", lambda sl: sl < 1),
        ("1 ≤ SL < 2", lambda sl: (sl >= 1) & (sl < 2)),
        ("2 ≤ SL < 3", lambda sl: (sl >= 2) & (sl < 3)),
        ("3 ≤ SL < 4", lambda sl: (sl >= 3) & (sl < 4)),
        ("4 ≤ SL < 5", lambda sl: (sl >= 4) & (sl < 5)),
        ("5 ≤ SL < 10", lambda sl: (sl >= 5) & (sl < 10)),
        ("10 ≤ SL < 15", lambda sl: (sl >= 10) & (sl < 15)),
        ("SL ≥ 15", lambda sl: sl >= 15),
    ]

    sl_rows = []

    for range_name, range_filter in sl_ranges:
        # Filter trades in this SL range
        mask = range_filter(df["SL"])
        range_df = df[mask]

        total_trades = len(range_df)

        if total_trades > 0:
            # Calculate wins (TP > 0 means profitable trade)
            wins = (range_df['TP'] > 0).sum()
            losses = total_trades - wins
            win_percentage = (wins / total_trades * 100)

            # Format for display
            win_percentage_display = f"{win_percentage:.1f}%"
        else:
            wins = 0
            losses = 0
            win_percentage_display = "N/A"

        sl_rows.append({
            "SL Range": range_name,
            "Total Trades": total_trades,
            "Wins": wins,
            "Losses": losses,
            "Win %": win_percentage_display
        })

    # Create DataFrame from results
    result_df = pd.DataFrame(sl_rows)

    return {"Stop Loss Distribution": result_df}


def analyze_tp_distribution(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze trade distribution and profitability by take profit size ranges.

    Creates a table showing wins, losses, and win percentage for different TP ranges.

    Args:
        df: Trading data with SL and TP columns

    Returns:
        Dictionary containing the TP distribution analysis DataFrame
    """
    # Define TP ranges for analysis
    tp_ranges = [
        ("TP < 10", lambda tp: tp < 10),
        ("10 ≤ TP < 15", lambda tp: (tp >= 10) & (tp < 15)),
        ("15 ≤ TP < 20", lambda tp: (tp >= 15) & (tp < 20)),
        ("20 ≤ TP < 25", lambda tp: (tp >= 20) & (tp < 25)),
        ("25 ≤ TP < 30", lambda tp: (tp >= 25) & (tp < 30)),
        ("TP ≥ 30", lambda tp: tp >= 30),
    ]

    tp_rows = []

    for range_name, range_filter in tp_ranges:
        # Filter trades in this TP range
        mask = range_filter(df["TP"])
        range_df = df[mask]

        total_trades = len(range_df)

        if total_trades > 0:
            # Calculate wins: SL != Pullback AND TP > SL
            wins = ((range_df['SL'] != range_df['Pullback']) & (range_df['TP'] > range_df['SL'])).sum()
            losses = total_trades - wins
            win_percentage = (wins / total_trades * 100)

            # Format for display
            win_percentage_display = f"{win_percentage:.1f}%"
        else:
            wins = 0
            losses = 0
            win_percentage_display = "N/A"

        tp_rows.append({
            "TP Range": range_name,
            "Total Trades": total_trades,
            "Wins": wins,
            "Losses": losses,
            "Win %": win_percentage_display
        })

    # Create DataFrame from results
    result_df = pd.DataFrame(tp_rows)

    return {"Take Profit Distribution": result_df}


def analyze_sl_reduction_profitability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze how reducing stop loss size affects trade profitability.

    Creates a table showing profitability statistics for different SL reduction types and RRR ratios.

    Args:
        df: Trading data with SL, TP, and Pullback columns

    Returns:
        Dictionary containing the SL reduction analysis DataFrame
    """
    sl_reduction_configs = [
        ("No adjustment", lambda sl: sl),
        ("1 pip reduction", lambda sl: sl - 1),
        ("2 pips reduction", lambda sl: sl - 2),
        ("1 pip reduction or max 4 pip SL", lambda sl: np.where(sl > 4, 4, sl - 1)),
        ("1 pip reduction or max 5 pip SL", lambda sl: np.where(sl > 5, 5, sl - 1)),
        ("1 pip reduction or max 6 pip SL", lambda sl: np.where(sl > 6, 6, sl - 1)),
        ("1 pip reduction or max 7 pip SL", lambda sl: np.where(sl > 7, 7, sl - 1)),
        ("1 pip reduction or max 10 pip SL", lambda sl: np.where(sl > 10, 10, sl - 1)),
        ("1 pip reduction or max 15 pip SL", lambda sl: np.where(sl > 15, 15, sl - 1)),
    ]

    sl_reduction_rows = []

    for config_name, sl_adjust_func in sl_reduction_configs:
        working_df = df.copy()
        adjusted_sl = sl_adjust_func(working_df["SL"])
        adjusted_sl = np.maximum(adjusted_sl, 1.1)  # Minimum 1.1 pip (broker limit)

        for ratio, breakeven_rate in RRR_CONFIGS:
            total_trades = len(working_df)

            if total_trades > 0:
                profitable = working_df[
                    (working_df["SL"] != working_df["Pullback"])
                    & (working_df["Pullback"] < adjusted_sl)
                    & (working_df["TP"] >= (ratio * adjusted_sl))
                ]
                wins = len(profitable)
                losses = total_trades - wins
                win_rate = wins / total_trades * 100
            else:
                wins = 0
                losses = 0
                win_rate = 0.0

            sl_reduction_rows.append({
                'Type': config_name if ratio == 1 else '',
                'RRR': f'1:{ratio}',
                'Total Trades': total_trades,
                'Wins': wins,
                'Losses': losses,
                'Win %': f"{win_rate:.1f}%"
            })

    final_table = pd.DataFrame(sl_reduction_rows)
    return {"SL Reduction Analysis": final_table}


# ============================================================================
# STRATEGY CREATION AND EVALUATION
# ============================================================================


def _create_technical_strategies() -> List[Tuple[str, Callable, str]]:
    """Create technical indicator based strategies."""
    return [
        (
            "EMA + Trade Direction",
            lambda df: df[df["EMA"] == df["Direction"]],
            "Trade with EMA trend",
        ),
        (
            "EMA + Opposite Trade Direction",
            lambda df: df[df["EMA"] != df["Direction"]],
            "Counter-trend trades",
        ),
        (
            "EMA + BOS",
            lambda df: df[(df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")],
            "Trend + Break of Structure",
        ),
        (
            "EMA + CH",
            lambda df: df[(df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")],
            "Trend + Change of Character",
        ),
        ("BOS", lambda df: df[df["BOS/CH"] == "BOS"], "Break of Structure trades only"),
        ("CH", lambda df: df[df["BOS/CH"] == "CH"], "Change of Character trades only"),
    ]


def _create_risk_management_strategies() -> List[Tuple[str, Callable, str]]:
    """Create risk management based strategies."""
    return [
        (
            "Conservative: SL <= 2 pips",
            lambda df: df[df["SL"] <= 2],
            "Very tight stop losses",
        ),
        (
            "Moderate Risk: SL 3-6 pips",
            lambda df: df[(df["SL"] >= 3) & (df["SL"] <= 6)],
            "Medium stop losses",
        ),
        ("Aggressive: SL >= 7 pips", lambda df: df[df["SL"] >= 7], "Wide stop losses"),
        (
            "BOS + Conservative SL <= 2 pips",
            lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] <= 2)],
            "BOS with tight stops",
        ),
        (
            "BOS + Moderate SL 3-6 pips",
            lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] >= 3) & (df["SL"] <= 6)],
            "BOS with medium stops",
        ),
    ]


def _create_30m_trend_strategies() -> List[Tuple[str, Callable, str]]:
    """Create 30-minute timeframe trend alignment strategies."""
    strategies = []

    # Basic 30M trend
    strategies.append(
        (
            "With 30M Trend",
            lambda df: df[
                (
                    (
                        df["30M Leg"].isin(["Above H", "Above L"])
                        & (df["Direction"] == "Buy")
                    )
                    | (
                        df["30M Leg"].isin(["Below H", "Below L"])
                        & (df["Direction"] == "Sell")
                    )
                )
            ],
            "Higher timeframe trend alignment",
        )
    )

    # 30M + Technical indicators
    strategies.extend(
        [
            (
                "30M Trend + EMA",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["EMA"] == df["Direction"])
                ],
                "30M trend with EMA confirmation",
            ),
            (
                "30M Trend + BOS",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["BOS/CH"] == "BOS")
                ],
                "30M trend with Break of Structure",
            ),
            (
                "30M Trend + CH",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["BOS/CH"] == "CH")
                ],
                "30M trend with Change of Character",
            ),
            (
                "30M Trend + EMA + BOS",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["EMA"] == df["Direction"])
                    & (df["BOS/CH"] == "BOS")
                ],
                "Triple confirmation: 30M + EMA + BOS",
            ),
            (
                "30M Trend + EMA + CH",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["EMA"] == df["Direction"])
                    & (df["BOS/CH"] == "CH")
                ],
                "Triple confirmation: 30M + EMA + CH",
            ),
        ]
    )

    # 30M + Risk management
    strategies.extend(
        [
            (
                "30M Trend + SL < 10 pips",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["SL"] < 10)
                ],
                "30M trend excluding large stops",
            ),
            (
                "30M Trend + SL < 15 pips",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["SL"] < 15)
                ],
                "30M trend excluding very large stops",
            ),
            (
                "30M Trend + SL > 3 pips",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["SL"] > 3)
                ],
                "30M trend excluding tiny stops",
            ),
            (
                "30M Trend + SL > 5 pips",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["SL"] > 5)
                ],
                "30M trend excluding small stops",
            ),
            (
                "30M Trend + 3 < SL < 10",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["SL"] > 3)
                    & (df["SL"] < 10)
                ],
                "30M trend with medium stops only",
            ),
            (
                "30M Trend + 5 < SL < 15",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["SL"] > 5)
                    & (df["SL"] < 15)
                ],
                "30M trend with moderate stops",
            ),
        ]
    )

    # Complex multi-factor with 30M
    strategies.extend(
        [
            (
                "30M Trend + BOS + SL < 10",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["BOS/CH"] == "BOS")
                    & (df["SL"] < 10)
                ],
                "30M + BOS with risk control",
            ),
            (
                "30M Trend + CH + SL < 10",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["BOS/CH"] == "CH")
                    & (df["SL"] < 10)
                ],
                "30M + CH with risk control",
            ),
            (
                "30M Trend + EMA + SL < 10",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["EMA"] == df["Direction"])
                    & (df["SL"] < 10)
                ],
                "30M + EMA with risk control",
            ),
            (
                "30M Trend + EMA + BOS + SL < 10",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["EMA"] == df["Direction"])
                    & (df["BOS/CH"] == "BOS")
                    & (df["SL"] < 10)
                ],
                "Full confluence with risk limit",
            ),
        ]
    )

    # 30M + News filters
    strategies.extend(
        [
            (
                "30M Trend + No News",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & df["News Event"].isna()
                ],
                "30M trend avoiding news",
            ),
            (
                "30M Trend + News > 2hrs",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (~df["News Event"].isna())
                    & (df["Hours Until News"] >= 2)
                ],
                "30M trend with safe news distance",
            ),
        ]
    )

    # Additional combinations
    strategies.extend(
        [
            (
                "30M Trend + EMA + 3 < SL < 10",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["EMA"] == df["Direction"])
                    & (df["SL"] > 3)
                    & (df["SL"] < 10)
                ],
                "30M + EMA with optimal stops",
            ),
            (
                "30M Trend + CH + No News",
                lambda df: df[
                    (
                        (
                            df["30M Leg"].isin(["Above H", "Above L"])
                            & (df["Direction"] == "Buy")
                        )
                        | (
                            df["30M Leg"].isin(["Below H", "Below L"])
                            & (df["Direction"] == "Sell")
                        )
                    )
                    & (df["BOS/CH"] == "CH")
                    & df["News Event"].isna()
                ],
                "30M + CH in clean conditions",
            ),
        ]
    )

    return strategies


def _create_news_strategies() -> List[Tuple[str, Callable, str]]:
    """Create news event based strategies."""
    return [
        (
            "No News Events",
            lambda df: df[df["News Event"].isna()],
            "Avoid news volatility",
        ),
        (
            "News Event Present",
            lambda df: df[~df["News Event"].isna()],
            "Trade during news",
        ),
        (
            "News in 2+ Hours",
            lambda df: df[(~df["News Event"].isna()) & (df["Hours Until News"] >= 2)],
            "Trade before news impact",
        ),
    ]


def _create_double_setup_strategies() -> List[Tuple[str, Callable, str]]:
    """Create double setup strategies (exactly 2 factors combined)."""
    strategies = []

    # EMA + BOS/CH combinations
    strategies.extend([
        ("EMA + BOS", lambda df: df[(df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")], "EMA trend + Break of Structure"),
        ("EMA + CH", lambda df: df[(df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")], "EMA trend + Change of Character"),
    ])

    # 30M Trend + Technical indicators
    strategies.extend([
        ("30M Trend + EMA", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"])
        ], "30M trend + EMA confirmation"),
        ("30M Trend + BOS", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["BOS/CH"] == "BOS")
        ], "30M trend + Break of Structure"),
        ("30M Trend + CH", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["BOS/CH"] == "CH")
        ], "30M trend + Change of Character"),
    ])

    # 30M Trend + Risk management
    strategies.extend([
        ("30M Trend + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] < 10)
        ], "30M trend + tight stops"),
        ("30M Trend + SL < 15", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] < 15)
        ], "30M trend + moderate stops"),
        ("30M Trend + SL > 3", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 3)
        ], "30M trend excluding tiny stops"),
        ("30M Trend + SL > 5", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 5)
        ], "30M trend excluding small stops"),
    ])

    # 30M + News filters
    strategies.extend([
        ("30M Trend + No News", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            df["News Event"].isna()
        ], "30M trend avoiding news"),
        ("30M Trend + News > 2hrs", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)
        ], "30M trend + safe news distance"),
    ])

    # BOS/CH + Risk management
    strategies.extend([
        ("BOS + SL ≤ 2", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] <= 2)], "BOS with very tight stops"),
        ("BOS + SL 3-6", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] >= 3) & (df["SL"] <= 6)], "BOS with medium stops"),
        ("CH + SL ≤ 2", lambda df: df[(df["BOS/CH"] == "CH") & (df["SL"] <= 2)], "CH with very tight stops"),
        ("CH + SL 3-6", lambda df: df[(df["BOS/CH"] == "CH") & (df["SL"] >= 3) & (df["SL"] <= 6)], "CH with medium stops"),
    ])

    # BOS/CH + News filters
    strategies.extend([
        ("BOS + No News", lambda df: df[(df["BOS/CH"] == "BOS") & df["News Event"].isna()], "BOS avoiding news"),
        ("CH + No News", lambda df: df[(df["BOS/CH"] == "CH") & df["News Event"].isna()], "CH avoiding news"),
        ("BOS + News > 2hrs", lambda df: df[(df["BOS/CH"] == "BOS") & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)], "BOS with safe news distance"),
        ("CH + News > 2hrs", lambda df: df[(df["BOS/CH"] == "CH") & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)], "CH with safe news distance"),
    ])

    # EMA + Risk management
    strategies.extend([
        ("EMA + SL ≤ 2", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] <= 2)], "EMA trend with tight stops"),
        ("EMA + SL 3-6", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] >= 3) & (df["SL"] <= 6)], "EMA trend with medium stops"),
        ("EMA + SL < 10", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] < 10)], "EMA trend excluding large stops"),
    ])

    # EMA + News filters
    strategies.extend([
        ("EMA + No News", lambda df: df[(df["EMA"] == df["Direction"]) & df["News Event"].isna()], "EMA trend avoiding news"),
        ("EMA + News > 2hrs", lambda df: df[(df["EMA"] == df["Direction"]) & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)], "EMA trend with safe news distance"),
    ])

    return strategies


def _create_triple_setup_strategies() -> List[Tuple[str, Callable, str]]:
    """Create triple setup strategies (3+ factors combined)."""
    strategies = []

    # Triple combinations with 30M + EMA + BOS/CH
    strategies.extend([
        ("30M Trend + EMA + BOS", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")
        ], "Triple confirmation: 30M + EMA + BOS"),
        ("30M Trend + EMA + CH", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")
        ], "Triple confirmation: 30M + EMA + CH"),
    ])

    # EMA + BOS + SL variations (tight to moderate risk management)
    strategies.extend([
        ("EMA + BOS + SL < 8", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] < 8)
        ], "EMA + BOS with very tight stops"),
        ("EMA + BOS + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] < 10)
        ], "EMA + BOS with tight stops"),
        ("EMA + BOS + SL < 12", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] < 12)
        ], "EMA + BOS with controlled stops"),
        ("EMA + BOS + SL < 15", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] < 15)
        ], "EMA + BOS with moderate stops"),
        ("EMA + BOS + SL > 3", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 3)
        ], "EMA + BOS excluding tiny stops"),
        ("EMA + BOS + skip SL <= 4", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 4)
        ], "EMA + BOS excluding small stops"),
        ("EMA + BOS + SL > 5", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 5)
        ], "EMA + BOS with meaningful stops"),
        ("EMA + BOS + 3 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 3) & (df["SL"] < 10)
        ], "EMA + BOS optimal stop range"),
        ("EMA + BOS + 4 < SL < 12", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 4) & (df["SL"] < 12)
        ], "EMA + BOS balanced stop range"),
        ("EMA + BOS + 5 < SL < 15", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 5) & (df["SL"] < 15)
        ], "EMA + BOS moderate stop range"),
        ("EMA + BOS + 3 < SL < 8", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 3) & (df["SL"] < 8)
        ], "EMA + BOS tight optimal range"),
    ])

    # EMA + BOS + 30M Trend alignment
    strategies.extend([
        ("EMA + BOS + 30M Trend", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell")))
        ], "EMA + BOS with 30M trend confirmation"),
    ])

    # EMA + BOS + News filtering
    strategies.extend([
        ("EMA + BOS + No News", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            df["News Event"].isna()
        ], "EMA + BOS avoiding all news"),
        ("EMA + BOS + News > 2hrs", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)
        ], "EMA + BOS with safe news distance"),
        ("EMA + BOS + No News or News > 2hrs", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["News Event"].isna() | (df["Hours Until News"] >= 2))
        ], "EMA + BOS with news safety"),
    ])

    # EMA + BOS + Combined SL and News
    strategies.extend([
        ("EMA + BOS + SL < 10 + No News", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] < 10) & df["News Event"].isna()
        ], "EMA + BOS tight stops avoiding news"),
        ("EMA + BOS + SL < 10 + News > 2hrs", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] < 10) & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)
        ], "EMA + BOS tight stops with news safety"),
        ("EMA + BOS + 3 < SL < 10 + No News", lambda df: df[
            (df["EMA"] == df["Direction"]) &
            (df["BOS/CH"] == "BOS") &
            (df["SL"] > 3) & (df["SL"] < 10) & df["News Event"].isna()
        ], "EMA + BOS optimal stops avoiding news"),
    ])

    # Triple combinations with risk management (non-EMA+BOS)
    strategies.extend([
        ("30M Trend + BOS + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["BOS/CH"] == "BOS") & (df["SL"] < 10)
        ], "30M + BOS with risk control"),
        ("30M Trend + CH + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["BOS/CH"] == "CH") & (df["SL"] < 10)
        ], "30M + CH with risk control"),
        ("30M Trend + EMA + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] < 10)
        ], "30M + EMA with risk control"),
    ])

    # Quadruple combination - ultimate confluence
    strategies.extend([
        ("30M Trend + EMA + BOS + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & (df["SL"] < 10)
        ], "Full confluence with risk limit"),
    ])

    # Complex SL range combinations
    strategies.extend([
        ("30M Trend + 3 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 3) & (df["SL"] < 10)
        ], "30M trend with optimal stops"),
        ("30M Trend + 5 < SL < 15", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 5) & (df["SL"] < 15)
        ], "30M trend with moderate stops"),
        ("30M Trend + EMA + 3 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] > 3) & (df["SL"] < 10)
        ], "30M + EMA with optimal stops"),
    ])

    # Triple combinations with news filtering
    strategies.extend([
        ("30M Trend + CH + No News", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["BOS/CH"] == "CH") & df["News Event"].isna()
        ], "30M + CH in clean conditions"),
    ])

    return strategies


def create_strategy_library() -> List[Strategy]:
    """
    Create a comprehensive library of trading strategies for backtesting.
    This includes all strategies (single, double, and triple setups).

    Returns:
        List of Strategy objects ready for backtesting
    """
    all_strategies = []

    # Combine all strategy categories
    all_strategies.extend(_create_technical_strategies())
    all_strategies.extend(_create_risk_management_strategies())
    all_strategies.extend(_create_30m_trend_strategies())
    all_strategies.extend(_create_news_strategies())

    # Convert to Strategy objects
    return [Strategy(name, func, desc) for name, func, desc in all_strategies]


def create_double_setup_strategy_library() -> List[Strategy]:
    """
    Create a library of double setup trading strategies (exactly 2 factors).

    Returns:
        List of double setup Strategy objects ready for backtesting
    """
    double_strategies = _create_double_setup_strategies()
    return [Strategy(name, func, desc) for name, func, desc in double_strategies]


def create_triple_setup_strategy_library() -> List[Strategy]:
    """
    Create a library of triple+ setup trading strategies (3+ factors).

    Returns:
        List of triple+ setup Strategy objects ready for backtesting
    """
    triple_strategies = _create_triple_setup_strategies()
    return [Strategy(name, func, desc) for name, func, desc in triple_strategies]


def _create_single_setup_strategies() -> List[Tuple[str, Callable, str]]:
    """Create single setup strategies (no combinations)."""
    strategies = []

    # Technical Indicators (Individual)
    strategies.extend([
        ("EMA Aligned", lambda df: df[df["EMA"] == df["Direction"]], "EMA matches trade direction"),
        ("EMA Counter-Trend", lambda df: df[df["EMA"] != df["Direction"]], "EMA opposite to trade direction"),
        ("BOS Only", lambda df: df[df["BOS/CH"] == "BOS"], "Break of Structure trades only"),
        ("CH Only", lambda df: df[df["BOS/CH"] == "CH"], "Change of Character trades only"),
    ])

    # Risk Management (Individual)
    strategies.extend([
        ("SL ≤ 2 pips", lambda df: df[df["SL"] <= 2], "Very tight stop losses"),
        ("SL 3-6 pips", lambda df: df[(df["SL"] >= 3) & (df["SL"] <= 6)], "Medium stop losses"),
        ("SL ≥ 7 pips", lambda df: df[df["SL"] >= 7], "Wide stop losses"),
        ("SL < 10 pips", lambda df: df[df["SL"] < 10], "Exclude very large stops"),
        ("SL > 3 pips", lambda df: df[df["SL"] > 3], "Exclude tiny stops"),
        ("SL > 5 pips", lambda df: df[df["SL"] > 5], "Exclude small stops"),
        ("SL < 15 pips", lambda df: df[df["SL"] < 15], "Exclude extremely large stops"),
    ])

    # 30M Trend (Individual)
    strategies.extend([
        ("30M Trend", lambda df: df[
            (df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
            (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))
        ], "Higher timeframe trend alignment only"),
    ])

    # News Events (Individual)
    strategies.extend([
        ("No News", lambda df: df[df["News Event"].isna()], "Avoid news volatility"),
        ("With News", lambda df: df[~df["News Event"].isna()], "Trade during news periods"),
        ("News > 2hrs", lambda df: df[(~df["News Event"].isna()) & (df["Hours Until News"] >= 2)], "Safe distance from news"),
    ])

    # Trade Direction (Individual)
    strategies.extend([
        ("Buy Trades Only", lambda df: df[df["Direction"] == "Buy"], "Long positions only"),
        ("Sell Trades Only", lambda df: df[df["Direction"] == "Sell"], "Short positions only"),
    ])

    # Additional SL ranges
    strategies.extend([
        ("SL 1-3 pips", lambda df: df[(df["SL"] >= 1) & (df["SL"] <= 3)], "Very tight SL range"),
        ("SL 4-8 pips", lambda df: df[(df["SL"] >= 4) & (df["SL"] <= 8)], "Moderate SL range"),
        ("SL 9-15 pips", lambda df: df[(df["SL"] >= 9) & (df["SL"] <= 15)], "Wide SL range"),
        ("SL > 10 pips", lambda df: df[df["SL"] > 10], "Large stops only"),
        ("SL > 15 pips", lambda df: df[df["SL"] > 15], "Very large stops only"),
    ])

    return strategies


def create_single_setup_strategy_library() -> List[Strategy]:
    """
    Create a library of single setup trading strategies for backtesting.

    Unlike the main strategy library, this focuses on individual components
    rather than combinations, allowing for analysis of single factors.

    Returns:
        List of single setup Strategy objects ready for backtesting
    """
    single_strategies = _create_single_setup_strategies()
    return [Strategy(name, func, desc) for name, func, desc in single_strategies]


def evaluate_all_strategies(
    df: pd.DataFrame, strategies: List[Strategy]
) -> Dict[str, pd.DataFrame]:
    """
    Run backtesting on all strategies and compile results.

    Args:
        df: Trading data
        strategies: List of Strategy objects

    Returns:
        Dictionary mapping strategy names to their performance DataFrames
    """
    strategy_results = {}
    for strategy in strategies:
        # Apply strategy filter
        filtered_df = strategy.apply(df)

        # Calculate normal RRR statistics
        summary_df = calculate_rrr_stats(filtered_df, strategy.name, 'SL')
        strategy_results[f"{strategy.name}[{ENTRY_TYPE_NAMES['SL']}]"] = (
            summary_df
        )

    return strategy_results


def _calculate_strategy_drawdown(df: pd.DataFrame, strategy: Strategy, ratio: int) -> float:
    """
    Calculate maximum drawdown in R for a strategy at a given RRR ratio.

    Args:
        df: Trading data
        strategy: Strategy object
        ratio: RRR ratio (1, 2, or 3)

    Returns:
        Maximum drawdown in R
    """
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        return 0.0

    # Calculate outcomes for each trade
    outcomes = []
    for _, trade in filtered_df.iterrows():
        is_win = (
            (trade['SL'] != trade['Pullback']) and
            (trade['Pullback'] < trade['SL']) and
            (trade['TP'] >= ratio * trade['SL'])
        )
        if is_win:
            outcomes.append(ratio)
        else:
            outcomes.append(-1)

    # Calculate cumulative returns
    cumulative = np.cumsum(outcomes)

    # Calculate drawdown: max drop from any peak
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns)

    return max_drawdown


def get_top_strategies_by_edge(
    strategy_results: Dict[str, pd.DataFrame],
    rrr_column: str,
    df: pd.DataFrame = None,
    strategies: List[Strategy] = None
) -> pd.DataFrame:
    """
    Extract top performing strategies for a specific RRR, sorted by Edge.

    Args:
        strategy_results: Dictionary of strategy results
        rrr_column: Column name for RRR (e.g., '1:2 RRR')
        df: Original trading data (optional, required for drawdown calculation)
        strategies: List of Strategy objects (optional, required for drawdown calculation)

    Returns:
        Top strategies ranked by edge
    """
    strategy_performance = []

    # Extract RRR ratio from column name (e.g., "1:2 RRR" -> 2)
    rrr_ratio = None
    if df is not None and strategies is not None:
        try:
            rrr_ratio = int(rrr_column.split(':')[1].split()[0])
        except:
            rrr_ratio = None

    # Create a mapping from strategy name to Strategy object
    strategy_map = {}
    if strategies is not None:
        for strategy in strategies:
            strategy_map[strategy.name] = strategy

    for strategy_name, df_result in strategy_results.items():
        # Skip if column doesn't exist
        if rrr_column not in df_result.columns:
            continue

        # Extract performance metrics
        total_trades = df_result[rrr_column].iloc[0]
        wins = df_result[rrr_column].iloc[1]
        losses = df_result[rrr_column].iloc[2]
        win_rate = df_result[rrr_column].iloc[3]
        edge = df_result[rrr_column].iloc[4]
        outcome_str = df_result[rrr_column].iloc[5]
        entry_str = df_result[rrr_column].iloc[6]

        # Parse edge value for sorting - handle both string and numeric
        try:
            if isinstance(edge, str):
                # Remove % sign and convert to float
                edge_str = edge.strip()
                if edge_str.endswith("%"):
                    edge_value = float(edge_str[:-1])
                else:
                    edge_value = float(edge_str)
            else:
                edge_value = float(edge) if edge else 0.0
        except (ValueError, TypeError, AttributeError):
            # If parsing fails, set to 0
            edge_value = 0.0

        # Calculate trades required to earn 1R
        trades_required = "N/A"
        try:
            if isinstance(outcome_str, str) and outcome_str.endswith("R"):
                r_value = float(outcome_str[:-1])
                if r_value > 0:
                    trades_required = f"{total_trades / r_value:.1f}"
        except (ValueError, TypeError, AttributeError):
            trades_required = "N/A"

        # Clean up display name
        display_name = strategy_name.split("[")[0].strip()

        # Calculate drawdown if possible
        drawdown_str = "N/A"
        if df is not None and rrr_ratio is not None and display_name in strategy_map:
            strategy = strategy_map[display_name]
            drawdown_value = _calculate_strategy_drawdown(df, strategy, rrr_ratio)
            drawdown_str = f"{int(drawdown_value)}"

        strategy_performance.append(
            {
                "Strategy": display_name,
                "Entry": entry_str,
                "Trades": total_trades,
                "Wins": wins,
                "Losses": losses,
                "Win Rate": win_rate,
                "Edge": edge,
                "Outcome": outcome_str,
                "Trades Required": trades_required,
                "Drawdown": drawdown_str,
                "edge_value": edge_value,
            }
        )

    # Filter positive edge strategies and sort
    filtered_strategies = [s for s in strategy_performance if s["edge_value"] > 0]

    # Sort by edge_value in descending order
    top_strategies = sorted(
        filtered_strategies, key=lambda x: x["edge_value"], reverse=True
    )

    # Remove sorting key from display
    for strat in top_strategies:
        del strat["edge_value"]

    return pd.DataFrame(top_strategies)


# ============================================================================
# VISUALIZATION AND DISPLAY FUNCTIONS
# ============================================================================


def display_hour_analysis(df: pd.DataFrame):
    """Display hour profitability analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>Hour Analysis</h2>"))
    hour_tables = analyze_hour_profitability(df)

    for table_name, table_df in hour_tables.items():
        html_table = create_sortable_table(table_df)
        display(HTML(html_table))
        print('')


def display_weekday_analysis(df: pd.DataFrame):
    """Display weekday profitability analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>Weekday Analysis</h2>"))
    weekday_tables = analyze_weekday_profitability(df)

    for table_name, table_df in weekday_tables.items():
        html_table = create_sortable_table(table_df)
        display(HTML(html_table))
        print('')


def display_pullback_analysis(df: pd.DataFrame):
    """Display pullback profitability analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>Pullback Analysis</h2>"))
    pullback_tables = analyze_pullback_profitability(df)

    for table_name, table_df in pullback_tables.items():
        html_table = create_sortable_table(table_df)
        display(HTML(html_table))
        print('')


def display_sl_distribution_analysis(df: pd.DataFrame):
    """Display stop loss distribution analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>Stop Loss Distribution Analysis</h2>"))
    sl_distribution_tables = analyze_sl_distribution(df)

    for table_name, table_df in sl_distribution_tables.items():
        html_table = create_sortable_table(table_df)
        display(HTML(html_table))
        print('')


def display_tp_distribution_analysis(df: pd.DataFrame):
    """Display take profit distribution analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>Take Profit Distribution Analysis</h2>"))
    tp_distribution_tables = analyze_tp_distribution(df)

    for table_name, table_df in tp_distribution_tables.items():
        html_table = create_sortable_table(table_df)
        display(HTML(html_table))
        print('')


def analyze_ema_30m_trend_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how EMA and 30M Trend alignment affects trade profitability.

    Creates a table showing:
    1. EMA aligned with trade direction
    2. 30M Trend aligned with trade direction
    3. Both EMA and 30M Trend aligned with trade direction

    Args:
        df: Trading data with EMA, Direction, 30M Leg, Pullback, TP, and SL columns

    Returns:
        DataFrame containing the alignment analysis
    """
    rows = []

    # Configuration for different alignment scenarios
    scenarios = [
        ("EMA aligned with trade", lambda d: d[d["EMA"] == d["Direction"]]),
        ("30M Trend aligned with trade", lambda d: d[
            ((d["Direction"] == "Buy") & d["30M Leg"].isin(["Above H", "Above L"])) |
            ((d["Direction"] == "Sell") & d["30M Leg"].isin(["Below H", "Below L"]))
        ]),
        ("EMA + 30M Trend aligned with trade", lambda d: d[
            (d["EMA"] == d["Direction"]) &
            (
                ((d["Direction"] == "Buy") & d["30M Leg"].isin(["Above H", "Above L"])) |
                ((d["Direction"] == "Sell") & d["30M Leg"].isin(["Below H", "Below L"]))
            )
        ])
    ]

    # Analyze each scenario for 1:1 RRR
    for scenario_name, filter_func in scenarios:
        filtered_df = filter_func(df)
        total_trades = len(filtered_df)

        if total_trades > 0:
            # Win condition for 1:1 RRR: SL != Pullback AND Pullback < SL AND TP >= SL
            profitable = filtered_df[
                (filtered_df["SL"] != filtered_df["Pullback"])
                & (filtered_df["Pullback"] < filtered_df["SL"])
                & (filtered_df["TP"] >= filtered_df["SL"])
            ]
            wins = len(profitable)
            losses = total_trades - wins
            win_rate = wins / total_trades * 100
        else:
            wins = 0
            losses = 0
            win_rate = 0.0

        rows.append({
            'Scenario': scenario_name,
            'Total Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Win Rate': f"{win_rate:.1f}%"
        })

    return pd.DataFrame(rows)


def display_ema_30m_trend_analysis(df: pd.DataFrame):
    """Display EMA and 30M Trend alignment analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>EMA and 30M Trend Alignment Analysis</h2>"))
    alignment_df = analyze_ema_30m_trend_alignment(df)

    html_table = create_sortable_table(alignment_df)
    display(HTML(html_table))
    print('')


def display_sl_reduction_analysis(df: pd.DataFrame):
    """Display SL reduction profitability analysis with proper formatting."""
    from IPython.display import display, HTML

    display(HTML("<h2>SL Reduction Analysis</h2>"))
    sl_reduction_tables = analyze_sl_reduction_profitability(df)

    for table_name, table_df in sl_reduction_tables.items():
        html_table = create_sortable_table(table_df)
        display(HTML(html_table))
        print('')


def display_double_setup_strategy_analysis(df: pd.DataFrame):
    """Display double setup strategy analysis with proper formatting."""
    from IPython.display import display, HTML

    strategies = [
        Strategy(
            "Plain Strategy",
            lambda df: df,
            "Baseline: All trades without any filtering"
        )
    ]
    strategies.extend(create_double_setup_strategy_library())
    strategy_results = evaluate_all_strategies(df, strategies)

    # Collect all results for all RRR levels
    all_results = []

    rrr_configs = [
        ('1:1 RRR', '1:1'),
        ('1:2 RRR', '1:2'),
        ('1:3 RRR', '1:3'),
    ]

    for rrr_column, rrr_label in rrr_configs:
        top_df = get_top_strategies_by_edge(strategy_results, rrr_column, df, strategies)

        # Add RRR column to each row
        if not top_df.empty:
            top_df['RRR'] = f'{rrr_label} RRR'
            # Remove Entry column as it will be replaced by RRR
            if 'Entry' in top_df.columns:
                top_df = top_df.drop('Entry', axis=1)
            all_results.append(top_df)

    # Combine all results and sort by edge
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Extract edge values for sorting
        combined_df['edge_value'] = combined_df['Edge'].apply(
            lambda x: float(x.strip('%')) if isinstance(x, str) and x.strip().endswith('%') else 0.0
        )

        # Sort by edge value in descending order
        combined_df = combined_df.sort_values('edge_value', ascending=False)

        # Drop the temporary sorting column
        combined_df = combined_df.drop('edge_value', axis=1)

        # Reorder columns to put RRR second, Trades Required and Drawdown at the end
        cols = combined_df.columns.tolist()
        if 'RRR' in cols:
            cols.remove('RRR')
            cols.insert(1, 'RRR')
        if 'Trades Required' in cols:
            cols.remove('Trades Required')
            cols.append('Trades Required')
        if 'Drawdown' in cols:
            cols.remove('Drawdown')
            cols.append('Drawdown')
        combined_df = combined_df[cols]

        # Display the combined table with sortable columns
        display(HTML(f"<h2>Double Setup Strategies</h2>"))
        html_table = create_sortable_table(combined_df, first_column_width='300px', highlight_column='Edge', highlight_color='green')
        display(HTML(html_table))
        print()


def display_single_setup_strategy_analysis(df: pd.DataFrame):
    """Display single setup strategy analysis with proper formatting."""
    from IPython.display import display, HTML

    strategies = [
        Strategy(
            "Plain Strategy",
            lambda df: df,
            "Baseline: All trades without any filtering"
        )
    ]
    strategies.extend(create_single_setup_strategy_library())
    strategy_results = evaluate_all_strategies(df, strategies)

    # Collect all results for all RRR levels
    all_results = []

    rrr_configs = [
        ('1:1 RRR', '1:1'),
        ('1:2 RRR', '1:2'),
        ('1:3 RRR', '1:3'),
    ]

    for rrr_column, rrr_label in rrr_configs:
        top_df = get_top_strategies_by_edge(strategy_results, rrr_column, df, strategies)

        # Add RRR column to each row
        if not top_df.empty:
            top_df['RRR'] = f'{rrr_label} RRR'
            # Remove Entry column as it will be replaced by RRR
            if 'Entry' in top_df.columns:
                top_df = top_df.drop('Entry', axis=1)
            all_results.append(top_df)

    # Combine all results and sort by edge
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Extract edge values for sorting
        combined_df['edge_value'] = combined_df['Edge'].apply(
            lambda x: float(x.strip('%')) if isinstance(x, str) and x.strip().endswith('%') else 0.0
        )

        # Sort by edge value in descending order
        combined_df = combined_df.sort_values('edge_value', ascending=False)

        # Drop the temporary sorting column
        combined_df = combined_df.drop('edge_value', axis=1)

        # Reorder columns to put RRR second, Trades Required and Drawdown at the end
        cols = combined_df.columns.tolist()
        if 'RRR' in cols:
            cols.remove('RRR')
            cols.insert(1, 'RRR')
        if 'Trades Required' in cols:
            cols.remove('Trades Required')
            cols.append('Trades Required')
        if 'Drawdown' in cols:
            cols.remove('Drawdown')
            cols.append('Drawdown')
        combined_df = combined_df[cols]

        # Display the combined table with sortable columns
        display(HTML(f"<h2>Single Setup Strategies</h2>"))
        html_table = create_sortable_table(combined_df, first_column_width='300px', highlight_column='Edge', highlight_color='green')
        display(HTML(html_table))
        print()


def display_triple_setup_strategy_analysis(df: pd.DataFrame):
    """Display triple setup strategy analysis with proper formatting."""
    from IPython.display import display, HTML

    strategies = [
        Strategy(
            "Plain Strategy",
            lambda df: df,
            "Baseline: All trades without any filtering"
        )
    ]
    strategies.extend(create_triple_setup_strategy_library())
    strategy_results = evaluate_all_strategies(df, strategies)

    # Collect all results for all RRR levels
    all_results = []

    rrr_configs = [
        ('1:1 RRR', '1:1'),
        ('1:2 RRR', '1:2'),
        ('1:3 RRR', '1:3'),
    ]

    for rrr_column, rrr_label in rrr_configs:
        top_df = get_top_strategies_by_edge(strategy_results, rrr_column, df, strategies)

        # Add RRR column to each row
        if not top_df.empty:
            top_df['RRR'] = f'{rrr_label} RRR'
            # Remove Entry column as it will be replaced by RRR
            if 'Entry' in top_df.columns:
                top_df = top_df.drop('Entry', axis=1)
            all_results.append(top_df)

    # Combine all results and sort by edge
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Extract edge values for sorting
        combined_df['edge_value'] = combined_df['Edge'].apply(
            lambda x: float(x.strip('%')) if isinstance(x, str) and x.strip().endswith('%') else 0.0
        )

        # Sort by edge value in descending order
        combined_df = combined_df.sort_values('edge_value', ascending=False)

        # Drop the temporary sorting column
        combined_df = combined_df.drop('edge_value', axis=1)

        # Reorder columns to put RRR second, Trades Required and Drawdown at the end
        cols = combined_df.columns.tolist()
        if 'RRR' in cols:
            cols.remove('RRR')
            cols.insert(1, 'RRR')
        if 'Trades Required' in cols:
            cols.remove('Trades Required')
            cols.append('Trades Required')
        if 'Drawdown' in cols:
            cols.remove('Drawdown')
            cols.append('Drawdown')
        combined_df = combined_df[cols]

        # Display the combined table with sortable columns
        display(HTML(f"<h2>Triple Setup Strategies</h2>"))
        html_table = create_sortable_table(combined_df, first_column_width='300px', highlight_column='Edge', highlight_color='green')
        display(HTML(html_table))
        print()


def display_strategy_cumulative_chart(df: pd.DataFrame, strategy: Strategy):
    """
    Display cumulative performance chart for a specific strategy with all three RRR ratios.

    Args:
        df: Trading data
        strategy: Strategy object to analyze
    """
    from IPython.display import display, HTML

    # Apply the strategy filter
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        display(HTML(f"<h2>{strategy.name}</h2>"))
        display(HTML("<p>No trades found for this strategy.</p>"))
        return

    # Calculate outcomes for each RRR ratio
    rrr_ratios = [1, 2, 3]
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    cumulative_outcomes = {}

    for ratio in rrr_ratios:
        outcomes = []
        for _, trade in filtered_df.iterrows():
            # Win condition: all three must be true
            is_win = (
                (trade['SL'] != trade['Pullback']) and
                (trade['Pullback'] < trade['SL']) and
                (trade['TP'] >= ratio * trade['SL'])
            )
            if is_win:
                outcomes.append(ratio)  # Win = +ratio R
            else:
                outcomes.append(-1)  # Loss = -1R

        cumulative_outcomes[ratio] = np.cumsum(outcomes)

    # Create the chart
    plt.figure(figsize=(14, 7))
    trade_numbers = range(1, len(filtered_df) + 1)

    for i, ratio in enumerate(rrr_ratios):
        cumulative = cumulative_outcomes[ratio]
        plt.plot(trade_numbers, cumulative, linewidth=2, color=colors[i],
                label=f'1:{ratio} RRR (Final: {cumulative[-1]:.1f}R)', alpha=0.85)

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Cumulative Outcome (R)', fontsize=12)
    plt.title(strategy.name, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.show()


def _is_strategy_profitable(df: pd.DataFrame, strategy: Strategy) -> bool:
    """
    Check if a strategy has positive edge for at least one RRR ratio.

    Args:
        df: Trading data
        strategy: Strategy to evaluate

    Returns:
        True if strategy has positive edge for any RRR ratio
    """
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        return False

    # Check each RRR ratio for positive edge
    for ratio, breakeven_rate in RRR_CONFIGS:
        total_trades = len(filtered_df)
        wins, losses = _calculate_wins_and_losses(
            filtered_df, ratio, 'SL', _win_condition_normal
        )

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            edge = win_rate - breakeven_rate

            if edge > 0:
                return True

    return False


def display_single_setup_strategy_charts(df: pd.DataFrame):
    """Display cumulative performance charts for profitable single setup strategies only."""
    from IPython.display import display, HTML

    display(HTML("<h1>Single Setup Strategies - Performance Charts</h1>"))

    strategies = create_single_setup_strategy_library()

    # Filter to only profitable strategies
    profitable_strategies = [s for s in strategies if _is_strategy_profitable(df, s)]

    display(HTML(f"<p>Showing {len(profitable_strategies)} profitable strategies out of {len(strategies)} total strategies.</p>"))

    for strategy in profitable_strategies:
        display_strategy_cumulative_chart(df, strategy)


def display_double_setup_strategy_charts(df: pd.DataFrame):
    """Display cumulative performance charts for profitable double setup strategies only."""
    from IPython.display import display, HTML

    display(HTML("<h1>Double Setup Strategies - Performance Charts</h1>"))

    strategies = create_double_setup_strategy_library()

    # Filter to only profitable strategies
    profitable_strategies = [s for s in strategies if _is_strategy_profitable(df, s)]

    display(HTML(f"<p>Showing {len(profitable_strategies)} profitable strategies out of {len(strategies)} total strategies.</p>"))

    for strategy in profitable_strategies:
        display_strategy_cumulative_chart(df, strategy)


def display_triple_setup_strategy_charts(df: pd.DataFrame):
    """Display cumulative performance charts for profitable triple setup strategies only."""
    from IPython.display import display, HTML

    display(HTML("<h1>Triple Setup Strategies - Performance Charts</h1>"))

    strategies = create_triple_setup_strategy_library()

    # Filter to only profitable strategies
    profitable_strategies = [s for s in strategies if _is_strategy_profitable(df, s)]

    display(HTML(f"<p>Showing {len(profitable_strategies)} profitable strategies out of {len(strategies)} total strategies.</p>"))

    for strategy in profitable_strategies:
        display_strategy_cumulative_chart(df, strategy)


def style_table(
    table_df: pd.DataFrame,
    first_column_width: str = DEFAULT_COLUMN_WIDTH,
    highlight_column: Optional[str] = None,
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR,
):
    """
    Apply consistent styling to a DataFrame for display.

    Args:
        table_df: DataFrame to style
        first_column_width: Width for the first column
        highlight_column: Optional column to highlight
        highlight_color: Color for highlighted column

    Returns:
        Styled DataFrame ready for display
    """
    # Adjust index to start from 1
    table_df_copy = table_df.copy()
    table_df_copy.index = range(1, len(table_df_copy) + 1)

    first_column = table_df_copy.columns[0]
    styled_df = table_df_copy.style.set_properties(
        subset=[first_column], **{"width": first_column_width, "font-weight": "bold"}
    )

    if highlight_column and highlight_column in table_df_copy.columns:
        styled_df = styled_df.set_properties(
            subset=[highlight_column], **{"color": highlight_color}
        )

    return styled_df


def create_sortable_table(
    table_df: pd.DataFrame,
    first_column_width: str = DEFAULT_COLUMN_WIDTH,
    highlight_column: Optional[str] = None,
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR,
    table_id: Optional[str] = None,
) -> str:
    """
    Create an interactive HTML table with clickable column headers for sorting.

    Args:
        table_df: DataFrame to display
        first_column_width: Width for the first column
        highlight_column: Optional column to highlight
        highlight_color: Color for highlighted column
        table_id: Optional unique ID for the table (auto-generated if not provided)

    Returns:
        HTML string with interactive sortable table
    """
    import uuid
    from IPython.display import HTML

    # Generate unique table ID if not provided
    if table_id is None:
        table_id = f"sortable_table_{uuid.uuid4().hex[:8]}"

    # Adjust index to start from 1
    table_df_copy = table_df.copy()
    table_df_copy.index = range(1, len(table_df_copy) + 1)

    # Convert DataFrame to HTML with custom styling
    html_table = table_df_copy.to_html(classes='sortable-table', table_id=table_id, escape=False)

    # Prepare column indices for sortable columns
    sortable_columns = []
    if 'Edge' in table_df_copy.columns:
        sortable_columns.append(table_df_copy.columns.get_loc('Edge') + 1)  # +1 for index column
    if 'Outcome' in table_df_copy.columns:
        sortable_columns.append(table_df_copy.columns.get_loc('Outcome') + 1)  # +1 for index column
    if 'Trades Required' in table_df_copy.columns:
        sortable_columns.append(table_df_copy.columns.get_loc('Trades Required') + 1)  # +1 for index column
    if 'Drawdown' in table_df_copy.columns:
        sortable_columns.append(table_df_copy.columns.get_loc('Drawdown') + 1)  # +1 for index column

    # Build the complete HTML with JavaScript
    html = f"""
    <style>
        #{table_id} th.sortable {{
            cursor: pointer;
            user-select: none;
        }}

        #{table_id} th.sortable::after {{
            content: ' ⇅';
            color: #999;
            font-size: 12px;
        }}

        #{table_id} th.sorted-asc::after {{
            content: ' ↑';
            color: #999;
        }}

        #{table_id} th.sorted-desc::after {{
            content: ' ↓';
            color: #999;
        }}

        #{table_id} td:first-child {{
            font-weight: bold;
            width: {first_column_width};
        }}

        #{table_id} .highlight-column {{
            color: {highlight_color} !important;
            font-weight: bold;
        }}
    </style>

    {html_table}

    <script>
    (function() {{
        const table = document.getElementById('{table_id}');
        if (!table) return;

        const tbody = table.querySelector('tbody');
        const headers = table.querySelectorAll('thead th');
        const sortableColumns = {sortable_columns};

        // Store the currently highlighted column index
        let currentHighlightIndex = -1;
        const initialHighlightColumn = '{highlight_column if highlight_column else ""}';

        // Find the initial highlight column index
        if (initialHighlightColumn) {{
            const headerRow = table.querySelector('thead tr');
            currentHighlightIndex = Array.from(headerRow.children).findIndex(th =>
                th.textContent.trim() === initialHighlightColumn
            );
        }}

        // Function to apply highlighting to a specific column
        function applyHighlighting(columnIndex) {{
            // Clear all existing highlights
            headers.forEach((header, idx) => {{
                header.style.color = '';
                const rows = tbody.querySelectorAll('tr');
                rows.forEach(row => {{
                    const cells = row.querySelectorAll('td');
                    if (cells[idx]) {{
                        cells[idx].style.color = '';
                        cells[idx].style.fontWeight = '';
                    }}
                }});
            }});

            // Apply new highlight if valid column
            if (columnIndex !== -1 && columnIndex < headers.length) {{
                // Highlight header
                headers[columnIndex].style.color = '{highlight_color}';

                currentHighlightIndex = columnIndex;
            }}
        }}

        // Mark sortable columns
        sortableColumns.forEach(colIndex => {{
            if (headers[colIndex]) {{
                headers[colIndex].classList.add('sortable');
            }}
        }});

        // Apply initial highlighting
        applyHighlighting(currentHighlightIndex);

        // Set initial sort state for Edge column (data is pre-sorted descending by Edge)
        if (currentHighlightIndex !== -1 && sortableColumns.includes(currentHighlightIndex)) {{
            headers[currentHighlightIndex].classList.add('sorted-desc');
        }}

        // Add click handlers to sortable columns
        sortableColumns.forEach(colIndex => {{
            if (!headers[colIndex]) return;

            headers[colIndex].addEventListener('click', function() {{
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const columnName = this.textContent.trim();

                // Remove sorted classes from all headers
                headers.forEach(h => {{
                    h.classList.remove('sorted-asc', 'sorted-desc');
                }});

                // Determine sort direction based on column type
                let sortDescending = true; // Default to descending
                if (columnName === 'Trades Required' || columnName === 'Drawdown') {{
                    sortDescending = false; // Trades Required and Drawdown should be ascending (lowest to highest)
                }}
                // Edge and Outcome columns use descending (highest to lowest) - default behavior

                // Sort rows
                rows.sort((a, b) => {{
                    const aCell = a.children[colIndex];
                    const bCell = b.children[colIndex];

                    let aValue = aCell.textContent.trim();
                    let bValue = bCell.textContent.trim();

                    // Handle percentage values
                    if (aValue.includes('%')) {{
                        aValue = parseFloat(aValue.replace('%', '')) || 0;
                        bValue = parseFloat(bValue.replace('%', '')) || 0;
                    }}
                    // Handle currency values
                    else if (aValue.includes('$')) {{
                        aValue = parseFloat(aValue.replace(/[\\$,]/g, '')) || 0;
                        bValue = parseFloat(bValue.replace(/[\\$,]/g, '')) || 0;
                    }}
                    // Handle regular numbers
                    else {{
                        const aNum = parseFloat(aValue);
                        const bNum = parseFloat(bValue);
                        if (!isNaN(aNum) && !isNaN(bNum)) {{
                            aValue = aNum;
                            bValue = bNum;
                        }}
                    }}

                    if (aValue < bValue) return sortDescending ? 1 : -1;
                    if (aValue > bValue) return sortDescending ? -1 : 1;
                    return 0;
                }});

                // Update the class for visual indicator
                this.classList.add(sortDescending ? 'sorted-desc' : 'sorted-asc');

                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));

                // Highlight the clicked column
                applyHighlighting(colIndex);
            }});
        }});
    }})();
    </script>
    """

    return html


def export_strategy_trades_to_csv(
    df: pd.DataFrame,
    strategy_name: str,
    rrr_ratio: int,
    output_file: str = "strategy_trades.csv"
) -> str:
    """
    Export trades for a specific strategy and RRR ratio to CSV.

    Args:
        df: Trading data with all columns
        strategy_name: Name of the strategy to export (e.g., "EMA + BOS")
        rrr_ratio: RRR ratio to use for profitability calculation (1, 2, or 3)
        output_file: Path to the output CSV file

    Returns:
        Path to the exported CSV file
    """
    # Find the strategy
    all_strategies = []
    all_strategies.extend(create_single_setup_strategy_library())
    all_strategies.extend(create_double_setup_strategy_library())
    all_strategies.extend(create_triple_setup_strategy_library())

    # Find matching strategy
    strategy = None
    for s in all_strategies:
        if s.name == strategy_name:
            strategy = s
            break

    if strategy is None:
        raise ValueError(f"Strategy '{strategy_name}' not found. Please check the strategy name.")

    # Apply strategy filter
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        raise ValueError(f"No trades found for strategy '{strategy_name}'")

    # Prepare trade details
    trade_details = []

    for idx, trade in filtered_df.iterrows():
        # Determine if profitable based on selected RRR
        # Win condition: all three must be true
        is_win = (
            (trade['SL'] != trade['Pullback']) and
            (trade['Pullback'] < trade['SL']) and
            (trade['TP'] >= rrr_ratio * trade['SL'])
        )

        if is_win:
            profitable = 'Yes'
        else:
            profitable = 'No'

        trade_details.append({
            'Profitable': profitable,
            'Date': trade.get('Date', ''),
            'Trade': trade.get('Trade', ''),
            'Weekday': trade.get('Weekday', ''),
            'Hour': int(trade.get('Hour', 0)) if trade.get('Hour', 0) != 0 else '',
            'Direction': trade.get('Direction', ''),
            'EMA': trade.get('EMA', ''),
            'SL': trade.get('SL', ''),
            'Pullback': trade.get('Pullback', ''),
            'TP': trade.get('TP', ''),
            'Extra': trade.get('Extra', ''),
            'BOS/CH': trade.get('BOS/CH', ''),
            '30M Leg': trade.get('30M Leg', ''),
            'Hours Until News': trade.get('Hours Until News', ''),
            'News Event': trade.get('News Event', '')
        })

    # Create DataFrame and export to CSV
    trades_df = pd.DataFrame(trade_details)
    trades_df.to_csv(output_file, index=False)

    return output_file


def display_strategy_trade_details(df: pd.DataFrame):
    """
    Display an interactive dropdown to select a profitable strategy and view its trade details.

    Args:
        df: Trading data with all columns
    """
    from IPython.display import display, HTML, clear_output
    import ipywidgets as widgets

    # Collect all profitable strategies from all three categories
    all_strategies = []

    # Single setup strategies
    single_strategies = create_single_setup_strategy_library()
    for strategy in single_strategies:
        if _is_strategy_profitable(df, strategy):
            all_strategies.append(('Single', strategy))

    # Double setup strategies
    double_strategies = create_double_setup_strategy_library()
    for strategy in double_strategies:
        if _is_strategy_profitable(df, strategy):
            all_strategies.append(('Double', strategy))

    # Triple setup strategies
    triple_strategies = create_triple_setup_strategy_library()
    for strategy in triple_strategies:
        if _is_strategy_profitable(df, strategy):
            all_strategies.append(('Triple', strategy))

    # Create dropdown options
    strategy_options = [('Select a strategy...', None)] + [
        (f"[{setup_type}] {strategy.name}", (setup_type, strategy))
        for setup_type, strategy in all_strategies
    ]

    # Create dropdown widget
    strategy_dropdown = widgets.Dropdown(
        options=strategy_options,
        description='Strategy:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px')
    )

    # Create RRR dropdown
    rrr_dropdown = widgets.Dropdown(
        options=[
            ('1:1 RRR', 1),
            ('1:2 RRR', 2),
            ('1:3 RRR', 3)
        ],
        value=1,
        description='RRR:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )

    # Output widget for displaying trade details
    output = widgets.Output()

    def on_selection_change(change):
        """Handle dropdown selection changes"""
        with output:
            clear_output(wait=True)

            if strategy_dropdown.value is None:
                display(HTML("<p><i>Please select a strategy to view trade details.</i></p>"))
                return

            setup_type, strategy = strategy_dropdown.value
            rrr_ratio = rrr_dropdown.value

            # Apply strategy filter
            filtered_df = strategy.apply(df)

            if len(filtered_df) == 0:
                display(HTML(f"<p>No trades found for strategy: <b>{strategy.name}</b></p>"))
                return

            # Prepare trade details table
            trade_details = []

            for idx, trade in filtered_df.iterrows():
                # Determine if profitable based on selected RRR
                # Win condition: all three must be true
                is_win = (
                    (trade['SL'] != trade['Pullback']) and
                    (trade['Pullback'] < trade['SL']) and
                    (trade['TP'] >= rrr_ratio * trade['SL'])
                )

                if is_win:
                    profitable = '🟢'
                else:
                    profitable = '🔴'

                trade_details.append({
                    'Profitable': profitable,
                    'Date': trade.get('Date', ''),
                    'Trade': trade.get('Trade', ''),
                    'Weekday': trade.get('Weekday', ''),
                    'Hour': int(trade.get('Hour', 0)) if trade.get('Hour', 0) != 0 else '',
                    'Direction': trade.get('Direction', ''),
                    'EMA': trade.get('EMA', ''),
                    'SL': trade.get('SL', ''),
                    'Pullback': trade.get('Pullback', ''),
                    'TP': trade.get('TP', ''),
                    'Extra': trade.get('Extra', ''),
                    'BOS/CH': trade.get('BOS/CH', ''),
                    '30M Leg': trade.get('30M Leg', ''),
                    'Hours Until News': trade.get('Hours Until News', ''),
                    'News Event': trade.get('News Event', '')
                })

            # Create DataFrame from trade details
            trades_df = pd.DataFrame(trade_details)

            # Create sortable table
            html_table = create_sortable_table(trades_df, first_column_width='100px')

            # Combine all HTML into a single display call
            combined_html = f"""
                <h3>Strategy: {strategy.name} ({setup_type} Setup) - {rrr_ratio}:1 RRR</h3>
                <p>Total trades: <b>{len(trades_df)}</b></p>
                {html_table}
            """
            display(HTML(combined_html))

    # Display header
    display(HTML("<h2>Strategy Trade Details</h2>"))
    display(HTML("<p>Select a profitable strategy to view all its trades with profitability indicators.</p>"))

    # Display dropdowns in a horizontal box
    dropdown_box = widgets.HBox([strategy_dropdown, rrr_dropdown])
    display(dropdown_box)

    # Display output area
    display(output)

    # Attach event handlers
    strategy_dropdown.observe(on_selection_change, names='value')
    rrr_dropdown.observe(on_selection_change, names='value')
