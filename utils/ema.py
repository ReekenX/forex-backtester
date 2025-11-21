"""
EMA Strategy Analysis Module

This module provides functions to analyze trading performance for various EMA-based strategies.
"""

import pandas as pd
from typing import Dict, List, Tuple, Callable


# Risk-Reward Ratio configurations with breakeven win rates
RRR_RATIOS: List[Tuple[int, float]] = [
    (1, 50.0),  # 1:1 RRR - need 50% win rate to break even
    (2, 33.3),  # 1:2 RRR - need 33.3% win rate to break even
    (3, 25.0),  # 1:3 RRR - need 25% win rate to break even
]


def create_ema_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Create a comprehensive list of EMA-based trading strategies.
    Each strategy is tested with: original (no SL filter), 5 < SL < 10, and SL < 10.

    Returns:
        List of tuples containing (strategy_name, filter_function)
    """
    strategies = []

    # 1. EMA Aligned
    strategies.extend([
        ("EMA Aligned", lambda df: df[df["EMA"] == df["Direction"]]),
        ("EMA Aligned + 5 < SL < 10", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] > 5) & (df["SL"] < 10)]),
        ("EMA Aligned + SL < 10", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] < 10)]),
    ])

    # 2. EMA Counter-Trend
    strategies.extend([
        ("EMA Counter-Trend", lambda df: df[df["EMA"] != df["Direction"]]),
        ("EMA Counter-Trend + 5 < SL < 10", lambda df: df[(df["EMA"] != df["Direction"]) & (df["SL"] > 5) & (df["SL"] < 10)]),
        ("EMA Counter-Trend + SL < 10", lambda df: df[(df["EMA"] != df["Direction"]) & (df["SL"] < 10)]),
    ])

    # 3. EMA + BOS
    strategies.extend([
        ("EMA + BOS", lambda df: df[(df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")]),
        ("EMA + BOS + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + BOS + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & (df["SL"] < 10)
        ]),
    ])

    # 4. EMA + CH
    strategies.extend([
        ("EMA + CH", lambda df: df[(df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")]),
        ("EMA + CH + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + CH + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & (df["SL"] < 10)
        ]),
    ])

    # 5. EMA + 30M Trend
    strategies.extend([
        ("EMA + 30M Trend", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"])
        ]),
        ("EMA + 30M Trend + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + 30M Trend + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] < 10)
        ]),
    ])

    # 6. EMA + 30M Trend Continuation
    strategies.extend([
        ("EMA + 30M Trend Continuation", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"])
        ]),
        ("EMA + 30M Trend Continuation + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + 30M Trend Continuation + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] < 10)
        ]),
    ])

    # 7. EMA + 30M Trend Reversal
    strategies.extend([
        ("EMA + 30M Trend Reversal", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"])
        ]),
        ("EMA + 30M Trend Reversal + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + 30M Trend Reversal + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["SL"] < 10)
        ]),
    ])

    # 8. EMA + BOS + 30M Trend
    strategies.extend([
        ("EMA + BOS + 30M Trend", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")
        ]),
        ("EMA + BOS + 30M Trend + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + BOS + 30M Trend + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & (df["SL"] < 10)
        ]),
    ])

    # 9. EMA + BOS + 30M Trend Continuation
    strategies.extend([
        ("EMA + BOS + 30M Trend Continuation", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")
        ]),
        ("EMA + BOS + 30M Trend Continuation + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + BOS + 30M Trend Continuation + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & (df["SL"] < 10)
        ]),
    ])

    # 10. EMA + BOS + 30M Trend Reversal
    strategies.extend([
        ("EMA + BOS + 30M Trend Reversal", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS")
        ]),
        ("EMA + BOS + 30M Trend Reversal + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + BOS + 30M Trend Reversal + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & (df["SL"] < 10)
        ]),
    ])

    # 11. EMA + CH + 30M Trend
    strategies.extend([
        ("EMA + CH + 30M Trend", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")
        ]),
        ("EMA + CH + 30M Trend + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + CH + 30M Trend + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & (df["SL"] < 10)
        ]),
    ])

    # 12. EMA + CH + 30M Trend Continuation
    strategies.extend([
        ("EMA + CH + 30M Trend Continuation", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")
        ]),
        ("EMA + CH + 30M Trend Continuation + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + CH + 30M Trend Continuation + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below L"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & (df["SL"] < 10)
        ]),
    ])

    # 13. EMA + CH + 30M Trend Reversal
    strategies.extend([
        ("EMA + CH + 30M Trend Reversal", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH")
        ]),
        ("EMA + CH + 30M Trend Reversal + 5 < SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + CH + 30M Trend Reversal + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H"]) & (df["Direction"] == "Sell"))) &
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & (df["SL"] < 10)
        ]),
    ])

    # 14. EMA + No News
    strategies.extend([
        ("EMA + No News", lambda df: df[(df["EMA"] == df["Direction"]) & df["News Event"].isna()]),
        ("EMA + No News + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & df["News Event"].isna() &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + No News + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & df["News Event"].isna() & (df["SL"] < 10)
        ]),
    ])

    # 15. EMA + News > 2hrs
    strategies.extend([
        ("EMA + News > 2hrs", lambda df: df[
            (df["EMA"] == df["Direction"]) & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)
        ]),
        ("EMA + News > 2hrs + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2) &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + News > 2hrs + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (~df["News Event"].isna()) & (df["Hours Until News"] >= 2) &
            (df["SL"] < 10)
        ]),
    ])

    # 16. EMA + BOS + No News
    strategies.extend([
        ("EMA + BOS + No News", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & df["News Event"].isna()
        ]),
        ("EMA + BOS + No News + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & df["News Event"].isna() &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + BOS + No News + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") & df["News Event"].isna() &
            (df["SL"] < 10)
        ]),
    ])

    # 17. EMA + BOS + News > 2hrs
    strategies.extend([
        ("EMA + BOS + News > 2hrs", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)
        ]),
        ("EMA + BOS + News > 2hrs + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2) &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + BOS + News > 2hrs + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "BOS") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2) &
            (df["SL"] < 10)
        ]),
    ])

    # 18. EMA + CH + No News
    strategies.extend([
        ("EMA + CH + No News", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & df["News Event"].isna()
        ]),
        ("EMA + CH + No News + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & df["News Event"].isna() &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + CH + No News + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") & df["News Event"].isna() &
            (df["SL"] < 10)
        ]),
    ])

    # 19. EMA + CH + News > 2hrs
    strategies.extend([
        ("EMA + CH + News > 2hrs", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2)
        ]),
        ("EMA + CH + News > 2hrs + 5 < SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2) &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("EMA + CH + News > 2hrs + SL < 10", lambda df: df[
            (df["EMA"] == df["Direction"]) & (df["BOS/CH"] == "CH") &
            (~df["News Event"].isna()) & (df["Hours Until News"] >= 2) &
            (df["SL"] < 10)
        ]),
    ])

    return strategies


def calculate_ema_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trading statistics for all EMA-based strategies.

    Args:
        df: DataFrame with columns: EMA, Direction, BOS/CH, 30M Leg, SL, Pullback, TP, Date, News Event, Hours Until News

    Returns:
        DataFrame with strategy statistics including win rates and outcomes for each RRR ratio
    """
    strategies = create_ema_strategies()
    results = []

    for strategy_name, filter_func in strategies:
        # Apply strategy filter
        try:
            filtered_df = filter_func(df)
        except Exception:
            # If filter fails, skip this strategy
            continue

        # Calculate statistics for each RRR ratio
        for rrr_ratio, breakeven_rate in RRR_RATIOS:
            stats = _calculate_stats_for_strategy_and_rrr(
                filtered_df, strategy_name, rrr_ratio, breakeven_rate
            )
            results.append(stats)

    return pd.DataFrame(results)


def _calculate_stats_for_strategy_and_rrr(
    trades: pd.DataFrame,
    strategy_name: str,
    rrr_ratio: int,
    breakeven_rate: float
) -> Dict:
    """
    Calculate trading statistics for a specific strategy and RRR ratio.

    Args:
        trades: DataFrame containing trades for a specific strategy
        strategy_name: Name of the trading strategy
        rrr_ratio: Risk-reward ratio multiplier (1, 2, or 3)
        breakeven_rate: Breakeven win rate percentage for this RRR

    Returns:
        Dictionary with calculated statistics
    """
    total_trades = len(trades)

    if total_trades == 0:
        return _create_empty_stats(strategy_name, rrr_ratio, breakeven_rate)

    # Count winning trades
    # A trade wins if:
    # 1. Pullback is less than SL (didn't hit stop loss)
    # 2. Pullback is not equal to SL (not instant loss)
    # 3. TP is at least (rrr_ratio * SL) (target profit reached)
    winning_trades = trades[
        (trades['Pullback'] < trades['SL']) &
        (trades['Pullback'] != trades['SL']) &
        (trades['TP'] >= (rrr_ratio * trades['SL']))
    ]

    wins = len(winning_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven_rate
    outcome = (wins * rrr_ratio) - losses

    # Calculate unique days with at least one win
    days_with_wins = winning_trades['Date'].nunique() if 'Date' in winning_trades.columns and len(winning_trades) > 0 else 0

    # Calculate total unique trading days for this strategy (days with any trades)
    total_strategy_days = trades['Date'].nunique() if 'Date' in trades.columns else 0

    # Calculate days percentage
    days_percentage = (days_with_wins / total_strategy_days * 100) if total_strategy_days > 0 else 0.0

    # Calculate trades required to earn 1R
    trades_required = f"{total_trades / outcome:.1f}" if outcome > 0 else "N/A"

    return {
        'Strategy': strategy_name,
        'RRR': f'1:{rrr_ratio}',
        'Trades': total_trades,
        'Notation': f'{wins}W – {losses}L',
        'Win Rate': f'{win_rate:.1f}%',
        'Outcome': f'{outcome}R',
        'Edge': f'{edge:.1f}%',
        'Days': days_with_wins,
        'Days %': f'{int(round(days_percentage))}%',
        'Trades Required': trades_required
    }


def _create_empty_stats(strategy_name: str, rrr_ratio: int, breakeven_rate: float) -> Dict:
    """
    Create an empty statistics dictionary when no trades are found.

    Args:
        strategy_name: Name of the trading strategy
        rrr_ratio: Risk-reward ratio multiplier (1, 2, or 3)
        breakeven_rate: Breakeven win rate percentage for this RRR

    Returns:
        Dictionary with zero values
    """
    return {
        'Strategy': strategy_name,
        'RRR': f'1:{rrr_ratio}',
        'Trades': 0,
        'Notation': '0W – 0L',
        'Win Rate': '0.0%',
        'Outcome': '0R',
        'Edge': f'{-breakeven_rate:.1f}%',
        'Days': 0,
        'Days %': '0%',
        'Trades Required': 'N/A'
    }


def create_html_table(stats_df: pd.DataFrame) -> str:
    """
    Create an HTML table from statistics DataFrame with dark mode styling.

    Args:
        stats_df: DataFrame with strategy statistics

    Returns:
        HTML string representation of the table
    """
    if stats_df.empty:
        return "<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data available</p>"

    # Start building HTML
    html = """
    <style>
        .ema-analysis-table {
            border-collapse: collapse;
            width: 100%;
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .ema-analysis-table th {
            background-color: #2d2d2d;
            color: #e0e0e0;
            padding: 8px;
            text-align: left;
            border: 1px solid #404040;
            font-weight: bold;
        }
        .ema-analysis-table td {
            padding: 6px 8px;
            border: 1px solid #404040;
        }
        .ema-analysis-table tr:hover {
            background-color: #2a2a2a;
        }
        .ema-strategy-col {
            width: 300px;
        }
        .positive-edge {
            color: #4ade80;
        }
        .negative-edge {
            color: #f87171;
        }
    </style>
    <table class="ema-analysis-table">
        <thead>
            <tr>
                <th class="ema-strategy-col">Strategy</th>
                <th>RRR</th>
                <th>Trades</th>
                <th>Notation</th>
                <th>Win Rate</th>
                <th>Outcome</th>
                <th>Edge</th>
                <th>Days</th>
                <th>Days %</th>
                <th>Trades Required</th>
            </tr>
        </thead>
        <tbody>
    """

    # Add rows
    for _, row in stats_df.iterrows():
        # Determine edge color
        edge_value = float(row['Edge'].rstrip('%'))
        edge_class = 'positive-edge' if edge_value > 0 else 'negative-edge'

        html += f"""
            <tr>
                <td class="ema-strategy-col">{row['Strategy']}</td>
                <td>{row['RRR']}</td>
                <td>{row['Trades']}</td>
                <td>{row['Notation']}</td>
                <td>{row['Win Rate']}</td>
                <td>{row['Outcome']}</td>
                <td class="{edge_class}">{row['Edge']}</td>
                <td>{row['Days']}</td>
                <td>{row['Days %']}</td>
                <td>{row['Trades Required']}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html


def display_ema_analysis(df: pd.DataFrame):
    """
    Display EMA strategy analysis in a formatted HTML table.

    Args:
        df: Trading data DataFrame
    """
    from IPython.display import display, HTML

    # Calculate statistics
    stats_df = calculate_ema_statistics(df)

    # Sort by edge (descending)
    if not stats_df.empty:
        stats_df['edge_value'] = stats_df['Edge'].apply(
            lambda x: float(x.rstrip('%')) if isinstance(x, str) else 0.0
        )
        stats_df = stats_df.sort_values('edge_value', ascending=False)
        stats_df = stats_df.drop('edge_value', axis=1)

    # Create and display HTML table
    html_table = create_html_table(stats_df)

    # Display title
    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>EMA Strategy Analysis</h2>"
    display(HTML(title_html))
    display(HTML(html_table))
