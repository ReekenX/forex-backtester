"""
Double Setup Strategy Analysis Module

This module provides functions to analyze trading strategies that combine
exactly two factors (EMA + filters, 30M Trend + filters, BOS + filters).
"""

import pandas as pd
from typing import Dict, List, Tuple, Callable


# Risk-Reward Ratio configurations with breakeven win rates
RRR_RATIOS: List[Tuple[int, float]] = [
    (2, 33.3),  # 1:2 RRR - need 33.3% win rate to break even
    (3, 25.0),  # 1:3 RRR - need 25% win rate to break even
]


def get_double_setup_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get list of double setup strategy definitions (exactly 2 factors combined).

    Returns:
        List of tuples (strategy_name, filter_function)
    """
    strategies = []

    # EMA Aligned + various filters
    strategies.extend([
        ("EMA Aligned + SL < 10", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] < 10)]),
        ("EMA Aligned + SL 5-10", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] > 5) & (df["SL"] < 10)]),
        ("EMA Aligned + SL > 3", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] > 3)]),
        ("EMA Aligned + SL > 5", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] > 5)]),
        ("EMA Aligned + SL < 5", lambda df: df[(df["EMA"] == df["Direction"]) & (df["SL"] < 5)]),
        ("EMA Aligned + No News", lambda df: df[(df["EMA"] == df["Direction"]) & df["News Event"].isna()]),
        ("EMA Aligned + With News", lambda df: df[(df["EMA"] == df["Direction"]) & (~df["News Event"].isna())]),
        ("EMA Aligned + 50% Pullback", lambda df: df[(df["EMA"] == df["Direction"]) & (df["Pullback"] >= df["SL"] * 0.5) & (df["Pullback"] >= 2)]),
        ("EMA Aligned + 1 pip Pullback", lambda df: df[(df["EMA"] == df["Direction"]) & (df["Pullback"] >= 1)]),
        ("EMA Aligned + 80% Pullback", lambda df: df[(df["EMA"] == df["Direction"]) & (df["Pullback"] >= df["SL"] * 0.8)]),
    ])

    # 30M Trend + various filters
    strategies.extend([
        ("30M Trend + SL < 10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] < 10)
        ]),
        ("30M Trend + SL 5-10", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 5) & (df["SL"] < 10)
        ]),
        ("30M Trend + SL > 3", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 3)
        ]),
        ("30M Trend + SL > 5", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] > 5)
        ]),
        ("30M Trend + SL < 5", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["SL"] < 5)
        ]),
        ("30M Trend + No News", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            df["News Event"].isna()
        ]),
        ("30M Trend + With News", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (~df["News Event"].isna())
        ]),
        ("30M Trend + 50% Pullback", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["Pullback"] >= df["SL"] * 0.5) & (df["Pullback"] >= 2)
        ]),
        ("30M Trend + 1 pip Pullback", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["Pullback"] >= 1)
        ]),
        ("30M Trend + 80% Pullback", lambda df: df[
            ((df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
             (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))) &
            (df["Pullback"] >= df["SL"] * 0.8)
        ]),
    ])

    # BOS + various filters
    strategies.extend([
        ("BOS + SL < 10", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] < 10)]),
        ("BOS + SL 5-10", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] > 5) & (df["SL"] < 10)]),
        ("BOS + SL > 3", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] > 3)]),
        ("BOS + SL > 5", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] > 5)]),
        ("BOS + SL < 5", lambda df: df[(df["BOS/CH"] == "BOS") & (df["SL"] < 5)]),
        ("BOS + No News", lambda df: df[(df["BOS/CH"] == "BOS") & df["News Event"].isna()]),
        ("BOS + With News", lambda df: df[(df["BOS/CH"] == "BOS") & (~df["News Event"].isna())]),
        ("BOS + 50% Pullback", lambda df: df[(df["BOS/CH"] == "BOS") & (df["Pullback"] >= df["SL"] * 0.5) & (df["Pullback"] >= 2)]),
        ("BOS + 1 pip Pullback", lambda df: df[(df["BOS/CH"] == "BOS") & (df["Pullback"] >= 1)]),
        ("BOS + 80% Pullback", lambda df: df[(df["BOS/CH"] == "BOS") & (df["Pullback"] >= df["SL"] * 0.8)]),
    ])

    return strategies


def calculate_strategy_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for all double setup strategies.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with strategy statistics for all RRR ratios
    """
    strategies = get_double_setup_strategies()
    results = []

    # Calculate stats for each strategy and RRR ratio
    for strategy_name, filter_func in strategies:
        filtered_df = filter_func(df)

        for rrr_ratio, breakeven_rate in RRR_RATIOS:
            stats = _calculate_stats_for_strategy_and_rrr(
                filtered_df, strategy_name, rrr_ratio, breakeven_rate
            )
            results.append(stats)

    result_df = pd.DataFrame(results)

    # Filter to only show strategies with positive edge
    result_df = result_df[result_df['edge_value'] > 0].copy()

    # Sort by edge value descending
    result_df = result_df.sort_values('edge_value', ascending=False)

    # Drop the sorting column
    result_df = result_df.drop('edge_value', axis=1)

    # Reset index
    result_df = result_df.reset_index(drop=True)

    return result_df


def _calculate_stats_for_strategy_and_rrr(
    trades: pd.DataFrame,
    strategy_name: str,
    rrr_ratio: int,
    breakeven_rate: float
) -> Dict:
    """
    Calculate trading statistics for a specific strategy and RRR ratio.

    Args:
        trades: DataFrame containing filtered trades
        strategy_name: Name of the strategy
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

    # Calculate total unique trading days
    total_strategy_days = trades['Date'].nunique() if 'Date' in trades.columns else 0

    # Calculate days percentage
    days_percentage = (days_with_wins / total_strategy_days * 100) if total_strategy_days > 0 else 0.0

    # Calculate profit factor (total profits / total losses)
    profit_factor = (wins * rrr_ratio) / losses if losses > 0 else float('inf')

    # Create notation column (Wins – Losses format)
    notation = f"{wins}W – {losses}L"

    return {
        'Strategy': strategy_name,
        'RRR': f'1:{rrr_ratio}',
        'Trades': total_trades,
        'Notation': notation,
        'Win Rate': f"{win_rate:.1f}%",
        'Edge': f"{edge:.1f}%",
        'Outcome': f"{outcome}R",
        'Days': days_with_wins,
        'Days %': f"{days_percentage:.0f}%",
        'Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
        'edge_value': edge  # For sorting (will be dropped later)
    }


def _create_empty_stats(strategy_name: str, rrr_ratio: int, breakeven_rate: float) -> Dict:
    """Create empty statistics structure for strategy with no trades."""
    return {
        'Strategy': strategy_name,
        'RRR': f'1:{rrr_ratio}',
        'Trades': 0,
        'Notation': "0W – 0L",
        'Win Rate': "0.0%",
        'Edge': f"{-breakeven_rate:.1f}%",
        'Outcome': "0R",
        'Days': 0,
        'Days %': "0%",
        'Factor': "∞",
        'edge_value': -breakeven_rate
    }


def create_html_table(df: pd.DataFrame) -> str:
    """
    Create an HTML table with sortable columns and styled formatting.

    Args:
        df: DataFrame to convert to HTML table

    Returns:
        HTML string with styled, sortable table
    """
    if df.empty:
        return "<p>No profitable double setup strategies found</p>"

    # Start HTML table with styling (dark mode optimized)
    html = """
    <style>
        .doubles-analysis-table {
            border-collapse: collapse;
            width: 100%;
            font-family: monospace;
            font-size: 12px;
            color: #e0e0e0;
            background-color: #1e1e1e;
        }
        .doubles-analysis-table th {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            padding: 8px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            color: #e0e0e0;
        }
        .doubles-analysis-table th:hover {
            background-color: #3d3d3d;
        }
        .doubles-analysis-table td {
            border: 1px solid #404040;
            padding: 8px;
            color: #e0e0e0;
        }
        .doubles-analysis-table tr:nth-child(even) {
            background-color: #252525;
        }
        .doubles-analysis-table tr:nth-child(odd) {
            background-color: #1e1e1e;
        }
        .doubles-analysis-table .strategy-col {
            width: 300px;
            font-weight: bold;
        }
        .doubles-analysis-table .positive-edge {
            color: #4ade80;
            font-weight: bold;
        }
        .doubles-analysis-table .negative-edge {
            color: #f87171;
        }
    </style>
    <table class="doubles-analysis-table">
        <thead>
            <tr>
    """

    # Add headers
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    # Add data rows
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            value = row[col]

            # Apply special styling
            css_class = ""
            if col == 'Strategy':
                css_class = "strategy-col"
            elif col == 'Edge':
                # Highlight positive/negative edge
                try:
                    edge_val = float(str(value).replace('%', ''))
                    css_class = "positive-edge" if edge_val > 0 else "negative-edge"
                except:
                    pass
            elif col == 'Days %':
                # Highlight days % above 50%
                try:
                    days_pct_val = float(str(value).replace('%', ''))
                    css_class = "positive-edge" if days_pct_val > 50 else ""
                except:
                    pass

            html += f'<td class="{css_class}">{value}</td>'
        html += "</tr>"

    html += "</tbody></table>"

    return html


def display_double_setup_analysis(df: pd.DataFrame):
    """
    Display double setup strategy analysis with HTML formatting in Jupyter notebook.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    display(HTML("<h2>Double Setup Strategies</h2>"))

    stats_df = calculate_strategy_statistics(df)

    if stats_df.empty:
        display(HTML("<p>No profitable double setup strategies found</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))
