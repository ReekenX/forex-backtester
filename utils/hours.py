"""
Hour Analysis Module

This module provides simple functions to analyze trading performance by hour of day.
"""

import pandas as pd
from typing import Dict, List, Tuple


# Risk-Reward Ratio configurations with breakeven win rates
RRR_RATIOS: List[Tuple[int, float]] = [
    (1, 50.0),  # 1:1 RRR - need 50% win rate to break even
    (2, 33.3),  # 1:2 RRR - need 33.3% win rate to break even
    (3, 25.0),  # 1:3 RRR - need 25% win rate to break even
]


def calculate_hour_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trading statistics for each hour of the day.

    Args:
        df: DataFrame with columns: Hour, SL, Pullback, TP

    Returns:
        DataFrame with hour statistics including win rates and outcomes for each RRR ratio
    """
    # Filter out invalid hours
    df_clean = df.dropna(subset=['Hour'])
    df_clean = df_clean[df_clean['Hour'] != 0].copy()

    if len(df_clean) == 0:
        return pd.DataFrame()

    # Get sorted list of unique hours
    hours = sorted(df_clean['Hour'].unique())

    results = []

    for hour in hours:
        # Filter trades for this specific hour
        hour_trades = df_clean[df_clean['Hour'] == hour]

        # Calculate statistics for each RRR ratio
        for rrr_ratio, breakeven_rate in RRR_RATIOS:
            stats = _calculate_stats_for_hour_and_rrr(hour_trades, hour, rrr_ratio, breakeven_rate)
            results.append(stats)

    return pd.DataFrame(results)


def _calculate_stats_for_hour_and_rrr(
    trades: pd.DataFrame,
    hour: int,
    rrr_ratio: int,
    breakeven_rate: float
) -> Dict:
    """
    Calculate trading statistics for a specific hour and RRR ratio.

    Args:
        trades: DataFrame containing trades for a specific hour
        hour: Hour of the day (0-23)
        rrr_ratio: Risk-reward ratio multiplier (1, 2, or 3)
        breakeven_rate: Breakeven win rate percentage for this RRR

    Returns:
        Dictionary with calculated statistics
    """
    total_trades = len(trades)

    if total_trades == 0:
        return _create_empty_stats(hour, rrr_ratio, breakeven_rate)

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

    # Calculate trades required to earn 1R
    if outcome > 0:
        trades_required = f"{total_trades / outcome:.1f}"
    else:
        trades_required = "N/A"

    # Show hour label only on first RRR row (1:1)
    hour_label = f"{int(hour):02d}h" if rrr_ratio == 1 else ''

    return {
        'Strategy': hour_label,
        'RRR': f'1:{rrr_ratio}',
        'Trades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Win Rate': f"{win_rate:.1f}%",
        'Edge': f"{edge:.1f}%",
        'Outcome': f"{outcome}R",
        'Trades Required': trades_required,
        'Drawdown': "N/A"
    }


def _create_empty_stats(hour: int, rrr_ratio: int, breakeven_rate: float) -> Dict:
    """Create empty statistics structure for hour with no trades."""
    hour_label = f"{int(hour):02d}h" if rrr_ratio == 1 else ''

    return {
        'Strategy': hour_label,
        'RRR': f'1:{rrr_ratio}',
        'Trades': 0,
        'Wins': 0,
        'Losses': 0,
        'Win Rate': "0.0%",
        'Edge': f"{-breakeven_rate:.1f}%",
        'Outcome': "0R",
        'Trades Required': "N/A",
        'Drawdown': "N/A"
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
        return "<p>No data available</p>"

    # Start HTML table with styling (dark mode optimized)
    html = """
    <style>
        .hour-analysis-table {
            border-collapse: collapse;
            width: 100%;
            font-family: monospace;
            font-size: 12px;
            color: #e0e0e0;
            background-color: #1e1e1e;
        }
        .hour-analysis-table th {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            padding: 8px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            color: #e0e0e0;
        }
        .hour-analysis-table th:hover {
            background-color: #3d3d3d;
        }
        .hour-analysis-table td {
            border: 1px solid #404040;
            padding: 8px;
            color: #e0e0e0;
        }
        .hour-analysis-table tr:nth-child(even) {
            background-color: #252525;
        }
        .hour-analysis-table tr:nth-child(odd) {
            background-color: #1e1e1e;
        }
        .hour-analysis-table .strategy-col {
            width: 300px;
            font-weight: bold;
        }
        .hour-analysis-table .positive-edge {
            color: #4ade80;
            font-weight: bold;
        }
        .hour-analysis-table .negative-edge {
            color: #f87171;
        }
    </style>
    <table class="hour-analysis-table">
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
        for col_idx, col in enumerate(df.columns):
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

            html += f'<td class="{css_class}">{value}</td>'
        html += "</tr>"

    html += "</tbody></table>"

    return html


def display_hour_analysis(df: pd.DataFrame):
    """
    Display hour analysis with HTML formatting in Jupyter notebook.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    display(HTML("<h2>Hour Analysis</h2>"))

    stats_df = calculate_hour_statistics(df)

    if stats_df.empty:
        display(HTML("<p>No hour data available for analysis</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))
