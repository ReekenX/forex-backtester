"""
Weekday Analysis Module

This module provides simple functions to analyze trading performance by day of the week.
"""

import pandas as pd
from typing import Dict, List, Tuple


# Risk-Reward Ratio configurations with breakeven win rates
RRR_RATIOS: List[Tuple[int, float]] = [
    (1, 50.0),  # 1:1 RRR - need 50% win rate to break even
    # (2, 33.3),  # 1:2 RRR - need 33.3% win rate to break even
    # (3, 25.0),  # 1:3 RRR - need 25% win rate to break even
]

# Weekday names for display
WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def calculate_weekday_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trading statistics for each day of the week.

    Args:
        df: DataFrame with columns: Date, SL, Pullback, TP

    Returns:
        DataFrame with weekday statistics including win rates and outcomes for each RRR ratio
    """
    # Ensure Date column is datetime
    df_clean = df.copy()

    if 'Date' not in df_clean.columns:
        return pd.DataFrame()

    # Convert Date to datetime if not already
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')

    # Remove rows with invalid dates
    df_clean = df_clean.dropna(subset=['Date'])

    if len(df_clean) == 0:
        return pd.DataFrame()

    # Extract weekday (0=Monday, 6=Sunday)
    df_clean['Weekday'] = df_clean['Date'].dt.dayofweek

    # Get sorted list of unique weekdays
    weekdays = sorted(df_clean['Weekday'].unique())

    results = []

    for weekday in weekdays:
        # Filter trades for this specific weekday
        weekday_trades = df_clean[df_clean['Weekday'] == weekday]

        # Calculate statistics for each RRR ratio
        for rrr_ratio, breakeven_rate in RRR_RATIOS:
            stats = _calculate_stats_for_weekday_and_rrr(weekday_trades, weekday, rrr_ratio, breakeven_rate)
            results.append(stats)

    return pd.DataFrame(results)


def _calculate_stats_for_weekday_and_rrr(
    trades: pd.DataFrame,
    weekday: int,
    rrr_ratio: int,
    breakeven_rate: float
) -> Dict:
    """
    Calculate trading statistics for a specific weekday and RRR ratio.

    Args:
        trades: DataFrame containing trades for a specific weekday
        weekday: Day of week (0=Monday, 6=Sunday)
        rrr_ratio: Risk-reward ratio multiplier (1, 2, or 3)
        breakeven_rate: Breakeven win rate percentage for this RRR

    Returns:
        Dictionary with calculated statistics
    """
    total_trades = len(trades)

    if total_trades == 0:
        return _create_empty_stats(weekday, rrr_ratio, breakeven_rate)

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
    days_with_wins = winning_trades['Date'].nunique() if len(winning_trades) > 0 else 0

    # Calculate total unique trading days for this weekday
    total_strategy_days = trades['Date'].nunique()

    # Calculate days percentage
    days_percentage = (days_with_wins / total_strategy_days * 100) if total_strategy_days > 0 else 0.0

    # Calculate trades required to earn 1R
    if outcome > 0:
        trades_required = f"{total_trades / outcome:.1f}"
    else:
        trades_required = ""

    # Show weekday label only on first RRR row (1:1)
    weekday_label = WEEKDAY_NAMES[weekday] if rrr_ratio == 1 else ''

    # Create notation column (Wins – Losses format)
    notation = f"{wins}W – {losses}L"

    # Highlight if Days % >= 50%
    days_percentage_str = f"{days_percentage:.0f}%"

    return {
        'Strategy': weekday_label,
        'RRR': f'1:{rrr_ratio}',
        'Trades': total_trades,
        'Notation': notation,
        'Win Rate': f"{win_rate:.1f}%",
        'Outcome': f"{outcome}R",
        'Edge': f"{edge:.1f}%",
        'Days': days_with_wins,
        'Days %': days_percentage_str,
        'Trades Required': trades_required,
        '_days_percentage_value': days_percentage  # Hidden field for styling
    }


def _create_empty_stats(weekday: int, rrr_ratio: int, breakeven_rate: float) -> Dict:
    """Create empty statistics structure for weekday with no trades."""
    weekday_label = WEEKDAY_NAMES[weekday] if rrr_ratio == 1 else ''

    return {
        'Strategy': weekday_label,
        'RRR': f'1:{rrr_ratio}',
        'Trades': 0,
        'Notation': "0W – 0L",
        'Win Rate': "0.0%",
        'Outcome': "0R",
        'Edge': f"{-breakeven_rate:.1f}%",
        'Days': 0,
        'Days %': "0%",
        'Trades Required': "",
        '_days_percentage_value': 0.0
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

    # Remove hidden columns from display
    display_df = df.drop(columns=['_days_percentage_value'], errors='ignore')

    # Start HTML table with styling (dark mode optimized)
    html = """
    <style>
        .weekday-analysis-table {
            border-collapse: collapse;
            width: 100%;
            font-family: monospace;
            font-size: 12px;
            color: #e0e0e0;
            background-color: #1e1e1e;
        }
        .weekday-analysis-table th {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            padding: 8px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            color: #e0e0e0;
        }
        .weekday-analysis-table th:hover {
            background-color: #3d3d3d;
        }
        .weekday-analysis-table td {
            border: 1px solid #404040;
            padding: 8px;
            color: #e0e0e0;
        }
        .weekday-analysis-table tr:nth-child(even) {
            background-color: #252525;
        }
        .weekday-analysis-table tr:nth-child(odd) {
            background-color: #1e1e1e;
        }
        .weekday-analysis-table .strategy-col {
            width: 300px;
            font-weight: bold;
        }
        .weekday-analysis-table .positive-edge {
            color: #4ade80;
            font-weight: bold;
        }
        .weekday-analysis-table .negative-edge {
            color: #f87171;
        }
        .weekday-analysis-table .high-days-percentage {
            color: #4ade80;
            font-weight: bold;
        }
    </style>
    <table class="weekday-analysis-table">
        <thead>
            <tr>
    """

    # Add headers
    for col in display_df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    # Add data rows
    for idx, row in df.iterrows():
        html += "<tr>"
        for col in display_df.columns:
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
                # Highlight if days percentage >= 50%
                try:
                    days_pct_val = row.get('_days_percentage_value', 0.0)
                    if days_pct_val >= 50.0:
                        css_class = "high-days-percentage"
                except:
                    pass

            html += f'<td class="{css_class}">{value}</td>'
        html += "</tr>"

    html += "</tbody></table>"

    return html


def display_weekday_analysis(df: pd.DataFrame):
    """
    Display weekday analysis with HTML formatting in Jupyter notebook.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    display(HTML("<h2>Weekday Analysis</h2>"))

    stats_df = calculate_weekday_statistics(df)

    if stats_df.empty:
        display(HTML("<p>No weekday data available for analysis</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))
