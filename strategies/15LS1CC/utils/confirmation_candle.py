"""
1M Confirmation Candle Analysis Module

Analyzes trading strategies based on 1H (higher timeframe) alignment.
All strategies are evaluated at 1:1 and 1:2 RRR.

CSV columns: Date, Weekday, Trade, Direction, 1H, SL, Pullback, TP, R

1H column represents higher timeframe alignment:
- Buy: trade idea is above High or bounced from Low
- Sell: trade idea is below Low or bounced from High
"""

import pandas as pd
from typing import Dict, List, Tuple, Callable


# RRR ratios to test
RRR_RATIOS = [1, 2]

# Extra pip buffer values to test
BUFFER_PIPS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def load_data(filepath: str = "../strategies/15LS1CC/data.csv") -> pd.DataFrame:
    """
    Load 1M confirmation candle data from CSV and clean NaN values.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame with trading data
    """
    df = pd.read_csv(filepath)

    for col in ["SL", "TP", "Pullback", "R"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def get_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get all strategy definitions.

    Returns:
        List of tuples (strategy_name, filter_function)
    """
    strategies = []

    # === Base strategies ===
    strategies.extend([
        ("All Trades", lambda df: df),
    ])

    # === 1H alignment strategies ===
    strategies.extend([
        ("1H Aligned", lambda df: df[df["Direction"] == df["1H"]]),
        ("1H Against", lambda df: df[df["Direction"] != df["1H"]]),
    ])

    # === SL filter strategies ===
    sl_filters = [
        ("SL < 3", lambda df: df[df["SL"] < 3]),
        ("SL < 5", lambda df: df[df["SL"] < 5]),
        ("SL 3-10", lambda df: df[(df["SL"] > 3) & (df["SL"] < 10)]),
        ("SL 5-10", lambda df: df[(df["SL"] > 5) & (df["SL"] < 10)]),
        ("SL > 3", lambda df: df[df["SL"] > 3]),
        ("SL > 5", lambda df: df[df["SL"] > 5]),
    ]

    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"All Trades + {sl_name}",
            lambda df, f=sl_func: f(df)
        ))

    # === 1H Aligned + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"1H Aligned + {sl_name}",
            lambda df, f=sl_func: f(df[df["Direction"] == df["1H"]])
        ))

    # === 1H Against + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"1H Against + {sl_name}",
            lambda df, f=sl_func: f(df[df["Direction"] != df["1H"]])
        ))

    return strategies


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for all strategies at 1:1 and 1:2 RRR.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with strategy statistics, sorted by edge descending, positive edge only
    """
    strategies = get_strategies()
    results = []

    for strategy_name, filter_func in strategies:
        filtered_df = filter_func(df)
        for rrr in RRR_RATIOS:
            stats = _calculate_stats(filtered_df, strategy_name, rrr)
            results.append(stats)

    result_df = pd.DataFrame(results)

    # Filter to only show strategies with positive edge
    result_df = result_df[result_df["edge_value"] > 0].copy()

    # Sort by edge descending
    result_df = result_df.sort_values("edge_value", ascending=False)

    # Drop sorting column
    result_df = result_df.drop("edge_value", axis=1)

    # Rename columns to include totals
    total_trades = len(df)
    total_days = df["Date"].nunique() if "Date" in df.columns else 0
    result_df = result_df.rename(columns={
        "Trades": f"Trades ({total_trades})",
        "Days": f"Days ({total_days})",
    })

    # Reset index
    result_df = result_df.reset_index(drop=True)

    return result_df


def _breakeven_rate(rrr_ratio: int) -> float:
    """
    Calculate the breakeven win rate for a given RRR ratio.

    For 1:N RRR, breakeven = 100 / (1 + N).
    1:1 → 50%, 1:2 → 33.3%.

    Args:
        rrr_ratio: The reward multiplier (1 for 1:1, 2 for 1:2)

    Returns:
        Breakeven win rate as a percentage
    """
    return 100.0 / (1 + rrr_ratio)


def _calculate_stats(trades: pd.DataFrame, strategy_name: str, rrr_ratio: int = 1) -> Dict:
    """
    Calculate trading statistics for a strategy at a given RRR.

    Args:
        trades: DataFrame containing filtered trades
        strategy_name: Name of the strategy
        rrr_ratio: Risk-reward ratio (1 for 1:1, 2 for 1:2)

    Returns:
        Dictionary with calculated statistics
    """
    breakeven = _breakeven_rate(rrr_ratio)
    rrr_label = f"1:{rrr_ratio}"
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "Strategy": strategy_name,
            "RRR": rrr_label,
            "Trades": 0,
            "Notation": "0W – 0L",
            "Win Rate": "0.0%",
            "Outcome": "0R",
            "Edge": f"{-breakeven:.1f}%",
            "Days": 0,
            "Days %": "0%",
            "Trades Required": "N/A",
            "edge_value": -breakeven,
        }

    # Win condition: Pullback < SL AND TP >= rrr_ratio * SL
    winning_trades = trades[
        (trades["Pullback"] < trades["SL"]) &
        (trades["TP"] >= rrr_ratio * trades["SL"])
    ]

    wins = len(winning_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven
    outcome = (wins * rrr_ratio) - losses

    # Days with wins
    days_with_wins = winning_trades["Date"].nunique() if "Date" in winning_trades.columns and len(winning_trades) > 0 else 0
    total_days = trades["Date"].nunique() if "Date" in trades.columns else 0
    days_pct = (days_with_wins / total_days * 100) if total_days > 0 else 0.0

    # Trades required to earn 1R
    trades_required = (total_trades / outcome) if outcome > 0 else float("inf")

    return {
        "Strategy": strategy_name,
        "RRR": rrr_label,
        "Trades": total_trades,
        "Notation": f"{wins}W – {losses}L",
        "Win Rate": f"{win_rate:.1f}%",
        "Outcome": f"{outcome}R",
        "Edge": f"{edge:.1f}%",
        "Days": days_with_wins,
        "Days %": f"{days_pct:.0f}%",
        "Trades Required": f"{trades_required:.1f}" if outcome > 0 else "N/A",
        "edge_value": edge,
    }


def create_html_table(df: pd.DataFrame) -> str:
    """
    Create a dark-mode HTML table with styled formatting.

    Args:
        df: DataFrame to convert to HTML table

    Returns:
        HTML string with styled table
    """
    if df.empty:
        return "<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No profitable strategies found</p>"

    html = """
    <style>
        .analysis-table {
            border-collapse: collapse;
            width: 100%;
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .analysis-table th {
            background-color: #2d2d2d;
            color: #e0e0e0;
            padding: 8px;
            text-align: left;
            border: 1px solid #404040;
            font-weight: bold;
        }
        .analysis-table td {
            padding: 6px 8px;
            border: 1px solid #404040;
        }
        .analysis-table tr:hover {
            background-color: #2a2a2a;
        }
        .strategy-col {
            width: 300px;
        }
        .positive-edge {
            color: #4ade80;
        }
        .negative-edge {
            color: #f87171;
        }
    </style>
    <table class="analysis-table">
        <thead>
            <tr>
    """

    for col in df.columns:
        cls = ' class="strategy-col"' if col == "Strategy" else ""
        html += f"<th{cls}>{col}</th>"
    html += """
            </tr>
        </thead>
        <tbody>
    """

    for _, row in df.iterrows():
        html += "            <tr>\n"
        for col in df.columns:
            value = row[col]
            css_class = ""

            if col == "Strategy":
                css_class = "strategy-col"
            elif col == "Edge" or col.startswith("Edge "):
                try:
                    edge_val = float(str(value).replace("%", "").replace("R", "").replace("+", ""))
                    css_class = "positive-edge" if edge_val > 0 else "negative-edge"
                except (ValueError, TypeError):
                    pass
            elif col == "Pass":
                try:
                    css_class = "positive-edge" if int(value) > 0 else ""
                except (ValueError, TypeError):
                    pass
            elif col == "Fail":
                try:
                    css_class = "negative-edge" if int(value) > 0 else ""
                except (ValueError, TypeError):
                    pass

            cls_attr = f' class="{css_class}"' if css_class else ""
            html += f"                <td{cls_attr}>{value}</td>\n"
        html += "            </tr>\n"

    html += """
        </tbody>
    </table>
    """
    return html


def simulate_challenge(winning_mask, rrr_ratio: float, pass_threshold: float = 10, drawdown_threshold: float = 10) -> Tuple[int, int]:
    """
    Simulate prop firm challenge attempts by walking through trades sequentially.

    Tracks cumulative R. On reaching +pass_threshold, counts a pass and resets.
    On reaching -drawdown_threshold, counts a drawdown and resets.

    Args:
        winning_mask: Boolean series/list where True = win, False = loss
        rrr_ratio: R gained per win (e.g., 1 for 1:1, 2 for 1:2)
        pass_threshold: R needed to pass challenge
        drawdown_threshold: R loss that fails challenge

    Returns:
        Tuple of (pass_count, drawdown_count)
    """
    running_sum = 0.0
    pass_count = 0
    drawdown_count = 0

    for is_win in winning_mask:
        if is_win:
            running_sum += rrr_ratio
        else:
            running_sum -= 1

        if running_sum >= pass_threshold:
            pass_count += 1
            running_sum = 0.0
        elif running_sum <= -drawdown_threshold:
            drawdown_count += 1
            running_sum = 0.0

    return pass_count, drawdown_count


def _calculate_stats_with_buffer(trades: pd.DataFrame, strategy_name: str, buffer: float, rrr_ratio: float = 1) -> Dict:
    """
    Calculate trading statistics with extra pips added to SL.

    With buffer, effective SL = SL + buffer. Trade survives if Pullback < effective SL.
    Trade wins if TP >= rrr_ratio * effective SL.

    Includes prop firm challenge simulation: walks through trades tracking cumulative R,
    counting passes (+10R) and drawdowns (-10R).

    Args:
        trades: DataFrame containing filtered trades
        strategy_name: Name of the strategy
        buffer: Extra pips to add to SL
        rrr_ratio: Risk-reward ratio (1 for 1:1, 2 for 1:2)

    Returns:
        Dictionary with calculated statistics
    """
    rrr_label = f"1:{rrr_ratio:g}"
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "Strategy": strategy_name,
            "Buffer": f"+{buffer}",
            "RRR": rrr_label,
            "Trades": 0,
            "Notation": "0W – 0L",
            "Win Rate": "0.0%",
            "Fail": 0,
            "Pass": 0,
        }

    effective_sl = trades["SL"] + buffer

    # Win condition with buffer: Pullback < effective_sl AND TP >= rrr_ratio * effective_sl
    winning_mask = (trades["Pullback"] < effective_sl) & (trades["TP"] >= rrr_ratio * effective_sl)

    wins = winning_mask.sum()
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100

    pass_count, drawdown_count = simulate_challenge(winning_mask, rrr_ratio)

    return {
        "Strategy": strategy_name,
        "Buffer": f"+{buffer}",
        "RRR": rrr_label,
        "Trades": total_trades,
        "Notation": f"{wins}W – {losses}L",
        "Win Rate": f"{win_rate:.1f}%",
        "Fail": drawdown_count,
        "Pass": pass_count,
    }


def get_buffer_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get key strategies to test with SL buffers.

    Returns:
        List of tuples (strategy_name, filter_function)
    """
    base_strategies = [
        ("All Trades", lambda df: df),
        ("1H Aligned", lambda df: df[df["Direction"] == df["1H"]]),
        ("1H Against", lambda df: df[df["Direction"] != df["1H"]]),
    ]

    strategies = list(base_strategies)

    # sl_caps = [3, 4, 5]
    # for name, base_func in base_strategies:
    #     for cap in sl_caps:
    #         strategies.append((
    #             f"{name} + SL < {cap}",
    #             lambda df, f=base_func, c=cap: f(df[df["SL"] < c])
    #         ))

    return strategies


def calculate_buffer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for key strategies with each SL buffer value at 1:1 and 1:2 RRR.

    For each strategy, tests what happens if extra pips (0, 0.5, 1, 1.5, 2, 3, 4, 5)
    are added to the SL. A wider SL means more trades survive but the target is also higher.

    Includes prop firm challenge simulation: walks through trades tracking cumulative R,
    counting passes (+10R) and drawdowns (-10R).

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with buffer statistics, filtered to strategies with at least one pass
    """
    strategies = get_buffer_strategies()
    results = []

    for strategy_name, filter_func in strategies:
        filtered_df = filter_func(df)
        for rrr in RRR_RATIOS:
            for buffer in BUFFER_PIPS:
                stats = _calculate_stats_with_buffer(filtered_df, strategy_name, buffer, rrr)
                results.append(stats)

    result_df = pd.DataFrame(results)

    # Filter to only show strategies with at least one pass
    result_df = result_df[result_df["Pass"] > 0].copy()

    # Sort by Pass descending, then Drawdown ascending
    result_df = result_df.sort_values(["Pass", "Fail"], ascending=[False, True])

    # Reset index
    result_df = result_df.reset_index(drop=True)

    return result_df


def _calculate_buffer_statistics_filtered(df: pd.DataFrame, strategy_names: List[str]) -> pd.DataFrame:
    """
    Calculate buffer statistics for a subset of strategies.

    Args:
        df: DataFrame with trading data
        strategy_names: List of strategy names to include

    Returns:
        DataFrame with buffer statistics, filtered to strategies with at least one pass
    """
    strategies = [(n, f) for n, f in get_buffer_strategies() if n in strategy_names]
    results = []

    for strategy_name, filter_func in strategies:
        filtered_df = filter_func(df)
        for rrr in RRR_RATIOS:
            for buffer in BUFFER_PIPS:
                stats = _calculate_stats_with_buffer(filtered_df, strategy_name, buffer, rrr)
                results.append(stats)

    result_df = pd.DataFrame(results)
    result_df = result_df[result_df["Pass"] > 0].copy()
    result_df = result_df.sort_values(["Pass", "Fail"], ascending=[False, True])
    result_df = result_df.reset_index(drop=True)

    return result_df


def _display_analysis_table(df: pd.DataFrame, title: str, strategy_names: List[str]):
    """
    Display a buffer analysis table for given strategies.

    Args:
        df: DataFrame with trading data
        title: Table title
        strategy_names: Strategy names to include
    """
    from IPython.display import display, HTML

    display(HTML("""<style>
        .jp-OutputArea-child { max-height: none !important; }
        .jp-OutputArea-output { max-height: none !important; overflow: visible !important; }
        .output_scroll { box-shadow: none !important; border: none !important; }
    </style>"""))

    title_html = f"<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>{title}</h2>"
    display(HTML(title_html))

    stats_df = _calculate_buffer_statistics_filtered(df, strategy_names)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No profitable strategies found</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


def display_analysis_all(df: pd.DataFrame):
    """
    Display buffer analysis for All Trades.

    Args:
        df: DataFrame with trading data
    """
    _display_analysis_table(df, "All Trades Analysis", ["All Trades"])


def display_analysis_1h(df: pd.DataFrame):
    """
    Display buffer analysis for 1H Aligned and 1H Against trades.

    Args:
        df: DataFrame with trading data
    """
    _display_analysis_table(df, "1H Alignment Analysis", ["1H Aligned", "1H Against"])


FIXED_SL_SIZES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def _calculate_fixed_sl_stats(trades: pd.DataFrame, fixed_sl: float, rrr_ratio: int = 1) -> Dict:
    """
    Calculate trading statistics using a fixed SL size instead of the original SL.

    The original SL from the CSV is ignored. Instead, a fixed SL is used:
    - Trade survives if Pullback < fixed_sl
    - Trade wins if Pullback < fixed_sl AND TP >= rrr_ratio * fixed_sl

    Args:
        trades: DataFrame containing filtered trades
        fixed_sl: Fixed stop loss in pips (replaces original SL)
        rrr_ratio: Risk-reward ratio (1 for 1:1, 2 for 1:2)

    Returns:
        Dictionary with calculated statistics
    """
    breakeven = _breakeven_rate(rrr_ratio)
    rrr_label = f"1:{rrr_ratio}"
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "Fixed SL": f"{fixed_sl}",
            "RRR": rrr_label,
            "Trades": 0,
            "Notation": "0W – 0L",
            "Win Rate": "0.0%",
            "Outcome": "0R",
            "Edge": f"{-breakeven:.1f}%",
            "Days": 0,
            "Days %": "0%",
            "Trades Required": "N/A",
            "edge_value": -breakeven,
        }

    winning_trades = trades[
        (trades["Pullback"] < fixed_sl) &
        (trades["TP"] >= rrr_ratio * fixed_sl)
    ]

    wins = len(winning_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven
    outcome = (wins * rrr_ratio) - losses

    days_with_wins = winning_trades["Date"].nunique() if "Date" in winning_trades.columns and len(winning_trades) > 0 else 0
    total_days = trades["Date"].nunique() if "Date" in trades.columns else 0
    days_pct = (days_with_wins / total_days * 100) if total_days > 0 else 0.0
    trades_required = (total_trades / outcome) if outcome > 0 else float("inf")

    return {
        "Fixed SL": f"{fixed_sl}",
        "RRR": rrr_label,
        "Trades": total_trades,
        "Notation": f"{wins}W – {losses}L",
        "Win Rate": f"{win_rate:.1f}%",
        "Outcome": f"{outcome}R",
        "Edge": f"{edge:.1f}%",
        "Days": days_with_wins,
        "Days %": f"{days_pct:.0f}%",
        "Trades Required": f"{trades_required:.1f}" if outcome > 0 else "N/A",
        "edge_value": edge,
    }


def calculate_fixed_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for fixed SL sizes from 1.5 to 5.0 pips.

    Instead of using the original SL from the CSV, each trade is evaluated
    with a fixed SL. A trade survives if Pullback < fixed_SL, and wins if
    TP >= RRR * fixed_SL.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with fixed SL statistics, sorted by outcome descending
    """
    results = []
    total_trades = len(df)

    for fixed_sl in FIXED_SL_SIZES:
        for rrr in RRR_RATIOS:
            stats = _calculate_fixed_sl_stats(df, fixed_sl, rrr)
            results.append(stats)

    result_df = pd.DataFrame(results)

    # Filter to only show strategies with positive edge
    result_df = result_df[result_df["edge_value"] > 0].copy()

    # Sort by outcome descending, then edge descending
    result_df = result_df.sort_values("edge_value", ascending=False)

    # Drop sorting column
    result_df = result_df.drop("edge_value", axis=1)

    # Rename columns to include totals
    total_days = df["Date"].nunique() if "Date" in df.columns else 0
    result_df = result_df.rename(columns={
        "Trades": f"Trades ({total_trades})",
        "Days": f"Days ({total_days})",
    })

    result_df = result_df.reset_index(drop=True)

    return result_df


def display_fixed_sl(df: pd.DataFrame):
    """
    Display fixed SL size analysis.

    Tests what happens when using a fixed SL (1.5 to 5.0 pips) instead of the
    original safe stop. Trade survives if Pullback < fixed SL, wins if TP reaches target.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Fixed SL Size Analysis</h2>"
    subtitle_html = "<p style='color: #a0a0a0; background-color: #1e1e1e; padding: 0 10px 10px;'>Uses a fixed SL (1.5-5.0 pips) instead of the original safe stop. Trade survives if Pullback &lt; fixed SL.</p>"
    display(HTML(title_html + subtitle_html))

    stats_df = calculate_fixed_sl_statistics(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data available</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


def _calculate_fixed_sl_stats_with_strategy(trades: pd.DataFrame, strategy_name: str, fixed_sl: float, rrr_ratio: int = 1) -> Dict:
    """
    Calculate fixed SL statistics for a named strategy filter.

    Same logic as _calculate_fixed_sl_stats but includes a Strategy column.

    Args:
        trades: DataFrame containing filtered trades
        strategy_name: Name of the strategy filter applied
        fixed_sl: Fixed stop loss in pips
        rrr_ratio: Risk-reward ratio (1 for 1:1, 2 for 1:2)

    Returns:
        Dictionary with calculated statistics including Strategy column
    """
    breakeven = _breakeven_rate(rrr_ratio)
    rrr_label = f"1:{rrr_ratio}"
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "Strategy": strategy_name,
            "Fixed SL": f"{fixed_sl}",
            "RRR": rrr_label,
            "Trades": 0,
            "Notation": "0W – 0L",
            "Win Rate": "0.0%",
            "Outcome": "0R",
            "Edge": f"{-breakeven:.1f}%",
            "Days": 0,
            "Days %": "0%",
            "Trades Required": "N/A",
            "edge_value": -breakeven,
        }

    winning_trades = trades[
        (trades["Pullback"] < fixed_sl) &
        (trades["TP"] >= rrr_ratio * fixed_sl)
    ]

    wins = len(winning_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven
    outcome = (wins * rrr_ratio) - losses

    days_with_wins = winning_trades["Date"].nunique() if "Date" in winning_trades.columns and len(winning_trades) > 0 else 0
    total_days = trades["Date"].nunique() if "Date" in trades.columns else 0
    days_pct = (days_with_wins / total_days * 100) if total_days > 0 else 0.0
    trades_required = (total_trades / outcome) if outcome > 0 else float("inf")

    return {
        "Strategy": strategy_name,
        "Fixed SL": f"{fixed_sl}",
        "RRR": rrr_label,
        "Trades": total_trades,
        "Notation": f"{wins}W – {losses}L",
        "Win Rate": f"{win_rate:.1f}%",
        "Outcome": f"{outcome}R",
        "Edge": f"{edge:.1f}%",
        "Days": days_with_wins,
        "Days %": f"{days_pct:.0f}%",
        "Trades Required": f"{trades_required:.1f}" if outcome > 0 else "N/A",
        "edge_value": edge,
    }


def _get_1h_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get 1H-based strategy filters.

    Returns:
        List of tuples (strategy_name, filter_function)
    """
    return [
        ("All Trades", lambda df: df),
        ("1H Aligned", lambda df: df[df["Direction"] == df["1H"]]),
        ("1H Against", lambda df: df[df["Direction"] != df["1H"]]),
    ]


def calculate_fixed_sl_1h_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fixed SL statistics filtered by 1H alignment.

    For each 1H strategy (All, Aligned, Against), tests all fixed SL sizes
    at each RRR ratio.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with fixed SL + 1H statistics, sorted by edge descending
    """
    strategies = _get_1h_strategies()
    results = []
    total_trades = len(df)

    for strategy_name, filter_func in strategies:
        filtered_df = filter_func(df)
        for fixed_sl in FIXED_SL_SIZES:
            for rrr in RRR_RATIOS:
                stats = _calculate_fixed_sl_stats_with_strategy(filtered_df, strategy_name, fixed_sl, rrr)
                results.append(stats)

    result_df = pd.DataFrame(results)

    # Filter to only show strategies with positive edge
    result_df = result_df[result_df["edge_value"] > 0].copy()

    # Sort by edge descending
    result_df = result_df.sort_values("edge_value", ascending=False)

    # Drop sorting column
    result_df = result_df.drop("edge_value", axis=1)

    # Rename columns to include totals
    total_days = df["Date"].nunique() if "Date" in df.columns else 0
    result_df = result_df.rename(columns={
        "Trades": f"Trades ({total_trades})",
        "Days": f"Days ({total_days})",
    })

    result_df = result_df.reset_index(drop=True)

    return result_df


def display_fixed_sl_1h(df: pd.DataFrame):
    """
    Display fixed SL analysis filtered by 1H alignment.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Fixed SL + 1H Alignment Analysis</h2>"
    subtitle_html = "<p style='color: #a0a0a0; background-color: #1e1e1e; padding: 0 10px 10px;'>Fixed SL (1.5-5.0 pips) combined with 1H higher timeframe direction filter.</p>"
    display(HTML(title_html + subtitle_html))

    stats_df = calculate_fixed_sl_1h_statistics(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data available</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


def create_r_histogram_combined(df: pd.DataFrame) -> str:
    """
    Create an HTML bar chart showing cumulative R value distribution (1R-10R).

    A trade that reached 3R also passed through 1R and 2R, so higher R trades
    are counted in all lower R buckets. For example:
    - 1R count = trades that reached 1R + 2R + 3R + ... + 10R
    - 2R count = trades that reached 2R + 3R + ... + 10R
    - 10R count = only trades that reached exactly 10R

    Args:
        df: DataFrame with trading data (must have 'R' column)

    Returns:
        HTML string with a styled horizontal bar chart
    """
    r_values = df["R"].dropna().abs().apply(lambda x: int(x))
    r_values = r_values[(r_values >= 1) & (r_values <= 10)]
    raw_counts = r_values.value_counts().reindex(range(1, 11), fill_value=0)

    # Cumulative: each R level includes all trades that reached that R or higher
    cumulative_counts = pd.Series(
        {r: raw_counts.loc[r:].sum() for r in range(1, 11)}
    )

    max_count = cumulative_counts.max() if cumulative_counts.max() > 0 else 1

    html = """
    <div style="background-color: #1e1e1e; padding: 20px; font-family: 'Courier New', monospace;">
        <h2 style="color: #e0e0e0; margin-top: 0;">R Distribution</h2>
    """

    for r_val in range(1, 11):
        count = cumulative_counts[r_val]
        bar_width = (count / max_count) * 100
        html += f"""
        <div style="display: flex; align-items: center; margin: 4px 0;">
            <span style="color: #e0e0e0; width: 40px; text-align: right; margin-right: 10px;">{r_val}R</span>
            <div style="background-color: #4ade80; height: 24px; width: {bar_width}%; min-width: {'2px' if count > 0 else '0'}; border-radius: 3px;"></div>
            <span style="color: #a0a0a0; margin-left: 8px;">{count}</span>
        </div>
        """

    html += "</div>"
    return html


def display_r_histogram_combined(df: pd.DataFrame):
    """
    Display a cumulative histogram of R value distribution (1R-10R).

    Each R level includes all trades that reached that R or higher,
    since a trade reaching e.g. 3R must have passed through 1R and 2R.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML
    html = create_r_histogram_combined(df)
    display(HTML(html))


def display_buffer_analysis(df: pd.DataFrame):
    """
    Display SL buffer analysis - what if extra pips were added to the stop loss.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>SL Buffer Analysis (1:1 RRR) - What if extra pips were added to SL?</h2>"
    display(HTML(title_html))

    stats_df = calculate_buffer_statistics(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No profitable buffer strategies found</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def _format_wl(wins: int, losses: int, total: int) -> str:
    """
    Format wins/losses/win rate into a compact string like '13W - 12L (52.0%)'.

    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        total: Total number of trades

    Returns:
        Formatted string, or '0W - 0L (0.0%)' when total is 0
    """
    win_rate = (wins / total * 100) if total > 0 else 0.0
    return f"{wins}W - {losses}L ({win_rate:.1f}%)"


def calculate_weekday_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for each weekday (Monday-Friday).

    Shows regular win/loss and with +2 pip buffer added to SL.
    Regular win: Pullback < SL AND TP > 0.
    Buffer win: Pullback < SL + 2 AND TP > 0.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Day, Trades, Regular, With 2 pips buffer
    """
    buffer = 2.0
    results = []

    for day in WEEKDAY_ORDER:
        day_trades = df[df['Weekday'] == day]
        total = len(day_trades)

        if total == 0:
            results.append({
                'Day': day,
                'Trades': 0,
                'Regular': _format_wl(0, 0, 0),
                'With 2 pips buffer': _format_wl(0, 0, 0),
            })
            continue

        wins = len(day_trades[
            (day_trades['Pullback'] < day_trades['SL']) &
            (day_trades['TP'] > 0)
        ])
        losses = total - wins

        buf_wins = len(day_trades[
            (day_trades['Pullback'] < day_trades['SL'] + buffer) &
            (day_trades['TP'] > 0)
        ])
        buf_losses = total - buf_wins

        results.append({
            'Day': day,
            'Trades': total,
            'Regular': _format_wl(wins, losses, total),
            'With 2 pips buffer': _format_wl(buf_wins, buf_losses, total),
        })

    return pd.DataFrame(results)


def display_weekday(df: pd.DataFrame):
    """
    Display win/loss statistics broken down by weekday.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Weekday Statistics</h2>"
    display(HTML(title_html))

    stats_df = calculate_weekday_statistics(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


SL_RANGES = [
    ("0-3", 0, 3),
    ("3-5", 3, 5),
    ("< 5", 0, 5),
    ("5-10", 5, 10),
    ("10+", 10, float("inf")),
]


def calculate_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for SL pip ranges.

    Shows regular win/loss and with +2 pip buffer added to SL.
    Regular win: Pullback < SL AND TP > 0.
    Buffer win: Pullback < SL + 2 AND TP > 0.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: SL Range, Trades, Regular, With 2 pips buffer
    """
    buffer = 2.0
    results = []

    for label, low, high in SL_RANGES:
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)

        if total == 0:
            results.append({
                'SL Range': label,
                'Trades': 0,
                'Regular': _format_wl(0, 0, 0),
                'With 2 pips buffer': _format_wl(0, 0, 0),
            })
            continue

        wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL']) &
            (range_trades['TP'] > 0)
        ])
        losses = total - wins

        buf_wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL'] + buffer) &
            (range_trades['TP'] > 0)
        ])
        buf_losses = total - buf_wins

        results.append({
            'SL Range': label,
            'Trades': total,
            'Regular': _format_wl(wins, losses, total),
            'With 2 pips buffer': _format_wl(buf_wins, buf_losses, total),
        })

    return pd.DataFrame(results)


def display_analysis_sl(df: pd.DataFrame):
    """
    Display win/loss statistics broken down by SL pip range.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>SL Range Statistics</h2>"
    display(HTML(title_html))

    stats_df = calculate_sl_statistics(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


PULLBACK_RANGES = [
    ("0-3", 0, 3),
    ("3-5", 3, 5),
    ("5-10", 5, 10),
    ("10+", 10, float("inf")),
]


def calculate_pullback_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for Pullback pip ranges.

    Shows regular win/loss and with +2 pip buffer added to SL.
    Regular win: Pullback < SL AND TP > 0.
    Buffer win: Pullback < SL + 2 AND TP > 0.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Pullback Range, Trades, Regular, With 2 pips buffer
    """
    buffer = 2.0
    results = []

    for label, low, high in PULLBACK_RANGES:
        range_trades = df[(df['Pullback'] >= low) & (df['Pullback'] < high)]
        total = len(range_trades)

        if total == 0:
            results.append({
                'Pullback Range': label,
                'Trades': 0,
                'Regular': _format_wl(0, 0, 0),
                'With 2 pips buffer': _format_wl(0, 0, 0),
            })
            continue

        wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL']) &
            (range_trades['TP'] > 0)
        ])
        losses = total - wins

        buf_wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL'] + buffer) &
            (range_trades['TP'] > 0)
        ])
        buf_losses = total - buf_wins

        results.append({
            'Pullback Range': label,
            'Trades': total,
            'Regular': _format_wl(wins, losses, total),
            'With 2 pips buffer': _format_wl(buf_wins, buf_losses, total),
        })

    return pd.DataFrame(results)


def display_analysis_pullback(df: pd.DataFrame):
    """
    Display win/loss statistics broken down by Pullback pip range.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Pullback Range Statistics</h2>"
    display(HTML(title_html))

    stats_df = calculate_pullback_statistics(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


TP_RANGES = [
    ("0-10", 0, 10),
    ("10-20", 10, 20),
    ("20-30", 20, 30),
    ("30-50", 30, 50),
    ("50+", 50, float("inf")),
]


def calculate_tp_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for TP pip ranges.

    Shows regular win/loss and with +2 pip buffer added to SL.
    Regular win: Pullback < SL AND TP > 0.
    Buffer win: Pullback < SL + 2 AND TP > 0.
    Trades with TP == 0 (no profit target reached) go into the 0-10 bucket.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: TP Range, Trades, Regular, With 2 pips buffer
    """
    buffer = 2.0
    results = []

    for label, low, high in TP_RANGES:
        range_trades = df[(df['TP'] >= low) & (df['TP'] < high)]
        total = len(range_trades)

        if total == 0:
            results.append({
                'TP Range': label,
                'Trades': 0,
                'Regular': _format_wl(0, 0, 0),
                'With 2 pips buffer': _format_wl(0, 0, 0),
            })
            continue

        wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL']) &
            (range_trades['TP'] > 0)
        ])
        losses = total - wins

        buf_wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL'] + buffer) &
            (range_trades['TP'] > 0)
        ])
        buf_losses = total - buf_wins

        results.append({
            'TP Range': label,
            'Trades': total,
            'Regular': _format_wl(wins, losses, total),
            'With 2 pips buffer': _format_wl(buf_wins, buf_losses, total),
        })

    return pd.DataFrame(results)


def display_analysis_tp(df: pd.DataFrame):
    """
    Display win/loss statistics broken down by TP pip range.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>TP Range Statistics</h2>"
    display(HTML(title_html))

    stats_df = calculate_tp_statistics(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


# ---------------------------------------------------------------------------
# Improvement-search analyses
#
# All evaluated on 1H Aligned trades with a +2 pip SL buffer, at 1:2 and 1:3 RRR.
# Win condition: Pullback < SL+buffer AND TP >= rrr * (SL+buffer).
# ---------------------------------------------------------------------------

IMPROVEMENT_BUFFER = 2.0
IMPROVEMENT_RRRS = [2, 3]
TRADE_NUMBERS = ['#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9']
PREV_OUTCOMES = ['First of Day', 'After Win', 'After Loss']


def _aligned(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to trades where Direction matches the 1H bias."""
    return df[df['Direction'] == df['1H']].copy()


def _wins_at(sub: pd.DataFrame, buffer: float, rrr: float) -> pd.Series:
    """Boolean Series: True where the trade wins at the given buffer and RRR."""
    eff = sub['SL'] + buffer
    return (sub['Pullback'] < eff) & (sub['TP'] >= rrr * eff)


def _outcome_cells(sub: pd.DataFrame, buffer: float, rrr: float) -> Tuple[str, str]:
    """
    Compute (notation, edge) cells for a slice at the given RRR.

    notation: '12W - 33L (26.6%)'
    edge: '+0.234R' net R per trade (or '0.000R' for empty slice).
    """
    n = len(sub)
    if n == 0:
        return _format_wl(0, 0, 0), '0.000R'
    wins = int(_wins_at(sub, buffer, rrr).sum())
    losses = n - wins
    net = (wins * rrr - losses) / n
    return _format_wl(wins, losses, n), f"{net:+.3f}R"


def _two_rrr_row(label_key: str, label_val: str, sub: pd.DataFrame, buffer: float) -> Dict:
    """Build a row with Trades + 1:2 and 1:3 columns."""
    row: Dict = {label_key: label_val, 'Trades': len(sub)}
    for rrr in IMPROVEMENT_RRRS:
        notation, edge = _outcome_cells(sub, buffer, rrr)
        row[f'1:{rrr} W/L'] = notation
        row[f'Edge 1:{rrr}'] = edge
    return row


def _display_section(df: pd.DataFrame, title: str, stats_fn: Callable[[pd.DataFrame], pd.DataFrame]):
    """Shared title + table display for the improvement analyses."""
    from IPython.display import display, HTML
    title_html = (
        "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>"
        f"{title}</h2>"
    )
    display(HTML(title_html))
    display(HTML(create_html_table(stats_fn(df))))


def calculate_direction_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Win/loss by trade Direction (Buy vs Sell) for 1H Aligned + 2 pip buffer."""
    aligned = _aligned(df)
    rows = [
        _two_rrr_row('Direction', d, aligned[aligned['Direction'] == d], IMPROVEMENT_BUFFER)
        for d in ['Buy', 'Sell']
    ]
    return pd.DataFrame(rows)


def display_analysis_direction(df: pd.DataFrame):
    """Display win/loss broken down by Direction (Buy vs Sell)."""
    _display_section(df, "Direction (1H Aligned + 2 pip buffer)", calculate_direction_statistics)


def calculate_trade_number_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Win/loss by Trade # of the day for 1H Aligned + 2 pip buffer."""
    aligned = _aligned(df)
    rows = []
    for t in TRADE_NUMBERS:
        sub = aligned[aligned['Trade'] == t]
        if len(sub) == 0:
            continue
        rows.append(_two_rrr_row('Trade #', t, sub, IMPROVEMENT_BUFFER))
    return pd.DataFrame(rows)


def display_analysis_trade_number(df: pd.DataFrame):
    """Display win/loss broken down by Trade # within the trading day."""
    _display_section(df, "Trade # of the Day (1H Aligned + 2 pip buffer)", calculate_trade_number_statistics)


def _annotate_prev_outcome(aligned: pd.DataFrame, buffer: float, rrr: float) -> pd.DataFrame:
    """
    Add a 'Context' column to aligned trades: First of Day / After Win / After Loss.

    Win/loss for the context lookup is computed at the given (buffer, rrr).
    """
    aligned = aligned.sort_values(['Date', 'Trade']).reset_index(drop=True).copy()
    aligned['_won'] = _wins_at(aligned, buffer, rrr)
    aligned['_prev_won'] = aligned.groupby('Date')['_won'].shift(1)

    def label(row):
        if pd.isna(row['_prev_won']):
            return 'First of Day'
        return 'After Win' if row['_prev_won'] else 'After Loss'

    aligned['Context'] = aligned.apply(label, axis=1)
    return aligned


def calculate_prev_outcome_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win/loss by previous trade outcome on the same day.

    Context is derived from the 1:2 RRR outcome, then each context is evaluated
    at both 1:2 and 1:3 RRR.
    """
    aligned = _aligned(df)
    annotated = _annotate_prev_outcome(aligned, IMPROVEMENT_BUFFER, 2)
    rows = [
        _two_rrr_row('Context', ctx, annotated[annotated['Context'] == ctx], IMPROVEMENT_BUFFER)
        for ctx in PREV_OUTCOMES
    ]
    return pd.DataFrame(rows)


def display_analysis_prev_outcome(df: pd.DataFrame):
    """Display win/loss broken down by previous trade outcome on the same day."""
    _display_section(
        df,
        "Previous Trade Outcome (1H Aligned + 2 pip buffer; context from 1:2)",
        calculate_prev_outcome_statistics,
    )


RRR_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0]


def calculate_rrr_sweep_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate and net R/trade across RRR ratios from 1:1 to 1:3 (1H Aligned + buffer).
    """
    aligned = _aligned(df)
    n = len(aligned)
    rows = []
    for rrr in RRR_SWEEP:
        if n == 0:
            rows.append({
                'RRR': f"1:{rrr:g}", 'Trades': 0,
                'W/L': _format_wl(0, 0, 0), 'Edge R/Trade': '0.000R',
            })
            continue
        wins = int(_wins_at(aligned, IMPROVEMENT_BUFFER, rrr).sum())
        losses = n - wins
        net = (wins * rrr - losses) / n
        rows.append({
            'RRR': f"1:{rrr:g}",
            'Trades': n,
            'W/L': _format_wl(wins, losses, n),
            'Edge R/Trade': f"{net:+.3f}R",
        })
    return pd.DataFrame(rows)


def display_analysis_rrr_sweep(df: pd.DataFrame):
    """Display win rate and net R/trade across RRR ratios."""
    _display_section(df, "RRR Sweep (1H Aligned + 2 pip buffer)", calculate_rrr_sweep_statistics)


STOP_AFTER_LOSS_RULES = [
    ('No Cap', None),
    ('Stop after 1st Loss', 1),
    ('Stop after 2nd Loss', 2),
    ('Stop after 3rd Loss', 3),
]


def _apply_loss_cap(aligned: pd.DataFrame, buffer: float, rrr: float, max_losses: int) -> pd.DataFrame:
    """
    Keep trades up to (but not including) the day's max_losses-th loss at the given RRR.

    Example with max_losses=1: keep trades until the first loss of the day occurs.
    """
    sorted_df = aligned.sort_values(['Date', 'Trade']).reset_index(drop=True).copy()
    sorted_df['_lost'] = ~_wins_at(sorted_df, buffer, rrr)
    losses_before = sorted_df.groupby('Date')['_lost'].cumsum() - sorted_df['_lost'].astype(int)
    keep = sorted_df[losses_before < max_losses]
    return keep.drop(columns=['_lost'])


def calculate_stop_after_loss_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Effect of stopping trading for the day after N losses.

    Loss for the cap is detected at each RRR independently (so 'stop after 1st loss'
    in the 1:3 column uses 1:3 losses to decide when to stop).
    """
    aligned = _aligned(df)
    rows = []
    for label, cap in STOP_AFTER_LOSS_RULES:
        row: Dict = {'Rule': label}
        for rrr in IMPROVEMENT_RRRS:
            sub = aligned if cap is None else _apply_loss_cap(aligned, IMPROVEMENT_BUFFER, rrr, cap)
            notation, edge = _outcome_cells(sub, IMPROVEMENT_BUFFER, rrr)
            row[f'Trades 1:{rrr}'] = len(sub)
            row[f'1:{rrr} W/L'] = notation
            row[f'Edge 1:{rrr}'] = edge
        rows.append(row)
    return pd.DataFrame(rows)


def display_analysis_stop_after_loss(df: pd.DataFrame):
    """Display effect of stopping after N losses per day."""
    _display_section(df, "Stop After N Losses (1H Aligned + 2 pip buffer)", calculate_stop_after_loss_statistics)


# ---------------------------------------------------------------------------
# PD (Premium / Discount) analyses
#
# PD column values:
#   - "Buy"  -> price is in Discount  (favors Buys)
#   - "Sell" -> price is in Premium   (favors Sells)
#   - empty  -> PD bias not yet labeled; excluded from PD analyses
#
# All evaluated with the +2 pip SL buffer at 1:2 and 1:3 RRR.
# ---------------------------------------------------------------------------

PD_ZONES = [
    ('Discount (PD = Buy)', 'Buy'),
    ('Premium (PD = Sell)', 'Sell'),
]


def _pd_known(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to trades where the PD column has a Buy/Sell value."""
    if 'PD' not in df.columns:
        return df.iloc[0:0].copy()
    return df[df['PD'].isin(['Buy', 'Sell'])].copy()


def calculate_pd_alignment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win/loss by PD alignment for trades with a known PD bias (+2 pip buffer).

    Rows: All PD-Known, PD Aligned (Direction == PD), PD Against.
    """
    known = _pd_known(df)
    aligned = known[known['Direction'] == known['PD']]
    against = known[known['Direction'] != known['PD']]
    rows = [
        _two_rrr_row('Filter', 'All PD-Known', known, IMPROVEMENT_BUFFER),
        _two_rrr_row('Filter', 'PD Aligned', aligned, IMPROVEMENT_BUFFER),
        _two_rrr_row('Filter', 'PD Against', against, IMPROVEMENT_BUFFER),
    ]
    return pd.DataFrame(rows)


def display_analysis_pd_alignment(df: pd.DataFrame):
    """Display PD alignment analysis (with PD vs against PD)."""
    _display_section(df, "PD Alignment (+2 pip buffer)", calculate_pd_alignment_statistics)


def calculate_pd_zone_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win/loss broken down by PD zone (Discount / Premium), +2 pip buffer.
    """
    known = _pd_known(df)
    rows = [
        _two_rrr_row('Zone', label, known[known['PD'] == val], IMPROVEMENT_BUFFER)
        for label, val in PD_ZONES
    ]
    return pd.DataFrame(rows)


def display_analysis_pd_zone(df: pd.DataFrame):
    """Display win/loss broken down by PD zone."""
    _display_section(df, "PD Zone (+2 pip buffer)", calculate_pd_zone_statistics)


def calculate_pd_1h_combined_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    2x2 confluence of 1H bias and PD bias on trades with a known PD value.
    """
    known = _pd_known(df)
    combos = [
        ('1H Aligned + PD Aligned',
         (known['Direction'] == known['1H']) & (known['Direction'] == known['PD'])),
        ('1H Aligned + PD Against',
         (known['Direction'] == known['1H']) & (known['Direction'] != known['PD'])),
        ('1H Against + PD Aligned',
         (known['Direction'] != known['1H']) & (known['Direction'] == known['PD'])),
        ('1H Against + PD Against',
         (known['Direction'] != known['1H']) & (known['Direction'] != known['PD'])),
    ]
    rows = [
        _two_rrr_row('Combination', label, known[mask], IMPROVEMENT_BUFFER)
        for label, mask in combos
    ]
    return pd.DataFrame(rows)


def display_analysis_pd_1h_combined(df: pd.DataFrame):
    """Display the 2x2 1H x PD confluence table."""
    _display_section(df, "1H x PD Confluence (+2 pip buffer)", calculate_pd_1h_combined_statistics)


def calculate_pd_direction_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    PD-Aligned trades split by Direction (Buy = Discount entries, Sell = Premium entries).
    """
    known = _pd_known(df)
    aligned = known[known['Direction'] == known['PD']]
    rows = [
        _two_rrr_row('Direction', d, aligned[aligned['Direction'] == d], IMPROVEMENT_BUFFER)
        for d in ['Buy', 'Sell']
    ]
    return pd.DataFrame(rows)


def display_analysis_pd_direction(df: pd.DataFrame):
    """Display PD-Aligned trades broken down by Direction."""
    _display_section(df, "PD Aligned by Direction (+2 pip buffer)", calculate_pd_direction_statistics)
