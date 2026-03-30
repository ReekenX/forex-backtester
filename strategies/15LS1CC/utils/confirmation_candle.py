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

    for col in ["SL", "TP", "Pullback"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

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
            elif col == "Edge":
                try:
                    edge_val = float(str(value).replace("%", ""))
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


def calculate_bruteforce(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bruteforce scan of all buffer (extra SL pips) and RRR combinations.

    Tests buffer values from 0.0 to 10.0 in 0.5 pip steps, combined with
    RRR from 1:1 to 1:3, to find which combination gives the best outcome.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with results for every buffer x RRR combination, sorted by outcome descending
    """
    buffers = [round(x * 0.5, 1) for x in range(21)]  # 0.0 to 10.0 in 0.5 steps
    rrr_range = [1, 2, 3]

    total_trades = len(df)
    results = []

    for buffer in buffers:
        effective_sl = df["SL"] + buffer

        for rrr in rrr_range:
            breakeven = 100.0 / (1 + rrr)

            winning_trades = df[
                (df["Pullback"] < effective_sl) &
                (df["TP"] >= rrr * effective_sl)
            ]

            wins = len(winning_trades)
            losses = total_trades - wins
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            edge = win_rate - breakeven
            outcome = (wins * rrr) - losses

            days_with_wins = winning_trades["Date"].nunique() if "Date" in winning_trades.columns and len(winning_trades) > 0 else 0
            total_days = df["Date"].nunique() if "Date" in df.columns else 0
            days_pct = (days_with_wins / total_days * 100) if total_days > 0 else 0.0
            trades_required = (total_trades / outcome) if outcome > 0 else float("inf")

            results.append({
                "Buffer": f"+{buffer}",
                "RRR": f"1:{rrr}",
                "Trades": total_trades,
                "Notation": f"{wins}W – {losses}L",
                "Win Rate": f"{win_rate:.1f}%",
                "Outcome": f"{outcome}R",
                "Edge": f"{edge:.1f}%",
                "Days": days_with_wins,
                "Days %": f"{days_pct:.0f}%",
                "Trades Required": f"{trades_required:.1f}" if outcome > 0 else "N/A",
                "outcome_value": outcome,
                "edge_value": edge,
            })

    result_df = pd.DataFrame(results)

    # Filter to only show strategies with positive edge
    result_df = result_df[result_df["edge_value"] > 0].copy()

    # Sort by outcome descending, then edge descending
    result_df = result_df.sort_values(["outcome_value", "edge_value"], ascending=[False, False])

    # Drop sorting columns
    result_df = result_df.drop(["outcome_value", "edge_value"], axis=1)

    # Rename columns to include totals
    total_days = df["Date"].nunique() if "Date" in df.columns else 0
    result_df = result_df.rename(columns={
        "Trades": f"Trades ({total_trades})",
        "Days": f"Days ({total_days})",
    })

    result_df = result_df.reset_index(drop=True)

    return result_df


def display_analysis_buffer(df: pd.DataFrame):
    """
    Display bruteforce analysis scanning all buffer x RRR combinations.

    Tests extra SL pips from 0.0 to 10.0 (0.5 steps) with RRR 1:1 to 1:3
    to find which combination gives the best outcome for all trades.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Buffer Analysis</h2>"
    display(HTML(title_html))

    stats_df = calculate_bruteforce(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data available</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


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
    ("5-10", 5, 10),
    ("10+", 10, float("inf")),
]


def calculate_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for SL pip ranges.

    Win condition: Pullback < SL AND TP > 0.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: SL Range, Trades, Wins, Losses, Win Rate
    """
    results = []

    for label, low, high in SL_RANGES:
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)

        if total == 0:
            results.append({'SL Range': label, 'Trades': 0, 'Wins': 0, 'Losses': 0, 'Win Rate': '0.0%'})
            continue

        wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL']) &
            (range_trades['TP'] > 0)
        ])
        losses = total - wins
        win_rate = (wins / total) * 100

        results.append({
            'SL Range': label,
            'Trades': total,
            'Wins': wins,
            'Losses': losses,
            'Win Rate': f'{win_rate:.1f}%',
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

    Win condition: Pullback < SL AND TP > 0.
    Trades with TP == 0 (no profit target reached) go into the 0-10 bucket.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: TP Range, Trades, Wins, Losses, Win Rate
    """
    results = []

    for label, low, high in TP_RANGES:
        range_trades = df[(df['TP'] >= low) & (df['TP'] < high)]
        total = len(range_trades)

        if total == 0:
            results.append({'TP Range': label, 'Trades': 0, 'Wins': 0, 'Losses': 0, 'Win Rate': '0.0%'})
            continue

        wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL']) &
            (range_trades['TP'] > 0)
        ])
        losses = total - wins
        win_rate = (wins / total) * 100

        results.append({
            'TP Range': label,
            'Trades': total,
            'Wins': wins,
            'Losses': losses,
            'Win Rate': f'{win_rate:.1f}%',
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
