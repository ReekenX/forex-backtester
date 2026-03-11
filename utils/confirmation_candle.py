"""
1M Confirmation Candle Analysis Module

Analyzes trading strategies based on EMA(50), EMA(200) alignment, and engulfing candle patterns.
All strategies are evaluated at 1:1 and 1:2 RRR.

CSV columns: Date, Weekday, Trade, Direction, EMA(50), EMA(200), Engulfing, SL, Pullback, TP, R
"""

import pandas as pd
from typing import Dict, List, Tuple, Callable


# RRR ratios to test
RRR_RATIOS = [1, 2]

# Extra pip buffer values to test
BUFFER_PIPS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def load_data(filepath: str = "../data/eurusd_2026_1m_confirmation_candle.csv") -> pd.DataFrame:
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

    # === EMA alignment strategies ===
    strategies.extend([
        ("EMA(50) Aligned", lambda df: df[df["Direction"] == df["EMA(50)"]]),
        ("EMA(50) Against", lambda df: df[df["Direction"] != df["EMA(50)"]]),
        ("EMA(200) Aligned", lambda df: df[df["Direction"] == df["EMA(200)"]]),
        ("EMA(200) Against", lambda df: df[df["Direction"] != df["EMA(200)"]]),
        ("Both EMAs Aligned", lambda df: df[
            (df["Direction"] == df["EMA(50)"]) & (df["Direction"] == df["EMA(200)"])
        ]),
        ("Both EMAs Against", lambda df: df[
            (df["Direction"] != df["EMA(50)"]) & (df["Direction"] != df["EMA(200)"])
        ]),
        ("EMA(50) Aligned + EMA(200) Against", lambda df: df[
            (df["Direction"] == df["EMA(50)"]) & (df["Direction"] != df["EMA(200)"])
        ]),
        ("EMA(50) Against + EMA(200) Aligned", lambda df: df[
            (df["Direction"] != df["EMA(50)"]) & (df["Direction"] == df["EMA(200)"])
        ]),
    ])

    # === EMA market state ===
    strategies.extend([
        ("EMAs Agree (trending)", lambda df: df[df["EMA(50)"] == df["EMA(200)"]]),
        ("EMAs Disagree (transition)", lambda df: df[df["EMA(50)"] != df["EMA(200)"]]),
        ("EMAs Agree + Direction Aligned", lambda df: df[
            (df["EMA(50)"] == df["EMA(200)"]) & (df["Direction"] == df["EMA(50)"])
        ]),
        ("EMAs Agree + Direction Against", lambda df: df[
            (df["EMA(50)"] == df["EMA(200)"]) & (df["Direction"] != df["EMA(50)"])
        ]),
    ])

    # === Engulfing strategies ===
    strategies.extend([
        ("Engulfing: Yes", lambda df: df[df["Engulfing"] == "Yes"]),
        ("Engulfing: No", lambda df: df[df["Engulfing"] == "No"]),
        ("Engulfing: Similar", lambda df: df[df["Engulfing"] == "Similar"]),
        ("Engulfing: Yes or Similar", lambda df: df[df["Engulfing"].isin(["Yes", "Similar"])]),
        ("Has Engulfing Data", lambda df: df[df["Engulfing"].notna()]),
    ])

    # === Engulfing + EMA combinations ===
    engulfing_filters = [
        ("Engulfing: Yes", lambda df: df[df["Engulfing"] == "Yes"]),
        ("Engulfing: No", lambda df: df[df["Engulfing"] == "No"]),
        ("Engulfing: Yes or Similar", lambda df: df[df["Engulfing"].isin(["Yes", "Similar"])]),
    ]

    ema_filters = [
        ("EMA(50) Aligned", lambda df: df[df["Direction"] == df["EMA(50)"]]),
        ("EMA(200) Aligned", lambda df: df[df["Direction"] == df["EMA(200)"]]),
        ("Both EMAs Aligned", lambda df: df[
            (df["Direction"] == df["EMA(50)"]) & (df["Direction"] == df["EMA(200)"])
        ]),
    ]

    for eng_name, eng_func in engulfing_filters:
        for ema_name, ema_func in ema_filters:
            strategies.append((
                f"{eng_name} + {ema_name}",
                lambda df, ef=eng_func, emf=ema_func: emf(ef(df))
            ))

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

    # === EMA(50) Aligned + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"EMA(50) Aligned + {sl_name}",
            lambda df, f=sl_func: f(df[df["Direction"] == df["EMA(50)"]])
        ))

    # === EMA(200) Aligned + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"EMA(200) Aligned + {sl_name}",
            lambda df, f=sl_func: f(df[df["Direction"] == df["EMA(200)"]])
        ))

    # === Both EMAs Aligned + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"Both EMAs Aligned + {sl_name}",
            lambda df, f=sl_func: f(df[
                (df["Direction"] == df["EMA(50)"]) & (df["Direction"] == df["EMA(200)"])
            ])
        ))

    # === EMA(50) Against + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"EMA(50) Against + {sl_name}",
            lambda df, f=sl_func: f(df[df["Direction"] != df["EMA(50)"]])
        ))

    # === EMA(200) Against + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"EMA(200) Against + {sl_name}",
            lambda df, f=sl_func: f(df[df["Direction"] != df["EMA(200)"]])
        ))

    # === Both EMAs Against + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"Both EMAs Against + {sl_name}",
            lambda df, f=sl_func: f(df[
                (df["Direction"] != df["EMA(50)"]) & (df["Direction"] != df["EMA(200)"])
            ])
        ))

    # === Engulfing: Yes + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"Engulfing: Yes + {sl_name}",
            lambda df, f=sl_func: f(df[df["Engulfing"] == "Yes"])
        ))

    # === Engulfing: No + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"Engulfing: No + {sl_name}",
            lambda df, f=sl_func: f(df[df["Engulfing"] == "No"])
        ))

    # === Engulfing: Yes or Similar + SL filters ===
    for sl_name, sl_func in sl_filters:
        strategies.append((
            f"Engulfing: Yes or Similar + {sl_name}",
            lambda df, f=sl_func: f(df[df["Engulfing"].isin(["Yes", "Similar"])])
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

            cls_attr = f' class="{css_class}"' if css_class else ""
            html += f"                <td{cls_attr}>{value}</td>\n"
        html += "            </tr>\n"

    html += """
        </tbody>
    </table>
    """
    return html


def _calculate_stats_with_buffer(trades: pd.DataFrame, strategy_name: str, buffer: float, rrr_ratio: int = 1) -> Dict:
    """
    Calculate trading statistics with extra pips added to SL.

    With buffer, effective SL = SL + buffer. Trade survives if Pullback < effective SL.
    Trade wins if TP >= rrr_ratio * effective SL.

    Args:
        trades: DataFrame containing filtered trades
        strategy_name: Name of the strategy
        buffer: Extra pips to add to SL
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
            "Buffer": f"+{buffer}",
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

    effective_sl = trades["SL"] + buffer

    # Win condition with buffer: Pullback < effective_sl AND TP >= rrr_ratio * effective_sl
    winning_trades = trades[
        (trades["Pullback"] < effective_sl) &
        (trades["TP"] >= rrr_ratio * effective_sl)
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
        "Buffer": f"+{buffer}",
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


def get_buffer_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get key strategies to test with SL buffers.

    Returns:
        List of tuples (strategy_name, filter_function)
    """
    base_strategies = [
        ("All Trades", lambda df: df),
        ("EMA(50) Aligned", lambda df: df[df["Direction"] == df["EMA(50)"]]),
        ("EMA(50) Against", lambda df: df[df["Direction"] != df["EMA(50)"]]),
        ("EMA(200) Aligned", lambda df: df[df["Direction"] == df["EMA(200)"]]),
        ("EMA(200) Against", lambda df: df[df["Direction"] != df["EMA(200)"]]),
        ("Both EMAs Aligned", lambda df: df[
            (df["Direction"] == df["EMA(50)"]) & (df["Direction"] == df["EMA(200)"])
        ]),
        ("Both EMAs Against", lambda df: df[
            (df["Direction"] != df["EMA(50)"]) & (df["Direction"] != df["EMA(200)"])
        ]),
        ("EMAs Agree + Direction Aligned", lambda df: df[
            (df["EMA(50)"] == df["EMA(200)"]) & (df["Direction"] == df["EMA(50)"])
        ]),
        ("EMAs Agree + Direction Against", lambda df: df[
            (df["EMA(50)"] == df["EMA(200)"]) & (df["Direction"] != df["EMA(50)"])
        ]),
        ("Engulfing: Yes", lambda df: df[df["Engulfing"] == "Yes"]),
        ("Engulfing: No", lambda df: df[df["Engulfing"] == "No"]),
        ("Engulfing: Yes or Similar", lambda df: df[df["Engulfing"].isin(["Yes", "Similar"])]),
        ("Has Engulfing Data", lambda df: df[df["Engulfing"].notna()]),
    ]

    sl_caps = [3, 4, 5]
    strategies = list(base_strategies)

    for name, base_func in base_strategies:
        for cap in sl_caps:
            strategies.append((
                f"{name} + SL < {cap}",
                lambda df, f=base_func, c=cap: f(df[df["SL"] < c])
            ))

    return strategies


def calculate_buffer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for key strategies with each SL buffer value at 1:1 and 1:2 RRR.

    For each strategy, tests what happens if extra pips (0, 0.5, 1, 1.5, 2, 3, 4, 5)
    are added to the SL. A wider SL means more trades survive but the target is also higher.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with buffer statistics, sorted by edge descending
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

    # Filter to only show strategies with positive edge
    # result_df = result_df[result_df["edge_value"] > 0].copy()

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


def display_analysis(df: pd.DataFrame):
    """
    Display strategy analysis with SL buffers in a single table.

    Shows each strategy at every buffer level (0, 0.5, 1.0, ..., 5.0 pips).
    Buffer 0 means no extra pips added to SL (original strategy).

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    # Disable Jupyter output scrolling so the full table is visible
    display(HTML("""<style>
        .jp-OutputArea-child { max-height: none !important; }
        .jp-OutputArea-output { max-height: none !important; overflow: visible !important; }
        .output_scroll { box-shadow: none !important; border: none !important; }
    </style>"""))

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>1M Confirmation Candle Analysis</h2>"
    display(HTML(title_html))

    stats_df = calculate_buffer_statistics(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No profitable strategies found</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


def calculate_bruteforce(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bruteforce scan of all buffer (extra SL pips) and RRR combinations.

    Tests buffer values from 0.0 to 10.0 in 0.5 pip steps, combined with
    RRR from 1:1 to 1:5, to find which combination gives the best outcome.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with results for every buffer × RRR combination, sorted by outcome descending
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


def display_bruteforce(df: pd.DataFrame):
    """
    Display bruteforce analysis scanning all buffer × RRR combinations.

    Tests extra SL pips from 0.0 to 10.0 (0.5 steps) with RRR 1:1 to 1:5
    to find which combination gives the best outcome for all trades.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Bruteforce: Buffer × RRR Scan (All Trades)</h2>"
    display(HTML(title_html))

    stats_df = calculate_bruteforce(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data available</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


def _calculate_limit_order_stats(trades: pd.DataFrame, strategy_name: str, rrr_ratio: int = 1) -> Dict:
    """
    Calculate statistics for limit order at half SL price.

    Limit order is placed SL/2 pips from original entry (halfway to SL).
    - Trade triggers only if Pullback >= SL/2
    - New SL from limit entry = SL/2
    - New TP from limit entry = TP + SL/2
    - Trade survives if Pullback < SL (original)
    - Win at 1:N RRR: TP + SL/2 >= N * (SL/2), i.e., TP >= (N-1) * SL/2
    - Minimum broker SL applies: SL/2 >= 1.1, so only trades with SL >= 2.2

    Args:
        trades: DataFrame containing filtered trades
        strategy_name: Name of the strategy
        rrr_ratio: Risk-reward ratio (1 for 1:1, 2 for 1:2)

    Returns:
        Dictionary with calculated statistics
    """
    breakeven = _breakeven_rate(rrr_ratio)
    rrr_label = f"1:{rrr_ratio}"

    # Only trades where SL/2 >= 1.1 (minimum broker SL)
    eligible = trades[trades["SL"] >= 2.2].copy()

    # Only triggered trades: Pullback >= SL/2
    half_sl = eligible["SL"] / 2
    triggered = eligible[eligible["Pullback"] >= half_sl]
    total_trades = len(triggered)

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

    triggered_half_sl = triggered["SL"] / 2

    # Win: survives (Pullback < SL) AND TP reaches target
    winning_trades = triggered[
        (triggered["Pullback"] < triggered["SL"]) &
        (triggered["TP"] >= (rrr_ratio - 1) * triggered_half_sl)
    ]

    # Exclude trades with TP == 0 (no TP means loss)
    winning_trades = winning_trades[winning_trades["TP"] > 0]

    wins = len(winning_trades)
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven
    outcome = (wins * rrr_ratio) - losses

    days_with_wins = winning_trades["Date"].nunique() if "Date" in winning_trades.columns and len(winning_trades) > 0 else 0
    total_days = triggered["Date"].nunique() if "Date" in triggered.columns else 0
    days_pct = (days_with_wins / total_days * 100) if total_days > 0 else 0.0
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


def calculate_limit_order_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for limit order at half SL price across all strategies.

    Places a limit order at SL/2 pullback. Only trades where price pulls back
    deep enough (>= SL/2) are triggered. The new SL is SL/2 and new TP is TP + SL/2.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with limit order statistics, sorted by edge descending
    """
    strategies = get_buffer_strategies()
    results = []

    for strategy_name, filter_func in strategies:
        filtered_df = filter_func(df)
        for rrr in RRR_RATIOS:
            stats = _calculate_limit_order_stats(filtered_df, strategy_name, rrr)
            results.append(stats)

    result_df = pd.DataFrame(results)

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

    result_df = result_df.reset_index(drop=True)

    return result_df


def display_limit_order(df: pd.DataFrame):
    """
    Display limit order at half SL price analysis.

    Strategy: place limit order at SL/2 pullback. If price pulls back deep enough,
    the entry is at a better price with SL/2 risk and TP + SL/2 reward.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Limit Order at Half SL Price</h2>"
    subtitle_html = "<p style='color: #a0a0a0; background-color: #1e1e1e; padding: 0 10px 10px;'>Entry at SL/2 pullback. New SL = SL/2, New TP = TP + SL/2. Only trades with SL &ge; 2.2 (broker min 1.1 pip).</p>"
    display(HTML(title_html + subtitle_html))

    stats_df = calculate_limit_order_statistics(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data available</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))


def display_buffer_analysis(df: pd.DataFrame):
    """
    Display SL buffer analysis - what if extra pips were added to the stop loss.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>SL Buffer Analysis (1:1 RRR) — What if extra pips were added to SL?</h2>"
    display(HTML(title_html))

    stats_df = calculate_buffer_statistics(df)

    if stats_df.empty:
        display(HTML("<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No profitable buffer strategies found</p>"))
    else:
        html_table = create_html_table(stats_df)
        display(HTML(html_table))
