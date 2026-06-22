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
BUFFER_PIPS = [0, 1, 2, 3, 4, 5]


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
            elif col == "Win Rate":
                try:
                    wr = float(str(value).rstrip("%"))
                    rrr_val = 1.0
                    if "RRR" in df.columns:
                        try:
                            rrr_val = float(str(row["RRR"]).split(":")[-1])
                        except (ValueError, IndexError):
                            pass
                    breakeven = 100.0 / (1 + rrr_val)
                    css_class = "positive-edge" if wr > breakeven else "negative-edge"
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


def _calculate_stats_with_buffer(trades: pd.DataFrame, strategy_name: str, buffer: float, rrr_ratio: float = 1) -> Dict:
    """
    Calculate trading statistics with extra pips added to SL.

    With buffer, effective SL = SL + buffer. Trade survives if Pullback < effective SL.
    Trade wins if TP >= rrr_ratio * effective SL.
    """
    rrr_label = f"1:{rrr_ratio:g}"
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "Strategy": strategy_name,
            "Buffer": f"+{buffer}",
            "Min SL": 0,
            "Max SL": 0,
            "RRR": rrr_label,
            "Trades": 0,
            "Notation": "0W – 0L",
            "Win Rate": "0.0%",
        }

    effective_sl = trades["SL"] + buffer
    winning_mask = (trades["Pullback"] < effective_sl) & (trades["TP"] >= rrr_ratio * effective_sl)

    wins = winning_mask.sum()
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100

    return {
        "Strategy": strategy_name,
        "Buffer": f"+{buffer}",
        "Min SL": 0,
        "Max SL": 0,
        "RRR": rrr_label,
        "Trades": total_trades,
        "Notation": f"{wins}W – {losses}L",
        "Win Rate": f"{win_rate:.1f}%",
    }


MIN_SL_VALUES = [0, 1, 2, 3]
MAX_SL_VALUES = [0, 5, 10, 15, 20]
FIXED_SL_STRATEGY_VALUES = list(range(2, 11))


def _apply_min_sl(df: pd.DataFrame, min_sl: int) -> pd.DataFrame:
    """Keep only trades whose original SL is strictly greater than min_sl pips."""
    return df if min_sl == 0 else df[df["SL"] > min_sl]


def _apply_max_sl(df: pd.DataFrame, max_sl: int) -> pd.DataFrame:
    """Keep only trades whose original SL is less than or equal to max_sl (0 disables)."""
    return df if max_sl == 0 else df[df["SL"] <= max_sl]


def _fixed_sl_filter(x: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a filter that replaces the SL column with a fixed value of x pips."""
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["SL"] = float(x)
        return out
    return _filter


def get_buffer_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get key strategies to test with SL buffers.

    "Fixed SL X" replaces SL with X and runs with buffer 0 only.
    """
    strategies: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("All Trades", lambda df: df),
        ("1H Aligned", lambda df: df[df["Direction"] == df["1H"]]),
        ("1H Against", lambda df: df[df["Direction"] != df["1H"]]),
    ]
    strategies.extend(
        (f"Fixed SL {x}", _fixed_sl_filter(x)) for x in FIXED_SL_STRATEGY_VALUES
    )
    return strategies


def _buffers_for(strategy_name: str) -> List[float]:
    """Fixed-SL strategies only run with buffer 0; everything else uses BUFFER_PIPS."""
    return [0] if strategy_name.startswith("Fixed SL ") else BUFFER_PIPS


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
        for min_sl in MIN_SL_VALUES:
            for max_sl in MAX_SL_VALUES:
                gated = _apply_max_sl(_apply_min_sl(df, min_sl), max_sl)
                filtered_df = filter_func(gated)
                for rrr in RRR_RATIOS:
                    for buffer in _buffers_for(strategy_name):
                        stats = _calculate_stats_with_buffer(filtered_df, strategy_name, buffer, rrr)
                        stats["Min SL"] = min_sl
                        stats["Max SL"] = max_sl
                        results.append(stats)

    result_df = pd.DataFrame(results)
    result_df = _sort_strategy_rows(result_df)
    return result_df


def _sort_strategy_rows(result_df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by Strategy (natural order, numbers compared numerically) then RRR ascending."""
    import re

    def strategy_key(name: str):
        return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', str(name))]

    def rrr_key(rrr: str):
        try:
            return float(str(rrr).split(':')[-1])
        except (ValueError, IndexError):
            return float('inf')

    sort_index = sorted(
        result_df.index,
        key=lambda i: (
            strategy_key(result_df.at[i, 'Strategy']),
            rrr_key(result_df.at[i, 'RRR']),
            int(result_df.at[i, 'Min SL']) if 'Min SL' in result_df.columns else 0,
            int(result_df.at[i, 'Max SL']) if 'Max SL' in result_df.columns else 0,
        ),
    )
    return result_df.loc[sort_index].reset_index(drop=True)


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
        for min_sl in MIN_SL_VALUES:
            for max_sl in MAX_SL_VALUES:
                gated = _apply_max_sl(_apply_min_sl(df, min_sl), max_sl)
                filtered_df = filter_func(gated)
                for rrr in RRR_RATIOS:
                    for buffer in _buffers_for(strategy_name):
                        stats = _calculate_stats_with_buffer(filtered_df, strategy_name, buffer, rrr)
                        stats["Min SL"] = min_sl
                        stats["Max SL"] = max_sl
                        results.append(stats)

    result_df = pd.DataFrame(results)
    result_df = _sort_strategy_rows(result_df)
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


def display_analysis_strategies(df: pd.DataFrame):
    """
    Display buffer analysis for every configured strategy in a single table,
    followed by a bar chart of each row's win rate (0–60%).
    """
    import matplotlib.pyplot as plt

    names = [name for name, _ in get_buffer_strategies()]
    _display_analysis_table(df, "Strategies", names)

    stats_df = _calculate_buffer_statistics_filtered(df, names)
    if stats_df.empty:
        return

    win_rates = stats_df['Win Rate'].str.rstrip('%').astype(float).tolist()
    labels = [
        f"{row['Strategy']} {row['Buffer']} min{row['Min SL']} max{row['Max SL']} {row['RRR']}"
        for _, row in stats_df.iterrows()
    ]
    breakevens = [
        100.0 / (1 + float(str(rrr).split(':')[-1]))
        for rrr in stats_df['RRR']
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.25), 5))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    colors = ['#4ade80' if wr > be else '#f87171' for wr, be in zip(win_rates, breakevens)]
    ax.bar(range(len(labels)), win_rates, color=colors, edgecolor='#404040')

    # Per-bar breakeven markers (50% for 1:1, 33.3% for 1:2, ...).
    for i, be in enumerate(breakevens):
        ax.hlines(be, i - 0.4, i + 0.4, colors='#e0e0e0', linewidth=1, alpha=0.6)

    ax.set_ylim(0, 60)
    ax.set_ylabel('Win Rate (%)', color='#e0e0e0')
    ax.set_title('Strategies — Win Rate', color='#e0e0e0', loc='left')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, color='#e0e0e0', fontsize=8)
    ax.tick_params(axis='y', colors='#e0e0e0')
    for spine in ax.spines.values():
        spine.set_color('#404040')
    ax.grid(axis='y', color='#404040', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


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
    (f"0-{x}", 0, x) for x in range(1, 11)
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
    # 80% of trades
    ("0-35", 0, 35),
    # 20% of trades
    ("35+", 35, float("inf")),
]


def calculate_tp_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trade counts per TP pip range.

    Trades column shows "X of Y" where X is trades in the range and
    Y is total profitable trades (TP > 0) in the entire dataset.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: TP Range, Trades
    """
    total_profitable = int((df['TP'] > 0).sum())
    results = []

    for label, low, high in TP_RANGES:
        range_trades = df[(df['TP'] > 0) & (df['TP'] >= low) & (df['TP'] < high)]
        results.append({
            'TP Range': label,
            'Trades': f"{len(range_trades)} of {total_profitable}",
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
# SL × Pullback correlation analyses
# ---------------------------------------------------------------------------

SL_PB_SL_RANGES = [
    ("0-2", 0, 2),
    ("2-3", 2, 3),
    ("3-5", 3, 5),
    ("5-10", 5, 10),
    ("10+", 10, float("inf")),
]

SL_PB_PB_RANGES = [
    ("0-1", 0, 1),
    ("1-2", 1, 2),
    ("2-3", 2, 3),
    ("3-5", 3, 5),
    ("5-10", 5, 10),
    ("10+", 10, float("inf")),
]

SL_PB_BUFFERS = [1.0, 2.0, 3.0]


def calculate_sl_pb_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tab of trade counts by SL range (rows) and Pullback range (columns).

    Each cell shows "N (P%)" where P is the percentage of that SL row.

    Returns:
        DataFrame with columns: SL Range, <one per PB range>, Total
    """
    pb_labels = [label for label, _, _ in SL_PB_PB_RANGES]
    results = []

    for sl_label, sl_lo, sl_hi in SL_PB_SL_RANGES:
        sl_trades = df[(df['SL'] >= sl_lo) & (df['SL'] < sl_hi)]
        total = len(sl_trades)
        row = {'SL Range': sl_label}
        for pb_label, pb_lo, pb_hi in SL_PB_PB_RANGES:
            n = len(sl_trades[(sl_trades['Pullback'] >= pb_lo) & (sl_trades['Pullback'] < pb_hi)])
            pct = (n / total * 100) if total > 0 else 0.0
            row[pb_label] = f"{n} ({pct:.0f}%)"
        row['Total'] = total
        results.append(row)

    return pd.DataFrame(results, columns=['SL Range', *pb_labels, 'Total'])


def display_analysis_sl_pb_crosstab(df: pd.DataFrame):
    """
    Display a cross-tab of trade counts: SL ranges (rows) × Pullback ranges (cols).

    Useful for spotting where the mass of (SL, Pullback) pairs sits.
    """
    from IPython.display import display, HTML

    title_html = (
        "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>"
        "SL × Pullback Cross-Tab</h2>"
    )
    display(HTML(title_html))

    stats_df = calculate_sl_pb_crosstab(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


def calculate_sl_pb_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-SL-bucket distribution stats of Pullback.

    Returns:
        DataFrame with columns: SL Range, Trades, PB Mean, PB Median, PB p75,
        PB p90, PB Max, PB/SL Median, % PB ≥ SL
    """
    results = []

    for sl_label, sl_lo, sl_hi in SL_PB_SL_RANGES:
        sl_trades = df[(df['SL'] >= sl_lo) & (df['SL'] < sl_hi)]
        n = len(sl_trades)

        if n == 0:
            results.append({
                'SL Range': sl_label,
                'Trades': 0,
                'PB Mean': '-',
                'PB Median': '-',
                'PB p75': '-',
                'PB p90': '-',
                'PB Max': '-',
                'PB/SL Median': '-',
                '% PB ≥ SL': '-',
            })
            continue

        pb = sl_trades['Pullback']
        ratio = sl_trades['Pullback'] / sl_trades['SL']
        loss_pct = (sl_trades['Pullback'] >= sl_trades['SL']).sum() / n * 100

        results.append({
            'SL Range': sl_label,
            'Trades': n,
            'PB Mean': f"{pb.mean():.2f}",
            'PB Median': f"{pb.median():.2f}",
            'PB p75': f"{pb.quantile(0.75):.2f}",
            'PB p90': f"{pb.quantile(0.90):.2f}",
            'PB Max': f"{pb.max():.2f}",
            'PB/SL Median': f"{ratio.median():.2f}",
            '% PB ≥ SL': f"{loss_pct:.0f}%",
        })

    return pd.DataFrame(results)


def display_analysis_sl_pb_distribution(df: pd.DataFrame):
    """
    Display per-SL-bucket pullback summary stats: mean, median, p75, p90, max,
    PB/SL ratio, and loss rate (PB ≥ SL).
    """
    from IPython.display import display, HTML

    title_html = (
        "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>"
        "Pullback Distribution per SL Bucket</h2>"
    )
    display(HTML(title_html))

    stats_df = calculate_sl_pb_distribution(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


def calculate_sl_pb_buffer_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-SL-bucket loss rate and how often a +N pip buffer flips a loss to a win.

    A trade is a loss when Pullback ≥ SL. A buffer of N "saves" a trade when
    SL ≤ Pullback < SL + N. Save percentages are reported relative to all trades
    in the SL bucket, so loss rate and save rates are directly comparable.

    Returns:
        DataFrame with columns: SL Range, Trades, Loss Rate, +1 Saves, +2 Saves, +3 Saves
    """
    results = []

    for sl_label, sl_lo, sl_hi in SL_PB_SL_RANGES:
        sl_trades = df[(df['SL'] >= sl_lo) & (df['SL'] < sl_hi)]
        n = len(sl_trades)

        if n == 0:
            row = {'SL Range': sl_label, 'Trades': 0, 'Loss Rate': '-'}
            for buf in SL_PB_BUFFERS:
                row[f'+{buf:g} Saves'] = '-'
            results.append(row)
            continue

        loss_mask = sl_trades['Pullback'] >= sl_trades['SL']
        loss_pct = loss_mask.sum() / n * 100

        row = {
            'SL Range': sl_label,
            'Trades': n,
            'Loss Rate': f"{loss_pct:.0f}%",
        }
        for buf in SL_PB_BUFFERS:
            saved = loss_mask & (sl_trades['Pullback'] < sl_trades['SL'] + buf)
            row[f'+{buf:g} Saves'] = f"{saved.sum() / n * 100:.0f}%"
        results.append(row)

    return pd.DataFrame(results)


def display_analysis_sl_pb_buffer_impact(df: pd.DataFrame):
    """
    Display per-SL-bucket loss rate and the % of trades that a +1/+2/+3 pip
    buffer would flip from a loss to a win.
    """
    from IPython.display import display, HTML

    title_html = (
        "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>"
        "Buffer Impact per SL Bucket</h2>"
    )
    display(HTML(title_html))

    stats_df = calculate_sl_pb_buffer_impact(df)
    html_table = create_html_table(stats_df)
    display(HTML(html_table))


