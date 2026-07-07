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
from typing import Dict, List, Optional, Tuple, Callable


# RRR ratios to test
RRR_RATIOS = [1, 2]

# Extra pip buffer values to test
BUFFER_PIPS = [0, 1, 2, 3]


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


def create_html_table(df: pd.DataFrame, sort_id: Optional[str] = None) -> str:
    """
    Create a dark-mode HTML table with styled formatting.

    Args:
        df: DataFrame to convert to HTML table
        sort_id: When set, gives the table this DOM id and makes every column
            whose cells carry a percentage (e.g. a "Win Rate" of "23.4%")
            click-to-sort by that percentage, descending only.

    Returns:
        HTML string with styled table
    """
    import re

    if df.empty:
        return "<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No profitable strategies found</p>"

    pct_re = re.compile(r"\d+(?:\.\d+)?%")
    sortable_cols = set()
    if sort_id:
        sortable_cols = {
            col for col in df.columns
            if any(pct_re.search(str(v)) for v in df[col])
        }

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
        .analysis-table th.sortable {
            cursor: pointer;
            user-select: none;
        }
        .analysis-table th.sortable:hover {
            background-color: #3a3a3a;
        }
        .analysis-table th.sorted-desc {
            color: #4ade80;
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
    """

    if sort_id:
        html += """
    <script>
        function sortAnalysisTable(tableId, colIndex, th) {
            var table = document.getElementById(tableId);
            var tbody = table.tBodies[0];
            var rows = Array.prototype.slice.call(tbody.rows);
            function pct(row) {
                var m = row.cells[colIndex].textContent.match(/([\\d.]+)%/);
                return m ? parseFloat(m[1]) : -1;
            }
            rows.sort(function(a, b) { return pct(b) - pct(a); });
            rows.forEach(function(r) { tbody.appendChild(r); });
            var headers = table.tHead.rows[0].cells;
            for (var i = 0; i < headers.length; i++) {
                headers[i].classList.remove('sorted-desc');
            }
            th.classList.add('sorted-desc');
        }
    </script>
    """

    id_attr = f' id="{sort_id}"' if sort_id else ""
    html += f'<table class="analysis-table"{id_attr}>\n        <thead>\n            <tr>\n'

    for idx, col in enumerate(df.columns):
        if col in sortable_cols:
            html += (
                f'<th class="sortable" title="Sort by win rate (desc)" '
                f'onclick="sortAnalysisTable(\'{sort_id}\', {idx}, this)">{col} ↓</th>'
            )
        else:
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
MAX_SL_VALUES = [0, 10, 15, 20]
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


def _one_h_location_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only trades that were taken per the "1H Location" column (value TRUE).

    Handles the column being parsed as a bool or a string, and returns no trades
    if the column is absent (e.g. legacy data without it).
    """
    if "1H Location" not in df.columns:
        return df.iloc[0:0]
    return df[df["1H Location"].astype(str).str.upper() == "TRUE"]


def get_buffer_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get key strategies to test with SL buffers.

    "Fixed SL X" replaces SL with X and runs with buffer 0 only.
    """
    strategies: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("All Trades", lambda df: df),
        ("1H Aligned", lambda df: df[df["Direction"] == df["1H"]]),
        ("1H Against", lambda df: df[df["Direction"] != df["1H"]]),
        ("1H Location", _one_h_location_filter),
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


def _display_analysis_table(df: pd.DataFrame, title: str, strategy_names: List[str],
                            sort_id: Optional[str] = None):
    """
    Display a buffer analysis table for given strategies.

    Args:
        df: DataFrame with trading data
        title: Table title
        strategy_names: Strategy names to include
        sort_id: When set, make the table's win-rate column click-to-sort (desc)
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
        html_table = create_html_table(stats_df, sort_id=sort_id)
        display(HTML(html_table))


def display_analysis_strategies(df: pd.DataFrame):
    """
    Display buffer analysis for every configured strategy in a single table.

    The Win Rate column header is click-to-sort by win rate, descending.
    """
    names = [name for name, _ in get_buffer_strategies()]
    _display_analysis_table(df, "Strategies", names, sort_id="strategies-table")


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
] + [
    ("1-10", 1, 10),
    ("2-10", 2, 10),
    ("3-10", 3, 10),
    ("4-10", 4, 10),
    ("5-10", 5, 10),
]


def calculate_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for SL pip ranges.

    Win (Notation): TP > 0.
    Win (Notation 1:2 RRR): TP >= 2 * SL.
    Win (Notation 1:3 RRR): TP >= 3 * SL.
    Win (Notation 1:4 RRR): TP >= 4 * SL.
    The Pullback < SL condition is intentionally not checked in any column.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: SL Range, Trades, Notation,
        Notation (1:2 RRR), Notation (1:3 RRR), Notation (1:4 RRR)
    """
    results = []

    for label, low, high in SL_RANGES:
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)

        if total == 0:
            results.append({
                'SL Range': label,
                'Trades': 0,
                'Notation': _format_wl(0, 0, 0),
                'Notation (1:2 RRR)': _format_wl(0, 0, 0),
                'Notation (1:3 RRR)': _format_wl(0, 0, 0),
                'Notation (1:4 RRR)': _format_wl(0, 0, 0),
            })
            continue

        wins = len(range_trades[range_trades['TP'] > 0])
        wins_2r = len(range_trades[range_trades['TP'] >= 2 * range_trades['SL']])
        wins_3r = len(range_trades[range_trades['TP'] >= 3 * range_trades['SL']])
        wins_4r = len(range_trades[range_trades['TP'] >= 4 * range_trades['SL']])

        results.append({
            'SL Range': label,
            'Trades': total,
            'Notation': _format_wl(wins, total - wins, total),
            'Notation (1:2 RRR)': _format_wl(wins_2r, total - wins_2r, total),
            'Notation (1:3 RRR)': _format_wl(wins_3r, total - wins_3r, total),
            'Notation (1:4 RRR)': _format_wl(wins_4r, total - wins_4r, total),
        })

    return pd.DataFrame(results)


def _create_sl_sortable_table(df: pd.DataFrame, table_id: str) -> str:
    """
    Build an SL Range table HTML with click-to-sort headers on every win-rate
    column. A column is sortable when its cells carry a win-rate percentage
    (e.g. "42W - 61L (40.8%)") - this covers both the "Notation*" columns and
    the buffer columns ("1 pip", "2 pips", "3 pips"). Clicking a header sorts
    rows by that column's win rate, descending only (each click re-sorts DESC).

    Args:
        df: DataFrame with an 'SL Range' column and one or more win-rate columns
        table_id: Unique DOM id for this table (avoids clashes when several
            sortable tables render in the same notebook)

    Returns:
        HTML string with a sortable, dark-mode styled table
    """
    import re

    if df.empty:
        return "<p style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>No data</p>"

    columns = list(df.columns)
    win_rate_re = re.compile(r"\(\d+(?:\.\d+)?%\)")
    sortable_cols = {
        col for col in columns
        if any(win_rate_re.search(str(v)) for v in df[col])
    }

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
        .analysis-table th.sortable {
            cursor: pointer;
            user-select: none;
        }
        .analysis-table th.sortable:hover {
            background-color: #3a3a3a;
        }
        .analysis-table th.sorted-desc {
            color: #4ade80;
        }
    </style>
    <script>
        function sortSlRange(tableId, colIndex, th) {
            var table = document.getElementById(tableId);
            var tbody = table.tBodies[0];
            var rows = Array.prototype.slice.call(tbody.rows);
            function pct(row) {
                var m = row.cells[colIndex].textContent.match(/\\(([\\d.]+)%\\)/);
                return m ? parseFloat(m[1]) : -1;
            }
            rows.sort(function(a, b) { return pct(b) - pct(a); });
            rows.forEach(function(r) { tbody.appendChild(r); });
            var headers = table.tHead.rows[0].cells;
            for (var i = 0; i < headers.length; i++) {
                headers[i].classList.remove('sorted-desc');
            }
            th.classList.add('sorted-desc');
        }
    </script>
    """
    html += f'<table class="analysis-table" id="{table_id}">\n        <thead>\n            <tr>\n'

    for idx, col in enumerate(columns):
        if col in sortable_cols:
            html += (
                f'<th class="sortable" title="Sort by win rate (desc)" '
                f'onclick="sortSlRange(\'{table_id}\', {idx}, this)">{col} ↓</th>'
            )
        else:
            html += f"<th>{col}</th>"

    html += "\n            </tr>\n        </thead>\n        <tbody>\n"

    for _, row in df.iterrows():
        html += "            <tr>\n"
        for col in columns:
            html += f"                <td>{row[col]}</td>\n"
        html += "            </tr>\n"

    html += "        </tbody>\n    </table>\n"
    return html


def display_analysis_sl(df: pd.DataFrame):
    """
    Display win/loss statistics broken down by SL pip range.

    Each Notation column header is click-to-sort by win rate, descending.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>SL Range Statistics</h2>"
    display(HTML(title_html))

    stats_df = calculate_sl_statistics(df)
    html_table = _create_sl_sortable_table(stats_df, "sl-range-stats")
    display(HTML(html_table))


SL_BUFFER_COLS = [
    ("1 pip", 1.0),
    ("2 pips", 2.0),
    ("3 pips", 3.0),
]


def calculate_sl_buffer_impact_statistics(df: pd.DataFrame, rrr: int = 1) -> pd.DataFrame:
    """
    Calculate win/loss statistics for SL pip ranges at a given RRR, showing the
    impact of padding the stop loss with a safety buffer.

    The reward target equals RRR times the risk. Adding a buffer widens the stop
    to SL + buffer. A trade wins only if it BOTH survives the (widened) stop and
    reaches the target:
        Pullback < SL + buffer   AND   TP >= RRR * (SL + buffer)

    The survival check matters here: a buffer's whole purpose is to keep trades
    from being stopped out, so ignoring Pullback would both inflate win rates and
    hide the buffer's actual effect. This matches _calculate_stats_with_buffer.

    Columns (for RRR = R):
        Notation: Pullback < SL     AND TP >= R * SL         (no buffer)
        1 pip:    Pullback < SL + 1 AND TP >= R * (SL + 1)
        2 pips:   Pullback < SL + 2 AND TP >= R * (SL + 2)
        3 pips:   Pullback < SL + 3 AND TP >= R * (SL + 3)

    Args:
        df: DataFrame with trading data
        rrr: Risk-reward ratio (1 for 1:1, 2 for 1:2, 3 for 1:3)

    Returns:
        DataFrame with columns: SL Range, Trades, Notation, 1 pip, 2 pips, 3 pips
    """
    results = []

    for label, low, high in SL_RANGES:
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)

        row = {'SL Range': label, 'Trades': total}

        if total == 0:
            row['Notation'] = _format_wl(0, 0, 0)
            for col, _ in SL_BUFFER_COLS:
                row[col] = _format_wl(0, 0, 0)
            results.append(row)
            continue

        def _wins(buffer: float) -> int:
            effective_sl = range_trades['SL'] + buffer
            return len(range_trades[
                (range_trades['Pullback'] < effective_sl)
                & (range_trades['TP'] >= rrr * effective_sl)
            ])

        wins = _wins(0.0)
        row['Notation'] = _format_wl(wins, total - wins, total)

        for col, buffer in SL_BUFFER_COLS:
            buf_wins = _wins(buffer)
            row[col] = _format_wl(buf_wins, total - buf_wins, total)

        results.append(row)

    return pd.DataFrame(results)


def display_analysis_sl_buffer_impact(df: pd.DataFrame, rrr: int = 1):
    """
    Display win/loss statistics by SL pip range at the given RRR, with the impact
    of padding the stop with a 1, 2, or 3 pip safety buffer.

    Each win-rate column header (Notation, 1 pip, 2 pips, 3 pips) is
    click-to-sort by win rate, descending.

    Args:
        df: DataFrame with trading data
        rrr: Risk-reward ratio (1 for 1:1, 2 for 1:2, 3 for 1:3)
    """
    from IPython.display import display, HTML

    title_html = (
        f"<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>"
        f"SL Range Buffer Impact (1:{rrr} RRR)</h2>"
    )
    display(HTML(title_html))

    stats_df = calculate_sl_buffer_impact_statistics(df, rrr)
    html_table = _create_sl_sortable_table(stats_df, f"sl-buffer-impact-1-{rrr}")
    display(HTML(html_table))


def calculate_sl_vs_buffer_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare 1:1 RRR levers: each alone, then the SL floors combined with a buffer.

    * Limiting SL ("0-X SL" caps and "X-10 SL" floors): take only trades whose
      SL is inside the range, with no buffer. Win: TP >= SL.
    * Adding buffer ("N pip buffer"): take every trade, padding the stop by
      N pips. Win: TP >= SL + N.
    * Combined ("X-10 SL and N pip buffer"): take only floored-SL trades AND
      pad the stop by N pips. Win: TP >= SL + N.

    The Pullback < SL condition is intentionally not checked (matching the
    other SL tables).

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Hypothesis, Trades, Notation
    """
    results = []

    # Limiting SL: cumulative 0-X caps then X-10 floors (no buffer),
    # 1:1 win = TP >= SL.
    for label, low, high in SL_RANGES:
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)
        wins = len(range_trades[range_trades['TP'] >= range_trades['SL']]) if total else 0
        results.append({
            'Hypothesis': f'{label} SL',
            'Trades': total,
            'Notation': _format_wl(wins, total - wins, total),
        })

    # Adding buffer: every trade (no SL cap), 1:1 win = TP >= SL + buffer.
    all_trades = df[df['SL'].notna()]
    total = len(all_trades)
    for col, buffer in SL_BUFFER_COLS:
        wins = len(all_trades[all_trades['TP'] >= all_trades['SL'] + buffer]) if total else 0
        results.append({
            'Hypothesis': f'{col} buffer',
            'Trades': total,
            'Notation': _format_wl(wins, total - wins, total),
        })

    # Combined: SL floor AND buffer, 1:1 win = TP >= SL + buffer.
    for label, low, high in SL_RANGES:
        if low == 0:  # only the X-10 floors
            continue
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)
        for col, buffer in SL_BUFFER_COLS:
            wins = len(range_trades[range_trades['TP'] >= range_trades['SL'] + buffer]) if total else 0
            results.append({
                'Hypothesis': f'{label} SL and {col} buffer',
                'Trades': total,
                'Notation': _format_wl(wins, total - wins, total),
            })

    return pd.DataFrame(results)


def display_analysis_sl_vs_buffer(df: pd.DataFrame):
    """
    Display a head-to-head comparison of limiting SL versus adding a stop
    buffer at 1:1 RRR, each lever alone and then the two combined.

    The Notation column header is click-to-sort by win rate, descending.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Limiting SL vs Adding Buffer (1:1 RRR)</h2>"
    display(HTML(title_html))

    stats_df = calculate_sl_vs_buffer_statistics(df)
    html_table = _create_sl_sortable_table(stats_df, "sl-vs-buffer")
    display(HTML(html_table))


PULLBACK_ENTRY_PIPS = [1, 2, 3]


def _format_wlm(wins: int, losses: int, missed: int) -> str:
    """
    Format winners/losers/missed-winners into '1W – 2L – 3M (50.0%)'.

    Win rate is over ENTERED trades only (W / (W + L)); missed winners never
    filled so they are excluded from the rate.
    """
    entered = wins + losses
    win_rate = (wins / entered * 100) if entered > 0 else 0.0
    return f"{wins}W – {losses}L – {missed}M ({win_rate:.1f}%)"


def calculate_pullback_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate limit-order pullback-entry opportunities at 1, 2 and 3 pip pullbacks.

    A limit order placed N pips into the pullback fills only if price pulled back
    at least N pips (Pullback >= N). A trade is a real winner only if it both
    survives the safe stop and reaches a profitable target (Pullback < SL AND
    TP > 0) - a trade that reached TP but pulled back past its SL would have been
    stopped out, so it does not count as a win.

    For each pullback level N:
        Entered (Trades) = Pullback >= N
        W = entered AND Pullback < SL AND TP > 0   (filled + real winner)
        L = entered - W                            (filled but stopped / unprofitable)
        M = missed winners = Pullback < N AND Pullback < SL AND TP > 0
            (real winners that never pulled back N pips, so the limit never filled)

    Missed winners are excluded from Trades because those trades were never entered.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Pullback, Trades, Notation
    """
    results = []

    real_winner = (df['Pullback'] < df['SL']) & (df['TP'] > 0)

    for n in PULLBACK_ENTRY_PIPS:
        entered = df['Pullback'] >= n
        wins = int((entered & real_winner).sum())
        entered_total = int(entered.sum())
        losses = entered_total - wins
        missed = int((~entered & real_winner).sum())

        results.append({
            'Pullback': f"{n} pip" if n == 1 else f"{n} pips",
            'Trades': entered_total,
            'Notation': _format_wlm(wins, losses, missed),
        })

    return pd.DataFrame(results)


def display_analysis_pullback(df: pd.DataFrame):
    """
    Display limit-order pullback-entry opportunities at 1/2/3 pip pullbacks.

    Notation is W - L - M: winners, losers and missed winners (real winners that
    never pulled back far enough for the limit to fill).

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Pullback Entry Opportunities</h2>"
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
