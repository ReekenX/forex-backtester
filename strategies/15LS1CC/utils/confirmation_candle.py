"""
1M Confirmation Candle Analysis Module

Analyzes trading strategies based on stop loss sizing and SL buffers.
All strategies are evaluated at 1:1 and 1:2 RRR.

CSV columns: Date, Weekday, Trade, Direction, SL, Pullback, TP, R

The R column is exported from the spreadsheet with an "R" suffix (e.g. "7R")
and is negative when the trade was stopped out before reaching that target
(e.g. "-6R" means 6R was available but Pullback exceeded SL). Its header cell
holds a computed win-rate value instead of the name "R", so load_data renames
the trailing column.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable


# RRR ratios to test (1:3 is the ceiling - see CLAUDE.md)
RRR_RATIOS = [1, 2, 3]

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

    # The spreadsheet export labels the R column with a computed win-rate cell
    # (e.g. "47.3%") instead of "R", so recover it from the trailing column.
    if "R" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[-1]: "R"})

    if "R" in df.columns:
        df["R"] = df["R"].astype(str).str.replace("R", "", regex=False).str.strip()

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


MIN_SL_VALUES = [0]
MAX_SL_VALUES = [0, 5]
FIXED_SL_STRATEGY_VALUES = list(range(2, 11))
MAX_SL_STRATEGY_VALUES = list(range(3, 11))


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


def _max_sl_filter(x: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a filter that caps the SL at x pips: effective SL = min(SL, x).

    Every trade is kept; a trade whose safe stop is wider than x now uses a
    tighter x-pip stop, so it is stopped out whenever Pullback >= x. A trade
    whose safe stop is already <= x is unchanged.
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["SL"] = out["SL"].clip(upper=float(x))
        return out
    return _filter


def get_buffer_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Get key strategies to test with SL buffers.

    "Fixed SL X" replaces SL with X; "Max SL X" caps SL at X (min(SL, X)).
    Both run with buffer 0 only.
    """
    strategies: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("All Trades", lambda df: df),
    ]
    strategies.extend(
        (f"Fixed SL {x}", _fixed_sl_filter(x)) for x in FIXED_SL_STRATEGY_VALUES
    )
    strategies.extend(
        (f"Max SL {x}", _max_sl_filter(x)) for x in MAX_SL_STRATEGY_VALUES
    )
    return strategies


def _buffers_for(strategy_name: str) -> List[float]:
    """Fixed-SL and Max-SL strategies only run with buffer 0 (a buffer would
    undo the fixed/capped stop); everything else uses BUFFER_PIPS."""
    if strategy_name.startswith("Fixed SL ") or strategy_name.startswith("Max SL "):
        return [0]
    return BUFFER_PIPS


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

    Win: Pullback < SL AND TP > 0 - the trade survived its stop and finished
    profitable.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Day, Trades, Notation, Win Rate
    """
    results = []

    for day in WEEKDAY_ORDER:
        day_trades = df[df['Weekday'] == day]
        total = len(day_trades)

        wins = len(day_trades[
            (day_trades['Pullback'] < day_trades['SL']) &
            (day_trades['TP'] > 0)
        ]) if total else 0
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0.0

        results.append({
            'Day': day,
            'Trades': total,
            'Notation': f"{wins}W - {losses}L",
            'Win Rate': f"{win_rate:.1f}%",
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
    (f"0-{x}", 0, x) for x in range(5, 11)
]

def calculate_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss statistics for SL pip ranges at 1:1 RRR.

    Win: Pullback < SL AND TP >= SL - the trade survived its stop and reached
    a 1:1 target. Skipping the survival check would score trades that were
    stopped out before running to target as wins - the data marks those with a
    negative R - and inflate every win rate. This matches _calculate_stats.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: SL Range, Trades, Notation, Win Rate
    """
    results = []

    for label, low, high in SL_RANGES:
        range_trades = df[(df['SL'] >= low) & (df['SL'] < high)]
        total = len(range_trades)

        wins = len(range_trades[
            (range_trades['Pullback'] < range_trades['SL'])
            & (range_trades['TP'] >= range_trades['SL'])
        ]) if total else 0
        win_rate = (wins / total * 100) if total > 0 else 0.0

        results.append({
            'SL Range': label,
            'Trades': total,
            'Notation': f"{wins}W - {total - wins}L",
            'Win Rate': f"{win_rate:.1f}%",
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
    # Matches a bare "48.6%" and a wrapped "18W - 19L (48.6%)" alike.
    win_rate_re = re.compile(r"\d+(?:\.\d+)?%")
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

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px 10px 0;'>SL Range Statistics</h2>"
    display(HTML(title_html))

    stats_df = calculate_sl_statistics(df)
    html_table = _create_sl_sortable_table(stats_df, "sl-range-stats")
    display(HTML(html_table))


# Pips to shave off the safe stop. 0 means the stop is left as recorded.
SL_REDUCTION_PIPS = [0, 1, 2, 3, 4]


def calculate_sl_reduction_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1:1 win/loss statistics for progressively tighter stops.

    Each row shaves N pips off every trade's safe stop:
        effective SL = SL - N
    A trade wins when it survives that tighter stop and still reaches a 1:1
    target on it:
        Pullback < effective SL   AND   TP >= effective SL

    Example: SL 3.6, Pullback 1.2. At -1 the stop is 2.6 and the trade
    survives; at -3 the stop is 0.6, which the 1.2 pullback takes out, so it
    becomes a loss.

    Reductions that drive the effective stop to zero or below leave no room for
    any pullback, so those trades count as losses - which is what adopting that
    reduction as a rule would actually cost. Note the broker minimum stop is
    1.1 pips, so rows whose effective stop falls under that are informational
    rather than tradeable.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: SL Reduction, Trades, Notation, Win Rate
    """
    total = len(df)
    results = []

    for reduction in SL_REDUCTION_PIPS:
        effective_sl = df['SL'] - reduction
        wins = int((
            (df['Pullback'] < effective_sl) & (df['TP'] >= effective_sl)
        ).sum()) if total else 0
        win_rate = (wins / total * 100) if total > 0 else 0.0

        label = "0 pips" if reduction == 0 else (
            "1 pip" if reduction == 1 else f"{reduction} pips")

        results.append({
            'SL Reduction': label,
            'Trades': total,
            'Notation': f"{wins}W - {total - wins}L",
            'Win Rate': f"{win_rate:.1f}%",
        })

    return pd.DataFrame(results)


def display_analysis_sl_reduction(df: pd.DataFrame):
    """
    Display 1:1 win/loss statistics as the safe stop is tightened.

    The Win Rate column header is click-to-sort, descending.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = ("<h2 style='color: #e0e0e0; background-color: #1e1e1e; "
                  "padding: 10px;'>Reducing SL Statistics</h2>")
    display(HTML(title_html))

    stats_df = calculate_sl_reduction_statistics(df)
    html_table = _create_sl_sortable_table(stats_df, "sl-reduction-table")
    display(HTML(html_table))


PULLBACK_ENTRY_PIPS = [0, 1, 2, 3]
PULLBACK_BUFFER_COLS = [("Notation", 0), ("+1 pip", 1), ("+2 pips", 2), ("+3 pips", 3)]


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
    Calculate limit-order pullback-entry statistics at 1:1 RRR across all trades.

    A limit order placed N pips into the pullback fills only if price pulled
    back at least N pips (Pullback >= N). The 0-pip level means every trade is
    taken at the signal (no limit order, nothing missed). Two extra levels
    scale the fill threshold to each trade's own stop: "Half" fills when the
    pullback reached at least half the SL, "Full" when it reached the SL itself
    (such a trade only survives if a buffer widens the stop past the pullback).

    Each notation column re-scores the same entered trades with an SL buffer
    (effective SL = SL + buffer, same semantics as the buffer strategies):
        winner = Pullback < effective SL AND TP >= effective SL   (1:1 RRR)
        W = entered winners; L = entered - W; M = missed winners (winners whose
        pullback never reached the fill threshold, so the limit never filled)
    "Notation" is buffer 0; "+1 pip" / "+2 pips" / "+3 pips" add that buffer.

    Missed winners are excluded from Trades because those trades were never entered.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Pullback, Trades, Notation, +1 pip, +2 pips, +3 pips
    """
    results = []

    levels = [
        (f"{n} pip" if n == 1 else f"{n} pips", df['Pullback'] >= n)
        for n in PULLBACK_ENTRY_PIPS
    ]
    levels.append(('Half', df['Pullback'] >= df['SL'] / 2))
    levels.append(('Full', df['Pullback'] >= df['SL']))

    for label, entered in levels:
        entered_total = int(entered.sum())
        row = {
            'Pullback': label,
            'Trades': entered_total,
        }
        for col, buffer in PULLBACK_BUFFER_COLS:
            effective_sl = df['SL'] + buffer
            winner = (df['Pullback'] < effective_sl) & (df['TP'] >= effective_sl)
            wins = int((entered & winner).sum())
            losses = entered_total - wins
            missed = int((~entered & winner).sum())
            row[col] = _format_wlm(wins, losses, missed)
        results.append(row)

    return pd.DataFrame(results)


def display_analysis_pullback(df: pd.DataFrame):
    """
    Display limit-order pullback-entry statistics at 1:1 RRR for fixed
    (0/1/2/3 pip) and SL-relative (Half/Full) pullback fill levels.

    Notation is W - L - M: winners, losers and missed winners (real winners that
    never pulled back far enough for the limit to fill). The +1/+2/+3 pip
    columns re-score the same trades with that buffer added to the SL.
    Each notation column header is click-to-sort by win rate, descending.

    Args:
        df: DataFrame with trading data
    """
    from IPython.display import display, HTML

    title_html = "<h2 style='color: #e0e0e0; background-color: #1e1e1e; padding: 10px;'>Pullback Analysis (1:1 RRR)</h2>"
    display(HTML(title_html))

    stats_df = calculate_pullback_statistics(df)
    html_table = _create_sl_sortable_table(stats_df, "pullback-analysis")
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
