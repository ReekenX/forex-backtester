"""
30MCR fixed-stop report analysis module.

Backs the static HTML page rendered by utils.report: every table on that page
is computed here, and nothing here needs a Jupyter kernel.

CSV columns (strategies/30MCR-fixed/trades.csv):
    Date, Weekday, Trade, Direction, 30M, SL, Pullback, Reward, Scaling

This is the 30MCR strategy traded with one stop size for every trade, so the
export differs from 30MCR's in two ways that reshape the page:

- SL is the same 5 pips on every trade. There is no recorded safe stop to
  widen, tighten or cap, so the stop-scenario tables 30MCR carries (SL Range,
  Fixed SL, Cap SL, Adding Buffer, Pullback, Half SL and the + 1 pip
  readings) have nothing to vary and are not built here.
- TP (pips) is replaced by Reward: the most the trade could have made, in R,
  after commissions, exported with an "R" suffix ("3R"). A blank Reward means
  the trade never reached 1R and loads as 0. Because costs are already taken
  out, a Reward cannot be turned back into pips to re-score the trade on
  another stop - another reason the stop scenarios are gone.
- Scaling is a second trade, opened when the first reached 1R, exported in R
  like Reward ("3R"). A positive value is how far that second trade ran; -1R
  means it was stopped out. A blank means no second trade was opened, so it
  loads as NaN, not 0.

The win rule keeps its shape with Reward in place of TP / SL:

    Pullback < SL   AND   Reward >= RRR

Signal is `Reward > 0`. The header cells still carry spreadsheet scratch on a
second line ("Trade\\n176"), so every header is normalised to its first line,
and everything right of Reward is the sheet's own scratch work, dropped at
load.

The 30M column names the 30-minute level the signal formed at - "High" or
"Low". The setup is that level read against Direction: buying a High or selling
a Low is a Continuation, the other two are Reversals. Both are known before the
trade is taken, so the setup is a tradeable filter.
"""

import math
import re
from html import escape
from typing import Dict, List, Optional, Tuple

import pandas as pd

# RRR ratios to test (1:3 is the ceiling - see CLAUDE.md)
RRR_RATIOS = [1, 2, 3]

# RRRs whose Strategies tables also run every trend with the stop moved to
# entry at 1R. At 1:1 the target is the trigger, so the rule changes nothing.
BE_AT_1R_RRRS = [2, 3]

# Appended to a trend's label on the rows that apply that rule.
BE_AT_1R_SUFFIX = ' + BE at 1R'

# The same RRRs again, with the stop pulled to half the risk at 1R instead.
HALF_SL_AT_1R_RRRS = BE_AT_1R_RRRS
HALF_SL_AT_1R_SUFFIX = ' + Half SL at 1R'

# What a trade stopped out on the half stop loses, in R.
HALF_SL_LOSS = -0.5

# Setup types the Strategies tables filter on. "All" keeps every trade; the
# rest keep the setups that end that way, e.g. Reversal = High + Low Reversal.
TREND_FILTERS = ['All', 'Continuation', 'Reversal']

WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Column naming the 30-minute level (High or Low) the signal formed at.
SETUP_COLUMN = '30M'

# (level, direction) pairs that continue through the level. Every other
# pairing is a Reversal.
CONTINUATIONS = {('High', 'Buy'), ('Low', 'Sell')}

TYPE_ORDER = ['Continuation', 'Reversal']

# The last trade columns. Everything right of them is the spreadsheet's
# scratch work, whose headers change between exports and can repeat, so they
# are never named. Scaling is optional; older exports end on Reward.
LAST_TRADE_COLUMNS = ('Reward', 'Scaling')

# The second trade's column. See _parse_scaling.
SCALING_COLUMN = 'Scaling'

DEFAULT_CSV = "../strategies/30MCR-fixed/trades.csv"

# The win-rate columns of every Signal/Strategy table. See _rate_columns.
RATE_COLUMNS = ['Signal', 'Strategy']


def _normalise_header(name: object) -> str:
    """
    Reduce a header cell to its first line, trimmed.

    The export wraps a computed figure under the column name in the same cell,
    so "Trade\\n196" and "Signal \\n54.1%" arrive as one header each. Only the
    first line is the name.
    """
    return str(name).split("\n")[0].strip()


def _drop_scratch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns up to and including Reward, or Scaling when the
    export carries it.

    The sheet precomputes win rates and R totals to the right of the trade
    data. This page computes those itself, so reading them would double up on
    the same question with a rule this module does not control.
    """
    columns = list(df.columns)
    positions = [columns.index(name) for name in LAST_TRADE_COLUMNS
                 if name in columns]
    if not positions:
        return df
    return df.iloc[:, :max(positions) + 1]


def _parse_r(values: pd.Series) -> pd.Series:
    """'3R' -> 3.0, '-1R' -> -1.0. A blank cell, or anything unreadable, is
    NaN."""
    text = values.astype(str).str.strip().str.rstrip('Rr').str.strip()
    return pd.to_numeric(text, errors="coerce")


def _parse_reward(values: pd.Series) -> pd.Series:
    """'3R' -> 3.0. A blank cell, or anything unreadable, is 0."""
    return _parse_r(values).fillna(0.0)


def _parse_scaling(values: pd.Series) -> pd.Series:
    """
    '3R' -> 3.0, '-1R' -> -1.0. A blank stays NaN: no second trade was
    opened, which is not the same as a 0R second trade.
    """
    return _parse_r(values)


def load_data(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load 30MCR fixed-stop data from CSV, clean it and derive the R and setup
    columns.

    R is Reward signed the way the 15LS1CC sheet signs it: negative when
    Pullback reached SL, i.e. the reach was there but the stop did not survive
    to collect it.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame with added R, Setup and Type columns
    """
    df = pd.read_csv(filepath)
    df.columns = [_normalise_header(col) for col in df.columns]
    df = _drop_scratch_columns(df).copy()
    # A row without a date is not a trade.
    df = df[df["Date"].notna() & df["Date"].astype(str).str.strip().ne("")]

    for col in ["SL", "Pullback"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Reward"] = _parse_reward(df["Reward"])
    df[SCALING_COLUMN] = (_parse_scaling(df[SCALING_COLUMN])
                          if SCALING_COLUMN in df.columns else float("nan"))

    # A 0R reward stays 0 rather than flipping to -0.
    stopped_out = (df["Pullback"] >= df["SL"]) & (df["Reward"] > 0)
    df["R"] = df["Reward"].where(~stopped_out, -df["Reward"])

    level = (df[SETUP_COLUMN].astype(str).str.strip()
             if SETUP_COLUMN in df.columns else pd.Series("", index=df.index))
    direction = df["Direction"].astype(str).str.strip()
    df["Type"] = [
        "Continuation" if (lvl, d) in CONTINUATIONS else "Reversal"
        for lvl, d in zip(level, direction)
    ]
    df["Setup"] = level + " " + df["Type"]

    return df


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

_TABLE_CSS = """
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

# Matches a bare "48.6%" and a wrapped "18W - 19L (48.6%)" alike.
_PCT_RE = re.compile(r"\d+(?:\.\d+)?%")

_SORT_SCRIPT = """
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


def _percentage_columns(df: pd.DataFrame) -> set:
    """Columns whose cells carry a percentage, so they can be sorted by it."""
    return {col for col in df.columns
            if any(_PCT_RE.search(str(v)) for v in df[col])}


def _win_rate_class(value, rrr: Optional[str]) -> str:
    """Colour a win rate against the breakeven rate for its RRR."""
    try:
        wr = float(str(value).rstrip("%"))
    except (ValueError, TypeError):
        return ""
    rrr_val = 1.0
    if rrr is not None:
        try:
            rrr_val = float(str(rrr).split(":")[-1])
        except (ValueError, IndexError):
            pass
    return "positive-edge" if wr > 100.0 / (1 + rrr_val) else "negative-edge"


def _breakeven_class(value) -> str:
    """Green above the 1:1 breakeven of 50%, red below, plain on it."""
    match = _PCT_RE.search(str(value))
    if not match:
        return ""
    rate = float(match.group(0).rstrip("%"))
    if rate > 50:
        return "positive-edge"
    return "negative-edge" if rate < 50 else ""


def create_html_table(df: pd.DataFrame, sort_id: Optional[str] = None,
                      first_col_width: Optional[str] = None,
                      colour_columns: Tuple[str, ...] = (),
                      rrr: Optional[str] = None,
                      column_widths: Optional[Dict[str, str]] = None) -> str:
    """
    Create a dark-mode HTML table with styled formatting.

    Args:
        df: DataFrame to convert to an HTML table
        sort_id: When set, gives the table this DOM id and makes every column
            whose cells carry a percentage click-to-sort by it, descending only
        first_col_width: When set, pins the label column to this width (e.g.
            "40%") and switches the table to fixed layout so it is honoured
        colour_columns: Columns of 1:1 win rates to colour against 50%
        rrr: The RRR (e.g. "1:2") to colour the Win Rate column against when
            the table carries no RRR column of its own
        column_widths: Widths (e.g. {"Buffer": "7%"}) for further columns,
            so short ones give their room to wide ones in a fixed layout

    Returns:
        HTML string with a styled table
    """
    if df.empty:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    sortable_cols = _percentage_columns(df) if sort_id else set()

    html = _TABLE_CSS
    if sort_id:
        html += _SORT_SCRIPT

    id_attr = f' id="{sort_id}"' if sort_id else ""
    table_style = ' style="table-layout: fixed;"' if first_col_width else ""
    html += (f'<table class="analysis-table"{id_attr}{table_style}>'
             '\n        <thead>\n            <tr>\n')

    column_widths = column_widths or {}
    for idx, col in enumerate(df.columns):
        width = first_col_width if idx == 0 else column_widths.get(col)
        width_style = f' style="width: {width};"' if width else ""
        if col in sortable_cols:
            html += (
                f'<th class="sortable"{width_style} title="Sort by win rate (desc)" '
                f'onclick="sortAnalysisTable(\'{sort_id}\', {idx}, this)">{col} ↓</th>'
            )
        else:
            # strategy-col is the 300px strategy-NAME column of the Strategies
            # tables, where it leads the row. A "Strategy" result column further
            # right is a different thing and must not inherit that width.
            cls = ' class="strategy-col"' if col == "Strategy" and idx == 0 else ""
            html += f"<th{cls}{width_style}>{col}</th>"

    html += "\n            </tr>\n        </thead>\n        <tbody>\n"

    rrr_col = "RRR" if "RRR" in df.columns else None
    for _, row in df.iterrows():
        html += "            <tr>\n"
        for col_idx, col in enumerate(df.columns):
            value = row[col]
            css_class = ""
            if col == "Strategy" and col_idx == 0:
                css_class = "strategy-col"
            elif col == "Win Rate":
                css_class = _win_rate_class(
                    value, row[rrr_col] if rrr_col else rrr)
            elif col in colour_columns:
                css_class = _breakeven_class(value)
            cls_attr = f' class="{css_class}"' if css_class else ""
            html += f"                <td{cls_attr}>{value}</td>\n"
        html += "            </tr>\n"

    html += "        </tbody>\n    </table>\n"
    return html


def create_sortable_table(df: pd.DataFrame, table_id: str,
                          sortable: bool = True,
                          first_col_width: Optional[str] = None,
                          colour_columns: Tuple[str, ...] = ()) -> str:
    """
    Build a plain label/statistics table, optionally with click-to-sort headers.

    Args:
        df: DataFrame with a label column and one or more win-rate columns
        table_id: Unique DOM id for this table
        sortable: Set False for tables whose row order carries meaning - the
            stop tables read as a progression from Default outwards, which a
            re-sort would scramble
        first_col_width: CSS width for the label column (e.g. "40%"), which
            also fixes the table layout so same-shaped tables line up
        colour_columns: Columns of 1:1 win rates to colour against 50%

    Returns:
        HTML string with a dark-mode styled table
    """
    if df.empty:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    columns = list(df.columns)
    sortable_cols = _percentage_columns(df) if sortable else set()

    html = _TABLE_CSS
    if sortable_cols:
        html += _SORT_SCRIPT

    table_style = ' style="table-layout: fixed;"' if first_col_width else ""
    html += (f'<table class="analysis-table" id="{table_id}"{table_style}>'
             '\n        <thead>\n            <tr>\n')

    for idx, col in enumerate(columns):
        width_style = (f' style="width: {first_col_width};"'
                       if idx == 0 and first_col_width else "")
        if col in sortable_cols:
            html += (
                f'<th class="sortable" title="Sort by win rate (desc)"{width_style} '
                f'onclick="sortAnalysisTable(\'{table_id}\', {idx}, this)">{col} ↓</th>'
            )
        else:
            html += f"<th{width_style}>{col}</th>"

    html += "\n            </tr>\n        </thead>\n        <tbody>\n"

    for _, row in df.iterrows():
        html += "            <tr>\n"
        for col in columns:
            css_class = (_breakeven_class(row[col])
                         if col in colour_columns else "")
            cls_attr = f' class="{css_class}"' if css_class else ""
            html += f"                <td{cls_attr}>{row[col]}</td>\n"
        html += "            </tr>\n"

    html += "        </tbody>\n    </table>\n"
    return html


# ---------------------------------------------------------------------------
# Signal / Strategy grouping tables
# ---------------------------------------------------------------------------

def strategy_wins(trades: pd.DataFrame, rrr: float = 1) -> pd.Series:
    """
    The page's one win rule: the trade survived its stop and its Reward
    reached the target.

        Pullback < SL   AND   Reward >= RRR

    Every table, chart and equity curve scores trades through here.
    """
    return (trades['Pullback'] < trades['SL']) & (trades['Reward'] >= rrr)


def _win_rate(wins: int, total: int) -> str:
    """'52.7%' - wins over trades."""
    return f"{(wins / total * 100) if total else 0.0:.1f}%"


def _rate_columns(trades: pd.DataFrame) -> Dict[str, str]:
    """
    The two readings every Signal/Strategy table carries, each a bare "52.7%"
    at 1:1.

    Signal: Reward > 0 - the trade idea was right and price reached a target.
    The stop is ignored on purpose, so this counts trades whose Pullback
    reached SL: the direction was correct but the entry was too early to
    survive it.

    Strategy: the 1:1 win rule (strategy_wins) - what trading it would
    actually have returned. Always a subset of Signal; the gap between the two
    is the cost of entry timing.
    """
    total = len(trades)
    return {
        'Signal': _win_rate(int((trades['Reward'] > 0).sum()), total),
        'Strategy': _win_rate(int(strategy_wins(trades).sum()), total),
    }


def _signal_strategy_rows(groups: List[Tuple[str, pd.DataFrame]],
                          column: str) -> pd.DataFrame:
    """
    Score each group of trades under the readings of _rate_columns. Every
    grouping table on the page comes through here so they cannot drift apart
    on the rule.

    Args:
        groups: (label, trades) tuples, one per row of the table
        column: Name for the leading column (e.g. 'Day', 'Setup')

    Returns:
        DataFrame with columns: <column>, Trades, then RATE_COLUMNS
    """
    return pd.DataFrame([
        {column: label, 'Trades': len(trades), **_rate_columns(trades)}
        for label, trades in groups
    ])


def calculate_weekday_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-weekday win rates under the readings of _rate_columns.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Day, Trades, then RATE_COLUMNS
    """
    groups = [(day, df[df['Weekday'] == day]) for day in WEEKDAY_ORDER]
    return _signal_strategy_rows(groups, 'Day')


# Series colours for the Signal/Strategy charts: categorical slots 1-2 of the
# dark-mode palette, validated against the #1e1e1e surface. Text never wears
# them - values and labels stay in the page's text greys.
CHART_SERIES = [('Signal', '#3987e5'), ('Strategy', '#d95926')]

# Strategy is scored at 1:1, where a win rate has to clear 50% to pay.
CHART_BREAKEVEN = 50

_CHART_CSS = """
    <style>
        .wr-chart { font-family: 'Courier New', monospace; font-size: 12px;
                    color: #e0e0e0; margin-top: 16px; padding: 14px 16px;
                    background-color: #1e1e1e; border: 1px solid #404040; }
        .wr-legend { display: flex; gap: 18px; margin: 0 0 8px 120px;
                     color: #a0a0a0; }
        .wr-swatch { display: inline-block; width: 10px; height: 10px;
                     border-radius: 2px; margin-right: 6px;
                     vertical-align: -1px; }
        .wr-body { position: relative; }
        .wr-grid { position: absolute; top: 0; bottom: 0; left: 120px;
                   right: 64px; pointer-events: none; }
        .wr-gridline { position: absolute; top: 0; bottom: 0;
                       border-left: 1px solid #2d2d2d; }
        .wr-breakeven { position: absolute; top: 0; bottom: 0;
                        border-left: 1px dashed #a0a0a0; }
        .wr-row { display: grid; grid-template-columns: 120px 1fr 64px;
                  align-items: center; padding: 6px 0; }
        .wr-row:hover { background-color: #2a2a2a; }
        .wr-label { padding-right: 10px; }
        .wr-count { display: block; color: #808080; font-size: 11px; }
        .wr-track { position: relative; display: flex;
                    flex-direction: column; gap: 2px; }
        .wr-bar-line { position: relative; height: 12px; }
        .wr-bar { height: 12px; border-radius: 0 4px 4px 0; }
        .wr-value { position: absolute; top: -1px; margin-left: 4px;
                    padding: 0 3px; background-color: #1e1e1e;
                    color: #e0e0e0; white-space: nowrap; font-size: 11px; }
        .wr-row:hover .wr-value { background-color: #2a2a2a; }
        .wr-axis { position: relative; height: 16px; margin: 4px 64px 0 120px;
                   color: #808080; font-size: 11px; }
        .wr-tick { position: absolute; transform: translateX(-50%); }
    </style>
"""


def _cell_rate(cell: str) -> float:
    """Read the win rate back out of a '13W - 12L (52.0%)' or '52.0%' cell."""
    return float(_PCT_RE.search(str(cell)).group(0).rstrip('%'))


def create_signal_strategy_chart(stats: pd.DataFrame) -> str:
    """
    Paired horizontal bars of Signal and Strategy win rate, one pair per row.

    Drawn from the very table it sits under - a `<label>, Trades` frame with
    the RATE_COLUMNS - so the chart and the table cannot disagree. Both bars
    share one 0-100% axis, with a dashed line at the 1:1 breakeven, so the
    gap between Signal and Strategy - the cost of entry timing - and the
    distance to breakeven read at a glance.
    Every bar carries its value at the tip and a hover tooltip.

    Args:
        stats: DataFrame whose first column labels the rows and which carries
            Trades and the CHART_SERIES columns as bare win rates

    Returns:
        HTML string with the chart, or a "No data" note for an empty frame
    """
    if stats.empty:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    label_column = stats.columns[0]

    html = _CHART_CSS + '<div class="wr-chart" role="img" aria-label="'
    html += f'Signal and Strategy win rate by {label_column}">'
    series = [(name, colour) for name, colour in CHART_SERIES
              if name in stats.columns]
    html += '<div class="wr-legend">'
    for name, colour in series:
        html += (f'<span><span class="wr-swatch" '
                 f'style="background-color: {colour};"></span>{name}</span>')
    html += (f'<span><span class="wr-swatch" style="border: 1px dashed '
             f'#a0a0a0; background: none;"></span>{CHART_BREAKEVEN}% '
             'breakeven (1:1)</span></div>')

    html += '<div class="wr-body"><div class="wr-grid">'
    for tick in (0, 25, 75, 100):
        html += f'<div class="wr-gridline" style="left: {tick}%;"></div>'
    html += (f'<div class="wr-breakeven" style="left: {CHART_BREAKEVEN}%;">'
             '</div></div>')

    for _, row in stats.iterrows():
        label = row[label_column]
        html += (f'<div class="wr-row"><div class="wr-label">{label}'
                 f'<span class="wr-count">{int(row["Trades"])} trades</span>'
                 '</div><div class="wr-track">')
        for name, colour in series:
            rate = _cell_rate(row[name])
            html += (f'<div class="wr-bar-line" title="{label} - {name}: '
                     f'{row[name]}"><div class="wr-bar" style="width: '
                     f'{rate:.1f}%; background-color: {colour};"></div>'
                     f'<span class="wr-value" style="left: {rate:.1f}%;">'
                     f'{rate:.1f}%</span></div>')
        html += '</div><div></div></div>'

    html += '</div><div class="wr-axis">'
    for tick in (0, 25, 50, 75, 100):
        html += f'<span class="wr-tick" style="left: {tick}%;">{tick}%</span>'
    html += '</div></div>'
    return html



def _ordered_labels(values: pd.Series, preferred: List[str]) -> List[str]:
    """
    List `preferred` first, then any other label present in the data.

    A setup the export starts recording tomorrow still gets a row rather than
    being dropped for not being on the list.
    """
    present = {str(v) for v in values.dropna().unique() if str(v).strip()}
    extras = sorted(present - set(preferred))
    return [label for label in preferred if label in present] + extras


def calculate_setup_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signal and Strategy win rates for Continuation against Reversal.

    The type - continuation through the 30-minute level or reversal off it -
    is known at entry, so it is a tradeable filter. "Default" is every trade;
    the two type rows split it, so they sum to it.

    Args:
        df: DataFrame with trading data, including a Type column

    Returns:
        DataFrame with columns: Setup, Trades, then RATE_COLUMNS
    """
    types = _ordered_labels(df['Type'], TYPE_ORDER)
    groups: List[Tuple] = [('Default', df)]
    groups.extend((label, df[df['Type'] == label]) for label in types)
    return _signal_strategy_rows(groups, 'Setup')


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

def _longest_run(won: pd.Series, dates: pd.Series, value: bool) -> str:
    """
    Length of the longest unbroken run of `value` in `won`, with its dates.

    Trades are read in the order given, so the caller sorts them first. The
    cell reads "7 (2026-03-10 - 2026-03-14)" - the run's first and last trade
    dates - or a bare "0" when the run never happened. On a tie the earliest
    run is the one named.
    """
    best, best_start, best_end = 0, None, None
    length, start = 0, None
    for date, hit in zip(dates, won):
        if hit == value:
            if length == 0:
                start = date
            length += 1
            if length > best:
                best, best_start, best_end = length, start, date
        else:
            length = 0
    if best == 0:
        return "0"
    return f"{best} ({best_start} - {best_end})"


def calculate_streak_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Longest winning and losing runs at 1:1, for every trade and for each type.

    A trade is a win under strategy_wins, and anything else breaks a winning
    run, so the stopped-out winner-that-wasn't counts as a loss here. Trades
    are read in date order; the export's own order breaks ties within a day.
    The Continuation and Reversal rows run through only their own trades, so a
    loss in one type does not break the other type's streak - that is what
    trading one type alone would have felt like.

    Args:
        df: DataFrame with trading data, including a Type column

    Returns:
        DataFrame with columns: Trend, Trades, Longest Wins, Longest Losses
    """
    ordered = df.sort_values('Date', kind='stable') if len(df) else df
    won = strategy_wins(ordered)
    groups: List[Tuple[str, pd.Series]] = [
        ('Default', pd.Series(True, index=ordered.index))]
    groups.extend((label, ordered['Type'] == label) for label in TYPE_ORDER)

    results = []
    for label, mask in groups:
        dates = ordered.loc[mask, 'Date']
        results.append({
            'Trend': label,
            'Trades': int(mask.sum()),
            'Longest Wins': _longest_run(won[mask], dates, True),
            'Longest Losses': _longest_run(won[mask], dates, False),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Strategies tables
# ---------------------------------------------------------------------------

def _apply_trend(df: pd.DataFrame, trend: str) -> pd.DataFrame:
    """Keep only trades of one setup type ('All' keeps every trade)."""
    return df if trend == 'All' else df[df['Type'] == trend]


def be_at_1r_results(trades: pd.DataFrame, rrr: float) -> pd.Series:
    """
    Each trade's R at 1:RRR with the stop moved to entry once it reached 1R.

    The stop is checked first, as in strategy_wins: a trade with Pullback >= SL
    never reached 1R alive, and one with Reward under 1R never triggered the
    rule, so both lose 1R. Past 1R, Scaling - the second trade opened there,
    with its stop at this trade's entry - says whether price came back:

    - Scaling -1R: price returned to entry first. Break-even, 0R, whatever
      Reward the trade reached later.
    - Scaling positive: price never returned, so the trade could not lose. It
      wins if Reward reached the target, otherwise it is break-even, 0R.
    - Scaling blank: the export does not say, so the normal win rule applies.

    At 1:1 the target is the trigger, so the trade is closed at 1R first.

    Returns:
        Series of +RRR (win), 0.0 (break-even) or -1.0 (loss), on trades' index
    """
    scaling = (trades[SCALING_COLUMN] if SCALING_COLUMN in trades.columns
               else pd.Series(float('nan'), index=trades.index))
    reached = (trades['Pullback'] < trades['SL']) & (trades['Reward'] >= 1)
    hit = trades['Reward'] >= rrr
    returned = scaling == -1
    protected = scaling.notna() & ~returned

    results = pd.Series(-1.0, index=trades.index)
    results[reached & hit] = float(rrr)
    if rrr > 1:
        results[reached & returned] = 0.0
        results[reached & protected & ~hit] = 0.0
    return results


def half_sl_at_1r_results(trades: pd.DataFrame, rrr: float) -> pd.Series:
    """
    Each trade's R at 1:RRR with the stop pulled to SL / 2 once it reached 1R.

    The stop is checked first, as in strategy_wins: a trade with Pullback >= SL
    or Reward under 1R never triggered the rule and loses 1R. Past 1R the
    trade risks only HALF_SL_LOSS. It wins if Reward reached the target and
    Pullback stayed under SL / 2; otherwise the half stop took it, -0.5R.

    Pullback is the deepest the trade went at any point, not only after 1R,
    so a pullback past SL / 2 that came before 1R - when the full stop was
    still in place - is scored as a half-stop loss too. That makes the rule
    read conservatively.

    At 1:1 the target is the trigger, so the trade is closed at 1R first.

    Returns:
        Series of +RRR (win), -0.5 (half-stop loss) or -1.0 (loss), on
        trades' index
    """
    reached = (trades['Pullback'] < trades['SL']) & (trades['Reward'] >= 1)
    hit = trades['Reward'] >= rrr
    results = pd.Series(-1.0, index=trades.index)
    if rrr <= 1:
        results[reached & hit] = float(rrr)
        return results
    results[reached] = HALF_SL_LOSS
    results[reached & hit & (trades['Pullback'] < trades['SL'] / 2)] = float(rrr)
    return results


# How each row of a Strategies table is scored: None is the plain win rule.
_RULES = {
    'be': (BE_AT_1R_SUFFIX, BE_AT_1R_RRRS, be_at_1r_results),
    'half': (HALF_SL_AT_1R_SUFFIX, HALF_SL_AT_1R_RRRS, half_sl_at_1r_results),
}


def _strategy_rows(rrr: int) -> List[Tuple[str, str, Optional[str]]]:
    """(label, trend, rule) for every row of one RRR's Strategies table: the
    trends, then the trends again under each rule that applies at that RRR."""
    rows = [(trend, trend, None) for trend in TREND_FILTERS]
    for rule, (suffix, rrrs, _) in _RULES.items():
        if rrr in rrrs:
            rows += [(trend + suffix, trend, rule) for trend in TREND_FILTERS]
    return rows


def _rule_results(trades: pd.DataFrame, rrr: int,
                  rule: Optional[str]) -> pd.Series:
    """Each trade's R at 1:RRR under one row's rule."""
    if rule is None:
        return strategy_wins(trades, rrr).map({True: float(rrr), False: -1.0})
    return _RULES[rule][2](trades, rrr)


def _format_r(value: float) -> str:
    """'+3R', or '+3.5R' when a half-stop loss leaves a half."""
    return f'{value:+.1f}R' if value % 1 else f'{value:+.0f}R'


def calculate_strategy_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every setup trend at every RRR.

    With one stop for every trade there is no buffer or cap to vary, so a row
    is a trend and an RRR. A win scores +RRR, anything else -1R:

        Outcome = W x RRR - L

    Reward is net of costs, so a win's R is already net; a loss is counted as
    the plain -1R.

    At the RRRs in BE_AT_1R_RRRS each trend repeats as '<trend> + BE at 1R',
    scored by be_at_1r_results. Its Notation is W – BE – L and its Win Rate
    W / (W + L): a break-even is 0R, so it counts on neither side.

    At the RRRs in HALF_SL_AT_1R_RRRS each trend repeats again as
    '<trend> + Half SL at 1R', scored by half_sl_at_1r_results. Its Notation
    is W – HL – L, HL the half-stop losses, and its Win Rate W / Trades: a
    half-stop loss is still a loss. Outcome = W x RRR - 0.5 x HL - L.

    Args:
        df: DataFrame with trading data, including a Type column

    Returns:
        DataFrame with columns: Trend, RRR, Trades, Notation, Win Rate,
        Outcome - RRRs ascending, the rows of _strategy_rows within each
    """
    results = []
    for rrr in RRR_RATIOS:
        for label, trend, rule in _strategy_rows(rrr):
            trades = _apply_trend(df, trend)
            outcome = _rule_results(trades, rrr, rule)
            won = int((outcome > 0).sum())
            lost = int((outcome == -1).sum())
            if rule == 'be':
                even = int((outcome == 0).sum())
                notation = f'{won}W – {even}BE – {lost}L'
                rate = _win_rate(won, won + lost)
            elif rule == 'half':
                half = int((outcome == HALF_SL_LOSS).sum())
                notation = f'{won}W – {half}HL – {lost}L'
                rate = _win_rate(won, len(trades))
            else:
                notation = f'{won}W – {lost}L'
                rate = _win_rate(won, len(trades))
            results.append({
                'Trend': label,
                'RRR': f'1:{rrr}',
                'Trades': len(trades),
                'Notation': notation,
                'Win Rate': rate,
                'Outcome': _format_r(float(outcome.sum())),
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def _scaled(trades: pd.DataFrame) -> pd.Series:
    """The second-trade results of the trades that opened one."""
    if SCALING_COLUMN not in trades.columns:
        return pd.Series(dtype=float)
    return trades[SCALING_COLUMN].dropna()


def _signed_r(value: float) -> str:
    """'+1.25R' - two decimals, signed, so an average reads as one."""
    return f'{value:+.2f}R'


def calculate_scaling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    What the second trade did, for every trade and for each setup type.

    A second trade is opened when the first reaches 1R. Its Scaling value is
    how far it ran, in R, or -1R when it was stopped out. A win is any
    positive value: the second trade reached at least 1R, so this is its 1:1
    win rate.

    Avg R and Avg Win read every winner at the most it reached - a ceiling no
    fixed target collects - so they describe the move, not a payout. The
    Scaling at each RRR table scores the same trades on fixed targets.

    Args:
        df: DataFrame with trading data, including Type and Scaling columns

    Returns:
        DataFrame with columns: Trend, Trades, Scaled, Notation, Win Rate,
        Avg R, Avg Win, Best - trends in TREND_FILTERS order
    """
    results = []
    for trend in TREND_FILTERS:
        trades = _apply_trend(df, trend)
        scaled = _scaled(trades)
        won = scaled[scaled > 0]
        share = len(scaled) / len(trades) * 100 if len(trades) else 0.0
        results.append({
            'Trend': trend,
            'Trades': len(trades),
            'Scaled': f'{len(scaled)} ({share:.1f}%)',
            'Notation': f'{len(won)}W – {len(scaled) - len(won)}L',
            'Win Rate': _win_rate(len(won), len(scaled)),
            'Avg R': _signed_r(scaled.mean()) if len(scaled) else 'N/A',
            'Avg Win': _signed_r(won.mean()) if len(won) else 'N/A',
            'Best': f'{won.max():.0f}R' if len(won) else 'N/A',
        })
    return pd.DataFrame(results)


def calculate_scaling_rrr_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    The second trade scored on a fixed target, next to the first trade's own
    result at that RRR and the two together.

    A second trade wins at 1:RRR when its Scaling reached RRR; anything else -
    including a positive Scaling short of the target - is -1R, the rule the
    Strategies tables apply to the first trade. Base is that first-trade
    Outcome, straight from calculate_strategy_statistics, so the two cannot
    disagree. Combined is Base plus the scaled Outcome: every trade taken
    alone, plus a second trade on each one that reached 1R.

    Args:
        df: DataFrame with trading data, including Type and Scaling columns

    Returns:
        DataFrame with columns: Trend, RRR, Scaled, Notation, Win Rate,
        Outcome, Base, Combined - trends in TREND_FILTERS order, RRRs
        ascending within each
    """
    base = calculate_strategy_statistics(df).set_index(['Trend', 'RRR'])
    results = []
    for trend in TREND_FILTERS:
        scaled = _scaled(_apply_trend(df, trend))
        for rrr in RRR_RATIOS:
            won = int((scaled >= rrr).sum())
            lost = len(scaled) - won
            outcome = won * rrr - lost
            base_outcome = int(base.loc[(trend, f'1:{rrr}'), 'Outcome']
                               .rstrip('R'))
            results.append({
                'Trend': trend,
                'RRR': f'1:{rrr}',
                'Scaled': len(scaled),
                'Notation': f'{won}W – {lost}L',
                'Win Rate': _win_rate(won, len(scaled)),
                'Outcome': f'{outcome:+d}R',
                'Base': f'{base_outcome:+d}R',
                'Combined': f'{base_outcome + outcome:+d}R',
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Strategy equity curves
# ---------------------------------------------------------------------------

# Lines coloured apart on the chart; the rest draw in one recessive grey and
# only come forward on hover. Five is the ceiling the categorical palette
# keeps distinct, so the five best outcomes get the colours.
EQUITY_HIGHLIGHT = 5
# Categorical slots 1-5 of the dark palette, in fixed order.
EQUITY_COLOURS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181']
EQUITY_OTHER_COLOUR = '#4a4a4a'

# SVG viewBox and the plot area inside it; the right margin holds the direct
# labels of the coloured lines.
EQUITY_WIDTH = 1100
EQUITY_HEIGHT = 340
EQUITY_MARGIN = {'top': 12, 'right': 300, 'bottom': 28, 'left': 52}

_EQUITY_CHART_CSS = """
    <style>
        .eq-chart { font-family: 'Courier New', monospace; font-size: 12px;
                    color: #e0e0e0; margin-top: 16px; padding: 14px 16px;
                    background-color: #1e1e1e; border: 1px solid #404040;
                    position: relative; }
        .eq-legend { display: flex; flex-wrap: wrap; gap: 6px 18px;
                     margin: 0 0 8px 0; color: #a0a0a0; }
        .eq-swatch { display: inline-block; width: 14px; height: 3px;
                     border-radius: 2px; margin-right: 6px;
                     vertical-align: 3px; }
        .eq-chart svg { display: block; width: 100%; height: auto;
                        overflow: visible; }
        .eq-grid { stroke: #2d2d2d; stroke-width: 1; }
        .eq-zero { stroke: #a0a0a0; stroke-width: 1;
                   stroke-dasharray: 4 4; }
        .eq-axis { fill: #808080; font-size: 11px; }
        .eq-hit { fill: none; stroke: transparent; stroke-width: 10;
                  cursor: crosshair; }
        .eq-line { fill: none; stroke-width: 1; stroke-linejoin: round;
                   stroke-linecap: round; pointer-events: none; }
        .eq-series.eq-top .eq-line { stroke-width: 2; }
        .eq-series:hover .eq-line, .eq-series.eq-active .eq-line {
            stroke-width: 3; }
        .eq-series.eq-other:hover .eq-line,
        .eq-series.eq-other.eq-active .eq-line { stroke: #e0e0e0; }
        .eq-label { fill: #e0e0e0; font-size: 11px; }
        .eq-leader { stroke: #606060; stroke-width: 1; }
        .eq-tip { position: absolute; display: none; padding: 4px 8px;
                  background-color: #2d2d2d; border: 1px solid #404040;
                  border-radius: 4px; color: #e0e0e0; font-size: 11px;
                  white-space: nowrap; pointer-events: none; z-index: 2; }
        .eq-caption { margin: 8px 0 0 0; color: #808080; font-size: 11px; }
    </style>
"""

# Binds every chart on the page that has not been bound yet, so each chart's
# own copy of the script picks up the charts rendered before it and the last
# copy leaves nothing unbound.
_EQUITY_CHART_SCRIPT = """
    <script>
    (function () {
        document.querySelectorAll('.eq-chart:not([data-bound])').forEach(function (chart) {
            chart.dataset.bound = '1';
            var svg = chart.querySelector('svg');
            var tip = chart.querySelector('.eq-tip');
            var left = parseFloat(svg.dataset.left);
            var step = parseFloat(svg.dataset.step);
            var rrr = svg.dataset.rrr;
            chart.querySelectorAll('.eq-series').forEach(function (series) {
                var values = series.dataset.values.split(',').map(Number);
                series.addEventListener('mousemove', function (event) {
                    var point = svg.createSVGPoint();
                    point.x = event.clientX;
                    point.y = event.clientY;
                    var local = point.matrixTransform(svg.getScreenCTM().inverse());
                    var index = Math.round((local.x - left) / step);
                    index = Math.max(0, Math.min(values.length - 1, index));
                    var value = values[index];
                    tip.textContent = series.dataset.label + ' · trade ' + index
                        + ' of ' + (values.length - 1) + ' · '
                        + (value > 0 ? '+' : '') + value.toFixed(1) + 'R at 1:' + rrr;
                    tip.style.display = 'block';
                    var box = chart.getBoundingClientRect();
                    tip.style.left = (event.clientX - box.left + 14) + 'px';
                    tip.style.top = (event.clientY - box.top - 28) + 'px';
                    series.classList.add('eq-active');
                    series.parentNode.appendChild(series);
                });
                series.addEventListener('mouseleave', function () {
                    tip.style.display = 'none';
                    series.classList.remove('eq-active');
                });
            });
        });
    })();
    </script>
"""


def calculate_equity_curves(df: pd.DataFrame, rrr: float = 1) -> List[Dict]:
    """
    Cumulative R of every Strategies table row, trade by trade, at one RRR.

    Trades are walked in date order; the export's own order breaks ties within
    a day. A win adds the RRR in R and a loss takes 1R, under strategy_wins -
    or under the row's rule on a '+ BE at 1R' or '+ Half SL at 1R' row, where
    a break-even adds 0 and a half-stop loss takes 0.5R.
    Every curve runs over every trade, so a Continuation-only row sits flat
    through the Reversal trades it would not have taken and the curves line
    up in time.

    Args:
        df: DataFrame with trading data, including a Type column
        rrr: Risk-reward ratio to score at

    Returns:
        One dict per trend, best final outcome first: label, trades (the
        trades the row took), outcome (final R) and curve (the cumulative R
        after each trade, starting from 0, so len(df) + 1 long)
    """
    ordered = df.sort_values('Date', kind='stable') if len(df) else df
    curves = []
    for label, trend, rule in _strategy_rows(rrr):
        taken = (pd.Series(True, index=ordered.index) if trend == 'All'
                 else ordered['Type'] == trend)
        per_trade = _rule_results(ordered, rrr, rule).copy()
        per_trade[~taken] = 0.0
        curve = [0.0] + per_trade.cumsum().round(6).tolist()
        curves.append({
            'label': (('All Trades' if trend == 'All' else trend)
                      + label[len(trend):]),
            'trades': int(taken.sum()),
            'outcome': curve[-1],
            'curve': curve,
        })
    return sorted(curves, key=lambda c: -c['outcome'])


def _nice_step(span: float, target_ticks: int = 6) -> float:
    """Pick a round tick step that gives about `target_ticks` across `span`."""
    raw = max(span, 1) / target_ticks
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if step >= raw:
            return step
    return 1000


def create_equity_chart(curves: List[Dict], rrr: float = 1,
                        chart_id: str = 'equity-chart') -> str:
    """
    Line chart of every Strategies row's cumulative R, one line per table row.

    The EQUITY_HIGHLIGHT best final outcomes are coloured, labelled at their
    right-hand end and listed in the legend; every other line draws in one
    recessive grey so the field reads as context. Hovering any line brings it
    forward and shows its name and its R at that trade. The dashed line is
    0R, where trading the strategy would have broken even.

    Args:
        curves: Output of calculate_equity_curves
        rrr: The RRR the curves were scored at, for the legend and tooltip
        chart_id: Unique DOM id for this chart

    Returns:
        HTML string with the chart, or a "No data" note when there are no
        curves or no trades
    """
    if not curves or len(curves[0]['curve']) < 2:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    n_points = len(curves[0]['curve'])
    n_trades = n_points - 1
    m = EQUITY_MARGIN
    plot_w = EQUITY_WIDTH - m['left'] - m['right']
    plot_h = EQUITY_HEIGHT - m['top'] - m['bottom']
    x_step = plot_w / n_trades

    lo = min(0.0, min(min(c['curve']) for c in curves))
    hi = max(0.0, max(max(c['curve']) for c in curves))
    tick = _nice_step(hi - lo)
    y_min = math.floor(lo / tick) * tick
    y_max = math.ceil(hi / tick) * tick
    if y_max == y_min:
        y_max = y_min + tick

    def x_of(i: int) -> float:
        return m['left'] + i * x_step

    def y_of(value: float) -> float:
        return m['top'] + (y_max - value) / (y_max - y_min) * plot_h

    top = curves[:EQUITY_HIGHLIGHT]
    rest = curves[EQUITY_HIGHLIGHT:]

    out = _EQUITY_CHART_CSS
    out += (f'<div class="eq-chart" id="{chart_id}" role="img" '
            f'aria-label="Cumulative R per strategy at 1:{rrr:g}">')
    out += '<div class="eq-legend">'
    for colour, curve in zip(EQUITY_COLOURS, top):
        out += (f'<span><span class="eq-swatch" style="background-color: '
                f'{colour};"></span>{escape(curve["label"])} '
                f'({_format_r(curve["outcome"])})</span>')
    if rest:
        out += (f'<span><span class="eq-swatch" style="background-color: '
                f'{EQUITY_OTHER_COLOUR};"></span>{len(rest)} other '
                'strategies (hover a line)</span>')
    out += ('<span><span class="eq-swatch" style="border-top: 1px dashed '
            '#a0a0a0; background: none; height: 0;"></span>0R breakeven'
            '</span></div>')

    out += (f'<svg viewBox="0 0 {EQUITY_WIDTH} {EQUITY_HEIGHT}" '
            f'data-left="{m["left"]}" data-step="{x_step:.4f}" '
            f'data-rrr="{rrr:g}">')

    # Gridlines and y-axis labels.
    value = y_min
    while value <= y_max + 1e-9:
        y = y_of(value)
        cls = 'eq-zero' if value == 0 else 'eq-grid'
        out += (f'<line class="{cls}" x1="{m["left"]}" y1="{y:.1f}" '
                f'x2="{m["left"] + plot_w}" y2="{y:.1f}"/>')
        out += (f'<text class="eq-axis" x="{m["left"] - 8}" y="{y + 4:.1f}" '
                f'text-anchor="end">{value:+.0f}R</text>')
        value += tick
    # x-axis labels: first, last and a few in between.
    x_tick = _nice_step(n_trades, 8)
    # The last trade always gets a label; a regular tick that would sit on
    # top of it is dropped.
    labels = [i for i in range(0, n_trades + 1, int(x_tick))
              if n_trades - i >= x_tick / 2]
    labels.append(n_trades)
    for i in labels:
        out += (f'<text class="eq-axis" x="{x_of(i):.1f}" '
                f'y="{EQUITY_HEIGHT - 8}" text-anchor="middle">{i}</text>')

    def series(curve: Dict, colour: str, kind: str) -> str:
        d = ' '.join(f'{"M" if i == 0 else "L"}{x_of(i):.1f},{y_of(v):.1f}'
                     for i, v in enumerate(curve['curve']))
        values = ','.join(f'{v:g}' for v in curve['curve'])
        return (f'<g class="eq-series {kind}" data-label="'
                f'{escape(curve["label"])}" data-values="{values}">'
                f'<path class="eq-hit" d="{d}"/>'
                f'<path class="eq-line" style="stroke: {colour};" d="{d}"/>'
                '</g>')

    out += '<g class="eq-lines">'
    for curve in rest:
        out += series(curve, EQUITY_OTHER_COLOUR, 'eq-other')
    for colour, curve in zip(EQUITY_COLOURS, top):
        out += series(curve, colour, 'eq-top')
    out += '</g>'

    # Direct labels at the right edge, nudged apart with leader lines so they
    # never overprint each other where lines converge.
    anchors = sorted(
        ((y_of(c['outcome']), colour, c) for colour, c in zip(EQUITY_COLOURS, top)),
        key=lambda a: a[0])
    min_gap = 14
    placed = []
    for y, colour, curve in anchors:
        slot = y if not placed else max(y, placed[-1] + min_gap)
        placed.append(slot)
    overflow = placed[-1] - (EQUITY_HEIGHT - m['bottom']) if placed else 0
    if overflow > 0:
        placed = [p - overflow for p in placed]
    x_end = x_of(n_trades)
    for (y, colour, curve), slot in zip(anchors, placed):
        out += (f'<line class="eq-leader" x1="{x_end:.1f}" y1="{y:.1f}" '
                f'x2="{x_end + 12}" y2="{slot:.1f}"/>')
        out += (f'<circle cx="{x_end:.1f}" cy="{y:.1f}" r="3" '
                f'fill="{colour}" stroke="#1e1e1e" stroke-width="2"/>')
        out += (f'<text class="eq-label" x="{x_end + 16}" y="{slot + 4:.1f}">'
                f'{escape(curve["label"])} {_format_r(curve["outcome"])}</text>')
    out += '</svg><div class="eq-tip"></div>'
    out += (f'<p class="eq-caption">Cumulative R over {n_trades} trades in '
            f'date order: +{rrr:g}R per win, -1R per loss. Trades outside a '
            'row\'s trend stay flat. '
            f'The best {len(top)} final outcomes are coloured.</p>')
    out += '</div>' + _EQUITY_CHART_SCRIPT
    return out


# ---------------------------------------------------------------------------
# R distribution
# ---------------------------------------------------------------------------

R_LEVELS = range(1, 11)


def calculate_r_counts(df: pd.DataFrame) -> pd.Series:
    """
    Count trades per whole R level (1R-10R), each trade counted once.

    R is truncated towards zero, so a trade at 3.8R lands in the 3R bucket.
    The sign is dropped: a negative R records the distance that was available
    before the stop was hit, which is the same reach as a positive one.

    Args:
        df: DataFrame with an R column (see load_data)

    Returns:
        Series indexed 1..10 with the number of trades at each R level
    """
    r_values = df["R"].dropna().abs().apply(lambda x: int(x))
    r_values = r_values[(r_values >= 1) & (r_values <= 10)]
    return r_values.value_counts().reindex(R_LEVELS, fill_value=0)


def _create_r_histogram(counts: pd.Series, title: str) -> str:
    """
    Render an R-level count series as a horizontal bar chart.

    Args:
        counts: Series indexed 1..10 holding the bar values
        title: Heading shown above the bars

    Returns:
        HTML string with a styled horizontal bar chart
    """
    max_count = counts.max() if counts.max() > 0 else 1

    html = f"""
    <div style="background-color: #1e1e1e; padding: 20px; font-family: 'Courier New', monospace;">
        <h2 style="color: #e0e0e0; margin-top: 0;">{title}</h2>
    """

    for r_val in R_LEVELS:
        count = int(counts[r_val])
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


def create_r_histogram_combined(df: pd.DataFrame) -> str:
    """
    Cumulative R distribution (1R-10R).

    A trade that reached 3R also passed through 1R and 2R, so higher-R trades
    are counted in every lower bucket too.

    Args:
        df: DataFrame with an R column

    Returns:
        HTML string with a styled horizontal bar chart
    """
    raw_counts = calculate_r_counts(df)
    cumulative = pd.Series({r: raw_counts.loc[r:].sum() for r in R_LEVELS})
    return _create_r_histogram(cumulative, "R Distribution")

