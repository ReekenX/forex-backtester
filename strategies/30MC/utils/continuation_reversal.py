"""
30MC report analysis module.

Backs the static HTML page rendered by utils.report: every table on that page
is computed here, and nothing here needs a Jupyter kernel.

CSV columns (strategies/30MC/trades.csv):
    Date, Weekday, Trade, Direction, 30M, SL, Pullback, TP

SL is a 5-pip placeholder on every trade for now; Pullback and TP are real.
There is no BE Rule column, so there is no break-even rule either.

Three things about this export shape the loader:

- Header cells carry spreadsheet scratch on a second line ("Trade\\n69"), so
  every header is normalised to its first line before anything reads it.
- Everything right of TP is the sheet's own scratch work - precomputed win
  rates and R totals for stop scenarios this page recomputes itself - so it is
  dropped at load rather than read. The scratch headers are renamed and
  repeated from one export to the next, so the cut is anchored on TP.
- There is no R column. R is derived as TP / SL - how many multiples of the
  recorded safe stop the move was worth - and signed the way the 15LS1CC sheet
  signs it: negative when Pullback reached SL, i.e. the reach was there but the
  stop did not survive to collect it.

A blank TP and a TP of 0 both mean the trade was not profitable, so both load
as 0 and `TP > 0` is the Signal rule here exactly as it is on the other pages.

The 30M column names the 30-minute level the signal formed at - "High" or
"Low". The setup is that level read against Direction: buying a High or selling
a Low is a Continuation, the other two are Reversals. Both are known before the
trade is taken, so the setup is a tradeable filter. Its Continuation/Reversal half is pooled in the same table, so the
setup can be read with the level and without it.
"""

import math
import re
from html import escape
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# RRR ratios to test (1:3 is the ceiling - see CLAUDE.md)
RRR_RATIOS = [1, 2, 3]

# Extra pip buffer values to test in the Strategies tables
BUFFER_PIPS = [0, 1]

# Setup types the Strategies tables filter on. "All" keeps every trade; the
# rest keep the setups that end that way, e.g. Reversal = High + Low Reversal.
TREND_FILTERS = ['All', 'Continuation', 'Reversal']

WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Column naming the 30-minute level (High or Low) the signal formed at.
SETUP_COLUMN = '30M'

# (level, direction) pairs that continue through the level. Every other
# pairing is a Reversal.
CONTINUATIONS = {('High', 'Buy'), ('Low', 'Sell')}

# The four setups this strategy trades, in the order the tables list them.
# A label outside this list is still shown - appended in sorted order - so a
# new setup in the export cannot vanish silently.
SETUP_ORDER = [
    'High Continuation',
    'High Reversal',
    'Low Continuation',
    'Low Reversal',
]

TYPE_ORDER = ['Continuation', 'Reversal']

# The last trade column. Everything right of it is the spreadsheet's scratch
# work, whose headers change between exports ("Signal", "Signal All") and can
# repeat, so they are never named.
LAST_TRADE_COLUMN = 'TP'

DEFAULT_CSV = "../strategies/30MC/trades.csv"

# Pips the "Strategy + 1 pip" column and the "<type> + 1 pip buffer" rows of
# the Streaks table pad the stop with.
TYPE_BUFFER_PIPS = 1

# The win-rate columns of every Signal/Strategy table. See _rate_columns.
RATE_COLUMNS = ['Signal', 'Strategy', 'Strategy + 1 pip']


def _normalise_header(name: object) -> str:
    """
    Reduce a header cell to its first line, trimmed.

    The export wraps a computed figure under the column name in the same cell,
    so "Trade\\n69" and "Signal \\n50.7%" arrive as one header each. Only the
    first line is the name.
    """
    return str(name).split("\n")[0].strip()


def _drop_scratch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns up to and including TP.

    The sheet precomputes win rates and R totals for its own stop scenarios to
    the right of the trade data. This page computes those itself, so reading
    them would double up on the same question with a rule this module does not
    control.
    """
    positions = [i for i, col in enumerate(df.columns)
                 if col == LAST_TRADE_COLUMN]
    if not positions:
        return df
    return df.iloc[:, :positions[0] + 1]


def load_data(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load 30MC data from CSV, clean it and derive the R and setup columns.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame with added R, Setup and Type columns
    """
    df = pd.read_csv(filepath)
    df.columns = [_normalise_header(col) for col in df.columns]
    df = _drop_scratch_columns(df).copy()
    # A row without a date is an empty sheet line, not a trade.
    df = df[df["Date"].astype(str).str.strip().ne("") & df["Date"].notna()]

    for col in ["SL", "Pullback", "TP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # A loser is exported with Pullback and TP both blank: the stop was hit
    # before any target. Read that Pullback as the stop, not as a clean 0-pip
    # run, or the half-SL limit would score these trades as never filled.
    if {"SL", "Pullback", "TP"} <= set(df.columns):
        stopped = df["Pullback"].isna() & df["TP"].isna()
        df.loc[stopped, "Pullback"] = df.loc[stopped, "SL"]

    for col in ["SL", "Pullback", "TP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Guard the division: a 0-pip stop is not a trade, and would otherwise
    # produce an infinite R.
    reach = df["TP"].where(df["SL"] > 0, 0) / df["SL"].where(df["SL"] > 0, 1)
    stopped_out = df["Pullback"] >= df["SL"]
    df["R"] = reach.where(~stopped_out, -reach)

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

def _strategy_wins(trades: pd.DataFrame, effective_sl) -> pd.Series:
    """
    The family's 1:1 win rule: the trade survived its stop and reached a
    target as far away as that stop.

        Pullback < effective SL   AND   TP >= effective SL
    """
    return (trades['Pullback'] < effective_sl) & (trades['TP'] >= effective_sl)


def _rate_columns(trades: pd.DataFrame, effective_sl) -> Dict[str, str]:
    """
    The three readings every Signal/Strategy table carries, each a bare
    "52.7%" at 1:1.

    Signal: TP > 0 - the trade idea was right and price reached a target. The
    stop is ignored on purpose, so this counts trades whose Pullback exceeded
    SL: the direction was correct but the entry was too early to survive it.

    Strategy: the 1:1 win rule (_strategy_wins) on the row's stop - what
    trading it would actually have returned. Always a subset of Signal; the
    gap between the two is the cost of entry timing.

    Strategy + 1 pip: the same rule with the row's stop padded by
    TYPE_BUFFER_PIPS.

    Args:
        trades: The trades of one table row
        effective_sl: The row's stop - a per-trade Series or one number

    Returns:
        Dict of column name to win rate
    """
    total = len(trades)
    if not total:
        return {name: _win_rate(0, 0) for name in RATE_COLUMNS}
    return {
        'Signal': _win_rate(int((trades['TP'] > 0).sum()), total),
        'Strategy': _win_rate(
            int(_strategy_wins(trades, effective_sl).sum()), total),
        RATE_COLUMNS[2]: _win_rate(int(_strategy_wins(
            trades, effective_sl + TYPE_BUFFER_PIPS).sum()), total),
    }


def _signal_strategy_rows(groups: List[Tuple[str, pd.DataFrame]],
                          column: str) -> pd.DataFrame:
    """
    Score each group of trades on its recorded stops under the four readings
    of _rate_columns. Every grouping table on the page comes through here so
    they cannot drift apart on the rule.

    Args:
        groups: (label, trades) tuples, one per row of the table
        column: Name for the leading column (e.g. 'Day', 'Setup')

    Returns:
        DataFrame with columns: <column>, Trades, then RATE_COLUMNS
    """
    return pd.DataFrame([
        {column: label, 'Trades': len(trades),
         **_rate_columns(trades, trades['SL'])}
        for label, trades in groups
    ])


def _win_rate(wins: int, decided: int) -> str:
    """'52.7%' - wins over the trades that ended in a win or a loss."""
    return f"{(wins / decided * 100) if decided else 0.0:.1f}%"


def calculate_weekday_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-weekday win rates under the four readings of _rate_columns.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Day, Trades, then RATE_COLUMNS
    """
    groups = [(day, df[df['Weekday'] == day]) for day in WEEKDAY_ORDER]
    return _signal_strategy_rows(groups, 'Day')


# Series colours for the Signal/Strategy charts: categorical slots 1-3 of the
# dark-mode palette, validated against the #1e1e1e surface. Text never wears
# them - values and labels stay in the page's text greys.
CHART_SERIES = [('Signal', '#3987e5'), ('Strategy', '#d95926'),
                ('Strategy + 1 pip', '#199e70')]

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
    Grouped horizontal bars of Signal, Strategy and Strategy + 1 pip win
    rate, one group per row.

    Drawn from the very table it sits under - a `<label>, Trades` frame with
    the RATE_COLUMNS - so the chart and the table cannot disagree. All three
    bars share one 0-100% axis, with a dashed line at the 1:1 breakeven, so
    the gap between Signal and Strategy - the cost of entry timing - what a
    pip of stop buys back, and the distance to breakeven read at a glance.
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
    # A table may carry only some of the series - Fixed SL has no + 1 pip.
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
    the two type rows split it, so they sum to it. The four level-specific
    setups ("High Continuation" and so on) are not listed: the level alone
    says nothing, only the type it forms with Direction does.

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
# Stop tables
# ---------------------------------------------------------------------------

# Bands overlap on purpose: 0-10 is the union of 0-5 and 5-10, and each is read
# on its own. Upper bounds are exclusive, so 10+ picks up whatever 0-10 leaves.
SL_RANGES = [
    ("0-5 SL", 0, 5),
    ("0-10 SL", 0, 10),
    ("5-10 SL", 5, 10),
    ("10+ pips", 10, float('inf')),
]


def calculate_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-SL-band Signal and Strategy win rates.

    Reading a band on both tells you whether a stop size is losing trades that
    were directionally right: a wide Signal-to-Strategy gap in one band is an
    entry problem inside that band, not a bad band.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: SL Range, Trades, then RATE_COLUMNS
    """
    groups: List[Tuple[str, pd.DataFrame]] = [('Default', df)]
    groups.extend(
        (label, df[(df['SL'] >= low) & (df['SL'] < high)])
        for label, low, high in SL_RANGES
    )
    return _signal_strategy_rows(groups, 'SL Range')


# Stop sizes to substitute for every trade's recorded SL.
SL_FIXED_PIPS = [3, 4, 5, 6, 7, 8, 9, 10]


def _pip_label(pips: int) -> str:
    """Render a pip count for a table cell: '0 pips', '1 pip', '2 pips'."""
    return f"{pips} pip" if pips == 1 else f"{pips} pips"


def _sl_scenario_statistics(df: pd.DataFrame,
                            scenarios: List[Tuple[str, object]],
                            column: str,
                            with_signal: bool = False) -> pd.DataFrame:
    """
    Score every trade at 1:1 under each stop scenario.

    A trade wins when it survives the scenario's stop and reaches a 1:1 target
    on it:
        Pullback < effective SL   AND   TP >= effective SL

    Every stop table goes through here so they cannot drift apart on the rule.

    Args:
        df: DataFrame with trading data
        scenarios: (label, effective SL) pairs. The effective SL is either a
            per-trade Series or a single number applied to every trade
        column: Name for the leading column (e.g. 'Fixed SL')
        with_signal: Emit `<column>, Trades` and the RATE_COLUMNS instead
            of `<column>, Trades, Notation, Win Rate`, matching the grouping
            tables. Signal (TP > 0) does not depend on the stop, so it repeats
            down every row - it is the ceiling no stop size beats, and each
            row's Strategy says how much of it that stop captures

    Returns:
        DataFrame with columns: <column>, Trades, Notation, Win Rate - or
        <column>, Trades, then RATE_COLUMNS when with_signal is set
    """
    total = len(df)
    results = []

    for label, effective_sl in scenarios:
        row = {column: label, 'Trades': total}
        if with_signal:
            row.update(_rate_columns(df, effective_sl))
        else:
            wins = int(_strategy_wins(df, effective_sl).sum()) if total else 0
            win_rate = (wins / total * 100) if total > 0 else 0.0
            row['Notation'] = f"{wins}W - {total - wins}L"
            row['Win Rate'] = f"{win_rate:.1f}%"
        results.append(row)

    return pd.DataFrame(results)


def calculate_sl_fixed_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    1:1 win/loss statistics for a single stop size used on every trade.

    The recorded safe stop is discarded and replaced by a fixed number of pips,
    so both the survival check and the 1:1 target move to that size. The first
    row keeps the recorded stops, as a baseline to read the rest against.

    There is no Strategy + 1 pip column: a fixed stop padded by a pip is just
    the next fixed size, already the row below.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Fixed SL, Trades, Signal, Strategy
    """
    scenarios: List[Tuple[str, object]] = [('Default', df['SL'])]
    scenarios.extend((_pip_label(pips), float(pips)) for pips in SL_FIXED_PIPS)
    return _sl_scenario_statistics(df, scenarios, 'Fixed SL',
                                   with_signal=True).drop(
        columns=RATE_COLUMNS[2])


# ---------------------------------------------------------------------------
# TP distribution and pullback entries
# ---------------------------------------------------------------------------

# Ten-pip bands, each holding TP >= low and TP < high, so a trade lands in
# exactly one of them.
TP_RANGES = [
    ("0-10", 0, 10),
    ("10-20", 10, 20),
    ("20-30", 20, 30),
    ("30+", 30, float("inf")),
]


def calculate_tp_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trade counts per TP pip range.

    Trades reads "X of Y", where Y is every profitable trade (TP > 0) in the
    dataset. This is a distribution, not a win rate: every trade counted here
    already has TP > 0.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: TP Range, Trades
    """
    total_profitable = int((df['TP'] > 0).sum())
    results = []

    for label, low, high in TP_RANGES:
        in_band = df[(df['TP'] > 0) & (df['TP'] >= low) & (df['TP'] < high)]
        results.append({
            'TP Range': label,
            'Trades': f"{len(in_band)} of {total_profitable}",
        })

    return pd.DataFrame(results)


# The TP chart gives every whole pip from TP_CHART_FIRST to TP_CHART_LAST its
# own column and pools everything from TP_CHART_LAST + 1 up into one "30+".
TP_CHART_FIRST = 8
TP_CHART_LAST = 29

_TP_CHART_CSS = """
    <style>
        .tp-chart { font-family: 'Courier New', monospace; font-size: 11px;
                    color: #e0e0e0; margin-top: 16px; padding: 14px 16px;
                    background-color: #1e1e1e; border: 1px solid #404040; }
        .tp-plot { display: flex; align-items: flex-end; gap: 2px;
                   height: 180px; border-bottom: 1px solid #404040; }
        .tp-col { flex: 1; display: flex; flex-direction: column;
                  justify-content: flex-end; align-items: stretch;
                  height: 100%; }
        .tp-col:hover { background-color: #2a2a2a; }
        .tp-count { text-align: center; margin-bottom: 3px; }
        .tp-bar { background-color: #3987e5; border-radius: 4px 4px 0 0; }
        .tp-axis { display: flex; gap: 2px; margin-top: 4px; color: #808080; }
        .tp-axis span { flex: 1; text-align: center; }
        .tp-pool { margin-left: 10px; }
        .tp-caption { margin-top: 8px; color: #a0a0a0; }
    </style>
"""


def calculate_tp_pip_counts(df: pd.DataFrame) -> pd.Series:
    """
    Profitable trades per whole TP pip, for the TP Range chart.

    A trade lands in the column of its TP rounded down, so 12.6 counts under
    12. Every TP above TP_CHART_LAST pools into the last, "30+", column.
    Trades below TP_CHART_FIRST are not columns; create_tp_chart reports how
    many there are instead of dropping them silently.

    Args:
        df: DataFrame with trading data

    Returns:
        Series of counts indexed '8', '9', ..., '29', '30+'
    """
    pips = df.loc[df['TP'] > 0, 'TP'].astype(float).apply(math.floor)
    counts = {str(pip): int((pips == pip).sum())
              for pip in range(TP_CHART_FIRST, TP_CHART_LAST + 1)}
    counts[f'{TP_CHART_LAST + 1}+'] = int((pips > TP_CHART_LAST).sum())
    return pd.Series(counts)


def create_tp_chart(df: pd.DataFrame) -> str:
    """
    Column chart of how far the profitable trades ran, one column per pip.

    Every column carries its count on the cap, so there is no y-axis, and a
    hover tooltip names the pip and the count. The pooled "30+" column is set
    a little apart, since it spans many pips rather than one.

    Args:
        df: DataFrame with trading data

    Returns:
        HTML string with the chart, or a "No data" note when no trade has
        TP > 0
    """
    profitable = int((df['TP'] > 0).sum()) if not df.empty else 0
    if profitable == 0:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    counts = calculate_tp_pip_counts(df)
    tallest = max(int(counts.max()), 1)

    html = _TP_CHART_CSS + ('<div class="tp-chart" role="img" '
                            'aria-label="Profitable trades per TP pip">'
                            '<div class="tp-plot">')
    pooled = counts.index[-1]
    for label, count in counts.items():
        height = count / tallest * 85
        css = 'tp-col tp-pool' if label == pooled else 'tp-col'
        html += (f'<div class="{css}" title="TP {label} pips: {count} '
                 f'trade{"" if count == 1 else "s"}">'
                 f'<span class="tp-count">{count if count else ""}</span>'
                 f'<div class="tp-bar" style="height: {height:.1f}%;"></div>'
                 '</div>')
    html += '</div><div class="tp-axis">'
    for label in counts.index:
        css = ' class="tp-pool"' if label == pooled else ''
        html += f'<span{css}>{label}</span>'
    html += '</div>'

    below = profitable - int(counts.sum())
    caption = f'TP in pips, {profitable} profitable trades'
    if below:
        caption += f'; {below} below {TP_CHART_FIRST} pips not shown'
    html += f'<div class="tp-caption">{caption}.</div></div>'
    return html


def _format_wml(wins: int, missed: int, losses: int) -> str:
    """Format tapped winners/missed winners/losers into '1W - 2M - 3L'."""
    return f"{wins}W - {missed}M - {losses}L"


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
    Longest winning and losing runs at 1:1, for every trade and for
    Continuation trades.

    A trade is a win under the family's rule - Pullback < SL AND TP >= SL -
    and anything else breaks a winning run, so the stopped-out
    winner-that-wasn't counts as a loss here. Trades are read in date order;
    the export's own order breaks ties within a day. The Continuation row runs
    through only its own trades. Every 30MC trade is a Continuation, so there
    is no Reversal row.

    Args:
        df: DataFrame with trading data, including a Type column

    Returns:
        DataFrame with columns: Trend, Trades, Longest Wins, Longest Losses
    """
    ordered = df.sort_values('Date', kind='stable') if len(df) else df
    groups: List[Tuple[str, pd.Series]] = [
        ('Default', pd.Series(True, index=ordered.index)),
        ('Continuation', ordered['Type'] == 'Continuation'),
    ]

    won = (ordered['Pullback'] < ordered['SL']) & (ordered['TP'] >= ordered['SL'])
    results = []
    for label, mask in groups:
        results.append({
            'Trend': label,
            'Trades': int(mask.sum()),
            'Longest Wins': _longest_run(won[mask], ordered.loc[mask, 'Date'], True),
            'Longest Losses': _longest_run(won[mask], ordered.loc[mask, 'Date'], False),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Half SL entries
# ---------------------------------------------------------------------------

HALF_SL_ENTRIES = ['Default', 'Half SL', 'Limit order on half']


def calculate_half_sl_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    The recorded stop against two ways of halving it, at every RRR.

    - Default: entry at the signal, stop as recorded.
    - Half SL: entry at the signal, stop at SL / 2. A pullback of half the
      stop now takes it out.
    - Limit order on half: a limit SL / 2 into the pullback, stop left at the
      recorded level - so it is SL / 2 from the fill. It fills only when
      Pullback >= SL / 2. Measured from the fill, the pullback shrinks to
      Pullback - SL / 2 and the run grows to TP + SL / 2. A trade that never
      reached a target (TP = 0) is given no run from the fill either.

    All three rows score the trade at its own entry with the family's rule,
    Pullback < SL AND TP >= RRR x SL. Half SL and the limit share a stop
    size, so their R is directly comparable.

    Notation splits every trade three ways, like the Pullback table:
    W = won, M = limit never filled (no trade), L = lost. Win Rate is
    W / (W + L), the trades actually taken; Outcome is W x RRR - L.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Entry, RRR, Trades, Notation, Win Rate, Outcome
    """
    half = df['SL'] / 2
    # No target reached means no run from the fill either.
    run = (df['TP'] + half).where(df['TP'] > 0, 0.0)
    entries = {
        'Default': (pd.Series(True, index=df.index),
                    df['Pullback'], df['TP'], df['SL']),
        'Half SL': (pd.Series(True, index=df.index),
                    df['Pullback'], df['TP'], half),
        'Limit order on half': (df['Pullback'] >= half,
                                df['Pullback'] - half, run, half),
    }

    results = []
    for rrr in RRR_RATIOS:
        for label in HALF_SL_ENTRIES:
            filled, pullback, tp, sl = entries[label]
            won = filled & (pullback < sl) & (tp >= rrr * sl)
            wins = int(won.sum())
            missed = int((~filled).sum())
            losses = len(df) - wins - missed
            taken = wins + losses
            win_rate = (wins / taken * 100) if taken else 0.0
            results.append({
                'Entry': label,
                'RRR': f'1:{rrr}',
                'Trades': taken,
                'Notation': _format_wml(wins, missed, losses),
                'Win Rate': f'{win_rate:.1f}%',
                'Outcome': f'{wins * rrr - losses:+d}R',
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Strategies tables
# ---------------------------------------------------------------------------

FIXED_SL_STRATEGY_VALUES = list(range(2, 11))

# Stop caps for the Strategies tables' Cap SL column. 0 means no cap.
MAX_SL_CAPS = [0, 5, 6]

def _fixed_sl_filter(x: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a filter that replaces the SL column with a fixed x pips."""
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["SL"] = float(x)
        return out
    return _filter


def get_buffer_strategies() -> List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]:
    """
    Strategies the ranked tables test.

    "Fixed SL X" replaces SL with X. It runs with buffer 0 and no cap only -
    a buffer or a cap would just turn it into another fixed size. Capping is
    the Cap SL column's job, on All Trades.
    """
    strategies: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("All Trades", lambda df: df),
    ]
    strategies.extend(
        (f"Fixed SL {x}", _fixed_sl_filter(x)) for x in FIXED_SL_STRATEGY_VALUES
    )
    return strategies


def _buffers_for(strategy_name: str) -> List[float]:
    """Fixed-SL strategies only run with buffer 0."""
    if strategy_name.startswith("Fixed SL "):
        return [0]
    return BUFFER_PIPS


def _caps_for(strategy_name: str) -> List[int]:
    """Fixed-SL strategies only run uncapped."""
    if strategy_name.startswith("Fixed SL "):
        return [0]
    return MAX_SL_CAPS


def _apply_trend(df: pd.DataFrame, trend: str) -> pd.DataFrame:
    """Keep only trades of one setup type ('All' keeps every trade)."""
    return df if trend == 'All' else df[df['Type'] == trend]


def _strategy_row_wins(trades: pd.DataFrame, buffer: float, rrr: float,
                       max_sl: int) -> pd.Series:
    """
    The winning trades of one Strategies row; every other trade is a loss.

    Effective SL = SL + buffer, then capped at max_sl when max_sl > 0. The cap
    wins over the buffer: SL 3 + 1 under a 5 pip cap is 4, SL 5 + 1 is 5. A
    trade wins when it survives that stop and reaches the RRR target on it:
        Pullback < effective SL   AND   TP >= rrr * effective SL
    """
    effective_sl = trades["SL"] + buffer
    if max_sl:
        effective_sl = effective_sl.clip(upper=float(max_sl))
    return ((trades["Pullback"] < effective_sl)
            & (trades["TP"] >= rrr * effective_sl))


def _calculate_stats_with_buffer(trades: pd.DataFrame, strategy_name: str,
                                 buffer: float, rrr_ratio: float = 1,
                                 trend: str = 'All',
                                 max_sl: int = 0) -> Dict:
    """
    Score a filtered set of trades under one Strategies row's stop rule - see
    _strategy_row_wins. Notation is "W – L" and Win Rate W over every trade.
    """
    row = {
        "Strategy": strategy_name,
        "Buffer": f"+{buffer}",
        "Trend": trend,
        "Cap SL": max_sl,
        "RRR": f"1:{rrr_ratio:g}",
        "Trades": len(trades),
    }
    won = int(_strategy_row_wins(trades, buffer, rrr_ratio, max_sl).sum())
    lost = len(trades) - won
    row["Notation"] = f"{won}W – {lost}L"
    row["Win Rate"] = _win_rate(won, won + lost)
    return row


def _sort_strategy_rows(result_df: pd.DataFrame) -> pd.DataFrame:
    """Sort by Strategy (natural order, numbers numerically), Trend, Cap SL,
    then RRR."""
    if result_df.empty:
        return result_df

    def strategy_key(name: str):
        return [int(p) if p.isdigit() else p.lower()
                for p in re.split(r'(\d+)', str(name))]

    def rrr_key(rrr: str):
        try:
            return float(str(rrr).split(':')[-1])
        except (ValueError, IndexError):
            return float('inf')

    sort_index = sorted(
        result_df.index,
        key=lambda i: (
            strategy_key(result_df.at[i, 'Strategy']),
            TREND_FILTERS.index(result_df.at[i, 'Trend']),
            int(result_df.at[i, 'Cap SL']),
            rrr_key(result_df.at[i, 'RRR']),
        ),
    )
    return result_df.loc[sort_index].reset_index(drop=True)


def calculate_buffer_statistics(df: pd.DataFrame,
                                strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Score every strategy across SL buffers, setup trends, stop caps and every
    RRR.

    Trend filters on the setup type, which is known at entry, so it is a
    tradeable filter. The strategy's stop rule runs on the trend's trades.

    Args:
        df: DataFrame with trading data
        strategy_names: Restrict to these strategy names (default: all)

    Returns:
        DataFrame with one row per strategy / buffer / trend / cap / RRR
        combination
    """
    strategies = get_buffer_strategies()
    if strategy_names is not None:
        strategies = [(n, f) for n, f in strategies if n in strategy_names]

    results = []
    for strategy_name, filter_func in strategies:
        for trend in TREND_FILTERS:
            filtered_df = filter_func(_apply_trend(df, trend))
            for max_sl in _caps_for(strategy_name):
                for rrr in RRR_RATIOS:
                    for buffer in _buffers_for(strategy_name):
                        results.append(_calculate_stats_with_buffer(
                            filtered_df, strategy_name, buffer, rrr,
                            trend, max_sl))

    return _sort_strategy_rows(pd.DataFrame(results))


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


def _strategy_curve_label(strategy_name: str, buffer: float, trend: str,
                          max_sl: int) -> str:
    """One line of text naming a Strategies table row, e.g.
    'All Trades +1 / Reversal / cap 5' or 'Fixed SL 6 / Continuation'."""
    parts = [strategy_name]
    if not strategy_name.startswith("Fixed SL "):
        parts[0] += f" +{buffer:g}"
    if trend != 'All':
        parts.append(trend)
    if max_sl:
        parts.append(f"cap {max_sl}")
    return " / ".join(parts)


def calculate_equity_curves(df: pd.DataFrame, rrr: float = 1) -> List[Dict]:
    """
    Cumulative R of every Strategies table row, trade by trade, at one RRR.

    Trades are walked in date order; the export's own order breaks ties within
    a day. A win adds the RRR in R and a loss takes 1R, under the same rule as
    calculate_buffer_statistics (_strategy_row_wins). Every curve
    runs over every trade, so a Continuation-only strategy sits flat through
    the Reversal trades it would not have taken and the curves line up in
    time. Transaction costs are not deducted.

    Args:
        df: DataFrame with trading data, including a Type column
        rrr: Risk-reward ratio to score at

    Returns:
        One dict per strategy row, best final outcome first: label, trades
        (the trades the strategy took), outcome (final R) and curve (the
        cumulative R after each trade, starting from 0, so len(df) + 1 long)
    """
    ordered = df.sort_values('Date', kind='stable') if len(df) else df
    curves = []
    for strategy_name, filter_func in get_buffer_strategies():
        filtered = filter_func(ordered)
        for trend in TREND_FILTERS:
            taken = (pd.Series(True, index=ordered.index) if trend == 'All'
                     else ordered['Type'] == trend)
            for max_sl in _caps_for(strategy_name):
                for buffer in _buffers_for(strategy_name):
                    won = _strategy_row_wins(filtered, buffer, rrr, max_sl)
                    per_trade = pd.Series(-1.0, index=ordered.index)
                    per_trade[won] = float(rrr)
                    per_trade[~taken] = 0.0
                    curve = [0.0] + per_trade.cumsum().round(6).tolist()
                    curves.append({
                        'label': _strategy_curve_label(
                            strategy_name, buffer, trend, max_sl),
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
    Line chart of every strategy's cumulative R, one line per table row.

    The EQUITY_HIGHLIGHT best final outcomes are coloured, labelled at their
    right-hand end and listed in the legend; every other line draws in one
    recessive grey so the field reads as context. Hovering any line brings it
    forward and shows its name and its R at that trade. The dashed line is
    0R, where trading the strategy would have broken even before costs.

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
                f'({curve["outcome"]:+.0f}R)</span>')
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
                f'{escape(curve["label"])} {curve["outcome"]:+.0f}R</text>')
    out += '</svg><div class="eq-tip"></div>'
    out += (f'<p class="eq-caption">Cumulative R over {n_trades} trades in '
            f'date order: +{rrr:g}R per win, -1R per loss, before costs. '
            'Trades outside a strategy\'s trend stay flat. '
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

