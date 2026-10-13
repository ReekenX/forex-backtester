"""
5OB1CC report analysis module.

Backs the static HTML page rendered by utils.report: every table on that page
is computed here, and nothing here needs a Jupyter kernel.

CSV columns (strategies/5OB1CC/data.csv):
    Date, Trade, Range, Strength, Weekday, Hour, Direction, EMA, SL, Pullback,
    TP, Extra, BOS/CH, 30M Leg, Hours Until News, News Event

Two differences from the 15LS1CC export shape this page was modelled on:

- There is no R column. R is derived as TP / SL - how many multiples of the
  recorded safe stop the move was worth - and signed the way the 15LS1CC sheet
  signs it: negative when Pullback reached SL, i.e. the reach was there but the
  stop did not survive to collect it.
- A loss is recorded as TP 0 or a blank TP. Both load as 0, so `TP > 0` is the
  Signal rule here exactly as it is there.

Range, Strength and Extra are populated on a handful of rows only, so no table
reads them. Hour, EMA, BOS/CH and 30M Leg are known at entry on every row, so
each gets a Signal/Strategy table.
"""

import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# RRR ratios to test (1:3 is the ceiling - see CLAUDE.md)
RRR_RATIOS = [1, 2, 3]

# Extra pip buffer values to test in the Strategies tables
BUFFER_PIPS = [0, 1, 2, 3]

WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# London session hours in Lithuanian time, as exported.
HOUR_ORDER = list(range(10, 19))

DEFAULT_CSV = "../strategies/5OB1CC/data.csv"


def load_data(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load 5OB1CC data from CSV, clean it and derive the R column.

    A blank TP and a TP of 0 both mean the trade was not profitable, so both
    land on 0 and `TP > 0` reads as "price reached a target".

    R is not in the export, so it is computed as TP / SL and signed negative
    when the stop was taken out first (Pullback >= SL) - the same convention
    the 15LS1CC sheet uses, where a negative R records reach that the recorded
    stop could not hold on to.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame with an added R column
    """
    df = pd.read_csv(filepath)

    for col in ["SL", "Pullback", "TP", "Hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Guard the division: a 0-pip stop is not a trade, and would otherwise
    # produce an infinite R.
    reach = df["TP"].where(df["SL"] > 0, 0) / df["SL"].where(df["SL"] > 0, 1)
    stopped_out = df["Pullback"] >= df["SL"]
    df["R"] = reach.where(~stopped_out, -reach)

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


def create_html_table(df: pd.DataFrame, sort_id: Optional[str] = None,
                      first_col_width: Optional[str] = None) -> str:
    """
    Create a dark-mode HTML table with styled formatting.

    Args:
        df: DataFrame to convert to an HTML table
        sort_id: When set, gives the table this DOM id and makes every column
            whose cells carry a percentage click-to-sort by it, descending only
        first_col_width: When set, pins the label column to this width (e.g.
            "40%") and switches the table to fixed layout so it is honoured

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

    for idx, col in enumerate(df.columns):
        width_style = (f' style="width: {first_col_width};"'
                       if idx == 0 and first_col_width else "")
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
                    value, row[rrr_col] if rrr_col else None)
            cls_attr = f' class="{css_class}"' if css_class else ""
            html += f"                <td{cls_attr}>{value}</td>\n"
        html += "            </tr>\n"

    html += "        </tbody>\n    </table>\n"
    return html


def create_sortable_table(df: pd.DataFrame, table_id: str,
                          sortable: bool = True,
                          first_col_width: Optional[str] = None) -> str:
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
            html += f"                <td>{row[col]}</td>\n"
        html += "            </tr>\n"

    html += "        </tbody>\n    </table>\n"
    return html


# ---------------------------------------------------------------------------
# Signal / Strategy grouping tables
# ---------------------------------------------------------------------------

def _format_wl(wins: int, losses: int, total: int) -> str:
    """Format wins/losses/win rate into a compact '13W - 12L (52.0%)'."""
    win_rate = (wins / total * 100) if total > 0 else 0.0
    return f"{wins}W - {losses}L ({win_rate:.1f}%)"


def _signal_strategy_rows(groups: List[Tuple[str, pd.DataFrame]],
                          column: str) -> pd.DataFrame:
    """
    Score each group of trades under both readings of a trade.

    Signal: TP > 0 - the trade idea was right and price reached a target. The
    stop is ignored on purpose, so this counts trades whose Pullback exceeded
    SL: the direction was correct but the entry was too early to survive it.

    Strategy: Pullback < SL AND TP >= SL - what trading it at 1:1 would
    actually have returned. The trade had to survive its stop AND reach a 1:1
    target, so it is always a subset of Signal.

    The gap between the two is the cost of entry timing. Every grouping table
    on the page comes through here so they cannot drift apart on the rule.

    Args:
        groups: (label, trades) pairs, one per row of the table
        column: Name for the leading column (e.g. 'Day', 'Hour')

    Returns:
        DataFrame with columns: <column>, Trades, Signal, Strategy
    """
    results = []

    for label, trades in groups:
        total = len(trades)
        signals = int((trades['TP'] > 0).sum()) if total else 0
        strategy = int((
            (trades['Pullback'] < trades['SL']) & (trades['TP'] >= trades['SL'])
        ).sum()) if total else 0

        results.append({
            column: label,
            'Trades': total,
            'Signal': _format_wl(signals, total - signals, total),
            'Strategy': _format_wl(strategy, total - strategy, total),
        })

    return pd.DataFrame(results)


def calculate_weekday_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-weekday Signal and Strategy win rates.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Day, Trades, Signal, Strategy
    """
    groups = [(day, df[df['Weekday'] == day]) for day in WEEKDAY_ORDER]
    return _signal_strategy_rows(groups, 'Day')


def calculate_hour_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-hour Signal and Strategy win rates, in Lithuanian time.

    The hour is known before the trade is taken, so it is a tradeable filter -
    unlike Pullback or TP. The first row, "Default", covers every trade.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Hour, Trades, Signal, Strategy
    """
    groups: List[Tuple[str, pd.DataFrame]] = [('Default', df)]
    groups.extend((f"{hour}h", df[df['Hour'] == hour]) for hour in HOUR_ORDER)
    return _signal_strategy_rows(groups, 'Hour')


def calculate_ema_alignment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA alignment statistics under both readings of a trade.

    "Aligned" means the trade ran with the EMA signal, "Against" means it
    fought it. Both are known at entry, so either is a tradeable filter. This
    is the 5OB1CC counterpart of the 15LS1CC page's 4H Alignment table - the
    export carries an EMA signal rather than a 4H bias.

    Args:
        df: DataFrame with trading data, including an "EMA" column

    Returns:
        DataFrame with columns: EMA Alignment, Trades, Signal, Strategy
    """
    aligned = df['EMA'] == df['Direction']
    groups = [
        ('Default', df),
        ('Aligned', df[aligned]),
        ('Against', df[~aligned]),
    ]
    return _signal_strategy_rows(groups, 'EMA Alignment')


def calculate_structure_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break-of-structure vs change-of-character statistics.

    The structure type that produced the order block is known at entry, so it
    is a tradeable filter.

    Args:
        df: DataFrame with trading data, including a "BOS/CH" column

    Returns:
        DataFrame with columns: Structure, Trades, Signal, Strategy
    """
    groups: List[Tuple[str, pd.DataFrame]] = [('Default', df)]
    groups.extend(
        (label, df[df['BOS/CH'] == label])
        for label in ('BOS', 'CH')
    )
    return _signal_strategy_rows(groups, 'Structure')


# The 30M Leg export records where price sat relative to the 30-minute leg.
LEG_ORDER = ['Above H', 'Above L', 'Below H', 'Below L']


def calculate_htf_leg_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    30-minute leg position statistics under both readings of a trade.

    Args:
        df: DataFrame with trading data, including a "30M Leg" column

    Returns:
        DataFrame with columns: 30M Leg, Trades, Signal, Strategy
    """
    groups: List[Tuple[str, pd.DataFrame]] = [('Default', df)]
    groups.extend((label, df[df['30M Leg'] == label]) for label in LEG_ORDER)
    return _signal_strategy_rows(groups, '30M Leg')


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


def _parse_wl(cell: str) -> Tuple[int, int]:
    """Read wins and losses back out of a '13W - 12L (52.0%)' cell."""
    wins, losses = re.match(r'\s*(\d+)W - (\d+)L', cell).groups()
    return int(wins), int(losses)


def create_signal_strategy_chart(stats: pd.DataFrame) -> str:
    """
    Paired horizontal bars of Signal and Strategy win rate, one pair per row.

    Drawn from the very table it sits under - a `<label>, Trades, Signal,
    Strategy` frame - so the chart and the table cannot disagree. Both bars
    share one 0-100% axis, with a dashed line at the 1:1 breakeven, so the gap
    between the two readings - the cost of entry timing - and the distance to
    breakeven read at a glance. Every bar carries its value at the tip and a
    hover tooltip with the W/L counts.

    Args:
        stats: DataFrame whose first column labels the rows and which carries
            Trades, Signal and Strategy columns in the _format_wl shape

    Returns:
        HTML string with the chart, or a "No data" note for an empty frame
    """
    if stats.empty:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    label_column = stats.columns[0]

    html = _CHART_CSS + '<div class="wr-chart" role="img" aria-label="'
    html += f'Signal and Strategy win rate by {label_column}">'
    html += '<div class="wr-legend">'
    for name, colour in CHART_SERIES:
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
        for name, colour in CHART_SERIES:
            wins, losses = _parse_wl(row[name])
            total = wins + losses
            rate = wins / total * 100 if total else 0.0
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
        DataFrame with columns: SL Range, Trades, Signal, Strategy
    """
    groups: List[Tuple[str, pd.DataFrame]] = [('Default', df)]
    groups.extend(
        (label, df[(df['SL'] >= low) & (df['SL'] < high)])
        for label, low, high in SL_RANGES
    )
    return _signal_strategy_rows(groups, 'SL Range')


# Pips to pad the safe stop with. 0 means the stop is left as recorded.
SL_BUFFER_PIPS = [0, 1, 2, 3, 4, 5]

# Stop sizes to substitute for every trade's recorded SL.
SL_FIXED_PIPS = [3, 4, 5, 6, 7, 8, 9, 10]

# Caps for the Cap SL table, widest first.
SL_MAX_PIPS = [9, 8, 7, 6, 5, 4]


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
        with_signal: Emit `<column>, Trades, Signal, Strategy` instead of
            `<column>, Trades, Notation, Win Rate`, matching the grouping
            tables. Signal (TP > 0) does not depend on the stop, so it repeats
            down every row - it is the ceiling no stop size beats, and each
            row's Strategy says how much of it that stop captures

    Returns:
        DataFrame with columns: <column>, Trades, Notation, Win Rate - or
        <column>, Trades, Signal, Strategy when with_signal is set
    """
    total = len(df)
    signals = int((df['TP'] > 0).sum()) if total else 0
    results = []

    for label, effective_sl in scenarios:
        wins = int((
            (df['Pullback'] < effective_sl) & (df['TP'] >= effective_sl)
        ).sum()) if total else 0

        row = {column: label, 'Trades': total}
        if with_signal:
            row['Signal'] = _format_wl(signals, total - signals, total)
            row['Strategy'] = _format_wl(wins, total - wins, total)
        else:
            win_rate = (wins / total * 100) if total > 0 else 0.0
            row['Notation'] = f"{wins}W - {total - wins}L"
            row['Win Rate'] = f"{win_rate:.1f}%"
        results.append(row)

    return pd.DataFrame(results)


def calculate_sl_buffer_statistics(df: pd.DataFrame,
                                   with_signal: bool = False) -> pd.DataFrame:
    """
    1:1 win/loss statistics for progressively wider stops.

    Each row pads every trade's safe stop by N pips. A wider stop survives
    deeper pullbacks, but the 1:1 target moves out by the same amount, so a
    trade whose TP was only just enough can drop out.

    Args:
        df: DataFrame with trading data
        with_signal: Emit the Signal/Strategy shape the charts read, instead
            of the Notation/Win Rate shape the table shows - same scenarios,
            same win rule

    Returns:
        DataFrame with columns: SL Buffer, Trades, Notation, Win Rate - or
        SL Buffer, Trades, Signal, Strategy when with_signal is set
    """
    scenarios: List[Tuple[str, object]] = [
        ("Default" if pips == 0 else _pip_label(pips), df['SL'] + pips)
        for pips in SL_BUFFER_PIPS
    ]
    return _sl_scenario_statistics(df, scenarios, 'SL Buffer',
                                   with_signal=with_signal)


def calculate_sl_fixed_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    1:1 win/loss statistics for a single stop size used on every trade.

    The recorded safe stop is discarded and replaced by a fixed number of pips,
    so both the survival check and the 1:1 target move to that size. The first
    row keeps the recorded stops, as a baseline to read the rest against.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Fixed SL, Trades, Signal, Strategy
    """
    scenarios: List[Tuple[str, object]] = [('Default', df['SL'])]
    scenarios.extend((_pip_label(pips), float(pips)) for pips in SL_FIXED_PIPS)
    return _sl_scenario_statistics(df, scenarios, 'Fixed SL', with_signal=True)


def calculate_sl_max_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    1:1 win/loss statistics with the recorded stop capped at N pips.

    Effective SL = min(SL, N): a trade whose safe stop is already inside the
    cap keeps it, a wider one is tightened to N. Both the survival check and
    the 1:1 target use the capped stop, so a capped trade is stopped out
    sooner but needs less TP to win. The first row keeps the recorded stops.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Cap SL, Trades, Signal, Strategy
    """
    scenarios: List[Tuple[str, object]] = [('Default', df['SL'])]
    scenarios.extend(
        (_pip_label(pips), df['SL'].clip(upper=float(pips)))
        for pips in SL_MAX_PIPS
    )
    return _sl_scenario_statistics(df, scenarios, 'Cap SL', with_signal=True)


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


PULLBACK_ENTRY_PIPS = [0, 1, 2, 3]
PULLBACK_BUFFER_PIPS = 1


def _format_wml(wins: int, missed: int, losses: int) -> str:
    """Format tapped winners/missed winners/losers into '1W - 2M - 3L'."""
    return f"{wins}W - {missed}M - {losses}L"


def calculate_pullback_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limit-order pullback-entry statistics at 1:1 RRR.

    A limit order placed N pips into the pullback fills only if price pulled
    back at least N pips (Pullback >= N). One extra level scales the fill
    threshold to each trade's own stop: "Half" fills when the pullback reached
    at least half the SL.

    Every level except Default then repeats with the stop padded by
    PULLBACK_BUFFER_PIPS, scored with the Adding Buffer rule. The fill
    threshold does not move - "Half" still means half the recorded SL - so
    each buffered row differs from its plain twin by the stop alone.

    Every row splits the whole dataset three ways:

        winner = Pullback < SL + buffer AND TP >= SL + buffer
        W = tapped winners (winner AND the limit filled)
        M = missed winners (winner, but the pullback never reached the limit)
        L = losers - not winners whatever the pullback, so the same on every
            row that shares a buffer

    The first row, "Default", is the 0-pip level: no limit order, every trade
    taken at the signal, so nothing is missed. It matches the Default row of
    the other stop tables.

    Args:
        df: DataFrame with trading data

    Returns:
        DataFrame with columns: Pullback, Trades
    """
    levels = [
        ('Default' if n == 0 else _pip_label(n), df['Pullback'] >= n)
        for n in PULLBACK_ENTRY_PIPS
    ]
    levels.append(('Half', df['Pullback'] >= df['SL'] / 2))

    buffer = PULLBACK_BUFFER_PIPS
    rows = [(label, tapped, 0) for label, tapped in levels]
    rows.extend(
        (f"{label} + {_pip_label(buffer)} buffer", tapped, buffer)
        for label, tapped in levels[1:]
    )

    results = []
    for label, tapped, pad in rows:
        effective_sl = df['SL'] + pad
        winner = (df['Pullback'] < effective_sl) & (df['TP'] >= effective_sl)
        wins = int((tapped & winner).sum())
        missed = int((~tapped & winner).sum())
        losses = int((~winner).sum())
        results.append({
            'Pullback': label,
            'Trades': _format_wml(wins, missed, losses),
        })

    return pd.DataFrame(results)


# Segment colours for the pullback chart: W and M take categorical slots 1
# and 2 of the dark palette, the same pair as the Signal/Strategy charts. L is
# a recessive grey - it is the same on every row of a stop, so it is context
# rather than a series to compare.
PULLBACK_SEGMENTS = [
    ('W', 'Tapped winners', '#3987e5'),
    ('M', 'Missed winners', '#d95926'),
    ('L', 'Losers', '#4a4a4a'),
]

_PULLBACK_CHART_CSS = """
    <style>
        .pb-chart { font-family: 'Courier New', monospace; font-size: 12px;
                    color: #e0e0e0; margin-top: 16px; padding: 14px 16px;
                    background-color: #1e1e1e; border: 1px solid #404040; }
        .pb-legend { display: flex; gap: 18px; margin: 0 0 8px 200px;
                     color: #a0a0a0; }
        .pb-swatch { display: inline-block; width: 10px; height: 10px;
                     border-radius: 2px; margin-right: 6px;
                     vertical-align: -1px; }
        .pb-row { display: grid; grid-template-columns: 200px 1fr;
                  align-items: center; padding: 5px 0; }
        .pb-row:hover { background-color: #2a2a2a; }
        .pb-group { margin-top: 12px; border-top: 1px solid #2d2d2d;
                    padding-top: 11px; }
        .pb-stack { display: flex; gap: 2px; height: 20px; }
        .pb-seg { display: flex; align-items: center; justify-content: center;
                  overflow: hidden; white-space: nowrap; font-size: 11px;
                  color: #ffffff; }
        .pb-seg:first-child { border-radius: 4px 0 0 4px; }
        .pb-seg:last-child { border-radius: 0 4px 4px 0; }
    </style>
"""


def _parse_wml(cell: str) -> Tuple[int, int, int]:
    """Read W, M and L back out of a '25W - 12M - 11L' cell."""
    wins, missed, losses = re.match(
        r'\s*(\d+)W - (\d+)M - (\d+)L', cell).groups()
    return int(wins), int(missed), int(losses)


def create_pullback_chart(stats: pd.DataFrame) -> str:
    """
    One 100% stacked bar per Pullback row: tapped winners, missed winners and
    losers, in the order the table names them.

    Drawn from the table's own cells, so the two cannot disagree. Every row
    covers every trade once, so the bars share one length and the W segment
    shrinking into M as the limit goes deeper is the whole story. The buffered
    rows start a second group, since they score against a wider stop. Counts
    sit inside the segments that have room; every segment has a hover tooltip.

    Args:
        stats: DataFrame from calculate_pullback_statistics

    Returns:
        HTML string with the chart, or a "No data" note for an empty frame
    """
    if stats.empty:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    html = _PULLBACK_CHART_CSS + ('<div class="pb-chart" role="img" '
                                  'aria-label="Tapped winners, missed winners '
                                  'and losers by limit depth">'
                                  '<div class="pb-legend">')
    for key, name, colour in PULLBACK_SEGMENTS:
        html += (f'<span><span class="pb-swatch" style="background-color: '
                 f'{colour};"></span>{key} = {name}</span>')
    html += '</div>'

    grouped = False
    for _, row in stats.iterrows():
        label = row['Pullback']
        counts = _parse_wml(row['Trades'])
        total = sum(counts) or 1
        css = 'pb-row'
        if label.endswith('buffer') and not grouped:
            css += ' pb-group'
            grouped = True
        html += (f'<div class="{css}"><div>{label}</div>'
                 '<div class="pb-stack">')
        for (key, name, colour), count in zip(PULLBACK_SEGMENTS, counts):
            if count == 0:
                continue
            share = count / total * 100
            # Room for the count only when the segment is wide enough.
            text = str(count) if share >= 4 else ''
            html += (f'<div class="pb-seg" title="{label} - {name}: {count} '
                     f'({share:.1f}% of trades)" style="flex: {count}; '
                     f'background-color: {colour};">{text}</div>')
        html += '</div></div>'

    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Strategies tables
# ---------------------------------------------------------------------------

MIN_SL_VALUES = [0]
MAX_SL_VALUES = [0, 5]
FIXED_SL_STRATEGY_VALUES = list(range(2, 11))
MAX_SL_STRATEGY_VALUES = list(range(3, 11))


def _apply_min_sl(df: pd.DataFrame, min_sl: int) -> pd.DataFrame:
    """Keep only trades whose original SL is strictly greater than min_sl pips."""
    return df if min_sl == 0 else df[df["SL"] > min_sl]


def _apply_max_sl(df: pd.DataFrame, max_sl: int) -> pd.DataFrame:
    """Keep only trades whose original SL is <= max_sl pips (0 disables)."""
    return df if max_sl == 0 else df[df["SL"] <= max_sl]


def _fixed_sl_filter(x: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a filter that replaces the SL column with a fixed x pips."""
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["SL"] = float(x)
        return out
    return _filter


def _max_sl_filter(x: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Return a filter that caps the SL at x pips: effective SL = min(SL, x).

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
    Strategies the ranked tables test.

    "Fixed SL X" replaces SL with X; "Max SL X" caps SL at min(SL, X). Both run
    with buffer 0 only - a buffer would undo the fixed or capped stop.
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
    """Fixed-SL and Max-SL strategies only run with buffer 0."""
    if strategy_name.startswith("Fixed SL ") or strategy_name.startswith("Max SL "):
        return [0]
    return BUFFER_PIPS


def _calculate_stats_with_buffer(trades: pd.DataFrame, strategy_name: str,
                                 buffer: float, rrr_ratio: float = 1) -> Dict:
    """
    Score a filtered set of trades with extra pips added to every SL.

    Effective SL = SL + buffer. A trade wins when it survives that stop and
    reaches the RRR target on it:
        Pullback < effective SL   AND   TP >= rrr * effective SL
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
    winning_mask = (
        (trades["Pullback"] < effective_sl)
        & (trades["TP"] >= rrr_ratio * effective_sl)
    )

    wins = int(winning_mask.sum())
    losses = total_trades - wins

    return {
        "Strategy": strategy_name,
        "Buffer": f"+{buffer}",
        "Min SL": 0,
        "Max SL": 0,
        "RRR": rrr_label,
        "Trades": total_trades,
        "Notation": f"{wins}W – {losses}L",
        "Win Rate": f"{wins / total_trades * 100:.1f}%",
    }


def _sort_strategy_rows(result_df: pd.DataFrame) -> pd.DataFrame:
    """Sort by Strategy (natural order, numbers numerically) then RRR ascending."""
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
            rrr_key(result_df.at[i, 'RRR']),
            int(result_df.at[i, 'Min SL']),
            int(result_df.at[i, 'Max SL']),
        ),
    )
    return result_df.loc[sort_index].reset_index(drop=True)


def calculate_buffer_statistics(df: pd.DataFrame,
                                strategy_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Score every strategy across SL buffers, Min/Max SL gates and every RRR.

    Args:
        df: DataFrame with trading data
        strategy_names: Restrict to these strategy names (default: all)

    Returns:
        DataFrame with one row per strategy / gate / RRR / buffer combination
    """
    strategies = get_buffer_strategies()
    if strategy_names is not None:
        strategies = [(n, f) for n, f in strategies if n in strategy_names]

    results = []
    for strategy_name, filter_func in strategies:
        for min_sl in MIN_SL_VALUES:
            for max_sl in MAX_SL_VALUES:
                gated = _apply_max_sl(_apply_min_sl(df, min_sl), max_sl)
                filtered_df = filter_func(gated)
                for rrr in RRR_RATIOS:
                    for buffer in _buffers_for(strategy_name):
                        stats = _calculate_stats_with_buffer(
                            filtered_df, strategy_name, buffer, rrr)
                        stats["Min SL"] = min_sl
                        stats["Max SL"] = max_sl
                        results.append(stats)

    return _sort_strategy_rows(pd.DataFrame(results))


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


def create_r_histogram_exact(df: pd.DataFrame) -> str:
    """
    Plain R counts (1R-10R), not cumulative.

    Each trade is counted once, in the bucket of the highest whole R it
    reached, so the bars sum to the number of trades with an R value.

    Args:
        df: DataFrame with an R column

    Returns:
        HTML string with a styled horizontal bar chart
    """
    return _create_r_histogram(calculate_r_counts(df), "R Distribution (Exact)")


# ---------------------------------------------------------------------------
# Three setups comparison
# ---------------------------------------------------------------------------

# RRR the per-trade three-setups comparison is scored at.
THREE_SETUPS_RRR = 2

# (group header, column header, DataFrame key). A group header spans every
# consecutive column that repeats it, so the order here is the table's order.
THREE_SETUPS_COLUMNS = [
    ('', 'Date', 'Date'),
    ('', 'Weekday', 'Weekday'),
    ('', 'Hour', 'Hour'),
    ('', 'Trade', 'Trade'),
    ('', 'Direction', 'Direction'),
    ('Regular', 'SL', 'Regular SL'),
    ('Regular', 'Pullback', 'Regular Pullback'),
    ('Regular', 'TP', 'Regular TP'),
    ('Regular', 'Outcome', 'Regular Outcome'),
    ('Aggressive', 'SL', 'Aggressive SL'),
    ('Aggressive', 'ROI', 'Aggressive ROI'),
    ('Waiter', 'SL', 'Waiter SL'),
    ('Waiter', 'Pullback', 'Waiter Pullback'),
    ('Waiter', 'TP', 'Waiter TP'),
    ('Waiter', 'ROI', 'Waiter ROI'),
]

# Columns holding a running R total, coloured by sign.
THREE_SETUPS_CUMULATIVE_KEYS = ('Regular Outcome', 'Aggressive ROI', 'Waiter ROI')


def _pip_cell(value: Optional[float]) -> str:
    """
    Render a pip figure to one decimal; None renders as an empty cell.

    Halving a stop recorded to one decimal produces a second one (5.5 -> 2.75),
    which is finer than the data warrants, so the cell rounds back to one:
    2.75 -> '2.8'. Trailing zeros are trimmed, so the recorded figures still
    read as they do in the CSV (34.0 -> '34').

    Rounded through Decimal on the value's own repr rather than with format(),
    so a half-way figure goes up: 2.15 is a hair under 2.15 in binary and
    would otherwise render as '2.1'.
    """
    if value is None:
        return ""
    rounded = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded:f}".rstrip("0").rstrip(".")


def _whole_pip_cell(value: Optional[float]) -> str:
    """
    Render a pip figure rounded to whole pips: 25.15 -> '25', 54.85 -> '55'.

    Half-way values round up rather than to even, so a 16.5 pip target reads
    as 17 instead of Python's default 16.
    """
    if value is None:
        return ""
    return str(int(math.floor(value + 0.5)))


def _r_cell(total: int) -> str:
    """Render a running R total with an explicit sign: '+4R', '-1R', '0R'."""
    return f"+{total}R" if total > 0 else f"{total}R"


def calculate_three_setups_comparison(
        df: pd.DataFrame, rrr_ratio: int = THREE_SETUPS_RRR) -> pd.DataFrame:
    """
    Score every trade under three entry rules and track each one's running R.

    All three read the same recorded trade; they differ only in where the stop
    sits and where the entry is:

    - Regular: the signal entry on the recorded safe stop.
        win = Pullback < SL AND TP >= rrr * SL
    - Aggressive: the same entry, but risking half the safe stop. The target
      halves with the stop, so it needs less run but survives less pullback.
        win = Pullback < SL/2 AND TP >= rrr * SL/2
    - Waiter: a limit order SL/2 into the pullback. It only trades when the
      pullback actually reached that price (Pullback >= SL/2); otherwise the
      trade is missed and its cells stay empty. The safe stop does not move,
      so the entry is SL/2 closer to it and SL/2 further from the target:
        stop = SL/2, pullback = Pullback - SL/2, target distance = TP + SL/2
        win = (Pullback - SL/2) < SL/2 AND (TP + SL/2) >= rrr * SL/2

    Cells round for display only - SL and Pullback to one decimal, the
    Waiter's TP to whole pips. Every win test uses the exact figure.

    Halving a small safe stop can land under the 1.1 pip broker minimum; those
    rows are informational rather than tradeable.

    A win adds rrr R, a loss subtracts 1R, and a missed Waiter trade adds
    nothing - its running total carries forward so the column still reads as an
    equity curve.

    Args:
        df: DataFrame with trading data
        rrr_ratio: Risk-reward ratio every setup is scored at

    Returns:
        DataFrame with the columns listed in THREE_SETUPS_COLUMNS, one row per
        trade, in the order the trades were recorded
    """
    rows = []
    regular_r = aggressive_r = waiter_r = 0

    for _, trade in df.iterrows():
        sl = float(trade['SL'])
        pullback = float(trade['Pullback'])
        tp = float(trade['TP'])
        half = sl / 2

        row = {key: trade.get(key, '')
               for key in ('Date', 'Weekday', 'Trade', 'Direction')}
        hour = trade.get('Hour', '')
        row['Hour'] = f"{int(hour)}h" if hour != '' and not pd.isna(hour) else ""

        regular_won = pullback < sl and tp >= rrr_ratio * sl
        regular_r += rrr_ratio if regular_won else -1
        row['Regular SL'] = _pip_cell(sl)
        row['Regular Pullback'] = _pip_cell(pullback)
        row['Regular TP'] = _pip_cell(tp) if tp > 0 else ""
        row['Regular Outcome'] = _r_cell(regular_r)

        aggressive_won = pullback < half and tp >= rrr_ratio * half
        aggressive_r += rrr_ratio if aggressive_won else -1
        row['Aggressive SL'] = _pip_cell(half)
        row['Aggressive ROI'] = _r_cell(aggressive_r)

        if pullback >= half:
            waiter_pullback = pullback - half
            waiter_tp = tp + half
            waiter_won = waiter_pullback < half and waiter_tp >= rrr_ratio * half
            waiter_r += rrr_ratio if waiter_won else -1
            row['Waiter SL'] = _pip_cell(half)
            row['Waiter Pullback'] = _pip_cell(waiter_pullback)
            row['Waiter TP'] = _whole_pip_cell(waiter_tp) if tp > 0 else ""
        else:
            row['Waiter SL'] = ""
            row['Waiter Pullback'] = ""
            row['Waiter TP'] = ""
        row['Waiter ROI'] = _r_cell(waiter_r)

        rows.append(row)

    keys = [key for _, _, key in THREE_SETUPS_COLUMNS]
    return pd.DataFrame(rows, columns=keys)


def _three_setups_header_groups() -> List[Tuple[str, int]]:
    """Collapse THREE_SETUPS_COLUMNS into (group header, colspan) pairs."""
    groups: List[List] = []
    for group, _, _ in THREE_SETUPS_COLUMNS:
        if groups and groups[-1][0] == group:
            groups[-1][1] += 1
        else:
            groups.append([group, 1])
    return [(group, span) for group, span in groups]


def create_three_setups_table(stats: pd.DataFrame,
                              table_id: str = "three-setups-table") -> str:
    """
    Render the per-trade three-setups comparison with grouped headers.

    Rows are the trade log in recorded order and the R columns are cumulative,
    so the table is deliberately not sortable - re-ordering it would make the
    running totals meaningless.

    Args:
        stats: DataFrame from calculate_three_setups_comparison
        table_id: Unique DOM id for this table

    Returns:
        HTML string with a dark-mode styled table
    """
    if stats.empty:
        return ("<p style='color: #e0e0e0; background-color: #1e1e1e; "
                "padding: 10px;'>No data</p>")

    html = """
    <style>
        .setups-table {
            border-collapse: collapse;
            width: 100%;
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .setups-table th {
            background-color: #2d2d2d;
            color: #e0e0e0;
            padding: 6px 8px;
            text-align: left;
            border: 1px solid #404040;
            font-weight: bold;
            white-space: nowrap;
        }
        .setups-table td {
            padding: 4px 8px;
            border: 1px solid #404040;
            white-space: nowrap;
        }
        .setups-table tr:hover {
            background-color: #2a2a2a;
        }
        .setups-table .group-start {
            border-left: 2px solid #6b6b6b;
        }
        .setups-table .group-head {
            text-align: center;
        }
        .setups-table .positive-edge {
            color: #4ade80;
        }
        .setups-table .negative-edge {
            color: #f87171;
        }
    </style>
    """

    # A group boundary gets a heavier left border so the three setups read as
    # blocks rather than one 15-column run.
    boundaries = set()
    index = 0
    for _, span in _three_setups_header_groups():
        if index:
            boundaries.add(index)
        index += span

    html += f'<table class="setups-table" id="{table_id}">\n        <thead>\n'

    html += "            <tr>\n"
    index = 0
    for group, span in _three_setups_header_groups():
        cls = ' class="group-head group-start"' if index else ' class="group-head"'
        html += f'                <th colspan="{span}"{cls}>{group}</th>\n'
        index += span
    html += "            </tr>\n            <tr>\n"

    for idx, (_, header, _) in enumerate(THREE_SETUPS_COLUMNS):
        cls = ' class="group-start"' if idx in boundaries else ""
        html += f"                <th{cls}>{header}</th>\n"
    html += "            </tr>\n        </thead>\n        <tbody>\n"

    for _, row in stats.iterrows():
        html += "            <tr>\n"
        for idx, (_, _, key) in enumerate(THREE_SETUPS_COLUMNS):
            classes = ["group-start"] if idx in boundaries else []
            value = row[key]
            if key in THREE_SETUPS_CUMULATIVE_KEYS:
                # Flat at zero is neither, so it stays the default colour.
                total = float(str(value).rstrip("R"))
                if total > 0:
                    classes.append("positive-edge")
                elif total < 0:
                    classes.append("negative-edge")
            cls = f' class="{" ".join(classes)}"' if classes else ""
            html += f"                <td{cls}>{value}</td>\n"
        html += "            </tr>\n"

    html += "        </tbody>\n    </table>\n"
    return html
