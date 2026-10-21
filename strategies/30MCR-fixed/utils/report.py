"""
Static HTML report for the 30MCR strategy traded on a fixed stop.

The 30MCR page cut down to what a fixed stop and a Reward column can answer,
rendered from this strategy's trades.csv. Every analysis table comes from utils.continuation_reversal, so the
lab can be read in a browser without a Jupyter kernel. Pair it with a file
watcher to get a live-updating page:

    watchexec -w strategies/30MCR-fixed -e py,csv -- poetry run python labs/render_30MCR_fixed.py

The page reloads itself only when the build id actually changes - never on a
timer. Over http:// it polls build-id.txt with fetch(); as a file:// URL that
fetch is blocked by the browser, so it loads build-id.js through a <script> tag
instead, which file:// does allow. Sort state and scroll position are saved to
sessionStorage and replayed after the reload, so a rebuild never throws away a
column the reader sorted.

Every page shares a build directory, so the sidecar files are named after
this page rather than reusing another page's names.
"""

import hashlib
import html
from typing import Callable, List, Tuple

import pandas as pd

from utils.continuation_reversal import (
    BE_AT_1R_RRRS,
    HALF_SL_AT_1R_RRRS,
    QUARTER_SL_AT_1R_RRRS,
    SCALING_RRRS,
    RRR_RATIOS,
    RATE_COLUMNS,
    calculate_equity_curves,
    calculate_scaling_rrr_statistics,
    calculate_scaling_statistics,
    calculate_setup_statistics,
    calculate_strategy_statistics,
    calculate_streak_statistics,
    calculate_weekday_statistics,
    create_equity_chart,
    create_html_table,
    create_r_histogram_combined,
    create_signal_strategy_chart,
    create_sortable_table,
    strategy_wins,
)

# Palette from CLAUDE.md's standard table styling.
BG_PAGE = "#161616"
BG_PANEL = "#1e1e1e"
BG_RAISED = "#2d2d2d"
TEXT = "#e0e0e0"
TEXT_MUTED = "#a0a0a0"
POSITIVE = "#4ade80"
NEGATIVE = "#f87171"
BORDER = "#404040"

MAX_WIDTH = "1200px"

PAGE_TITLE = "30MCR Fixed"
PAGE_SUBTITLE = ("30 minute continuation and reversal setups on a fixed stop "
                 "&middot; EURUSD &middot; London session")
SOURCE_CSV = "strategies/30MCR-fixed/trades.csv"

# Named after this page: the other reports write their own pairs into
# the same build directory, and a shared name would have each page reload on
# another page's id.
BUILD_ID_FILENAME = "build-id-30mcr-fixed.txt"
# Same id as a script, so file:// pages can read it: fetch() is blocked there by
# the browser, but a <script src> with a cache-busting query still loads.
BUILD_ID_JS_FILENAME = "build-id-30mcr-fixed.js"

# Milliseconds between change checks. The page only reloads on a real change.
POLL_INTERVAL_MS = 1000

SCROLL_STORAGE_KEY = "30mcr-fixed-scroll"
SORT_STORAGE_KEY = "30mcr-fixed-sorts"


# Every table on the page pins its label column to this width, so they all
# line up and the result columns get the room.
FIRST_COL_WIDTH = "30%"


def _with_chart(stats: pd.DataFrame, table: str) -> str:
    """Put the Signal/Strategy chart under the table it was drawn from."""
    return table + create_signal_strategy_chart(stats)


def _rate_table(stats: pd.DataFrame, table_id: str,
                sortable: bool = True) -> str:
    """A Signal/Strategy table: win rates coloured against 50%, its chart
    underneath."""
    return _with_chart(stats, create_sortable_table(
        stats, table_id, sortable=sortable, first_col_width=FIRST_COL_WIDTH,
        colour_columns=tuple(RATE_COLUMNS)))


def _weekday_section(df: pd.DataFrame) -> str:
    return _rate_table(calculate_weekday_statistics(df), "weekday-table",
                       sortable=False)


def _setup_section(df: pd.DataFrame) -> str:
    return _rate_table(calculate_setup_statistics(df), "setup-table")


def _streaks_section(df: pd.DataFrame) -> str:
    return create_sortable_table(
        calculate_streak_statistics(df), "streaks-table", sortable=False,
        first_col_width=FIRST_COL_WIDTH
    )


def _scaling_section(df: pd.DataFrame) -> str:
    """The second trade's own record, then the same trades on fixed targets
    next to the first trade's result."""
    subheading = (f'<p style="color: {TEXT_MUTED}; font-size: 12px; '
                  'margin: 18px 0 8px;">At each RRR: the second trade on a '
                  'fixed target. <b>Base</b> is the first trade alone, as in '
                  'the Strategies tables; <b>Combined</b> adds the second '
                  'trade to it.</p>')
    return (
        create_html_table(calculate_scaling_statistics(df),
                          first_col_width=FIRST_COL_WIDTH, rrr="1:1")
        + subheading
        + create_html_table(calculate_scaling_rrr_statistics(df),
                            first_col_width=FIRST_COL_WIDTH)
    )


def _sort_by_win_rate(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Order rows by win rate, highest first.

    Sorted here rather than left to the click-to-sort script so the table is
    already ranked on first paint, before any JavaScript runs.
    """
    if stats.empty:
        return stats
    key = (
        stats["Win Rate"].astype(str).str.rstrip("%")
        .apply(lambda v: float(v) if v.replace(".", "", 1).isdigit() else -1.0)
    )
    return (
        stats.assign(_wr=key)
        .sort_values("_wr", ascending=False, kind="stable")
        .drop(columns="_wr")
        .reset_index(drop=True)
    )


def _strategies_section_for(rrr: int) -> Callable[[pd.DataFrame], str]:
    """Build the Strategies table for a single RRR, ranked by win rate."""
    def build(df: pd.DataFrame) -> str:
        stats = calculate_strategy_statistics(df)
        # One RRR per table, named in the heading, so the column is dropped;
        # the Win Rate is still coloured against that RRR's breakeven.
        stats = stats[stats["RRR"] == f"1:{rrr}"].drop(columns="RRR")
        # Distinct from the section anchor: a duplicate id would make
        # getElementById inside the sort script return the <section>.
        return create_html_table(
            _sort_by_win_rate(stats), sort_id=f"strategies-1-{rrr}-table",
            first_col_width=FIRST_COL_WIDTH, rrr=f"1:{rrr}",
        ) + create_equity_chart(
            calculate_equity_curves(df, rrr), rrr,
            chart_id=f"strategies-1-{rrr}-equity")
    return build


RATE_COLUMNS_NOTE = (
    "Two 1:1 win rates per row. <b>Signal</b> = Reward &gt; 0: the idea was "
    "right, stop ignored. <b>Strategy</b> = Pullback &lt; SL AND Reward &gt;= "
    "1R: what trading it would have returned. Green is above the 50% "
    "breakeven, red below."
)

# (anchor, nav label, heading, note, builder). Every trade has the same stop
# and its Reward is already net of costs, so the 30MCR page's stop-scenario
# tables have nothing to vary here and are left out.
SECTIONS: List[Tuple[str, str, str, str, Callable[[pd.DataFrame], str]]] = [
    (
        "weekday",
        "Weekday",
        "Weekday Analysis",
        RATE_COLUMNS_NOTE,
        _weekday_section,
    ),
    (
        "setup",
        "Setup",
        "Setup Analysis",
        "Trades split by what the 30-minute structure did before the signal: "
        "continuation through the level against reversal off it. The setup "
        "is known before the trade is taken, so it is a tradeable filter. "
        "Default is every trade, and the two type rows sum to it. Same two "
        "readings as the weekday table; every rate header is click-to-sort.",
        _setup_section,
    ),
    (
        "r-distribution",
        "R Distribution",
        "R Distribution",
        "Reward as exported: the most each trade could have made, in R, after "
        "commissions. Cumulative: a trade reaching 3R also passed through 1R "
        "and 2R. Stopped-out trades count by the reach they had.",
        create_r_histogram_combined,
    ),
    (
        "streaks",
        "Streaks",
        "Streaks",
        "Longest unbroken runs of winners and losers at 1:1, read in date "
        "order (win = Pullback &lt; SL AND Reward &gt;= 1R, so a stopped-out "
        "trade breaks a winning run). Each cell gives the run's length and "
        "the dates of its first and last trade. <b>Continuation</b> and "
        "<b>Reversal</b> run through only their own trades, so a loss in one "
        "type does not break the other's streak - the sequence trading that "
        "type alone would have produced.",
        _streaks_section,
    ),
    (
        "scaling",
        "Scaling",
        "Scaling",
        "A second trade, opened when the first reaches 1R. <b>Scaled</b> is "
        "how many trades opened one. A win is a second trade that ran 1R or "
        "more; -1R is a stopped-out one, so <b>Win Rate</b> is its 1:1 rate. "
        "<b>Avg R</b> and <b>Avg Win</b> read every winner at the most it "
        "reached - a ceiling no fixed target collects. <b>Best</b> is the "
        "furthest one ran.",
        _scaling_section,
    ),
]

BE_AT_1R_NOTE = (
    " <b>+ BE at 1R</b> rows move the stop to entry once the trade reaches "
    "1R, read off Scaling: -1R means price came back to entry first, 0R; "
    "any other value means it never did, so the trade wins if Reward reached "
    "the target and is 0R otherwise. A blank Scaling keeps the normal rule. "
    "Notation is W – BE – L and Win Rate W / (W + L).")

HALF_SL_AT_1R_NOTE = (
    " <b>+ Half SL at 1R</b> rows pull the stop to SL / 2 once the trade "
    "reaches 1R: a win needs Reward at the target and Pullback under SL / 2, "
    "anything else past 1R loses 0.5R. Pullback is the deepest point of the "
    "whole trade, so a dip past SL / 2 before 1R counts against it too. "
    "Notation is W – HL – L, HL the half-stop losses; Win Rate is W / Trades.")

QUARTER_SL_AT_1R_NOTE = (
    " <b>+ Quarter SL at 1R</b> rows do the same with the stop cut by a "
    "quarter, to SL x 0.75: Pullback must stay under SL x 0.75, and a "
    "stop-out past 1R loses 0.75R. Notation is W – QL – L.")

SCALING_NOTE = (
    " <b>+ Scaling</b> rows also take the second trade, opened at 1R: it "
    "wins RRR when Scaling reached RRR, otherwise loses 1R, so a trade whose "
    "two positions both lose costs 2R. A trade stopped out on its own stop "
    "opened no second trade. Notation and Win Rate count positions, so W + "
    "L is Trades plus the second trades.")

# One Strategies table per RRR, each ranked by win rate descending.
SECTIONS.extend(
    (
        f"strategies-1-{rrr}",
        f"Strategies 1:{rrr}",
        f"Strategies (1:{rrr} RRR)",
        f"Every trade, then each setup trend alone. Win = Pullback &lt; SL AND "
        f"Reward &gt;= {rrr}R; Outcome = W x {rrr} - L. "
        f"Ranked by win rate. Breakeven at 1:{rrr} is {100 / (1 + rrr):.1f}%. "
        "The chart under the table walks every row trade by trade in date "
        "order and plots its cumulative R."
        + (BE_AT_1R_NOTE if rrr in BE_AT_1R_RRRS else "")
        + (HALF_SL_AT_1R_NOTE if rrr in HALF_SL_AT_1R_RRRS else "")
        + (QUARTER_SL_AT_1R_NOTE if rrr in QUARTER_SL_AT_1R_RRRS else "")
        + (SCALING_NOTE if rrr in SCALING_RRRS else ""),
        _strategies_section_for(rrr),
    )
    for rrr in RRR_RATIOS
)


def compute_build_id(df: pd.DataFrame) -> str:
    """
    Short digest of the loaded data, used to decide whether to reload the page.

    Derived from the data rather than the clock so an unchanged rebuild does
    not bounce the browser.
    """
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _summary_cards(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Build the headline figures as (label, value, tone) triples.

    Tone is 'positive', 'negative' or 'neutral' and only drives text colour.
    """
    if df.empty:
        return [("Trades", "0", "neutral")]

    wins = int(strategy_wins(df).sum())
    win_rate = wins / len(df) * 100
    dates = df["Date"].astype(str)
    # Signal: the idea was right and price reached a target, whatever the
    # pullback did on the way. Always at or above the 1:1 win rate.
    signal_rate = int((df["Reward"] > 0).sum()) / len(df) * 100

    return [
        ("Trades", str(len(df)), "neutral"),
        ("Trading days", str(df["Date"].nunique()), "neutral"),
        ("Date range", f"{dates.min()} to {dates.max()}", "neutral"),
        ("Notation", f"{wins}W – {len(df) - wins}L", "neutral"),
        (
            "Signal (Reward > 0)",
            f"{signal_rate:.1f}%",
            "positive" if signal_rate >= 50 else "negative",
        ),
        (
            "Win rate (1:1)",
            f"{win_rate:.1f}%",
            "positive" if win_rate >= 50 else "negative",
        ),
    ]


def _render_summary(df: pd.DataFrame) -> str:
    tones = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": TEXT}
    cards = []
    for label, value, tone in _summary_cards(df):
        # Long values (the date range) would clip at the 1200px cap, so step the
        # type down and let that card claim the width it needs.
        font_size = 18 if len(value) <= 12 else 13
        cards.append(
            f"""<div style="background-color: {BG_PANEL}; border: 1px solid {BORDER};
                border-radius: 6px; padding: 12px 16px; flex: 1 1 auto;">
                <div style="color: {TEXT_MUTED}; font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 6px; white-space: nowrap;">
                    {html.escape(label)}</div>
                <div style="color: {tones[tone]}; font-size: {font_size}px; font-weight: bold;
                    white-space: nowrap;">{html.escape(value)}</div>
            </div>"""
        )
    return (
        '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 28px;">'
        + "".join(cards)
        + "</div>"
    )


def _render_nav() -> str:
    links = "".join(
        f'<a href="#{anchor}" style="color: {TEXT_MUTED}; text-decoration: none; '
        f'padding: 6px 10px; border: 1px solid {BORDER}; border-radius: 4px; '
        f'font-size: 12px; white-space: nowrap;">{html.escape(label)}</a>'
        for anchor, label, _, _, _ in SECTIONS
    )
    return (
        f'<nav style="position: sticky; top: 0; z-index: 10; background-color: {BG_PAGE}; '
        f'padding: 12px 0; margin-bottom: 24px; border-bottom: 1px solid {BORDER}; '
        f'display: flex; flex-wrap: wrap; gap: 8px;">{links}</nav>'
    )


def _render_section(anchor: str, heading: str, note: str, body: str) -> str:
    return f"""<section id="{anchor}" style="margin-bottom: 44px; scroll-margin-top: 70px;">
        <h2 style="color: {TEXT}; font-size: 18px; margin: 0 0 4px;">{html.escape(heading)}</h2>
        <p style="color: {TEXT_MUTED}; font-size: 12px; margin: 0 0 12px;">{note}</p>
        <div style="overflow-x: auto;">{body}</div>
    </section>"""


def _reload_script(build_id: str) -> str:
    """
    Reload the page only when the build id actually changes, and put the view
    back the way the reader left it.

    Change detection tries fetch() first (works over http://), then falls back
    to loading the build-id script through a <script> tag, which is the one way
    a file:// page can read a sibling file. If neither works the page simply
    stops watching rather than reloading on a timer.

    Sort state and scroll position are saved to sessionStorage and replayed
    after load, so a rebuild does not throw away a column the reader sorted.
    Both table sorts are descending-only and idempotent, so replaying one click
    on the same header restores the order exactly.
    """
    return f"""<script>
(function () {{
    var BUILD_ID = "{build_id}";
    var SCROLL_KEY = "{SCROLL_STORAGE_KEY}";
    var SORT_KEY = "{SORT_STORAGE_KEY}";
    var POLL = {POLL_INTERVAL_MS};

    var badge = document.getElementById("reload-mode");
    function setMode(text) {{ if (badge) badge.textContent = text; }}

    function readSorts() {{
        try {{ return JSON.parse(sessionStorage.getItem(SORT_KEY)) || {{}}; }}
        catch (e) {{ return {{}}; }}
    }}
    function writeSorts(v) {{
        try {{ sessionStorage.setItem(SORT_KEY, JSON.stringify(v)); }} catch (e) {{}}
    }}

    // Remember which header the reader sorted by, per table.
    document.addEventListener("click", function (ev) {{
        var el = ev.target;
        var th = el && el.closest ? el.closest("th.sortable") : null;
        if (!th) return;
        var table = th.closest("table");
        if (!table || !table.id) return;
        var sorts = readSorts();
        sorts[table.id] = th.cellIndex;
        writeSorts(sorts);
    }}, true);

    // Replay those sorts on load.
    var sorts = readSorts();
    Object.keys(sorts).forEach(function (id) {{
        var table = document.getElementById(id);
        if (!table || !table.tHead || !table.tHead.rows.length) return;
        var th = table.tHead.rows[0].cells[sorts[id]];
        if (th && th.classList.contains("sortable")) th.click();
    }});

    // Restore scroll last, so re-sorting cannot shift it.
    try {{
        var y = sessionStorage.getItem(SCROLL_KEY);
        if (y !== null) window.scrollTo(0, parseInt(y, 10) || 0);
        window.addEventListener("beforeunload", function () {{
            try {{ sessionStorage.setItem(SCROLL_KEY, String(window.scrollY)); }} catch (e) {{}}
        }});
    }} catch (e) {{}}

    function changed(id) {{ return id && id !== BUILD_ID; }}

    function pollOverHttp() {{
        fetch("{BUILD_ID_FILENAME}?t=" + Date.now(), {{ cache: "no-store" }})
            .then(function (r) {{
                if (!r.ok) throw new Error("bad status");
                return r.text();
            }})
            .then(function (id) {{
                if (changed(id.trim())) {{ location.reload(); return; }}
                setMode("watching for changes");
                setTimeout(pollOverHttp, POLL);
            }})
            .catch(probeOverFile);
    }}

    function probeOverFile() {{
        var s = document.createElement("script");
        s.src = "{BUILD_ID_JS_FILENAME}?t=" + Date.now();
        s.onload = function () {{
            s.remove();
            if (changed(window.__LAB_BUILD_ID__)) {{ location.reload(); return; }}
            setMode("watching for changes (file://)");
            setTimeout(probeOverFile, POLL);
        }};
        s.onerror = function () {{
            s.remove();
            setMode("auto-reload unavailable - refresh manually, "
                    + "or serve the folder over http:// (see labs/render_30MCR_fixed.py)");
        }};
        document.head.appendChild(s);
    }}

    pollOverHttp();
}})();
</script>"""


def build_report(df: pd.DataFrame, generated_at: str, build_id: str,
                 live_reload: bool = True) -> str:
    """
    Build the full HTML document for the 30MCR Fixed lab.

    Args:
        df: DataFrame with trading data
        generated_at: Human-readable build timestamp shown in the header
        build_id: Digest used by the page to detect a rebuild
        live_reload: Embed the self-reload script. Set False for a frozen
            snapshot to share, print or screenshot - a reloading page never
            settles for a headless browser.

    Returns:
        Complete self-contained HTML document
    """
    sections = "".join(
        _render_section(anchor, heading, note, builder(df))
        for anchor, _, heading, note, builder in SECTIONS
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PAGE_TITLE} Lab</title>
<style>
    body {{
        background-color: {BG_PAGE};
        color: {TEXT};
        font-family: 'Courier New', monospace;
        box-sizing: border-box;
        max-width: {MAX_WIDTH};
        margin: 0 auto;
        padding: 24px 28px 60px;
    }}
    a:hover {{ background-color: {BG_RAISED}; color: {TEXT} !important; }}
    ::selection {{ background: {BG_RAISED}; }}
</style>
</head>
<body>
<header style="margin-bottom: 24px;">
    <h1 style="color: {TEXT}; font-size: 22px; margin: 0 0 6px;">{PAGE_TITLE}</h1>
    <p style="color: {TEXT_MUTED}; font-size: 12px; margin: 0;">
        {PAGE_SUBTITLE}
    </p>
    <p style="color: {TEXT_MUTED}; font-size: 11px; margin: 8px 0 0;">
        Built {html.escape(generated_at)} &middot; build {html.escape(build_id)}
        {'&middot; <span id="reload-mode">starting up</span>' if live_reload else ''}
    </p>
</header>
{_render_summary(df)}
{_render_nav()}
{sections}
<footer style="color: {TEXT_MUTED}; font-size: 11px; border-top: 1px solid {BORDER};
    padding-top: 14px; margin-top: 20px;">
    Generated by labs/render_30MCR_fixed.py from {SOURCE_CSV}
</footer>
{_reload_script(build_id) if live_reload else ''}
</body>
</html>"""


def _write_build_id(out_dir, build_id: str) -> None:
    """Write the build id twice: as text for fetch(), as JS for file:// pages."""
    (out_dir / BUILD_ID_FILENAME).write_text(build_id, encoding="utf-8")
    (out_dir / BUILD_ID_JS_FILENAME).write_text(
        f'window.__LAB_BUILD_ID__ = "{build_id}";\n', encoding="utf-8")


def build_error_page(message: str, generated_at: str) -> str:
    """
    Page shown when the CSV cannot be read or the tables cannot be built.

    A half-written spreadsheet export raises inside pandas; without this the
    watcher would just exit and leave the last good page on screen, silently
    stale. Rendering the failure instead makes it obvious in the browser.
    """
    build_id = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PAGE_TITLE} Lab - build failed</title>
<style>
    body {{
        background-color: {BG_PAGE};
        color: {TEXT};
        font-family: 'Courier New', monospace;
        box-sizing: border-box;
        max-width: {MAX_WIDTH};
        margin: 0 auto;
        padding: 24px 28px;
    }}
</style>
</head>
<body>
<h1 style="color: {NEGATIVE}; font-size: 20px; margin: 0 0 6px;">Build failed</h1>
<p style="color: {TEXT_MUTED}; font-size: 12px; margin: 0 0 18px;">
    {html.escape(generated_at)} &middot;
    <span id="reload-mode">starting up</span> &middot;
    the page will refresh once the error is fixed
</p>
<pre style="background-color: {BG_PANEL}; border: 1px solid {NEGATIVE};
    border-radius: 6px; padding: 16px; overflow-x: auto; color: {TEXT};
    font-size: 12px; line-height: 1.5;">{html.escape(message)}</pre>
{_reload_script(build_id)}
</body>
</html>"""


def render_error_to_file(message: str, out_path, generated_at: str) -> str:
    """Write the failure page and bump the build id so the browser reloads."""
    build_id = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_error_page(message, generated_at), encoding="utf-8")
    _write_build_id(out_path.parent, build_id)
    return build_id


def render_to_file(df: pd.DataFrame, out_path, generated_at: str,
                   live_reload: bool = True) -> str:
    """
    Write the report and its build-id sidecar next to each other.

    Args:
        df: DataFrame with trading data
        out_path: pathlib.Path for the HTML file (parent dirs are created)
        generated_at: Human-readable build timestamp
        live_reload: Embed the self-reload script (see build_report)

    Returns:
        The build id that was written
    """
    build_id = compute_build_id(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_report(df, generated_at, build_id, live_reload), encoding="utf-8")
    _write_build_id(out_path.parent, build_id)
    return build_id
