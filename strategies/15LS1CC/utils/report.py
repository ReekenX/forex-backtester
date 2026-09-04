"""
Static HTML report for the 15LS1CC strategy.

Renders every analysis table from confirmation_candle into a single
self-contained HTML file, so the lab can be read in a browser without a
Jupyter kernel. Pair it with a file watcher to get a live-updating page:

    watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

The page reloads itself only when the build id actually changes - never on a
timer. Over http:// it polls build-id.txt with fetch(); as a file:// URL that
fetch is blocked by the browser, so it loads build-id.js through a <script> tag
instead, which file:// does allow. Sort state and scroll position are saved to
sessionStorage and replayed after the reload, so a rebuild never throws away a
column you sorted.
"""

import hashlib
import html
from typing import Callable, List, Tuple

import pandas as pd

from utils.confirmation_candle import (
    THREE_SETUPS_RRR,
    _calculate_buffer_statistics_filtered,
    _calculate_stats_with_buffer,
    _create_sl_sortable_table,
    calculate_pullback_statistics,
    calculate_sl_statistics,
    calculate_sl_buffer_small_sl_statistics,
    calculate_sl_fixed_statistics,
    calculate_sl_buffer_statistics,
    calculate_sl_reduction_statistics,
    calculate_tp_statistics,
    calculate_three_setups_comparison,
    calculate_weekday_statistics,
    create_html_table,
    create_three_setups_table,
    create_r_histogram_combined,
    create_r_histogram_exact,
    get_buffer_strategies,
    RRR_RATIOS,
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

BUILD_ID_FILENAME = "build-id.txt"
# Same id as a script, so file:// pages can read it: fetch() is blocked there by
# the browser, but a <script src> with a cache-busting query still loads.
BUILD_ID_JS_FILENAME = "build-id.js"

# Milliseconds between change checks. The page only reloads on a real change.
POLL_INTERVAL_MS = 1000


def _weekday_section(df: pd.DataFrame) -> str:
    return create_html_table(calculate_weekday_statistics(df))


def _sl_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(calculate_sl_statistics(df), "sl-range-stats", sortable=False, first_col_width="50%")


def _sl_reduction_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(
        calculate_sl_reduction_statistics(df), "sl-reduction-table", sortable=False, first_col_width="50%"
    )


def _sl_buffer_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(
        calculate_sl_buffer_statistics(df), "sl-buffer-table", sortable=False, first_col_width="50%"
    )


def _sl_buffer_small_sl_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(
        calculate_sl_buffer_small_sl_statistics(df), "sl-buffer-small-table", sortable=False, first_col_width="50%"
    )


def _sl_fixed_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(
        calculate_sl_fixed_statistics(df), "sl-fixed-table"
    )


def _tp_section(df: pd.DataFrame) -> str:
    return create_html_table(calculate_tp_statistics(df))


def _pullback_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(calculate_pullback_statistics(df), "pullback-analysis")


def _three_setups_section(df: pd.DataFrame) -> str:
    return create_three_setups_table(
        calculate_three_setups_comparison(df), "three-setups-table")


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
        names = [name for name, _ in get_buffer_strategies()]
        stats = _calculate_buffer_statistics_filtered(df, names)
        stats = stats[stats["RRR"] == f"1:{rrr}"]
        # Distinct from the section anchor: a duplicate id would make
        # getElementById inside the sort script return the <section>.
        return create_html_table(
            _sort_by_win_rate(stats), sort_id=f"strategies-1-{rrr}-table"
        )
    return build


# (anchor, nav label, heading, note, builder) - mirrors the notebook's cell order.
SECTIONS: List[Tuple[str, str, str, str, Callable[[pd.DataFrame], str]]] = [
    (
        "weekday",
        "Weekday",
        "Weekday Statistics",
        "Win = Pullback &lt; SL AND TP &gt; 0, i.e. the trade survived its stop and finished profitable at any distance.",
        _weekday_section,
    ),
    (
        "sl-range",
        "SL Range",
        "SL Range Statistics",
        "Win rate per safe-stop band at 1:1. Win = Pullback &lt; SL AND TP &gt;= SL.",
        _sl_section,
    ),
    (
        "sl-reduction",
        "Reducing SL",
        "Reducing SL Statistics",
        "Every stop shaved by N pips: win = Pullback &lt; SL - N AND TP &gt;= SL - N. "
        "The broker minimum stop is 1.1 pips.",
        _sl_reduction_section,
    ),
    (
        "sl-buffer",
        "Adding Buffer",
        "Adding Buffer To SL Statistics",
        "The mirror of the table above - every stop padded by N pips: "
        "win = Pullback &lt; SL + N AND TP &gt;= SL + N.",
        _sl_buffer_section,
    ),
    (
        "sl-buffer-small",
        "Adding Buffer (SL < 5)",
        "Adding Buffer To SL When SL < 5 Statistics",
        "The same buffer, but only on stops under 5 pips - trades with a stop of "
        "5.0 or wider keep it as recorded. All trades are still scored.",
        _sl_buffer_small_sl_section,
    ),
    (
        "sl-fixed",
        "Fixed SL",
        "Fixed SL Statistics",
        "The recorded stop replaced by one size for every trade, so both the "
        "survival check and the 1:1 target move to it. Default keeps the "
        "recorded stops as a baseline.",
        _sl_fixed_section,
    ),
    (
        "tp-range",
        "TP Range",
        "TP Range Statistics",
        "How far the profitable trades ran.",
        _tp_section,
    ),
    (
        "pullback",
        "Pullback",
        "Pullback Range Statistics",
        "Filling a limit order N pips into the pullback instead of taking the "
        "signal. Half fills at half the stop. M counts winners the limit never "
        "filled, so they are excluded from Trades and Win Rate.",
        _pullback_section,
    ),
    (
        "r-distribution",
        "R Distribution",
        "R Distribution",
        "Cumulative: a trade reaching 3R also passed through 1R and 2R.",
        create_r_histogram_combined,
    ),
    (
        "r-distribution-exact",
        "R Distribution (Exact)",
        "R Distribution (Exact)",
        "70% of all trades are 1-3 R ranges.",
        create_r_histogram_exact,
    ),
]

# One Strategies table per RRR, each ranked by win rate descending.
SECTIONS.extend(
    (
        f"strategies-1-{rrr}",
        f"Strategies 1:{rrr}",
        f"Strategies (1:{rrr} RRR)",
        "All Trades / Fixed SL / Max SL across SL buffers and Min/Max SL gates, "
        f"ranked by win rate. Breakeven at 1:{rrr} is {100 / (1 + rrr):.1f}%.",
        _strategies_section_for(rrr),
    )
    for rrr in RRR_RATIOS
)

# Trade log rather than a summary, so it sits after the ranked tables.
SECTIONS.append((
    "three-setups",
    "Three Setups",
    f"Three Setups Comparison on 1:{THREE_SETUPS_RRR} RRR",
    f"Every trade under three entry rules, scored at 1:{THREE_SETUPS_RRR}. "
    "<b>Regular</b> takes the signal on the recorded safe stop. "
    "<b>Aggressive</b> takes the same entry risking half that stop, so the "
    "target halves with it. <b>Waiter</b> rests a limit order half a stop into "
    "the pullback - it only trades when the pullback reached that price, and "
    "blank cells are the trades it never filled. Outcome and ROI are "
    f"cumulative R: &minus;1R per loss, +{THREE_SETUPS_RRR}R per win. Halved "
    "stops under the 1.1 pip broker minimum are informational.",
    _three_setups_section,
))


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

    baseline = _calculate_stats_with_buffer(df, "All Trades", 0, 1)
    win_rate = float(baseline["Win Rate"].rstrip("%"))
    dates = df["Date"].astype(str)

    return [
        ("Trades", str(len(df)), "neutral"),
        ("Trading days", str(df["Date"].nunique()), "neutral"),
        ("Date range", f"{dates.min()} to {dates.max()}", "neutral"),
        (
            "Win rate (1:1, no buffer)",
            baseline["Win Rate"],
            "positive" if win_rate >= 50 else "negative",
        ),
        ("Notation", baseline["Notation"], "neutral"),
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
    to loading build-id.js through a <script> tag, which is the one way a
    file:// page can read a sibling file. If neither works the page simply stops
    watching rather than reloading on a timer.

    Sort state and scroll position are saved to sessionStorage and replayed
    after load, so a rebuild does not throw away a column the reader sorted.
    Both table sorts are descending-only and idempotent, so replaying one click
    on the same header restores the order exactly.
    """
    return f"""<script>
(function () {{
    var BUILD_ID = "{build_id}";
    var SCROLL_KEY = "15ls1cc-scroll";
    var SORT_KEY = "15ls1cc-sorts";
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
                    + "or serve the folder over http:// (see labs/render.py)");
        }};
        document.head.appendChild(s);
    }}

    pollOverHttp();
}})();
</script>"""


def build_report(df: pd.DataFrame, generated_at: str, build_id: str,
                 live_reload: bool = True) -> str:
    """
    Build the full HTML document for the 15LS1CC lab.

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
<title>15C Lab</title>
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
    <h1 style="color: {TEXT}; font-size: 22px; margin: 0 0 6px;">15C</h1>
    <p style="color: {TEXT_MUTED}; font-size: 12px; margin: 0;">
        15-minute Leg Structure, 1-minute Confirmation Candle &middot; EURUSD, London session
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
    Generated by labs/render.py from strategies/15LS1CC/v5_data.csv
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
<title>15C Lab - build failed</title>
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
