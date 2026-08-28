"""
Static HTML report for the 15LS1CC strategy.

Renders every analysis table from confirmation_candle into a single
self-contained HTML file, so the lab can be read in a browser without a
Jupyter kernel. Pair it with a file watcher to get a live-updating page:

    watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

The page reloads itself when the build id changes. Served over HTTP it polls
build-id.txt and reloads only on a real change; opened as a file:// URL that
poll is blocked by the browser, so it falls back to a timed reload. Either way
the scroll position is preserved across reloads.
"""

import hashlib
import html
from typing import Callable, List, Tuple

import pandas as pd

from utils.confirmation_candle import (
    _calculate_buffer_statistics_filtered,
    _calculate_stats_with_buffer,
    _create_sl_sortable_table,
    calculate_pullback_statistics,
    calculate_sl_buffer_impact_statistics,
    calculate_sl_statistics,
    calculate_sl_vs_buffer_statistics,
    calculate_tp_statistics,
    calculate_weekday_statistics,
    create_html_table,
    create_r_histogram_combined,
    get_buffer_strategies,
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

# Seconds between reload checks (poll when served over HTTP, blind reload on file://).
POLL_INTERVAL_MS = 1000
FALLBACK_RELOAD_MS = 2000


def _weekday_section(df: pd.DataFrame) -> str:
    return create_html_table(calculate_weekday_statistics(df))


def _sl_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(calculate_sl_statistics(df), "sl-range-stats")


def _sl_buffer_impact_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(
        calculate_sl_buffer_impact_statistics(df), "sl-buffer-impact"
    )


def _sl_vs_buffer_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(calculate_sl_vs_buffer_statistics(df), "sl-vs-buffer")


def _tp_section(df: pd.DataFrame) -> str:
    return create_html_table(calculate_tp_statistics(df))


def _pullback_section(df: pd.DataFrame) -> str:
    return _create_sl_sortable_table(calculate_pullback_statistics(df), "pullback-analysis")


def _strategies_section(df: pd.DataFrame) -> str:
    names = [name for name, _ in get_buffer_strategies()]
    return create_html_table(
        _calculate_buffer_statistics_filtered(df, names), sort_id="strategies-table"
    )


# (anchor, nav label, heading, note, builder) - mirrors the notebook's cell order.
SECTIONS: List[Tuple[str, str, str, str, Callable[[pd.DataFrame], str]]] = [
    (
        "weekday",
        "Weekday",
        "Weekday Statistics",
        "Win rate per day of week, with and without a 2 pip stop buffer.",
        _weekday_section,
    ),
    (
        "sl-range",
        "SL Range",
        "SL Range Statistics",
        "Win rate per safe-stop band at 1:1 to 1:4. Win = Pullback &lt; SL AND TP &gt;= RRR x SL.",
        _sl_section,
    ),
    (
        "buffer-impact",
        "Buffer Impact",
        "SL Range Buffer Impact (1:1 RRR)",
        "How many winners survive as the stop is padded +1 / +2 / +3 pips.",
        _sl_buffer_impact_section,
    ),
    (
        "sl-vs-buffer",
        "SL vs Buffer",
        "Limiting SL vs Adding Buffer (1:1 RRR)",
        "Each lever alone. Same win rule as the Strategies table, so the numbers are comparable.",
        _sl_vs_buffer_section,
    ),
    (
        "tp-range",
        "TP Range",
        "TP Range Statistics",
        "How far the profitable trades ran.",
        _tp_section,
    ),
    (
        "r-distribution",
        "R Distribution",
        "R Distribution",
        "Cumulative: a trade reaching 3R also passed through 1R and 2R.",
        create_r_histogram_combined,
    ),
    (
        "pullback",
        "Pullback",
        "Pullback Range Statistics",
        "Entering on a pullback of 0 / 1 / 2 / 3 pips instead of at the signal.",
        _pullback_section,
    ),
    (
        "strategies",
        "Strategies",
        "Strategies",
        "All Trades / Fixed SL / Max SL across SL buffers and Min/Max SL gates.",
        _strategies_section,
    ),
]


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
    Reload the page when the build id changes.

    Over HTTP, poll build-id.txt and reload only on a real change. On file://
    the fetch is blocked, so fall back to a timed reload. Scroll position is
    saved either way so the page does not jump.
    """
    return f"""<script>
(function () {{
    var BUILD_ID = "{build_id}";
    var KEY = "15ls1cc-scroll";
    try {{
        var saved = sessionStorage.getItem(KEY);
        if (saved !== null) window.scrollTo(0, parseInt(saved, 10) || 0);
        window.addEventListener("beforeunload", function () {{
            try {{ sessionStorage.setItem(KEY, String(window.scrollY)); }} catch (e) {{}}
        }});
    }} catch (e) {{}}

    var badge = document.getElementById("reload-mode");
    function setMode(text) {{ if (badge) badge.textContent = text; }}

    function poll() {{
        fetch("{BUILD_ID_FILENAME}?t=" + Date.now(), {{ cache: "no-store" }})
            .then(function (r) {{
                if (!r.ok) throw new Error("bad status");
                return r.text();
            }})
            .then(function (id) {{
                id = id.trim();
                if (id && id !== BUILD_ID) {{ location.reload(); return; }}
                setMode("watching for changes");
                setTimeout(poll, {POLL_INTERVAL_MS});
            }})
            .catch(function () {{
                setMode("reloading every {FALLBACK_RELOAD_MS}ms (open over http:// to poll instead)");
                setTimeout(function () {{ location.reload(); }}, {FALLBACK_RELOAD_MS});
            }});
    }}
    poll();
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
<title>15LS1CC Lab</title>
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
    <h1 style="color: {TEXT}; font-size: 22px; margin: 0 0 6px;">15LS1CC</h1>
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
    Generated by labs/render.py from strategies/15LS1CC/data.csv
</footer>
{_reload_script(build_id) if live_reload else ''}
</body>
</html>"""


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
<title>15LS1CC Lab - build failed</title>
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
    (out_path.parent / BUILD_ID_FILENAME).write_text(build_id, encoding="utf-8")
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
    (out_path.parent / BUILD_ID_FILENAME).write_text(build_id, encoding="utf-8")
    return build_id
