"""
Tests for utils.report - the static HTML renderer for the 5OB1CC lab.

The page is a port of the 15LS1CC lab's 15C.html, so these tests pin the same
structural guarantees: one anchor per section, no duplicate DOM ids, a reload
that never fires on a timer, and a build id derived from the data.

Run with: poetry run python -m pytest strategies/5OB1CC/tests/ -v
"""

import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.order_block import (  # noqa: E402
    RRR_RATIOS,
    calculate_buffer_statistics,
    get_buffer_strategies,
    load_data,
)
from utils.report import (  # noqa: E402
    BUILD_ID_FILENAME,
    BUILD_ID_JS_FILENAME,
    SECTIONS,
    SORT_STORAGE_KEY,
    SCROLL_STORAGE_KEY,
    _sort_by_win_rate,
    _summary_cards,
    build_error_page,
    build_report,
    compute_build_id,
    render_error_to_file,
    render_to_file,
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data.csv')


def get_sample_data():
    """Six trades over three days, with one stopped-out winner-that-wasn't."""
    return pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-03', '2025-02-04',
                 '2025-02-04', '2025-02-05', '2025-02-05'],
        'Trade': ['#1', '#2', '#1', '#2', '#1', '#2'],
        'Weekday': ['Monday', 'Monday', 'Tuesday',
                    'Tuesday', 'Wednesday', 'Wednesday'],
        'Hour': [10, 12, 11, 17, 10, 18],
        'Direction': ['Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        'EMA': ['Buy', 'Buy', 'Sell', 'Sell', 'Buy', 'Sell'],
        'SL': [4.4, 7.1, 2.6, 3.0, 4.3, 5.8],
        'Pullback': [0.7, 7.1, 2.7, 0.0, 0.9, 2.4],
        'TP': [34.0, 0.0, 31.0, 9.0, 17.0, 7.0],
        'BOS/CH': ['BOS', 'CH', 'BOS', 'BOS', 'CH', 'BOS'],
        '30M Leg': ['Above H', 'Above L', 'Below H',
                    'Below L', 'Above H', 'Above L'],
        'R': [7.727, 0.0, -11.923, 3.0, 3.953, 1.207],
    })


def get_empty_data():
    return get_sample_data().iloc[0:0].copy()


# --- document shape -------------------------------------------------------

def test_build_report_is_a_full_document():
    html = build_report(get_sample_data(), '2025-09-12 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert html.rstrip().endswith('</html>')
    assert '<title>5OB Lab</title>' in html


def test_build_report_contains_every_section():
    import html as html_mod

    page = build_report(get_sample_data(), '2025-09-12 10:00:00', 'abc123')
    for anchor, label, heading, _, _ in SECTIONS:
        assert f'id="{anchor}"' in page, f'missing section {anchor}'
        assert f'href="#{anchor}"' in page, f'missing nav link for {anchor}'
        # Headings and nav labels are escaped once on the way in, so compare
        # against the escaped form (a heading may contain '<').
        assert html_mod.escape(heading) in page, f'missing heading {heading}'
        assert html_mod.escape(label) in page, f'missing nav label {label}'


def test_headings_are_not_double_escaped():
    """Headings and nav labels pass through html.escape, so they must be
    written as literal text, not pre-escaped entities."""
    page = build_report(get_sample_data(), 'now', 'abc123')
    assert '&amp;lt;' not in page
    assert '&amp;gt;' not in page
    for _, label, heading, _, _ in SECTIONS:
        assert '&lt;' not in label and '&lt;' not in heading, (
            f'pre-escaped entity in {label!r} / {heading!r}')


def test_sections_are_uniquely_anchored():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert len(anchors) == len(set(anchors))


def test_no_duplicate_dom_ids():
    """A section anchor must never collide with a table's sort id: the sort
    script does getElementById(tableId) and would get the <section> instead."""
    html = build_report(get_sample_data(), 'now', 'abc123')
    counts = Counter(re.findall(r'id="([^"]+)"', html))
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f'duplicate ids: {dupes}'


def test_build_report_shows_timestamp_and_build_id():
    html = build_report(get_sample_data(), '2025-09-12 10:00:00', 'abc123')
    assert '2025-09-12 10:00:00' in html
    assert 'abc123' in html


def test_build_report_names_the_strategy_and_its_source():
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert 'strategies/5OB1CC/data.csv' in html
    assert 'order block' in html


def test_build_report_renders_data_values():
    """The tables carry real numbers, not placeholders."""
    html = build_report(get_sample_data(), 'now', 'id')
    assert '<td>Monday</td>' in html
    assert '<td>Wednesday</td>' in html
    assert '<td>10h</td>' in html


def test_build_report_empty_dataset():
    """An empty CSV still renders a readable page rather than raising."""
    html = build_report(get_empty_data(), 'now', 'id')
    assert html.startswith('<!DOCTYPE html>')
    for anchor, _, _, _, _ in SECTIONS:
        assert f'id="{anchor}"' in html


# --- summary strip --------------------------------------------------------

def test_summary_cards_report_dataset_shape():
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    assert cards['Trades'] == '6'
    assert cards['Trading days'] == '3'
    assert cards['Date range'] == '2025-02-03 to 2025-02-05'


def test_summary_win_rate_excludes_stopped_out_trade():
    """Trade #1 on 02-04 has Pullback 2.7 > SL 2.6, so it is a loss even
    though TP is 31 - matching the Strategies table's rule."""
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    # Winners: 34/4.4, 9/3.0, 17/4.3, 7/5.8 -> 4 of 6.
    assert cards['Win rate (1:1, no buffer)'] == '66.7%'
    assert cards['Notation'] == '4W – 2L'


def test_summary_win_rate_tone_flips_at_50_percent():
    df = get_sample_data()
    tones = {label: tone for label, _, tone in _summary_cards(df)}
    assert tones['Win rate (1:1, no buffer)'] == 'positive'

    losing = df.copy()
    losing['TP'] = 0.0
    tones = {label: tone for label, _, tone in _summary_cards(losing)}
    assert tones['Win rate (1:1, no buffer)'] == 'negative'


def test_summary_cards_empty_dataset():
    assert _summary_cards(get_empty_data()) == [('Trades', '0', 'neutral')]


# --- build id -------------------------------------------------------------

def test_build_id_is_stable_for_identical_data():
    assert compute_build_id(get_sample_data()) == compute_build_id(get_sample_data())


def test_build_id_changes_when_data_changes():
    df = get_sample_data()
    changed = df.copy()
    changed.loc[0, 'TP'] = 99.0
    assert compute_build_id(df) != compute_build_id(changed)


def test_build_id_is_short_hex():
    assert re.fullmatch(r'[0-9a-f]{12}', compute_build_id(get_sample_data()))


# --- live reload ----------------------------------------------------------

def test_live_reload_script_present_by_default():
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert 'reload-mode' in html
    assert BUILD_ID_FILENAME in html
    assert 'location.reload()' in html


def test_reload_never_fires_on_a_timer():
    """Every location.reload() must sit behind a build-id comparison. A blind
    timed reload would wipe the reader's column sort every few seconds."""
    html = build_report(get_sample_data(), 'now', 'abc123')
    script = html[html.rindex('<script>\n(function ()'):]
    assert script.count('location.reload()') == 2
    assert script.count('if (changed(') == 2
    assert 'setTimeout(function () { location.reload()' not in script


def test_reload_script_probes_both_transports():
    """fetch() for http://, a <script> tag for file:// where fetch is blocked."""
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert f'fetch("{BUILD_ID_FILENAME}?t="' in html
    assert f'"{BUILD_ID_JS_FILENAME}?t="' in html
    assert 'window.__LAB_BUILD_ID__' in html


def test_build_id_sidecars_do_not_collide_with_the_15c_page():
    """Both reports write into labs/build, so a shared sidecar name would have
    each page reload on the other's data."""
    assert BUILD_ID_FILENAME != 'build-id.txt'
    assert BUILD_ID_JS_FILENAME != 'build-id.js'
    assert SORT_STORAGE_KEY.startswith('5ob1cc')
    assert SCROLL_STORAGE_KEY.startswith('5ob1cc')


def test_reload_script_persists_sort_and_scroll():
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert SORT_STORAGE_KEY in html
    assert SCROLL_STORAGE_KEY in html
    # Sorts are captured from header clicks and replayed by clicking again.
    assert 'th.sortable' in html
    assert 'th.cellIndex' in html


def test_reload_script_reports_when_it_cannot_watch():
    """If neither transport works the page must say so, not sit silently."""
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert 'auto-reload unavailable' in html


def test_no_reload_produces_a_frozen_page():
    """A reloading page never settles for a headless browser or a PDF print."""
    html = build_report(get_sample_data(), 'now', 'abc123', live_reload=False)
    assert 'reload-mode' not in html
    assert 'location.reload()' not in html


# --- writing to disk ------------------------------------------------------

def test_render_to_file_writes_html_and_build_id(tmp_path):
    out = tmp_path / 'nested' / '5OB.html'
    build_id = render_to_file(get_sample_data(), out, '2025-09-12 10:00:00')

    assert out.exists()
    assert out.read_text().startswith('<!DOCTYPE html>')

    sidecar = out.parent / BUILD_ID_FILENAME
    assert sidecar.exists()
    assert sidecar.read_text() == build_id
    assert build_id in out.read_text()

    js = out.parent / BUILD_ID_JS_FILENAME
    assert js.exists()
    assert js.read_text().strip() == f'window.__LAB_BUILD_ID__ = "{build_id}";'


def test_render_to_file_rewrites_on_second_call(tmp_path):
    out = tmp_path / '5OB.html'
    render_to_file(get_sample_data(), out, 'first')
    first = out.read_text()

    changed = get_sample_data()
    changed.loc[0, 'TP'] = 99.0
    second_id = render_to_file(changed, out, 'second')

    assert out.read_text() != first
    assert (out.parent / BUILD_ID_FILENAME).read_text() == second_id


def test_render_real_csv(tmp_path):
    """The project CSV renders end to end."""
    out = tmp_path / '5OB.html'
    render_to_file(load_data(CSV_PATH), out, 'now')

    html = out.read_text()
    assert html.startswith('<!DOCTYPE html>')
    for anchor, _, _, _, _ in SECTIONS:
        assert f'id="{anchor}"' in html


def test_render_real_csv_leaves_the_data_alone(tmp_path):
    """Rendering must never write back to the CSV it read."""
    before = open(CSV_PATH, 'rb').read()
    render_to_file(load_data(CSV_PATH), tmp_path / '5OB.html', 'now')
    assert open(CSV_PATH, 'rb').read() == before


# --- error page -----------------------------------------------------------

def test_error_page_shows_the_message():
    """A half-written CSV export should surface in the browser, not vanish."""
    html = build_error_page('ParserError: Expected 8 fields, saw 15', 'now')
    assert html.startswith('<!DOCTYPE html>')
    assert 'Build failed' in html
    assert 'Expected 8 fields, saw 15' in html


def test_error_page_escapes_the_message():
    html = build_error_page('<script>alert(1)</script>', 'now')
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;' in html


def test_error_page_keeps_reloading():
    """The page must poll so it recovers on its own once the CSV is fixed."""
    html = build_error_page('boom', 'now')
    assert 'reload-mode' in html
    assert BUILD_ID_FILENAME in html


def test_render_error_to_file_bumps_build_id(tmp_path):
    out = tmp_path / '5OB.html'
    good_id = render_to_file(get_sample_data(), out, 'now')
    err_id = render_error_to_file('boom', out, 'now')

    assert err_id != good_id
    assert (out.parent / BUILD_ID_FILENAME).read_text() == err_id
    assert err_id in (out.parent / BUILD_ID_JS_FILENAME).read_text()
    assert 'Build failed' in out.read_text()


def test_render_error_then_recover(tmp_path):
    """Once the CSV parses again the good page comes back."""
    out = tmp_path / '5OB.html'
    render_error_to_file('boom', out, 'now')
    assert 'Build failed' in out.read_text()

    render_to_file(get_sample_data(), out, 'now')
    assert 'Build failed' not in out.read_text()
    assert 'Weekday Analysis' in out.read_text()


# --- section order --------------------------------------------------------

def test_grouping_sections_lead_the_page_in_order():
    """Weekday first, then this export's entry-time filters, then the SL
    family - the 15C page's shape with EMA standing in for its 4H bias."""
    page = build_report(get_sample_data(), 'now', 'abc123')
    order = [page.index(f'id="{a}"') for a in
             ('weekday', 'hour', 'ema-alignment', 'structure', 'htf-leg',
              'sl-range', 'sl-fixed', 'tp-range', 'pullback', 'sl-buffer')]
    assert order == sorted(order)


def test_nav_order_matches_the_sections():
    """The nav is built from the same list, so it must reorder with it."""
    page = build_report(get_sample_data(), 'now', 'abc123')
    nav = page[:page.index('id="weekday"')]
    order = [nav.index(f'href="#{a}"') for a, _, _, _, _ in SECTIONS]
    assert order == sorted(order)


def test_signal_strategy_tables_pin_their_first_column():
    """Every Signal/Strategy table pins its label column to 40% so they all
    line up down the page."""
    page = build_report(get_sample_data(), 'now', 'abc123')

    tables = (
        ('weekday', 'Day'),
        ('hour', 'Hour'),
        ('ema-alignment', 'EMA Alignment'),
        ('structure', 'Structure'),
        ('htf-leg', '30M Leg'),
        ('sl-range', 'SL Range'),
        ('sl-fixed', 'Fixed SL'),
        ('tp-range', 'TP Range'),
        ('pullback', 'Pullback'),
    )
    for anchor, label in tables:
        start = page.index(f'id="{anchor}"')
        section = page[start:start + 6000]
        # A label column never carries a percentage, so it is never the
        # sortable variant of the header - the plain form is the only one.
        assert f'<th style="width: 40%;">{label}</th>' in section, (
            f'{anchor} not pinned')


def test_tp_range_keeps_neither_win_rate_column():
    """Every trade in this table already has TP > 0, so a Signal column would
    read 100% on every row - see "Signal vs Strategy" in CLAUDE.md."""
    page = build_report(get_sample_data(), 'now', 'abc123')
    section = page[page.index('id="tp-range"'):page.index('id="pullback"')]
    headers = re.findall(r'<th[^>]*>([^<]*)</th>', section)
    assert headers == ['TP Range', 'Trades']


def test_stop_tables_are_not_sortable_but_others_are():
    """The stop tables keep their Default-outwards order; Pullback, Hour and
    the Strategies tables stay click-to-sort."""
    page = build_report(get_sample_data(), 'now', 'abc123')

    for table_id in ('sl-range-table', 'sl-buffer-table'):
        assert f'id="{table_id}"' in page, f'{table_id} missing'
        assert f"sortAnalysisTable('{table_id}'" not in page, (
            f'{table_id} still sortable')

    assert "sortAnalysisTable('pullback-table'" in page
    assert "sortAnalysisTable('hour-table'" in page
    assert "sortAnalysisTable('strategies-1-1-table'" in page


def test_adding_buffer_keeps_the_wider_label_column():
    page = build_report(get_sample_data(), 'now', 'abc123')
    head = page[page.index('id="sl-buffer-table"'):][:400]
    assert 'table-layout: fixed' in head
    assert 'style="width: 50%;"' in head


# --- Strategies tables ----------------------------------------------------

def test_strategies_split_one_table_per_rrr():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    for rrr in RRR_RATIOS:
        assert f'strategies-1-{rrr}' in anchors

    html = build_report(get_sample_data(), 'now', 'abc123')
    for rrr in RRR_RATIOS:
        assert f'Strategies (1:{rrr} RRR)' in html
        assert f'id="strategies-1-{rrr}-table"' in html


def test_strategies_tables_are_pre_sorted_by_win_rate():
    """Ranked server-side, so the table is ordered on first paint."""
    df = get_sample_data()
    names = [n for n, _ in get_buffer_strategies()]
    for rrr in RRR_RATIOS:
        stats = calculate_buffer_statistics(df, names)
        stats = _sort_by_win_rate(stats[stats['RRR'] == f'1:{rrr}'])
        rates = [float(v.rstrip('%')) for v in stats['Win Rate']]
        assert rates == sorted(rates, reverse=True), f'1:{rrr} not ranked'


def test_strategies_tables_only_hold_their_own_rrr():
    df = get_sample_data()
    names = [n for n, _ in get_buffer_strategies()]
    for rrr in RRR_RATIOS:
        stats = calculate_buffer_statistics(df, names)
        stats = stats[stats['RRR'] == f'1:{rrr}']
        assert set(stats['RRR']) == {f'1:{rrr}'}
        assert len(stats) > 0


def test_sort_by_win_rate_handles_empty_and_malformed():
    assert _sort_by_win_rate(pd.DataFrame()).empty

    df = pd.DataFrame({'Win Rate': ['10.0%', 'n/a', '90.0%', '50.0%']})
    out = _sort_by_win_rate(df)
    assert list(out['Win Rate']) == ['90.0%', '50.0%', '10.0%', 'n/a']


# --- three setups ---------------------------------------------------------

def test_three_setups_section_follows_the_last_strategies_table():
    """It reads as a per-trade postscript to the ranked tables."""
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors.index('three-setups') > anchors.index(
        f'strategies-1-{RRR_RATIOS[-1]}')


def test_three_setups_section_renders_a_row_per_trade():
    df = get_sample_data()
    page = build_report(df, 'now', 'abc123')

    start = page.index('id="three-setups-table"')
    table = page[start:page.index('</table>', start)]
    assert table.count('<tr>') == len(df) + 2, 'two header rows + one per trade'
    assert 'Three Setups Comparison on 1:2 RRR' in page


def test_three_setups_table_id_differs_from_its_anchor():
    page = build_report(get_sample_data(), 'now', 'abc123')
    assert 'id="three-setups"' in page
    assert 'id="three-setups-table"' in page
