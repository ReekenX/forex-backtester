"""
Tests for utils.report - the static HTML renderer for the 30MCR lab.

The page is a port of the 15LS1CC lab's 15C.html, so these tests pin the same
structural guarantees: one anchor per section, no duplicate DOM ids, a reload
that never fires on a timer, sidecar names that do not collide with the other
two pages, and a build id derived from the data.

Run with: poetry run python -m pytest strategies/30MCR/tests/ -v
"""

import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.continuation_reversal import (  # noqa: E402
    RRR_RATIOS,
    calculate_buffer_statistics,
    get_buffer_strategies,
    load_data,
)
from utils.report import (  # noqa: E402
    BUILD_ID_FILENAME,
    BUILD_ID_JS_FILENAME,
    SECTIONS,
    SCROLL_STORAGE_KEY,
    SORT_STORAGE_KEY,
    SOURCE_CSV,
    _sort_by_win_rate,
    _summary_cards,
    build_error_page,
    build_report,
    compute_build_id,
    render_error_to_file,
    render_to_file,
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'trades.csv')


def get_sample_data():
    """Six trades over three days, with one stopped-out winner-that-wasn't."""
    setups = ['High Continuation', 'High Reversal', 'Low Continuation',
              'Low Reversal', 'High Continuation', 'Low Reversal']
    return pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-03', '2025-02-04',
                 '2025-02-04', '2025-02-05', '2025-02-05'],
        'Weekday': ['Monday', 'Monday', 'Tuesday',
                    'Tuesday', 'Wednesday', 'Wednesday'],
        'Trade': ['#1', '#2', '#1', '#2', '#1', '#2'],
        'Direction': ['Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        '30M': [f'30M {s}' for s in setups],
        'SL': [4.4, 7.1, 2.6, 3.0, 4.3, 5.8],
        'Pullback': [0.7, 7.1, 2.7, 0.0, 0.9, 2.4],
        'TP': [34.0, 0.0, 31.0, 9.0, 17.0, 7.0],
        'R': [7.727, 0.0, -11.923, 3.0, 3.953, 1.207],
        'Setup': setups,
        'Type': [s.split()[-1] for s in setups],
    })


def get_empty_data():
    return get_sample_data().iloc[0:0].copy()


# --- document shape -------------------------------------------------------

def test_build_report_is_a_full_document():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert html.rstrip().endswith('</html>')
    assert '<title>30MCR Lab</title>' in html


def test_build_report_contains_every_section():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    for anchor, label, heading, _, _ in SECTIONS:
        assert f'id="{anchor}"' in html, anchor
        assert f'href="#{anchor}"' in html, anchor
        assert heading in html, heading
        assert label in html, label


def test_headings_are_not_double_escaped():
    """Headings and nav labels go through html.escape once, so a literal '<'
    is written in the source - an entity there would render as '&lt;'."""
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert '&amp;lt;' not in html
    assert '&amp;gt;' not in html


def test_sections_are_uniquely_anchored():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert len(anchors) == len(set(anchors))


def test_no_duplicate_dom_ids():
    """A section anchor and a table id must differ: a collision makes
    getElementById in the sort script return the <section> and sorting throw."""
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    ids = re.findall(r'\sid="([^"]+)"', html)
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert duplicates == [], f'duplicate ids: {duplicates}'


def test_build_report_shows_timestamp_and_build_id():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert '2025-02-06 10:00:00' in html
    assert 'abc123' in html


def test_build_report_names_the_strategy_and_its_source():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert '>30MCR</h1>' in html
    assert SOURCE_CSV == 'strategies/30MCR/trades.csv'
    assert SOURCE_CSV in html
    assert 'labs/render_30MCR.py' in html


def test_build_report_renders_data_values():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert 'Monday' in html
    assert 'High Continuation' in html
    assert '>Reversal<' in html  # the pooled type row
    assert '2025-02-03' in html


def test_build_report_empty_dataset():
    html = build_report(get_empty_data(), '2025-02-06 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert 'No data' in html


# --- summary cards --------------------------------------------------------

def test_summary_cards_report_dataset_shape():
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    assert cards['Trades'] == '6'
    assert cards['Trading days'] == '3'
    assert cards['Date range'] == '2025-02-03 to 2025-02-05'


def test_summary_win_rate_excludes_stopped_out_trade():
    """Row 2 never reached a target and row 3's pullback took out its stop,
    so four of the six are 1:1 winners."""
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    assert cards['Notation'] == '4W – 2L'
    assert cards['Win rate (1:1, no buffer)'] == '66.7%'


def test_summary_win_rate_tone_flips_at_50_percent():
    losing = get_sample_data().copy()
    losing['TP'] = 0.0
    tones = {label: tone for label, _, tone in _summary_cards(losing)}
    assert tones['Win rate (1:1, no buffer)'] == 'negative'
    tones = {label: tone for label, _, tone in _summary_cards(get_sample_data())}
    assert tones['Win rate (1:1, no buffer)'] == 'positive'


def test_summary_cards_empty_dataset():
    assert _summary_cards(get_empty_data()) == [('Trades', '0', 'neutral')]


# --- build id -------------------------------------------------------------

def test_build_id_is_stable_for_identical_data():
    assert compute_build_id(get_sample_data()) == compute_build_id(get_sample_data())


def test_build_id_changes_when_data_changes():
    changed = get_sample_data()
    changed.loc[0, 'TP'] = 99.0
    assert compute_build_id(changed) != compute_build_id(get_sample_data())


def test_build_id_is_short_hex():
    build_id = compute_build_id(get_sample_data())
    assert len(build_id) == 12
    assert re.fullmatch(r'[0-9a-f]{12}', build_id)


# --- live reload ----------------------------------------------------------

def test_live_reload_script_present_by_default():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert BUILD_ID_FILENAME in html
    assert 'id="reload-mode"' in html


def test_reload_never_fires_on_a_timer():
    """The page reloads only when the build id actually changed."""
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    reloads = re.findall(r'location\.reload\(\)', html)
    assert reloads, 'no reload at all'
    for match in re.finditer(r'location\.reload\(\)', html):
        window = html[max(0, match.start() - 200):match.start()]
        assert 'changed(' in window


def test_reload_script_probes_both_transports():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert f'fetch("{BUILD_ID_FILENAME}' in html
    assert f's.src = "{BUILD_ID_JS_FILENAME}' in html


def test_build_id_sidecars_do_not_collide_with_the_other_pages():
    """All three pages land in labs/build. A shared sidecar name would have
    each page reload on another page's data."""
    assert BUILD_ID_FILENAME == 'build-id-30mcr.txt'
    assert BUILD_ID_JS_FILENAME == 'build-id-30mcr.js'
    for taken in ('build-id.txt', 'build-id.js',
                  'build-id-5ob.txt', 'build-id-5ob.js'):
        assert BUILD_ID_FILENAME != taken
        assert BUILD_ID_JS_FILENAME != taken


def test_storage_keys_are_namespaced_to_this_page():
    """Sort and scroll state is per page, for the same reason."""
    assert SCROLL_STORAGE_KEY == '30mcr-scroll'
    assert SORT_STORAGE_KEY == '30mcr-sorts'
    for taken in ('15ls1cc-scroll', '15ls1cc-sorts',
                  '5ob1cc-scroll', '5ob1cc-sorts'):
        assert SCROLL_STORAGE_KEY != taken
        assert SORT_STORAGE_KEY != taken


def test_reload_script_persists_sort_and_scroll():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert SORT_STORAGE_KEY in html
    assert SCROLL_STORAGE_KEY in html
    assert 'sessionStorage' in html


def test_reload_script_reports_when_it_cannot_watch():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert 'auto-reload unavailable' in html


def test_no_reload_produces_a_frozen_page():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123',
                        live_reload=False)
    assert 'location.reload()' not in html
    assert 'id="reload-mode"' not in html


# --- writing to disk ------------------------------------------------------

def test_render_to_file_writes_html_and_build_id(tmp_path):
    out = tmp_path / 'build' / '30MCR.html'
    build_id = render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')

    assert out.exists()
    assert build_id == compute_build_id(get_sample_data())
    assert (out.parent / BUILD_ID_FILENAME).read_text() == build_id
    assert build_id in (out.parent / BUILD_ID_JS_FILENAME).read_text()


def test_render_to_file_rewrites_on_second_call(tmp_path):
    out = tmp_path / '30MCR.html'
    first = render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')

    changed = get_sample_data()
    changed.loc[0, 'TP'] = 99.0
    second = render_to_file(changed, out, '2025-02-06 10:05:00')

    assert first != second
    assert (out.parent / BUILD_ID_FILENAME).read_text() == second


def test_render_real_csv(tmp_path):
    """The page builds from the real export, not just the sample."""
    df = load_data(CSV_PATH)
    out = tmp_path / '30MCR.html'
    render_to_file(df, out, '2025-02-06 10:00:00')

    html = out.read_text()
    for _, _, heading, _, _ in SECTIONS:
        assert heading in html
    assert 'No data' not in html


def test_render_real_csv_leaves_the_data_alone(tmp_path):
    """Rendering must never write back to the export."""
    before = open(CSV_PATH, 'rb').read()
    render_to_file(load_data(CSV_PATH), tmp_path / 'x.html', 'now')
    assert open(CSV_PATH, 'rb').read() == before


# --- error page -----------------------------------------------------------

def test_error_page_shows_the_message():
    html = build_error_page('Traceback: boom', '2025-02-06 10:00:00')
    assert 'Build failed' in html
    assert 'Traceback: boom' in html


def test_error_page_escapes_the_message():
    html = build_error_page('<script>alert(1)</script>', '2025-02-06 10:00:00')
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;' in html


def test_error_page_keeps_reloading():
    """Otherwise a fixed CSV would leave the failure on screen."""
    html = build_error_page('boom', '2025-02-06 10:00:00')
    assert 'location.reload()' in html


def test_render_error_to_file_bumps_build_id(tmp_path):
    out = tmp_path / '30MCR.html'
    render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')
    good = (out.parent / BUILD_ID_FILENAME).read_text()

    failed = render_error_to_file('boom', out, '2025-02-06 10:01:00')
    assert failed != good
    assert (out.parent / BUILD_ID_FILENAME).read_text() == failed
    assert 'Build failed' in out.read_text()


def test_render_error_then_recover(tmp_path):
    out = tmp_path / '30MCR.html'
    render_error_to_file('boom', out, '2025-02-06 10:00:00')
    render_to_file(get_sample_data(), out, '2025-02-06 10:01:00')
    assert 'Build failed' not in out.read_text()


# --- section order and table conventions ----------------------------------

def test_grouping_sections_lead_the_page_in_order():
    """Weekday first, then this export's one entry-time filter, then the stop
    tables - the 15C page's order with 30MCR's own filter in the middle."""
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors[:3] == ['weekday', 'setup', 'sl-range']


def test_nav_order_matches_the_sections():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    nav = html[html.index('<nav'):html.index('</nav>')]
    positions = [nav.index(f'href="#{anchor}"')
                 for anchor, _, _, _, _ in SECTIONS]
    assert positions == sorted(positions)


def _section_html(anchor, df=None):
    df = get_sample_data() if df is None else df
    builder = next(b for a, _, _, _, b in SECTIONS if a == anchor)
    return builder(df)


def test_signal_strategy_tables_pin_their_first_column():
    """Weekday, the grouping tables, SL Range, Fixed SL, TP Range and Pullback
    all pin 40% so they line up down the page."""
    for anchor in ('weekday', 'setup', 'sl-range', 'sl-fixed',
                   'tp-range', 'pullback'):
        html = _section_html(anchor)
        assert 'width: 40%' in html, anchor
        assert 'table-layout: fixed' in html, anchor


def test_grouping_tables_carry_both_readings():
    for anchor in ('weekday', 'setup', 'sl-range', 'sl-fixed'):
        html = _section_html(anchor)
        assert '>Signal<' in html or 'Signal ↓' in html, anchor
        assert '>Strategy<' in html or 'Strategy ↓' in html, anchor


def test_tp_range_keeps_neither_win_rate_column():
    """Every trade it buckets already has TP > 0, so a Signal column would
    read 100% on every row. It is a distribution, not a win rate."""
    html = _section_html('tp-range')
    assert 'Signal' not in html
    assert '>Strategy<' not in html


def test_stop_tables_are_not_sortable_but_others_are():
    """Row order carries meaning in SL Range and Adding Buffer. Fixed SL is
    the exception - its rows are stop sizes to compare."""
    for anchor in ('sl-range', 'sl-buffer'):
        assert 'class="sortable"' not in _section_html(anchor), anchor
    for anchor in ('setup', 'sl-fixed', 'pullback'):
        assert 'class="sortable"' in _section_html(anchor), anchor


def test_adding_buffer_keeps_the_wider_label_column():
    html = _section_html('sl-buffer')
    assert 'width: 50%' in html


def test_strategies_split_one_table_per_rrr():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    for rrr in RRR_RATIOS:
        assert f'strategies-1-{rrr}' in anchors


def test_strategies_tables_are_pre_sorted_by_win_rate():
    """Ranked before first paint, so the page reads right before any JS runs."""
    html = _section_html(f'strategies-1-{RRR_RATIOS[0]}')
    rates = [float(v) for v in re.findall(r'<td[^>]*>(\d+\.\d)%</td>', html)]
    assert rates == sorted(rates, reverse=True)


def test_strategies_tables_only_hold_their_own_rrr():
    for rrr in RRR_RATIOS:
        html = _section_html(f'strategies-1-{rrr}')
        others = {f'1:{r}' for r in RRR_RATIOS} - {f'1:{rrr}'}
        assert f'>1:{rrr}</td>' in html
        for other in others:
            assert f'>{other}</td>' not in html


def test_sort_by_win_rate_handles_empty_and_malformed():
    assert _sort_by_win_rate(pd.DataFrame()).empty
    messy = pd.DataFrame({'Strategy': ['a', 'b'], 'Win Rate': ['N/A', '10.0%']})
    assert list(_sort_by_win_rate(messy)['Strategy']) == ['b', 'a']


def test_strategies_table_ids_differ_from_their_anchors():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    for rrr in RRR_RATIOS:
        assert f'id="strategies-1-{rrr}"' in html
        assert f'id="strategies-1-{rrr}-table"' in html


def test_buffer_statistics_feed_the_strategies_tables():
    names = [name for name, _ in get_buffer_strategies()]
    stats = calculate_buffer_statistics(get_sample_data(), names)
    assert not stats.empty
    assert set(stats['Strategy']) == set(names)


def test_three_setups_section_follows_the_last_strategies_table():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors[-1] == 'three-setups'
    assert anchors[-2] == f'strategies-1-{RRR_RATIOS[-1]}'


def test_three_setups_section_renders_a_row_per_trade():
    html = _section_html('three-setups')
    assert html.count('<tr>') == len(get_sample_data()) + 2  # two header rows


def test_three_setups_table_id_differs_from_its_anchor():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert 'id="three-setups"' in html
    assert 'id="three-setups-table"' in html
