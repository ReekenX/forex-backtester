"""
Tests for utils.report - the static HTML renderer for the 30MC lab.

The page is a port of the 15LS1CC lab's 15C.html, so these tests pin the same
structural guarantees: one anchor per section, no duplicate DOM ids, a reload
that never fires on a timer, sidecar names that do not collide with the other
two pages, and a build id derived from the data.

Run with: poetry run python -m pytest strategies/30MC/tests/ -v
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
    create_html_table,
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
        '30M': [s.split()[0] for s in setups],
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
    assert '<title>30MC Lab</title>' in html


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
    assert '>30MC</h1>' in html
    assert SOURCE_CSV == 'strategies/30MC/trades.csv'
    assert SOURCE_CSV in html
    assert 'labs/render_30MC.py' in html


def test_build_report_renders_data_values():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert 'Monday' in html
    assert 'Continuation' in html and 'Reversal' in html
    assert 'High Continuation' not in html
    assert '>Reversal<' in html  # the pooled type row
    assert '2025-02-03' in html


def test_build_report_empty_dataset():
    html = build_report(get_empty_data(), '2025-02-06 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert '>0.0%<' in html


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


def test_summary_signal_ignores_the_stop():
    """Signal counts every trade with TP > 0: row 2 has none, so five of six
    - one more than the win rate, which also drops the stopped-out row 3."""
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    assert cards['Signal (TP > 0)'] == '83.3%'


def test_summary_signal_tone_flips_at_50_percent():
    losing = get_sample_data().copy()
    losing['TP'] = 0.0
    tones = {label: tone for label, _, tone in _summary_cards(losing)}
    assert tones['Signal (TP > 0)'] == 'negative'
    tones = {label: tone for label, _, tone in _summary_cards(get_sample_data())}
    assert tones['Signal (TP > 0)'] == 'positive'


def test_summary_win_rate_tone_flips_at_50_percent():
    losing = get_sample_data().copy()
    losing['TP'] = 0.0
    tones = {label: tone for label, _, tone in _summary_cards(losing)}
    assert tones['Win rate (1:1, no buffer)'] == 'negative'
    tones = {label: tone for label, _, tone in _summary_cards(get_sample_data())}
    assert tones['Win rate (1:1, no buffer)'] == 'positive'


def test_summary_cards_put_signal_between_notation_and_win_rate():
    labels = [label for label, _, _ in _summary_cards(get_sample_data())]
    assert labels[-3:] == ['Notation', 'Signal (TP > 0)', 'Win rate (1:1, no buffer)']


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
    assert BUILD_ID_FILENAME == 'build-id-30mc.txt'
    assert BUILD_ID_JS_FILENAME == 'build-id-30mc.js'
    for taken in ('build-id.txt', 'build-id.js',
                  'build-id-5ob.txt', 'build-id-5ob.js',
                  'build-id-30mcr.txt', 'build-id-30mcr.js'):
        assert BUILD_ID_FILENAME != taken
        assert BUILD_ID_JS_FILENAME != taken


def test_storage_keys_are_namespaced_to_this_page():
    """Sort and scroll state is per page, for the same reason."""
    assert SCROLL_STORAGE_KEY == '30mc-scroll'
    assert SORT_STORAGE_KEY == '30mc-sorts'
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
    out = tmp_path / 'build' / '30MC.html'
    build_id = render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')

    assert out.exists()
    assert build_id == compute_build_id(get_sample_data())
    assert (out.parent / BUILD_ID_FILENAME).read_text() == build_id
    assert build_id in (out.parent / BUILD_ID_JS_FILENAME).read_text()


def test_render_to_file_rewrites_on_second_call(tmp_path):
    out = tmp_path / '30MC.html'
    first = render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')

    changed = get_sample_data()
    changed.loc[0, 'TP'] = 99.0
    second = render_to_file(changed, out, '2025-02-06 10:05:00')

    assert first != second
    assert (out.parent / BUILD_ID_FILENAME).read_text() == second


def test_render_real_csv(tmp_path):
    """The page builds from the real export, not just the sample."""
    df = load_data(CSV_PATH)
    out = tmp_path / '30MC.html'
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
    out = tmp_path / '30MC.html'
    render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')
    good = (out.parent / BUILD_ID_FILENAME).read_text()

    failed = render_error_to_file('boom', out, '2025-02-06 10:01:00')
    assert failed != good
    assert (out.parent / BUILD_ID_FILENAME).read_text() == failed
    assert 'Build failed' in out.read_text()


def test_render_error_then_recover(tmp_path):
    out = tmp_path / '30MC.html'
    render_error_to_file('boom', out, '2025-02-06 10:00:00')
    render_to_file(get_sample_data(), out, '2025-02-06 10:01:00')
    assert 'Build failed' not in out.read_text()


# --- section order and table conventions ----------------------------------

def test_weekday_leads_the_page_straight_into_the_stop_tables():
    """Every 30MC trade is a Continuation, so there is no Setup table to
    split them - Weekday is followed by the stop tables."""
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors[:2] == ['weekday', 'sl-range']
    assert 'setup' not in anchors


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


def test_every_section_up_to_max_sl_puts_a_chart_below_its_table():
    for anchor in ('weekday', 'sl-range', 'sl-fixed'):
        html = _section_html(anchor)
        assert html.count('class="wr-chart"') == 1, anchor
        assert html.index('<table') < html.index('class="wr-chart"'), anchor


def test_adding_buffer_keeps_its_table_and_adds_a_chart_below():
    html = _section_html('sl-buffer')
    assert '>Notation<' in html and '>Win Rate<' in html
    assert html.count('class="wr-chart"') == 1
    assert html.index('<table') < html.index('class="wr-chart"')


def test_pullback_section_is_gone():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert 'pullback' not in anchors


def test_tp_range_carries_no_win_rate_chart():
    assert 'class="wr-chart"' not in _section_html('tp-range')


def test_tp_range_keeps_its_table_and_adds_a_pip_chart_below():
    html = _section_html('tp-range')
    assert '>0-10<' in html and '>30+<' in html
    assert html.index('<table') < html.index('class="tp-chart"')


def test_every_table_pins_its_first_column_to_30_percent():
    """Every table on the page pins one label width so they line up."""
    for anchor, _, _, _, builder in SECTIONS:
        html = builder(get_sample_data())
        tables = html.count('<table')
        if not tables:
            continue
        assert html.count('table-layout: fixed') == tables, anchor
        assert html.count('style="width: 30%;"') == tables, anchor
        assert 'width: 40%' not in html and 'width: 50%' not in html, anchor


def test_grouping_tables_carry_both_readings():
    for anchor in ('weekday', 'sl-range', 'sl-fixed'):
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
    """Row order carries meaning in SL Range and Adding Buffer.
    Fixed SL is the exception - its rows are stop sizes to compare."""
    for anchor in ('sl-range', 'sl-buffer'):
        assert 'class="sortable"' not in _section_html(anchor), anchor
    for anchor in ('sl-fixed',):
        assert 'class="sortable"' in _section_html(anchor), anchor


def test_strategies_split_one_table_per_rrr():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    for rrr in RRR_RATIOS:
        assert f'strategies-1-{rrr}' in anchors


def test_strategies_tables_are_pre_sorted_by_win_rate():
    """Ranked before first paint, so the page reads right before any JS runs."""
    html = _section_html(f'strategies-1-{RRR_RATIOS[0]}')
    rates = [float(v) for v in re.findall(r'<td[^>]*>(\d+\.\d)%</td>', html)]
    assert rates == sorted(rates, reverse=True)


def test_strategies_tables_carry_no_rrr_column():
    """Each table holds one RRR, already named in its heading."""
    for rrr in RRR_RATIOS:
        html = _section_html(f'strategies-1-{rrr}')
        table = html[html.index('<table'):html.index('</table>')]
        assert '>RRR<' not in table
        assert '>1:' not in table


def test_strategies_tables_name_the_cap_column_cap_sl():
    """Cap SL, not Max SL: the column caps the stop at min(SL + buffer,
    cap)."""
    for rrr in RRR_RATIOS:
        html = _section_html(f'strategies-1-{rrr}')
        table = html[html.index('<table'):html.index('</table>')]
        assert '>Cap SL<' in table
        assert 'Max SL' not in html


def test_strategies_tables_carry_no_be_rule():
    """The 30MC export has no BE Rule column - no column, no checkbox, no
    note."""
    notes = {anchor: note for anchor, _, _, note, _ in SECTIONS}
    for rrr in RRR_RATIOS:
        html = _section_html(f'strategies-1-{rrr}')
        assert 'BE Rule' not in html
        assert 'type="checkbox"' not in html
        assert 'BE Rule' not in notes[f'strategies-1-{rrr}']


def test_strategies_tables_narrow_their_short_columns():
    html = _section_html('strategies-1-3')
    assert 'style="width: 6%;">Buffer<' in html
    assert 'style="width: 16%;">Notation<' in html


def test_strategies_win_rate_is_coloured_against_its_own_rrr():
    """28.6% loses at 1:1 (breakeven 50%) and pays at 1:3 (breakeven
    25%), with no RRR column left to read it from."""
    stats = pd.DataFrame({'Strategy': ['a'], 'Win Rate': ['28.6%']})
    assert 'class="negative-edge">28.6%' in create_html_table(stats, rrr='1:1')
    assert 'class="positive-edge">28.6%' in create_html_table(stats, rrr='1:3')


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


def test_strategies_tables_close_the_page():
    """The Three Setups trade log was removed, so the last Strategies table
    is the final section."""
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors[-1] == f'strategies-1-{RRR_RATIOS[-1]}'
    assert 'three-setups' not in anchors


def test_cap_sl_section_is_gone():
    """With every stop at 5 pips a cap only repeats Fixed SL, so the page
    has no Cap SL Analysis."""
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert 'sl-max' not in anchors
    assert anchors.index('tp-range') == anchors.index('sl-fixed') + 1


def test_streaks_table_sits_above_the_strategies_tables():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert 'id="streaks-table"' in html
    assert html.index('id="streaks"') > html.index('id="r-distribution"')
    assert 'R Distribution (Exact)' not in html
    assert html.index('id="streaks"') < html.index('id="strategies-1-1"')
    for label in ('Default', 'Continuation', 'Reversal'):
        assert label in html
    assert 'Longest Wins' in html and 'Longest Losses' in html


def test_each_strategies_table_carries_an_equity_chart():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    for rrr in RRR_RATIOS:
        chart = f'id="strategies-1-{rrr}-equity"'
        assert chart in html
        assert html.index(f'id="strategies-1-{rrr}-table"') < html.index(chart)


def test_half_sl_table_sits_right_above_the_strategies_tables():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors.index('half-sl') == anchors.index('strategies-1-1') - 1


def test_half_sl_table_compares_the_three_entries_without_sorting():
    html = _section_html('half-sl')
    for entry in ('>Default<', '>Half SL<', '>Limit order on half<'):
        assert entry in html
    assert 'class="sortable"' not in html


def test_weekday_table_colours_every_rate_against_breakeven():
    """Sample Wednesday wins both trades (green), Monday wins one of two and
    sits on 50% (plain), Thursday has no trades and reads 0% (red)."""
    html = _section_html('weekday')
    table = html[html.index('<table'):html.index('</table>')]
    assert '>Notation<' not in table
    assert 'class="positive-edge"' in table
    assert 'class="negative-edge"' in table
    assert '<td>50.0%</td>' in table
