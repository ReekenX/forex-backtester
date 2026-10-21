"""
Tests for utils.report - the static HTML renderer for the 30MCR Fixed lab.

The page is the 30MCR page cut down to what a fixed stop and a Reward column
can answer, so these tests pin the same structural guarantees - one anchor per
section, no duplicate DOM ids, a reload that never fires on a timer, sidecar
names that do not collide with the other pages, a build id derived from the
data - and that the stop-scenario tables stay off the page.

Run with: poetry run python -m pytest strategies/30MCR-fixed/tests/ -v
"""

import os
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.continuation_reversal import (  # noqa: E402
    RRR_RATIOS,
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
    """Six trades over three days on a 5-pip stop: one never reached a
    target, one reached 6R after its pullback had taken out the stop."""
    levels = ['High', 'High', 'Low', 'Low', 'High', 'High']
    directions = ['Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell']
    types = ['Continuation', 'Reversal', 'Reversal',
             'Continuation', 'Continuation', 'Reversal']
    return pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-03', '2025-02-04',
                 '2025-02-04', '2025-02-05', '2025-02-05'],
        'Weekday': ['Monday', 'Monday', 'Tuesday',
                    'Tuesday', 'Wednesday', 'Wednesday'],
        'Trade': ['#1', '#2', '#1', '#2', '#1', '#2'],
        'Direction': directions,
        '30M': levels,
        'SL': [5.0] * 6,
        'Pullback': [0.7, 5.0, 5.4, 0.0, 0.9, 2.4],
        'Reward': [3.0, 0.0, 6.0, 1.0, 2.0, 1.0],
        'R': [3.0, 0.0, -6.0, 1.0, 2.0, 1.0],
        'Setup': [f'{lvl} {t}' for lvl, t in zip(levels, types)],
        'Type': types,
    })


def get_empty_data():
    return get_sample_data().iloc[0:0].copy()


def _section_html(anchor, df=None):
    df = get_sample_data() if df is None else df
    builder = next(b for a, _, _, _, b in SECTIONS if a == anchor)
    return builder(df)


# --- document shape -------------------------------------------------------

def test_build_report_is_a_full_document():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert html.rstrip().endswith('</html>')
    assert '<title>30MCR Fixed Lab</title>' in html


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
    assert '>30MCR Fixed</h1>' in html
    assert SOURCE_CSV == 'strategies/30MCR-fixed/trades.csv'
    assert SOURCE_CSV in html
    assert 'labs/render_30MCR_fixed.py' in html


def test_build_report_renders_data_values():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    assert 'Monday' in html
    assert '>Continuation<' in html and '>Reversal<' in html
    assert 'High Continuation' not in html
    assert '2025-02-03' in html


def test_build_report_empty_dataset():
    html = build_report(get_empty_data(), '2025-02-06 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert '>0.0%<' in html


def test_page_never_mentions_tp():
    """The export has no TP column - every rule reads Reward."""
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    text = re.sub(r'<(script|style)\b.*?</\1>', '', html, flags=re.S)
    assert not re.search(r'\bTP\b', text)


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
    assert cards['Win rate (1:1)'] == '66.7%'


def test_summary_signal_ignores_the_stop():
    """Signal counts every trade with Reward > 0: row 2 has none, so five of
    six - one more than the win rate, which also drops the stopped-out row 3."""
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    assert cards['Signal (Reward > 0)'] == '83.3%'


def test_summary_tones_flip_at_50_percent():
    losing = get_sample_data().copy()
    losing['Reward'] = 0.0
    tones = {label: tone for label, _, tone in _summary_cards(losing)}
    assert tones['Signal (Reward > 0)'] == 'negative'
    assert tones['Win rate (1:1)'] == 'negative'
    tones = {label: tone for label, _, tone in _summary_cards(get_sample_data())}
    assert tones['Signal (Reward > 0)'] == 'positive'
    assert tones['Win rate (1:1)'] == 'positive'


def test_summary_cards_put_signal_between_notation_and_win_rate():
    labels = [label for label, _, _ in _summary_cards(get_sample_data())]
    assert labels[-3:] == ['Notation', 'Signal (Reward > 0)', 'Win rate (1:1)']


def test_summary_cards_empty_dataset():
    assert _summary_cards(get_empty_data()) == [('Trades', '0', 'neutral')]


# --- build id -------------------------------------------------------------

def test_build_id_is_stable_for_identical_data():
    assert compute_build_id(get_sample_data()) == compute_build_id(get_sample_data())


def test_build_id_changes_when_data_changes():
    changed = get_sample_data()
    changed.loc[0, 'Reward'] = 9.0
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
    """Every page lands in labs/build. A shared sidecar name would have each
    page reload on another page's data."""
    assert BUILD_ID_FILENAME == 'build-id-30mcr-fixed.txt'
    assert BUILD_ID_JS_FILENAME == 'build-id-30mcr-fixed.js'
    for taken in ('build-id.txt', 'build-id.js',
                  'build-id-5ob.txt', 'build-id-5ob.js',
                  'build-id-30mcr.txt', 'build-id-30mcr.js'):
        assert BUILD_ID_FILENAME != taken
        assert BUILD_ID_JS_FILENAME != taken


def test_storage_keys_are_namespaced_to_this_page():
    """Sort and scroll state is per page, for the same reason."""
    assert SCROLL_STORAGE_KEY == '30mcr-fixed-scroll'
    assert SORT_STORAGE_KEY == '30mcr-fixed-sorts'
    for taken in ('15ls1cc-scroll', '15ls1cc-sorts',
                  '5ob1cc-scroll', '5ob1cc-sorts',
                  '30mcr-scroll', '30mcr-sorts'):
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
    out = tmp_path / 'build' / '30MCR-fixed.html'
    build_id = render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')

    assert out.exists()
    assert build_id == compute_build_id(get_sample_data())
    assert (out.parent / BUILD_ID_FILENAME).read_text() == build_id
    assert build_id in (out.parent / BUILD_ID_JS_FILENAME).read_text()


def test_render_to_file_rewrites_on_second_call(tmp_path):
    out = tmp_path / '30MCR-fixed.html'
    first = render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')

    changed = get_sample_data()
    changed.loc[0, 'Reward'] = 9.0
    second = render_to_file(changed, out, '2025-02-06 10:05:00')

    assert first != second
    assert (out.parent / BUILD_ID_FILENAME).read_text() == second


def test_render_real_csv(tmp_path):
    """The page builds from the real export, not just the sample."""
    df = load_data(CSV_PATH)
    out = tmp_path / '30MCR-fixed.html'
    render_to_file(df, out, '2025-02-06 10:00:00')

    html = out.read_text()
    for _, _, heading, _, _ in SECTIONS:
        assert heading in html
    assert 'No data' not in html


def test_render_real_csv_leaves_the_data_alone(tmp_path):
    """Rendering must never write back to the export."""
    before = Path(CSV_PATH).read_bytes()
    render_to_file(load_data(CSV_PATH), tmp_path / 'x.html', 'now')
    assert Path(CSV_PATH).read_bytes() == before


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
    out = tmp_path / '30MCR-fixed.html'
    render_to_file(get_sample_data(), out, '2025-02-06 10:00:00')
    good = (out.parent / BUILD_ID_FILENAME).read_text()

    failed = render_error_to_file('boom', out, '2025-02-06 10:01:00')
    assert failed != good
    assert (out.parent / BUILD_ID_FILENAME).read_text() == failed
    assert 'Build failed' in out.read_text()


def test_render_error_then_recover(tmp_path):
    out = tmp_path / '30MCR-fixed.html'
    render_error_to_file('boom', out, '2025-02-06 10:00:00')

# --- section order and table conventions ----------------------------------

def test_sections_in_page_order():
    anchors = [anchor for anchor, _, _, _, _ in SECTIONS]
    assert anchors == ['weekday', 'setup', 'r-distribution', 'streaks',
                       'scaling', 'strategies-1-1', 'strategies-1-2', 'strategies-1-3']


def test_stop_scenario_tables_are_gone():
    """One stop for every trade, and a Reward already net of costs, leave
    nothing for the stop tables to vary."""
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    for gone in ('SL Range', 'Fixed SL', 'Cap SL', 'Adding Buffer',
                 'Pullback Analysis', 'Half SL', 'TP Range', '+ 1 pip',
                 'BE Rule', 'Buffer'):
        assert gone not in html, gone


def test_grouping_tables_put_a_chart_below():
    for anchor in ('weekday', 'setup'):
        html = _section_html(anchor)
        assert html.count('class="wr-chart"') == 1, anchor
        assert html.index('<table') < html.index('class="wr-chart"'), anchor


def test_grouping_tables_carry_both_readings():
    for anchor in ('weekday', 'setup'):
        html = _section_html(anchor)
        assert '>Signal' in html and '>Strategy' in html, anchor
        assert 'Strategy + 1 pip' not in html, anchor


def test_weekday_is_not_sortable_but_setup_is():
    assert 'class="sortable"' not in _section_html('weekday')
    assert 'class="sortable"' in _section_html('setup')


def test_weekday_table_colours_every_rate_against_breakeven():
    """Sample Wednesday wins both trades (green), Monday wins one of two and
    sits on 50% (plain), Thursday has no trades and reads 0% (red)."""
    html = _section_html('weekday')
    table = html[html.index('<table'):html.index('</table>')]
    assert 'class="positive-edge"' in table
    assert 'class="negative-edge"' in table
    assert '<td>50.0%</td>' in table


def test_every_table_pins_its_first_column_to_30_percent():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    tables = re.findall(r'<table[^>]*>.*?</th>', html, flags=re.S)
    assert tables
    for table in tables:
        assert 'width: 30%;' in table


def test_streaks_table_lists_the_three_trends():
    html = _section_html('streaks')
    for label in ('>Default<', '>Continuation<', '>Reversal<'):
        assert label in html
    assert 'buffer' not in html
    assert 'Longest Wins' in html and 'Longest Losses' in html


def test_strategies_split_one_table_per_rrr():
    for rrr in RRR_RATIOS:
        html = _section_html(f'strategies-1-{rrr}')
        assert f'id="strategies-1-{rrr}-table"' in html
        assert '>RRR<' not in html
        for column in ('>Trend<', '>Trades<', '>Notation<', '>Win Rate',
                       '>Outcome<'):
            assert column in html


def test_strategies_tables_are_pre_sorted_by_win_rate():
    html = _section_html('strategies-1-1')
    table = html[html.index('<tbody>'):html.index('</tbody>')]
    rates = [float(r) for r in re.findall(r'>(\d+\.\d)%<', table)]
    assert rates == sorted(rates, reverse=True)


def test_strategies_win_rate_is_coloured_against_its_own_rrr():
    """Sample All Trades at 1:3 is 1W - 5L, 16.7%: below 25% breakeven."""
    html = _section_html('strategies-1-3')
    assert re.search(r'<td class="negative-edge">16\.7%</td>', html)


def test_each_strategies_table_carries_an_equity_chart():
    html = build_report(get_sample_data(), '2025-02-06 10:00:00', 'abc123')
    for rrr in RRR_RATIOS:
        chart = f'id="strategies-1-{rrr}-equity"'
        assert chart in html
        assert html.index(f'id="strategies-1-{rrr}-table"') < html.index(chart)


def test_sort_by_win_rate_handles_empty_and_malformed():
    assert _sort_by_win_rate(pd.DataFrame()).empty
    stats = pd.DataFrame({'Win Rate': ['n/a', '40.0%', '60.0%']})
    assert _sort_by_win_rate(stats)['Win Rate'].tolist() == ['60.0%', '40.0%', 'n/a']


# --- scaling --------------------------------------------------------------

def test_scaling_section_carries_both_tables():
    html = _section_html('scaling', _with_scaling(get_sample_data()))
    assert html.count('<table') == 2
    for column in ('Scaled', 'Avg R', 'Avg Win', 'Best', 'Base', 'Combined'):
        assert f'>{column}<' in html, column


def test_scaling_section_reads_the_second_trades():
    html = _section_html('scaling', _with_scaling(get_sample_data()))
    assert '2 (33.3%)' in html
    assert '1W – 1L' in html


def test_scaling_section_without_a_scaling_column():
    html = _section_html('scaling')
    assert '0 (0.0%)' in html
    assert 'N/A' in html


def test_scaling_section_on_empty_data():
    _section_html('scaling', get_empty_data())


def _with_scaling(df):
    """One second trade won 3R, one was stopped out, the rest never opened."""
    df = df.copy()
    df['Scaling'] = [3.0, -1.0] + [float('nan')] * (len(df) - 2)
    return df
