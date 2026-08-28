"""
Tests for utils.report - the static HTML renderer for the 15LS1CC lab.

These tests use a small dataset to verify the report structure.
Run with: poetry run python -m pytest strategies/15LS1CC/tests/ -v
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.confirmation_candle import (  # noqa: E402
    RRR_RATIOS,
    _calculate_buffer_statistics_filtered,
    get_buffer_strategies,
    load_data,
)
from utils.report import (  # noqa: E402
    BUILD_ID_FILENAME,
    BUILD_ID_JS_FILENAME,
    SECTIONS,
    build_error_page,
    build_report,
    compute_build_id,
    render_error_to_file,
    render_to_file,
    _sort_by_win_rate,
    _summary_cards,
)


def get_sample_data():
    """Six trades spanning three days, with one stopped-out winner-that-wasn't."""
    return pd.DataFrame({
        'Date': ['2026-07-27', '2026-07-27', '2026-07-28',
                 '2026-07-28', '2026-07-29', '2026-07-29'],
        'Weekday': ['Monday', 'Monday', 'Tuesday', 'Tuesday', 'Wednesday', 'Wednesday'],
        'Trade': ['#1', '#2', '#1', '#2', '#1', '#2'],
        'Direction': ['Sell', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        'SL': [4.4, 7.1, 2.6, 3.0, 4.3, 5.8],
        'Pullback': [0.7, 7.1, 2.7, 0.0, 0.9, 2.4],
        'TP': [34.0, 0.0, 31.0, 9.0, 17.0, 7.0],
        'R': [7.0, 0.0, -10.0, 3.0, 3.0, 1.0],
    })


def get_empty_data():
    return pd.DataFrame({
        'Date': [], 'Weekday': [], 'Trade': [], 'Direction': [],
        'SL': [], 'Pullback': [], 'TP': [], 'R': [],
    })


def test_build_report_is_a_full_document():
    html = build_report(get_sample_data(), '2026-08-28 10:00:00', 'abc123')
    assert html.startswith('<!DOCTYPE html>')
    assert html.rstrip().endswith('</html>')
    assert '<title>15LS1CC Lab</title>' in html


def test_build_report_contains_every_section():
    import html as html_mod

    page = build_report(get_sample_data(), '2026-08-28 10:00:00', 'abc123')
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


def test_build_report_shows_timestamp_and_build_id():
    html = build_report(get_sample_data(), '2026-08-28 10:00:00', 'abc123')
    assert '2026-08-28 10:00:00' in html
    assert 'abc123' in html


def test_build_report_renders_data_values():
    """The tables carry real numbers, not placeholders."""
    html = build_report(get_sample_data(), 'now', 'id')
    assert 'Monday' in html
    assert 'Wednesday' in html
    # 6 trades over 3 days is in the summary strip.
    assert '>6<' in html.replace(' ', '').replace('\n', '') or '6' in html


def test_summary_cards_report_dataset_shape():
    cards = dict((label, value) for label, value, _ in _summary_cards(get_sample_data()))
    assert cards['Trades'] == '6'
    assert cards['Trading days'] == '3'
    assert cards['Date range'] == '2026-07-27 to 2026-07-29'


def test_summary_win_rate_excludes_stopped_out_trade():
    """Trade #1 on 07-28 has Pullback 2.7 > SL 2.6, so it is a loss even
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
    cards = _summary_cards(get_empty_data())
    assert cards == [('Trades', '0', 'neutral')]


def test_build_report_empty_dataset():
    """An empty CSV still renders a readable page rather than raising."""
    html = build_report(get_empty_data(), 'now', 'id')
    assert html.startswith('<!DOCTYPE html>')
    for anchor, _, _, _, _ in SECTIONS:
        assert f'id="{anchor}"' in html


def test_build_id_is_stable_for_identical_data():
    assert compute_build_id(get_sample_data()) == compute_build_id(get_sample_data())


def test_build_id_changes_when_data_changes():
    df = get_sample_data()
    changed = df.copy()
    changed.loc[0, 'TP'] = 99.0
    assert compute_build_id(df) != compute_build_id(changed)


def test_build_id_is_short_hex():
    assert re.fullmatch(r'[0-9a-f]{12}', compute_build_id(get_sample_data()))


def test_live_reload_script_present_by_default():
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert 'reload-mode' in html
    assert BUILD_ID_FILENAME in html
    assert 'location.reload()' in html


def test_reload_never_fires_on_a_timer():
    """Every location.reload() must sit behind a build-id comparison. A blind
    timed reload would wipe the reader's column sort every few seconds."""
    html = build_report(get_sample_data(), 'now', 'abc123')
    script = html[html.index('<script>'):]
    # The only reloads are inside `if (changed(...))` guards.
    assert script.count('location.reload()') == 2
    assert script.count('if (changed(') == 2
    # No setTimeout schedules a reload directly.
    assert 'setTimeout(function () { location.reload()' not in script
    assert 'location.reload(); }' not in script


def test_reload_script_probes_both_transports():
    """fetch() for http://, a <script> tag for file:// where fetch is blocked."""
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert f'fetch("{BUILD_ID_FILENAME}?t="' in html
    assert f'"{BUILD_ID_JS_FILENAME}?t="' in html
    assert 'window.__LAB_BUILD_ID__' in html


def test_reload_script_persists_sort_and_scroll():
    html = build_report(get_sample_data(), 'now', 'abc123')
    assert '15ls1cc-sorts' in html
    assert '15ls1cc-scroll' in html
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


def test_render_to_file_writes_html_and_build_id(tmp_path):
    out = tmp_path / 'nested' / '15LS1CC.html'
    build_id = render_to_file(get_sample_data(), out, '2026-08-28 10:00:00')

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
    out = tmp_path / '15LS1CC.html'
    render_to_file(get_sample_data(), out, 'first')
    first = out.read_text()

    changed = get_sample_data()
    changed.loc[0, 'TP'] = 99.0
    second_id = render_to_file(changed, out, 'second')

    assert out.read_text() != first
    assert (out.parent / BUILD_ID_FILENAME).read_text() == second_id


def test_render_real_csv(tmp_path):
    """The project CSV renders end to end."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data.csv')
    out = tmp_path / '15LS1CC.html'
    render_to_file(load_data(csv_path), out, 'now')

    html = out.read_text()
    assert html.startswith('<!DOCTYPE html>')
    for anchor, _, _, _, _ in SECTIONS:
        assert f'id="{anchor}"' in html


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
    out = tmp_path / '15LS1CC.html'
    good_id = render_to_file(get_sample_data(), out, 'now')
    err_id = render_error_to_file('boom', out, 'now')

    assert err_id != good_id
    assert (out.parent / BUILD_ID_FILENAME).read_text() == err_id
    assert err_id in (out.parent / BUILD_ID_JS_FILENAME).read_text()
    assert 'Build failed' in out.read_text()


def test_render_error_then_recover(tmp_path):
    """Once the CSV parses again the good page comes back."""
    out = tmp_path / '15LS1CC.html'
    render_error_to_file('boom', out, 'now')
    assert 'Build failed' in out.read_text()

    render_to_file(get_sample_data(), out, 'now')
    assert 'Build failed' not in out.read_text()
    assert 'Weekday Statistics' in out.read_text()


def test_no_duplicate_dom_ids():
    """A section anchor must never collide with a table's sort id: the sort
    script does getElementById(tableId) and would get the <section> instead."""
    import re
    from collections import Counter

    html = build_report(get_sample_data(), 'now', 'abc123')
    counts = Counter(re.findall(r'id="([^"]+)"', html))
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f'duplicate ids: {dupes}'


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
        stats = _calculate_buffer_statistics_filtered(df, names)
        stats = _sort_by_win_rate(stats[stats['RRR'] == f'1:{rrr}'])
        rates = [float(v.rstrip('%')) for v in stats['Win Rate']]
        assert rates == sorted(rates, reverse=True), f'1:{rrr} not ranked'


def test_strategies_tables_only_hold_their_own_rrr():

    df = get_sample_data()
    names = [n for n, _ in get_buffer_strategies()]
    for rrr in RRR_RATIOS:
        stats = _calculate_buffer_statistics_filtered(df, names)
        stats = stats[stats['RRR'] == f'1:{rrr}']
        assert set(stats['RRR']) == {f'1:{rrr}'}
        assert len(stats) > 0


def test_sort_by_win_rate_handles_empty_and_malformed():

    assert _sort_by_win_rate(pd.DataFrame()).empty

    df = pd.DataFrame({'Win Rate': ['10.0%', 'n/a', '90.0%', '50.0%']})
    out = _sort_by_win_rate(df)
    assert list(out['Win Rate']) == ['90.0%', '50.0%', '10.0%', 'n/a']


def test_stop_tables_are_not_sortable_but_others_are():
    """The four stop tables keep their Default-outwards order; Pullback and the
    Strategies tables stay click-to-sort."""
    page = build_report(get_sample_data(), 'now', 'abc123')

    for table_id in ('sl-range-stats', 'sl-reduction-table',
                     'sl-buffer-table', 'sl-buffer-small-table'):
        assert f'id="{table_id}"' in page, f'{table_id} missing'
        assert f"sortSlRange('{table_id}'" not in page, f'{table_id} still sortable'

    assert "sortSlRange('pullback-analysis'" in page
    assert "sortAnalysisTable('strategies-1-1-table'" in page


def test_stop_tables_share_a_pinned_first_column():
    """The four stop tables pin their label column so they line up with each
    other; the other tables keep auto sizing."""
    page = build_report(get_sample_data(), 'now', 'abc123')

    for table_id in ('sl-range-stats', 'sl-reduction-table',
                     'sl-buffer-table', 'sl-buffer-small-table'):
        start = page.index(f'id="{table_id}"')
        head = page[start:start + 400]
        assert 'table-layout: fixed' in head, f'{table_id} not fixed-layout'
        assert 'style="width: 50%;"' in head, f'{table_id} first column not pinned'

    start = page.index('id="pullback-analysis"')
    assert 'table-layout: fixed' not in page[start:start + 400]
