"""
Tests for utils.continuation_reversal - the analysis behind the 30MCR report.

Run with: poetry run python -m pytest strategies/30MCR/tests/ -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.continuation_reversal import (  # noqa: E402
    BUFFER_PIPS,
    PULLBACK_ENTRY_PIPS,
    RRR_RATIOS,
    SETUP_ORDER,
    SIDE_ORDER,
    SL_BUFFER_PIPS,
    SL_FIXED_PIPS,
    SL_RANGES,
    THREE_SETUPS_COLUMNS,
    THREE_SETUPS_RRR,
    TP_RANGES,
    TYPE_ORDER,
    WEEKDAY_ORDER,
    _calculate_stats_with_buffer,
    _format_wl,
    _normalise_header,
    _pip_cell,
    _whole_pip_cell,
    calculate_buffer_statistics,
    calculate_direction_statistics,
    calculate_pullback_statistics,
    calculate_r_counts,
    calculate_setup_statistics,
    calculate_side_statistics,
    calculate_sl_buffer_statistics,
    calculate_sl_fixed_statistics,
    calculate_sl_statistics,
    calculate_three_setups_comparison,
    calculate_tp_statistics,
    calculate_type_statistics,
    calculate_weekday_statistics,
    create_html_table,
    create_r_histogram_combined,
    create_r_histogram_exact,
    create_sortable_table,
    create_three_setups_table,
    get_buffer_strategies,
    load_data,
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'trades.csv')


def get_sample_data():
    """
    Ten trades over three days, covering every case the tables must separate.

    Row 2 is the stopped-out winner-that-wasn't: Pullback 6.0 >= SL 5.0 with a
    TP of 20, so Signal counts it and Strategy does not. All four setups are
    present so the setup, side and type tables each have both of their groups.
    """
    setups = ['High Continuation', 'High Reversal', 'Low Continuation',
              'Low Reversal', 'High Continuation', 'Low Reversal',
              'High Reversal', 'Low Continuation', 'High Continuation',
              'Low Reversal']
    return pd.DataFrame({
        'Date': ['2025-02-03'] * 4 + ['2025-02-04'] * 3 + ['2025-02-05'] * 3,
        'Weekday': ['Monday'] * 4 + ['Tuesday'] * 3 + ['Wednesday'] * 3,
        'Trade': ['#1', '#2', '#3', '#4', '#1', '#2', '#3', '#1', '#2', '#3'],
        'Direction': ['Buy', 'Sell', 'Buy', 'Sell',
                      'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        '30M': [f'30M {s}' for s in setups],
        'SL': [4.0, 5.0, 2.0, 7.0, 3.0, 12.0, 6.0, 2.5, 8.0, 5.5],
        'Pullback': [1.0, 6.0, 2.0, 3.5, 0.5, 12.0, 2.0, 2.5, 4.0, 1.5],
        'TP': [16.0, 20.0, 0.0, 25.0, 3.0, 0.0, 35.0, 0.0, 9.0, 12.0],
        'R': [4.0, -4.0, 0.0, 3.571, 1.0, 0.0, 5.833, 0.0, 1.125, 2.182],
        'Setup': setups,
        'Side': [s.split()[0] for s in setups],
        'Type': [s.split()[-1] for s in setups],
    })


def get_empty_data():
    return get_sample_data().iloc[0:0].copy()


# --- load_data -------------------------------------------------------------

def test_load_data_reads_the_project_csv():
    df = load_data(CSV_PATH)
    assert len(df) > 0
    for col in ('Date', 'Weekday', 'Trade', 'Direction', '30M',
                'SL', 'Pullback', 'TP'):
        assert col in df.columns


def test_normalise_header_keeps_only_the_first_line():
    """The export wraps a computed figure under the column name in one cell."""
    assert _normalise_header('Trade\n69') == 'Trade'
    assert _normalise_header('Signal \n50.7%') == 'Signal'
    assert _normalise_header('WR \n37.7%\n122R') == 'WR'
    assert _normalise_header('SL') == 'SL'


def test_load_data_drops_everything_from_the_signal_column_on():
    """The sheet's own stop scenarios are scratch: this page recomputes them
    under a rule it controls, so reading them back would double up."""
    df = load_data(CSV_PATH)
    for col in df.columns:
        assert not str(col).startswith('Signal'), col
    for scratch in ('WR', '+1pip WR', '3 pips SL', '4 pips SL', '5 pips SL',
                    'Attributes'):
        assert scratch not in df.columns


def test_load_data_keeps_every_trade_column():
    """Dropping the scratch columns must not drop a row or a trade column."""
    raw = pd.read_csv(CSV_PATH)
    df = load_data(CSV_PATH)
    assert len(df) == len(raw)
    assert list(df.columns)[:8] == ['Date', 'Weekday', 'Trade', 'Direction',
                                    '30M', 'SL', 'Pullback', 'TP']


def test_load_data_fills_blank_numbers_with_zero():
    """A blank TP means the trade was not profitable, so it must read as 0 -
    the Signal rule is TP > 0 and a NaN would silently drop out of it."""
    df = load_data(CSV_PATH)
    for col in ('SL', 'Pullback', 'TP'):
        assert df[col].notna().all(), f'{col} still has NaN'
    assert (df['TP'] == 0).any(), 'expected losses to land on TP 0'


def test_load_data_derives_r_from_tp_over_sl():
    """This export has no R column, so R is TP / SL."""
    df = load_data(CSV_PATH)
    assert 'R' in df.columns

    survived = df[(df['Pullback'] < df['SL']) & (df['SL'] > 0)].iloc[0]
    assert survived['R'] == pytest.approx(survived['TP'] / survived['SL'])


def test_load_data_signs_r_negative_when_the_stop_was_hit():
    """A negative R records reach the recorded stop could not hold on to -
    the same convention the 15LS1CC sheet exports."""
    df = load_data(CSV_PATH)
    stopped = df[(df['Pullback'] >= df['SL']) & (df['TP'] > 0)]
    assert len(stopped) > 0, 'expected at least one stopped-out reach'
    assert (stopped['R'] < 0).all()


def test_load_data_splits_the_setup_label():
    """Setup drops the repeated '30M ' prefix; Side and Type are its halves."""
    df = load_data(CSV_PATH)
    assert set(df['Setup']) <= set(SETUP_ORDER)
    assert set(df['Side']) <= set(SIDE_ORDER)
    assert set(df['Type']) <= set(TYPE_ORDER)
    row = df.iloc[0]
    assert row['30M'] == f"30M {row['Setup']}"
    assert row['Setup'] == f"{row['Side']} {row['Type']}"


def test_load_data_matches_the_sheets_own_headline_rates():
    """The export labels its Signal and WR columns with the figures the sheet
    computed. They are dropped at load, but they are still the trader's own
    cross-check that this module scores the same trades the same way."""
    df = load_data(CSV_PATH)
    total = len(df)
    signal = (df['TP'] > 0).sum() / total * 100
    strategy = ((df['Pullback'] < df['SL'])
                & (df['TP'] >= df['SL'])).sum() / total * 100
    assert round(signal, 1) == 50.7
    assert round(strategy, 1) == 37.7


# --- Signal / Strategy grouping tables ------------------------------------

def test_format_wl_shape():
    assert _format_wl(3, 7, 10) == '3W - 7L (30.0%)'
    assert _format_wl(0, 0, 0) == '0W - 0L (0.0%)'


def test_weekday_statistics_has_a_row_per_weekday():
    result = calculate_weekday_statistics(get_sample_data())
    assert list(result['Day']) == WEEKDAY_ORDER
    assert list(result.columns) == ['Day', 'Trades', 'Signal', 'Strategy']


def test_weekday_statistics_counts_trades_per_day():
    result = calculate_weekday_statistics(get_sample_data())
    counts = dict(zip(result['Day'], result['Trades']))
    assert counts['Monday'] == 4
    assert counts['Tuesday'] == 3
    assert counts['Wednesday'] == 3
    assert counts['Friday'] == 0


def test_signal_counts_the_stopped_out_winner_and_strategy_does_not():
    """Row 2 reached 20 pips but its pullback took out the 5 pip stop first:
    the idea was right, the entry was too early."""
    monday = get_sample_data().iloc[1:2]
    result = calculate_weekday_statistics(monday)
    row = result[result['Day'] == 'Monday'].iloc[0]
    assert row['Signal'] == '1W - 0L (100.0%)'
    assert row['Strategy'] == '0W - 1L (0.0%)'


def test_setup_statistics_opens_with_default_and_lists_every_setup():
    result = calculate_setup_statistics(get_sample_data())
    assert list(result['Setup']) == ['Default'] + SETUP_ORDER
    assert result.iloc[0]['Trades'] == 10
    assert list(result.columns) == ['Setup', 'Trades', 'Signal', 'Strategy']


def test_setup_rows_sum_to_the_default_row():
    result = calculate_setup_statistics(get_sample_data())
    assert result.iloc[1:]['Trades'].sum() == result.iloc[0]['Trades']


def test_setup_statistics_keeps_an_unlisted_label():
    """A setup the export starts recording tomorrow still gets a row."""
    sample = get_sample_data()
    sample.loc[0, 'Setup'] = 'Mid Sweep'
    result = calculate_setup_statistics(sample)
    assert 'Mid Sweep' in list(result['Setup'])
    assert result[result['Setup'] == 'Mid Sweep'].iloc[0]['Trades'] == 1


def test_side_statistics_splits_high_from_low():
    result = calculate_side_statistics(get_sample_data())
    assert list(result['30M Side']) == ['Default'] + SIDE_ORDER
    counts = dict(zip(result['30M Side'], result['Trades']))
    assert counts['High'] + counts['Low'] == counts['Default'] == 10


def test_type_statistics_splits_continuation_from_reversal():
    result = calculate_type_statistics(get_sample_data())
    assert list(result['Setup Type']) == ['Default'] + TYPE_ORDER
    counts = dict(zip(result['Setup Type'], result['Trades']))
    assert counts['Continuation'] + counts['Reversal'] == counts['Default']


def test_direction_statistics_splits_buys_from_sells():
    result = calculate_direction_statistics(get_sample_data())
    assert list(result['Direction']) == ['Default', 'Buy', 'Sell']
    counts = dict(zip(result['Direction'], result['Trades']))
    assert counts['Buy'] + counts['Sell'] == counts['Default']


def test_grouping_tables_handle_an_empty_dataset():
    empty = get_empty_data()
    for fn in (calculate_weekday_statistics, calculate_setup_statistics,
               calculate_side_statistics, calculate_type_statistics,
               calculate_direction_statistics):
        result = fn(empty)
        assert (result['Trades'] == 0).all()
        assert (result['Signal'] == '0W - 0L (0.0%)').all()


# --- stop tables -----------------------------------------------------------

def test_sl_statistics_lists_every_band():
    result = calculate_sl_statistics(get_sample_data())
    assert list(result['SL Range']) == ['Default'] + [n for n, _, _ in SL_RANGES]
    assert list(result.columns) == ['SL Range', 'Trades', 'Signal', 'Strategy']


def test_sl_bands_overlap_on_purpose():
    """0-10 is the union of 0-5 and 5-10, and each is read on its own."""
    result = calculate_sl_statistics(get_sample_data()).set_index('SL Range')
    assert (result.loc['0-5 SL', 'Trades'] + result.loc['5-10 SL', 'Trades']
            == result.loc['0-10 SL', 'Trades'])


def test_sl_buffer_statistics_lists_every_buffer():
    result = calculate_sl_buffer_statistics(get_sample_data())
    assert len(result) == len(SL_BUFFER_PIPS)
    assert result.iloc[0]['SL Buffer'] == 'Default'
    assert list(result['SL Buffer'])[1:3] == ['1 pip', '2 pips']
    assert list(result.columns) == ['SL Buffer', 'Trades', 'Notation', 'Win Rate']


def test_sl_buffer_widens_the_target_with_the_stop():
    """Row 9 survives its 8 pip stop and just clears a 9 pip target. Pad the
    stop by 2 and the target moves to 10, which its 9 pip TP never reaches -
    a wider stop is not free."""
    one = get_sample_data().iloc[8:9]
    result = calculate_sl_buffer_statistics(one).set_index('SL Buffer')
    assert result.loc['Default', 'Notation'] == '1W - 0L'
    assert result.loc['2 pips', 'Win Rate'] == '0.0%'


def test_sl_fixed_statistics_reports_signal_and_strategy():
    result = calculate_sl_fixed_statistics(get_sample_data())
    assert list(result.columns) == ['Fixed SL', 'Trades', 'Signal', 'Strategy']
    assert result.iloc[0]['Fixed SL'] == 'Default'
    assert list(result['Fixed SL'])[1:] == [f'{p} pips' for p in SL_FIXED_PIPS]


def test_fixed_sl_signal_repeats_down_the_table():
    """Signal does not depend on the stop: it is the ceiling no size beats."""
    result = calculate_sl_fixed_statistics(get_sample_data())
    assert result['Signal'].nunique() == 1


def test_every_stop_table_opens_with_the_same_default_row():
    """SL Range, Adding Buffer, Fixed SL and Pullback all start from the
    recorded stops, so their Default rows must agree - one shared win rule."""
    sample = get_sample_data()
    tables = {
        'SL Range': (calculate_sl_statistics(sample), 'SL Range'),
        'Adding Buffer': (calculate_sl_buffer_statistics(sample), 'SL Buffer'),
        'Fixed SL': (calculate_sl_fixed_statistics(sample), 'Fixed SL'),
        'Pullback': (calculate_pullback_statistics(sample), 'Pullback'),
    }

    seen = {}
    for name, (result, column) in tables.items():
        first = result.iloc[0]
        assert first[column] == 'Default', f'{name} does not open with Default'
        # SL Range and Fixed SL report Signal/Strategy in one cell each; the
        # rest keep Notation and Win Rate apart. Compare on the combined form
        # so the shared rule stays pinned across both shapes.
        if 'Strategy' in result.columns:
            seen[name] = (first['Trades'], first['Strategy'])
        else:
            # Pullback's Default notation carries a 0M suffix - nothing is
            # missed when every trade is taken at the signal.
            notation = first['Notation'].replace(' - 0M', '')
            seen[name] = (first['Trades'],
                          f"{notation} ({first['Win Rate']})")

    assert len(set(seen.values())) == 1, f'Default rows disagree: {seen}'
    assert seen['SL Range'][0] == len(sample)


# --- TP distribution and pullback entries ---------------------------------

def test_tp_statistics_buckets_profitable_trades_only():
    result = calculate_tp_statistics(get_sample_data())
    assert list(result['TP Range']) == [n for n, _, _ in TP_RANGES]
    profitable = int((get_sample_data()['TP'] > 0).sum())
    assert all(row.endswith(f'of {profitable}') for row in result['Trades'])


def test_tp_bands_hold_each_trade_once():
    result = calculate_tp_statistics(get_sample_data())
    counted = sum(int(cell.split(' of ')[0]) for cell in result['Trades'])
    assert counted == int((get_sample_data()['TP'] > 0).sum())


def test_pullback_statistics_lists_every_level():
    result = calculate_pullback_statistics(get_sample_data())
    assert result.iloc[0]['Pullback'] == 'Default'
    assert list(result['Pullback'])[-1] == 'Half'
    assert len(result) == len(PULLBACK_ENTRY_PIPS) + 1
    assert list(result.columns) == ['Pullback', 'Trades', 'Notation', 'Win Rate']


def test_pullback_excludes_winners_the_limit_never_filled():
    """Row 5 pulled back 0.5 pips, so a 3 pip limit never fills - it is an M,
    not a loss, and it must stay out of Trades and Win Rate."""
    result = calculate_pullback_statistics(get_sample_data()).set_index('Pullback')
    default_trades = result.loc['Default', 'Trades']
    deep = result.loc['3 pips']
    wins, losses, missed = (int(p[:-1]) for p in deep['Notation'].split(' - '))
    assert deep['Trades'] == wins + losses < default_trades
    assert missed > 0


# --- Strategies tables ----------------------------------------------------

def test_get_buffer_strategies_covers_fixed_and_max_sl():
    names = [name for name, _ in get_buffer_strategies()]
    assert names[0] == 'All Trades'
    assert any(n.startswith('Fixed SL ') for n in names)
    assert any(n.startswith('Max SL ') for n in names)


def test_buffer_statistics_scores_every_rrr():
    result = calculate_buffer_statistics(get_sample_data(), ['All Trades'])
    assert set(result['RRR']) == {f'1:{r}' for r in RRR_RATIOS}
    assert set(result['Buffer']) == {f'+{b}' for b in BUFFER_PIPS}


def test_fixed_and_max_sl_strategies_run_without_a_buffer():
    """A buffer would undo the fixed or capped stop the row is testing."""
    names = [n for n, _ in get_buffer_strategies() if n != 'All Trades']
    result = calculate_buffer_statistics(get_sample_data(), names)
    assert set(result['Buffer']) == {'+0'}


def test_calculate_stats_with_buffer_applies_the_rrr_target():
    """Row 1: SL 4, TP 16 - a winner at 1:1 through 1:3, and 1:4 would need 16
    exactly, so the rule is >= and not >."""
    one = get_sample_data().iloc[0:1]
    assert _calculate_stats_with_buffer(one, 'x', 0, 3)['Notation'] == '1W – 0L'
    assert _calculate_stats_with_buffer(one, 'x', 0, 4)['Notation'] == '1W – 0L'
    assert _calculate_stats_with_buffer(one, 'x', 0, 5)['Notation'] == '0W – 1L'


def test_calculate_stats_with_buffer_handles_no_trades():
    stats = _calculate_stats_with_buffer(get_empty_data(), 'x', 0, 1)
    assert stats['Trades'] == 0
    assert stats['Win Rate'] == '0.0%'


# --- R distribution -------------------------------------------------------

def test_r_counts_truncate_towards_zero_and_drop_the_sign():
    counts = calculate_r_counts(get_sample_data())
    # 4.0 -> 4R, -4.0 -> 4R (the reach was there either way), 3.571 -> 3R.
    assert counts[4] == 2
    assert counts[3] == 1
    assert counts[5] == 1  # 5.833


def test_r_counts_ignore_sub_1r_trades():
    counts = calculate_r_counts(get_sample_data())
    assert counts.sum() == 7  # the three 0R rows fall outside 1R-10R


def test_r_histograms_render_every_level():
    combined = create_r_histogram_combined(get_sample_data())
    exact = create_r_histogram_exact(get_sample_data())
    for html in (combined, exact):
        for level in range(1, 11):
            assert f'>{level}R<' in html
    assert 'R Distribution (Exact)' in exact


def test_cumulative_histogram_never_falls_below_the_exact_one():
    sample = get_sample_data()
    raw = calculate_r_counts(sample)
    assert raw.loc[1:].sum() >= raw[1]


# --- three setups ---------------------------------------------------------

def test_three_setups_has_a_row_per_trade_in_recorded_order():
    sample = get_sample_data()
    result = calculate_three_setups_comparison(sample)
    assert len(result) == len(sample)
    assert list(result['Date']) == list(sample['Date'])
    assert list(result.columns) == [key for _, _, key in THREE_SETUPS_COLUMNS]


def test_three_setups_carries_the_setup_instead_of_an_hour():
    """This export has no Hour, and the setup is the column worth reading
    beside each trade."""
    headers = [header for _, header, _ in THREE_SETUPS_COLUMNS]
    assert 'Setup' in headers
    assert 'Hour' not in headers
    result = calculate_three_setups_comparison(get_sample_data())
    assert result.iloc[0]['Setup'] == 'High Continuation'


def test_three_setups_outcome_is_cumulative_r():
    result = calculate_three_setups_comparison(get_sample_data())
    outcomes = [int(v.rstrip('R')) for v in result['Regular Outcome']]
    steps = {b - a for a, b in zip(outcomes, outcomes[1:])}
    assert steps <= {THREE_SETUPS_RRR, -1}


def test_waiter_leaves_unfilled_trades_blank():
    """Row 5 pulled back 0.5 against a 3 pip stop, so a 1.5 pip limit never
    filled: the Waiter cells stay empty and its running total carries over."""
    result = calculate_three_setups_comparison(get_sample_data())
    unfilled = result.iloc[4]
    assert unfilled['Waiter SL'] == ''
    assert unfilled['Waiter Pullback'] == ''
    assert unfilled['Waiter ROI'] == result.iloc[3]['Waiter ROI']


def test_aggressive_halves_the_stop():
    result = calculate_three_setups_comparison(get_sample_data())
    assert result.iloc[0]['Aggressive SL'] == '2'  # 4.0 / 2


def test_pip_cells_round_half_up_and_trim_zeros():
    assert _pip_cell(2.75) == '2.8'
    assert _pip_cell(2.15) == '2.2'
    assert _pip_cell(34.0) == '34'
    assert _pip_cell(None) == ''
    assert _whole_pip_cell(16.5) == '17'
    assert _whole_pip_cell(None) == ''


def test_three_setups_table_renders_grouped_headers():
    stats = calculate_three_setups_comparison(get_sample_data())
    html = create_three_setups_table(stats, 'three-setups-table')
    assert 'id="three-setups-table"' in html
    for group in ('Regular', 'Aggressive', 'Waiter'):
        assert f'>{group}</th>' in html
    assert html.count('<tr>') == len(stats) + 2  # two header rows


def test_three_setups_table_handles_an_empty_dataset():
    stats = calculate_three_setups_comparison(get_empty_data())
    assert 'No data' in create_three_setups_table(stats, 'x')


# --- table rendering ------------------------------------------------------

def test_create_html_table_marks_percentage_columns_sortable():
    stats = calculate_setup_statistics(get_sample_data())
    html = create_html_table(stats, sort_id='setup-table')
    assert 'id="setup-table"' in html
    assert "sortAnalysisTable('setup-table'" in html
    assert 'class="sortable"' in html


def test_create_html_table_without_a_sort_id_is_not_sortable():
    html = create_html_table(calculate_setup_statistics(get_sample_data()))
    assert 'class="sortable"' not in html


def test_create_html_table_pins_the_first_column_width():
    html = create_html_table(calculate_weekday_statistics(get_sample_data()),
                             first_col_width='40%')
    assert 'table-layout: fixed' in html
    assert 'width: 40%' in html


def test_create_sortable_table_can_refuse_to_sort():
    """Row order carries meaning in the stop tables, so they opt out."""
    stats = calculate_sl_statistics(get_sample_data())
    assert 'class="sortable"' not in create_sortable_table(
        stats, 'sl-range-table', sortable=False)
    assert 'class="sortable"' in create_sortable_table(stats, 'sl-range-table')


def test_tables_render_no_data_when_empty():
    assert 'No data' in create_html_table(get_empty_data())
    assert 'No data' in create_sortable_table(get_empty_data(), 'x')
