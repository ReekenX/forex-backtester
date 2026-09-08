"""
Tests for utils.order_block - the analysis behind the 5OB1CC HTML report.

Run with: poetry run python -m pytest strategies/5OB1CC/tests/ -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.order_block import (  # noqa: E402
    BUFFER_PIPS,
    HOUR_ORDER,
    LEG_ORDER,
    PULLBACK_ENTRY_PIPS,
    RRR_RATIOS,
    SL_BUFFER_PIPS,
    SL_FIXED_PIPS,
    SL_RANGES,
    THREE_SETUPS_COLUMNS,
    THREE_SETUPS_RRR,
    TP_RANGES,
    WEEKDAY_ORDER,
    _calculate_stats_with_buffer,
    _format_wl,
    _pip_cell,
    _whole_pip_cell,
    calculate_buffer_statistics,
    calculate_ema_alignment_statistics,
    calculate_hour_statistics,
    calculate_htf_leg_statistics,
    calculate_pullback_statistics,
    calculate_r_counts,
    calculate_sl_buffer_statistics,
    calculate_sl_fixed_statistics,
    calculate_sl_statistics,
    calculate_structure_statistics,
    calculate_three_setups_comparison,
    calculate_tp_statistics,
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
    os.path.dirname(os.path.abspath(__file__)), '..', 'data.csv')


def get_sample_data():
    """
    Ten trades over three days, covering every case the tables must separate.

    Row 2 is the stopped-out winner-that-wasn't: Pullback 6.0 >= SL 5.0 with a
    TP of 20, so Signal counts it and Strategy does not.
    """
    return pd.DataFrame({
        'Date': ['2025-02-03'] * 4 + ['2025-02-04'] * 3 + ['2025-02-05'] * 3,
        'Trade': ['#1', '#2', '#3', '#4', '#1', '#2', '#3', '#1', '#2', '#3'],
        'Weekday': ['Monday'] * 4 + ['Tuesday'] * 3 + ['Wednesday'] * 3,
        'Hour': [10, 10, 12, 18, 11, 14, 17, 10, 13, 18],
        'Direction': ['Buy', 'Sell', 'Buy', 'Sell',
                      'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        'EMA': ['Buy', 'Buy', 'Buy', 'Sell',
                'Sell', 'Sell', 'Buy', 'Buy', 'Buy', 'Sell'],
        'SL': [4.0, 5.0, 2.0, 7.0, 3.0, 12.0, 6.0, 2.5, 8.0, 5.5],
        'Pullback': [1.0, 6.0, 2.0, 3.5, 0.5, 12.0, 2.0, 2.5, 4.0, 1.5],
        'TP': [16.0, 20.0, 0.0, 25.0, 3.0, 0.0, 35.0, 0.0, 9.0, 12.0],
        'BOS/CH': ['BOS', 'BOS', 'CH', 'BOS', 'CH', 'BOS', 'CH', 'BOS', 'BOS', 'CH'],
        '30M Leg': ['Above H', 'Above L', 'Below H', 'Below L', 'Above H',
                    'Above L', 'Below H', 'Below L', 'Above H', 'Above L'],
        'R': [4.0, -4.0, 0.0, 3.571, 1.0, 0.0, 5.833, 0.0, 1.125, 2.182],
    })


def get_empty_data():
    return get_sample_data().iloc[0:0].copy()


# --- load_data -------------------------------------------------------------

def test_load_data_reads_the_project_csv():
    df = load_data(CSV_PATH)
    assert len(df) > 0
    for col in ('Date', 'Weekday', 'Hour', 'Direction', 'EMA',
                'SL', 'Pullback', 'TP', 'BOS/CH', '30M Leg'):
        assert col in df.columns


def test_load_data_fills_blank_numbers_with_zero():
    """A blank TP means the trade was not profitable, so it must read as 0 -
    the Signal rule is TP > 0 and a NaN would silently drop out of it."""
    df = load_data(CSV_PATH)
    for col in ('SL', 'Pullback', 'TP', 'Hour'):
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
    assert len(stopped) > 0, 'expected stopped-out trades that reached a target'
    assert (stopped['R'] <= 0).all()


def test_load_data_never_divides_by_a_zero_stop():
    df = load_data(CSV_PATH)
    assert df['R'].notna().all()
    assert df['R'].abs().max() < float('inf')


# --- Signal / Strategy grouping tables ------------------------------------

def test_format_wl_shape():
    assert _format_wl(3, 7, 10) == '3W - 7L (30.0%)'
    assert _format_wl(0, 0, 0) == '0W - 0L (0.0%)'


def test_weekday_statistics_columns_and_rows():
    stats = calculate_weekday_statistics(get_sample_data())
    assert list(stats.columns) == ['Day', 'Trades', 'Signal', 'Strategy']
    assert list(stats['Day']) == WEEKDAY_ORDER


def test_weekday_trade_counts():
    stats = calculate_weekday_statistics(get_sample_data())
    counts = dict(zip(stats['Day'], stats['Trades']))
    assert counts == {'Monday': 4, 'Tuesday': 3, 'Wednesday': 3,
                      'Thursday': 0, 'Friday': 0}


def test_signal_counts_the_stopped_out_winner_and_strategy_does_not():
    """Monday trade #2: Pullback 6.0 >= SL 5.0, TP 20. The idea was right but
    the recorded stop was taken out first."""
    stats = calculate_weekday_statistics(get_sample_data())
    monday = stats[stats['Day'] == 'Monday'].iloc[0]
    # Signal winners: #1 (TP 16), #2 (TP 20), #4 (TP 25) -> 3 of 4.
    assert monday['Signal'] == '3W - 1L (75.0%)'
    # Strategy winners: #1 and #4 only - #2 was stopped out, #3 has TP 0.
    assert monday['Strategy'] == '2W - 2L (50.0%)'


def test_strategy_is_always_a_subset_of_signal():
    """Strategy adds the survival check to Signal, so it can never be higher."""
    df = load_data(CSV_PATH)
    tables = (
        calculate_weekday_statistics(df),
        calculate_hour_statistics(df),
        calculate_ema_alignment_statistics(df),
        calculate_structure_statistics(df),
        calculate_htf_leg_statistics(df),
        calculate_sl_statistics(df),
        calculate_sl_fixed_statistics(df),
    )
    for stats in tables:
        for _, row in stats.iterrows():
            signal = int(row['Signal'].split('W')[0])
            strategy = int(row['Strategy'].split('W')[0])
            assert strategy <= signal, row.to_dict()


def test_hour_statistics_opens_with_default_then_every_hour():
    stats = calculate_hour_statistics(get_sample_data())
    assert list(stats.columns) == ['Hour', 'Trades', 'Signal', 'Strategy']
    assert list(stats['Hour']) == ['Default'] + [f'{h}h' for h in HOUR_ORDER]
    assert stats.iloc[0]['Trades'] == 10


def test_hour_rows_partition_the_dataset():
    """Every trade falls in exactly one hour, so the hours sum to Default."""
    df = load_data(CSV_PATH)
    stats = calculate_hour_statistics(df)
    assert stats.iloc[1:]['Trades'].sum() == stats.iloc[0]['Trades'] == len(df)


def test_hour_statistics_counts():
    stats = calculate_hour_statistics(get_sample_data())
    counts = dict(zip(stats['Hour'], stats['Trades']))
    assert counts['10h'] == 3
    assert counts['18h'] == 2
    assert counts['15h'] == 0


def test_ema_alignment_splits_with_and_against():
    stats = calculate_ema_alignment_statistics(get_sample_data())
    assert list(stats.columns) == ['EMA Alignment', 'Trades', 'Signal', 'Strategy']
    assert list(stats['EMA Alignment']) == ['Default', 'Aligned', 'Against']
    counts = dict(zip(stats['EMA Alignment'], stats['Trades']))
    # Against rows are the three where Direction and EMA disagree: Monday #2
    # (Sell/Buy), Tuesday #1 (Buy/Sell) and Wednesday #1 (Sell/Buy).
    assert counts['Aligned'] == 7
    assert counts['Against'] == 3
    assert counts['Aligned'] + counts['Against'] == counts['Default'] == 10


def test_structure_statistics_splits_bos_and_ch():
    stats = calculate_structure_statistics(get_sample_data())
    assert list(stats.columns) == ['Structure', 'Trades', 'Signal', 'Strategy']
    assert list(stats['Structure']) == ['Default', 'BOS', 'CH']
    counts = dict(zip(stats['Structure'], stats['Trades']))
    assert counts['BOS'] == 6
    assert counts['CH'] == 4
    assert counts['BOS'] + counts['CH'] == counts['Default']


def test_htf_leg_statistics_covers_every_recorded_position():
    stats = calculate_htf_leg_statistics(get_sample_data())
    assert list(stats.columns) == ['30M Leg', 'Trades', 'Signal', 'Strategy']
    assert list(stats['30M Leg']) == ['Default'] + LEG_ORDER
    assert stats.iloc[1:]['Trades'].sum() == stats.iloc[0]['Trades']


def test_grouping_tables_handle_an_empty_dataset():
    empty = get_empty_data()
    for fn in (calculate_weekday_statistics, calculate_hour_statistics,
               calculate_ema_alignment_statistics, calculate_structure_statistics,
               calculate_htf_leg_statistics, calculate_sl_statistics):
        stats = fn(empty)
        assert (stats['Trades'] == 0).all()
        assert stats['Signal'].str.endswith('(0.0%)').all()


# --- stop tables ----------------------------------------------------------

def test_sl_ranges_constant():
    assert SL_RANGES == [("0-5 SL", 0, 5), ("0-10 SL", 0, 10),
                         ("5-10 SL", 5, 10), ("10+ pips", 10, float('inf'))]


def test_sl_statistics_opens_with_default_over_every_trade():
    stats = calculate_sl_statistics(get_sample_data())
    assert list(stats.columns) == ['SL Range', 'Trades', 'Signal', 'Strategy']
    assert stats.iloc[0]['SL Range'] == 'Default'
    assert stats.iloc[0]['Trades'] == 10


def test_sl_bands_are_exclusive_at_the_top():
    """A 5.0 pip stop belongs to 5-10, not 0-5."""
    df = pd.DataFrame({'SL': [5.0], 'Pullback': [1.0], 'TP': [10.0]})
    stats = calculate_sl_statistics(df).set_index('SL Range')
    assert stats.loc['0-5 SL', 'Trades'] == 0
    assert stats.loc['5-10 SL', 'Trades'] == 1


def test_sl_buffer_statistics_shape():
    stats = calculate_sl_buffer_statistics(get_sample_data())
    assert list(stats.columns) == ['SL Buffer', 'Trades', 'Notation', 'Win Rate']
    assert list(stats['SL Buffer']) == [
        'Default', '1 pip', '2 pips', '3 pips', '4 pips', '5 pips']


def test_sl_buffer_widens_the_target_with_the_stop():
    """SL 3.0, Pullback 3.5, TP 6: stopped out at +0, a winner at +1, and
    still a winner at +3 where the target is exactly 6.0."""
    df = pd.DataFrame({'SL': [3.0], 'Pullback': [3.5], 'TP': [6.0]})
    stats = calculate_sl_buffer_statistics(df).set_index('SL Buffer')
    assert stats.loc['Default', 'Notation'] == '0W - 1L'
    assert stats.loc['1 pip', 'Notation'] == '1W - 0L'
    assert stats.loc['3 pips', 'Notation'] == '1W - 0L'
    assert stats.loc['4 pips', 'Notation'] == '0W - 1L'


def test_sl_fixed_statistics_shape():
    stats = calculate_sl_fixed_statistics(get_sample_data())
    assert list(stats.columns) == ['Fixed SL', 'Trades', 'Signal', 'Strategy']
    assert list(stats['Fixed SL']) == (
        ['Default'] + [f'{p} pips' for p in SL_FIXED_PIPS])


def test_fixed_sl_signal_repeats_down_the_table():
    """The stop cannot change whether price reached a target, so Signal is the
    same ceiling on every row."""
    stats = calculate_sl_fixed_statistics(get_sample_data())
    assert stats['Signal'].nunique() == 1


def test_every_stop_table_opens_with_the_same_default_row():
    """SL Range, Adding Buffer and Fixed SL all start from the recorded stops,
    so their Default rows must agree - they share one win rule."""
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


# --- TP and pullback ------------------------------------------------------

def test_tp_ranges_constant():
    assert TP_RANGES == [("0-10", 0, 10), ("10-20", 10, 20),
                         ("20-30", 20, 30), ("30+", 30, float("inf"))]


def test_tp_statistics_counts_only_profitable_trades():
    stats = calculate_tp_statistics(get_sample_data())
    assert list(stats.columns) == ['TP Range', 'Trades']
    # Profitable: 16, 20, 25, 3, 35, 9, 12 -> 7 trades.
    assert list(stats['Trades']) == ['2 of 7', '2 of 7', '2 of 7', '1 of 7']


def test_tp_bands_partition_the_profitable_trades():
    df = load_data(CSV_PATH)
    stats = calculate_tp_statistics(df)
    counts = [int(v.split(' of ')[0]) for v in stats['Trades']]
    assert sum(counts) == int((df['TP'] > 0).sum())


def test_pullback_entry_pips_constant():
    assert PULLBACK_ENTRY_PIPS == [0, 1, 2, 3]


def test_pullback_statistics_shape_and_levels():
    stats = calculate_pullback_statistics(get_sample_data())
    assert list(stats.columns) == ['Pullback', 'Trades', 'Notation', 'Win Rate']
    assert list(stats['Pullback']) == [
        'Default', '1 pip', '2 pips', '3 pips', 'Half']


def test_pullback_default_takes_every_trade_and_misses_nothing():
    stats = calculate_pullback_statistics(get_sample_data())
    default = stats.iloc[0]
    assert default['Trades'] == 10
    assert default['Notation'].endswith('- 0M')


def test_pullback_deeper_entry_misses_more_winners():
    stats = calculate_pullback_statistics(get_sample_data()).set_index('Pullback')
    missed = [int(stats.loc[lvl, 'Notation'].split('- ')[-1].rstrip('M'))
              for lvl in ('Default', '1 pip', '2 pips', '3 pips')]
    assert missed == sorted(missed)


def test_pullback_win_rate_ignores_missed_winners():
    """A limit order that never filled is not a loss - it is not a trade."""
    df = pd.DataFrame({'SL': [4.0, 4.0], 'Pullback': [0.2, 3.0], 'TP': [10.0, 10.0]})
    stats = calculate_pullback_statistics(df).set_index('Pullback')
    row = stats.loc['2 pips']
    # Only the 3.0 pullback filled, and it won. The 0.2 one is a missed winner.
    assert row['Trades'] == 1
    assert row['Notation'] == '1W - 0L - 1M'
    assert row['Win Rate'] == '100.0%'


# --- R distribution -------------------------------------------------------

def test_r_counts_truncate_towards_zero():
    counts = calculate_r_counts(get_sample_data())
    assert list(counts.index) == list(range(1, 11))
    # R values 4.0, -4.0, 3.571, 1.0, 5.833, 1.125, 2.182 -> 1R x2, 2R, 3R,
    # 4R x2, 5R.
    assert counts[1] == 2
    assert counts[2] == 1
    assert counts[3] == 1
    assert counts[4] == 2
    assert counts[5] == 1


def test_r_counts_drop_the_sign():
    """A negative R records the same reach as a positive one."""
    df = pd.DataFrame({'R': [-3.5, 3.5]})
    assert calculate_r_counts(df)[3] == 2


def test_r_counts_ignore_levels_outside_1_to_10():
    df = pd.DataFrame({'R': [0.4, 11.0, 25.0, 2.0]})
    counts = calculate_r_counts(df)
    assert counts.sum() == 1
    assert counts[2] == 1


def test_r_histograms_render_every_level():
    df = get_sample_data()
    for html in (create_r_histogram_combined(df), create_r_histogram_exact(df)):
        for level in range(1, 11):
            assert f'>{level}R</span>' in html


def test_cumulative_histogram_is_monotonic():
    """A trade reaching 3R passed through 1R, so bars never rise with R."""
    import re
    html = create_r_histogram_combined(load_data(CSV_PATH))
    widths = [float(w) for w in re.findall(r'width: ([\d.]+)%;', html)]
    assert widths == sorted(widths, reverse=True)


# --- Strategies tables ----------------------------------------------------

def test_rrr_ratios_stop_at_1_3():
    """1:3 is the ceiling - see CLAUDE.md."""
    assert RRR_RATIOS == [1, 2, 3]
    assert BUFFER_PIPS == [0, 1, 2, 3]


def test_buffer_strategies_cover_all_fixed_and_max_stops():
    names = [name for name, _ in get_buffer_strategies()]
    assert names[0] == 'All Trades'
    assert 'Fixed SL 3' in names
    assert 'Max SL 10' in names


def test_stats_with_buffer_moves_the_target_with_the_stop():
    trades = pd.DataFrame({'SL': [3.0], 'Pullback': [3.5], 'TP': [6.0]})
    assert _calculate_stats_with_buffer(trades, 'x', 0, 1)['Notation'] == '0W – 1L'
    assert _calculate_stats_with_buffer(trades, 'x', 1, 1)['Notation'] == '1W – 0L'
    # At 1:2 the +1 stop needs 8 pips and only has 6.
    assert _calculate_stats_with_buffer(trades, 'x', 1, 2)['Notation'] == '0W – 1L'


def test_stats_with_buffer_on_no_trades():
    stats = _calculate_stats_with_buffer(get_empty_data(), 'x', 0, 1)
    assert stats['Trades'] == 0
    assert stats['Win Rate'] == '0.0%'


def test_buffer_statistics_covers_every_rrr():
    stats = calculate_buffer_statistics(get_sample_data(), ['All Trades'])
    assert set(stats['Strategy']) == {'All Trades'}
    assert set(stats['RRR']) == {f'1:{r}' for r in RRR_RATIOS}


def test_buffer_statistics_restricts_to_named_strategies():
    stats = calculate_buffer_statistics(get_sample_data(), ['Fixed SL 3'])
    assert set(stats['Strategy']) == {'Fixed SL 3'}
    # A fixed stop runs with buffer 0 only - a buffer would undo it.
    assert set(stats['Buffer']) == {'+0'}


def test_buffer_statistics_is_empty_for_an_unknown_strategy():
    assert calculate_buffer_statistics(get_sample_data(), ['nope']).empty


def test_max_sl_strategy_keeps_every_trade_but_tightens_wide_stops():
    """Max SL caps the stop; it does not filter trades out."""
    df = get_sample_data()
    capped = calculate_buffer_statistics(df, ['Max SL 5'])
    row = capped[(capped['RRR'] == '1:1') & (capped['Max SL'] == 0)].iloc[0]
    assert row['Trades'] == len(df)


# --- three setups ---------------------------------------------------------

def test_three_setups_is_scored_at_1_2():
    assert THREE_SETUPS_RRR == 2


def test_three_setups_keeps_one_row_per_trade_in_order():
    df = get_sample_data()
    stats = calculate_three_setups_comparison(df)
    assert len(stats) == len(df)
    assert list(stats.columns) == [key for _, _, key in THREE_SETUPS_COLUMNS]
    assert list(stats['Date']) == list(df['Date'])


def test_three_setups_carries_the_hour_column():
    """The 5OB1CC export records the entry hour, so the trade log shows it."""
    stats = calculate_three_setups_comparison(get_sample_data())
    assert list(stats['Hour'])[:3] == ['10h', '10h', '12h']


def test_three_setups_regular_columns_are_the_recorded_trade():
    stats = calculate_three_setups_comparison(get_sample_data())
    first = stats.iloc[0]
    assert first['Regular SL'] == '4'
    assert first['Regular Pullback'] == '1'
    assert first['Regular TP'] == '16'


def test_three_setups_blank_tp_renders_empty():
    stats = calculate_three_setups_comparison(get_sample_data())
    # Monday #3 has TP 0 - not profitable, so the cell is blank not '0'.
    assert stats.iloc[2]['Regular TP'] == ''


def test_three_setups_aggressive_halves_the_stop():
    stats = calculate_three_setups_comparison(get_sample_data())
    assert stats.iloc[0]['Aggressive SL'] == '2'
    assert stats.iloc[1]['Aggressive SL'] == '2.5'


def test_three_setups_outcomes_are_cumulative():
    stats = calculate_three_setups_comparison(get_sample_data())
    totals = [int(v.rstrip('R')) for v in stats['Regular Outcome']]
    # Each step is +2 for a win or -1 for a loss, never anything else.
    steps = [b - a for a, b in zip(totals, totals[1:])]
    assert set(steps) <= {2, -1}


def test_three_setups_waiter_misses_shallow_pullbacks():
    """A limit half a stop into the pullback never fills when the pullback did
    not reach it, so those cells stay blank and the ROI carries forward."""
    df = pd.DataFrame({
        'Date': ['2025-02-03'], 'Weekday': ['Monday'], 'Hour': [10],
        'Trade': ['#1'], 'Direction': ['Buy'],
        'SL': [4.0], 'Pullback': [0.5], 'TP': [12.0],
    })
    row = calculate_three_setups_comparison(df).iloc[0]
    assert row['Waiter SL'] == ''
    assert row['Waiter Pullback'] == ''
    assert row['Waiter ROI'] == '0R'


def test_three_setups_waiter_shifts_the_entry_into_the_pullback():
    df = pd.DataFrame({
        'Date': ['2025-02-03'], 'Weekday': ['Monday'], 'Hour': [10],
        'Trade': ['#1'], 'Direction': ['Buy'],
        'SL': [4.0], 'Pullback': [3.0], 'TP': [12.0],
    })
    row = calculate_three_setups_comparison(df).iloc[0]
    # Stop halves to 2.0, the pullback left to survive is 3.0 - 2.0 = 1.0, and
    # the target is 2.0 further away: 14 pips.
    assert row['Waiter SL'] == '2'
    assert row['Waiter Pullback'] == '1'
    assert row['Waiter TP'] == '14'
    assert row['Waiter ROI'] == '+2R'


def test_pip_cells_round_half_up():
    assert _pip_cell(2.75) == '2.8'
    assert _pip_cell(2.15) == '2.2'
    assert _pip_cell(34.0) == '34'
    assert _pip_cell(None) == ''
    assert _whole_pip_cell(16.5) == '17'
    assert _whole_pip_cell(25.15) == '25'
    assert _whole_pip_cell(None) == ''


# --- rendering ------------------------------------------------------------

def test_create_html_table_renders_headers_and_cells():
    html = create_html_table(calculate_weekday_statistics(get_sample_data()))
    assert '<th>Day</th>' in html
    assert '<td>Monday</td>' in html


def test_create_html_table_pins_the_first_column():
    html = create_html_table(
        calculate_weekday_statistics(get_sample_data()), first_col_width='40%')
    assert 'table-layout: fixed' in html
    assert '<th style="width: 40%;">Day</th>' in html


def test_create_html_table_sorts_only_percentage_columns():
    html = create_html_table(
        calculate_hour_statistics(get_sample_data()), sort_id='t')
    assert "sortAnalysisTable('t', 2, this)" in html  # Signal
    assert "sortAnalysisTable('t', 3, this)" in html  # Strategy
    assert "sortAnalysisTable('t', 1, this)" not in html  # Trades


def test_create_html_table_without_sort_id_has_no_sort_script():
    html = create_html_table(calculate_weekday_statistics(get_sample_data()))
    assert 'sortAnalysisTable' not in html


def test_create_html_table_colours_win_rate_against_breakeven():
    df = pd.DataFrame({'RRR': ['1:1', '1:2'], 'Win Rate': ['60.0%', '20.0%']})
    html = create_html_table(df)
    assert 'class="positive-edge">60.0%' in html
    assert 'class="negative-edge">20.0%' in html


def test_create_sortable_table_respects_the_sortable_flag():
    stats = calculate_sl_buffer_statistics(get_sample_data())
    assert "sortAnalysisTable('x'" in create_sortable_table(stats, 'x')
    assert "sortAnalysisTable('x'" not in create_sortable_table(
        stats, 'x', sortable=False)


def test_tables_render_a_message_when_there_is_nothing_to_show():
    empty = pd.DataFrame()
    assert 'No data' in create_html_table(empty)
    assert 'No data' in create_sortable_table(empty, 'x')
    assert 'No data' in create_three_setups_table(empty)


def test_three_setups_table_has_grouped_headers_and_a_row_per_trade():
    df = get_sample_data()
    html = create_three_setups_table(calculate_three_setups_comparison(df))
    assert 'colspan=' in html
    assert '>Regular</th>' in html and '>Aggressive</th>' in html
    assert '>Waiter</th>' in html
    assert html.count('<tr>') == len(df) + 2  # two header rows


def test_three_setups_table_colours_the_running_totals():
    html = create_three_setups_table(
        calculate_three_setups_comparison(get_sample_data()))
    assert 'positive-edge' in html
    assert 'negative-edge' in html
