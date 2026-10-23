"""
Tests for utils.continuation_reversal - the analysis behind the 30MC report.

Run with: poetry run python -m pytest strategies/30MC/tests/ -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.continuation_reversal import (  # noqa: E402
    RATE_COLUMNS,
    BUFFER_PIPS,
    MAX_SL_CAPS,
    TREND_FILTERS,
    RRR_RATIOS,
    SETUP_ORDER,
    SL_FIXED_PIPS,
    TP_RANGES,
    TYPE_BUFFER_PIPS,
    TYPE_ORDER,
    WEEKDAY_ORDER,
    _calculate_stats_with_buffer,
    _drop_scratch_columns,
    _normalise_header,
    EQUITY_HIGHLIGHT,
    HALF_SL_ENTRIES,
    calculate_buffer_statistics,
    calculate_half_sl_statistics,
    calculate_equity_curves,
    calculate_r_counts,
    calculate_setup_statistics,
    calculate_sl_fixed_statistics,
    calculate_streak_statistics,
    calculate_tp_pip_counts,
    calculate_tp_statistics,
    calculate_weekday_statistics,
    create_equity_chart,
    create_html_table,
    create_r_histogram_combined,
    create_signal_strategy_chart,
    create_sortable_table,
    create_tp_chart,
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
    present so the setup and type tables each have every group.
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
        '30M': [s.split()[0] for s in setups],
        'SL': [4.0, 5.0, 2.0, 7.0, 3.0, 12.0, 6.0, 2.5, 8.0, 5.5],
        'Pullback': [1.0, 6.0, 2.0, 3.5, 0.5, 12.0, 2.0, 2.5, 4.0, 1.5],
        'TP': [16.0, 20.0, 0.0, 25.0, 3.0, 0.0, 35.0, 0.0, 9.0, 12.0],
        'R': [4.0, -4.0, 0.0, 3.571, 1.0, 0.0, 5.833, 0.0, 1.125, 2.182],
        'Setup': setups,
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


def test_load_data_drops_everything_right_of_tp():
    """The sheet's own stop scenarios are scratch: this page recomputes them
    under a rule it controls, so reading them back would double up."""
    df = load_data(CSV_PATH)
    for col in df.columns:
        assert not str(col).startswith('Signal'), col
    for scratch in ('WR', '+1pip WR', '+1pip C WR', '+1pip R WR',
                    '4 pips SL', '5 pips SL'):
        assert scratch not in df.columns


def test_drop_scratch_columns_survives_renamed_and_repeated_headers():
    """One export renamed the first scratch column to "Signal All" and added
    two more both headed "Signal". Anchoring on a scratch name broke on
    that; anchoring on TP does not."""
    raw = pd.DataFrame(
        [[1.0, 2.0, 3.0, '6R', '', '4R']],
        columns=['SL', 'Pullback', 'TP', 'Signal All', 'Signal', 'Signal'])
    assert list(_drop_scratch_columns(raw).columns) == ['SL', 'Pullback', 'TP']


def test_drop_scratch_columns_leaves_a_clean_frame_alone():
    clean = pd.DataFrame([[1.0, 2.0, 3.0]], columns=['SL', 'Pullback', 'TP'])
    assert list(_drop_scratch_columns(clean).columns) == ['SL', 'Pullback', 'TP']


def test_load_data_keeps_every_trade_column():
    """Dropping the scratch columns must not drop a row or a trade column."""
    raw = pd.read_csv(CSV_PATH)
    df = load_data(CSV_PATH)
    assert len(df) == raw['Date'].notna().sum()
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


def test_load_data_derives_the_setup_from_level_and_direction():
    """30M holds only the level; Type comes from reading it against
    Direction, and Setup joins the two."""
    df = load_data(CSV_PATH)
    assert set(df['30M']) <= {'High', 'Low'}
    assert set(df['Setup']) <= set(SETUP_ORDER)
    assert set(df['Type']) <= set(TYPE_ORDER)
    row = df.iloc[0]
    assert row['Setup'] == f"{row['30M']} {row['Type']}"


@pytest.mark.parametrize('level, direction, setup', [
    ('High', 'Buy', 'High Continuation'),
    ('High', 'Sell', 'High Reversal'),
    ('Low', 'Sell', 'Low Continuation'),
    ('Low', 'Buy', 'Low Reversal'),
])
def test_load_data_maps_each_level_and_direction(tmp_path, level, direction,
                                                 setup):
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,30M,SL,Pullback,TP\n'
                   f'2025-02-03,Monday,#1,{direction},{level},4,1,8\n')
    row = load_data(str(csv)).iloc[0]
    assert row['Setup'] == setup
    assert row['Type'] == setup.split()[-1]


def test_load_data_drops_rows_without_a_date(tmp_path):
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,30M,SL,Pullback,TP\n'
                   '2025-02-03,Monday,#1,Buy,High,4,1,12\n'
                   ',,,,,,,\n'
                   ',,,,,,,\n')
    df = load_data(str(csv))
    assert len(df) == 1


def test_load_data_matches_the_sheets_own_headline_rates():
    """The export labels its Signal and WR columns with the figures the sheet
    computed. They are dropped at load, but they are still the trader's own
    cross-check that this module scores the same trades the same way."""
    df = load_data(CSV_PATH)
    total = len(df)
    signal = (df['TP'] > 0).sum() / total * 100
    strategy = ((df['Pullback'] < df['SL'])
                & (df['TP'] >= df['SL'])).sum() / total * 100
    assert round(signal, 1) == 68.5
    assert round(strategy, 1) == 49.3


def test_load_data_reads_a_blank_pullback_loser_as_stopped_out(tmp_path):
    """A loser exports Pullback and TP both blank. Its Pullback is the stop,
    so it loses on every stop scenario and never reads as an unfilled limit.
    A blank Pullback beside a real TP still loads as 0."""
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,30M,SL,Pullback,TP\n'
                   '2026-09-28,Monday,#1,Buy,High,5.0,,\n'
                   '2026-09-28,Monday,#2,Buy,High,5.0,,12\n')
    df = load_data(str(csv))
    assert list(df['Pullback']) == [5.0, 0.0]
    assert list(df['TP']) == [0.0, 12.0]


# --- Signal / Strategy grouping tables ------------------------------------

def test_chart_bars_follow_the_table_cells():
    """Monday holds rows 1-4: three reached a target (TP 16, 20, 25) and two
    survived their stop too (rows 1 and 4) - row 2 was stopped out first."""
    html = create_signal_strategy_chart(
        calculate_weekday_statistics(get_sample_data()))
    for day in WEEKDAY_ORDER:
        assert f'{day}<span class="wr-count">' in html
    assert html.count('class="wr-bar"') == 3 * len(WEEKDAY_ORDER)
    assert 'width: 75.0%' in html and 'width: 50.0%' in html


def test_chart_keeps_empty_rows_at_zero():
    """Friday has no trades in the sample: a zero-width bar, not a crash."""
    html = create_signal_strategy_chart(
        calculate_weekday_statistics(get_sample_data()))
    assert 'Friday<span class="wr-count">0 trades</span>' in html
    assert 'width: 0.0%' in html


def test_chart_has_a_legend_breakeven_and_tooltips():
    html = create_signal_strategy_chart(
        calculate_weekday_statistics(get_sample_data()))
    assert '>Signal</span>' in html and '>Strategy</span>' in html
    assert '>Strategy + 1 pip</span>' in html
    assert 'class="wr-breakeven" style="left: 50%;"' in html
    assert 'title="Monday - Strategy + 1 pip: 50.0%"' in html
    assert 'title="Monday - Strategy: 50.0%"' in html


def test_chart_draws_every_row_of_any_signal_strategy_table():
    for stats in (calculate_setup_statistics(get_sample_data()),
                  calculate_sl_fixed_statistics(get_sample_data())):
        html = create_signal_strategy_chart(stats)
        assert html.count('class="wr-row"') == len(stats)
        assert f'by {stats.columns[0]}"' in html


def test_chart_handles_an_empty_table():
    empty = calculate_weekday_statistics(get_sample_data()).iloc[0:0]
    assert 'No data' in create_signal_strategy_chart(empty)



def test_weekday_statistics_has_a_row_per_weekday():
    result = calculate_weekday_statistics(get_sample_data())
    assert list(result['Day']) == WEEKDAY_ORDER
    assert list(result.columns) == ['Day', 'Trades', 'Signal', 'Strategy',
                                    'Strategy + 1 pip']


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
    assert row['Signal'] == '100.0%'
    assert row['Strategy'] == '0.0%'


def test_weekday_statistics_carries_only_win_rates():
    result = calculate_weekday_statistics(get_sample_data())
    for col in result.columns[2:]:
        assert result[col].str.fullmatch(r'\d+\.\d%').all(), col


def test_weekday_buffer_column_matches_the_setup_tables_buffer_rule():
    """Monday at SL + 1: rows 1 and 4 survive and reach (TP 16 >= 5,
    25 >= 8), row 2 still fails (6 >= 6), row 3 never reached a target."""
    result = calculate_weekday_statistics(get_sample_data())
    monday = result[result['Day'] == 'Monday'].iloc[0]
    assert monday['Strategy + 1 pip'] == '50.0%'


def test_setup_statistics_lists_default_then_types_only():
    """Every trade, then Continuation and Reversal - no per-level setup rows,
    as the level alone says nothing, and no buffered rows, as the Strategy +
    1 pip column carries them."""
    result = calculate_setup_statistics(get_sample_data())
    assert list(result['Setup']) == ['Default'] + TYPE_ORDER
    assert result.iloc[0]['Trades'] == 10
    assert list(result.columns) == ['Setup', 'Trades'] + RATE_COLUMNS
    assert not set(result['Setup']) & set(SETUP_ORDER)


def test_setup_type_rows_sum_to_the_default_row():
    result = calculate_setup_statistics(get_sample_data()).set_index('Setup')
    assert (result.loc[TYPE_ORDER, 'Trades'].sum()
            == result.loc['Default', 'Trades'])


def test_setup_statistics_keeps_an_unlisted_type():
    """A type the data starts carrying tomorrow still gets a row, after the
    known ones."""
    sample = get_sample_data()
    sample.loc[0, 'Type'] = 'Sweep'
    labels = list(calculate_setup_statistics(sample)['Setup'])
    assert labels == ['Default'] + TYPE_ORDER + ['Sweep']


def test_setup_buffer_column_pads_the_stop_for_strategy_only():
    """Continuation is sample rows 1, 3, 5, 8, 9 - Strategy wins 1, 5, 9.
    At SL + 1 row 5 drops out (TP 3 < SL 3 + 1). Reversal is rows 2, 4, 6,
    7, 10 - wins 4, 7, 10 - and SL + 1 rescues none and drops none."""
    assert TYPE_BUFFER_PIPS == 1
    result = calculate_setup_statistics(get_sample_data()).set_index('Setup')
    assert result.loc['Continuation', 'Strategy'] == '60.0%'
    assert result.loc['Continuation', 'Strategy + 1 pip'] == '40.0%'
    assert (result.loc['Reversal', 'Strategy + 1 pip']
            == result.loc['Reversal', 'Strategy'])


def test_every_rate_table_carries_the_three_readings():
    sample = get_sample_data()
    for fn in (calculate_weekday_statistics, calculate_setup_statistics):
        result = fn(sample)
        assert list(result.columns)[2:] == RATE_COLUMNS, fn.__name__
        for col in RATE_COLUMNS:
            assert result[col].str.fullmatch(r'\d+\.\d%').all(), fn.__name__


def test_fixed_sl_has_no_buffer_column():
    """A fixed stop padded by a pip is the next fixed size - the row below -
    so the column would only repeat it. Its chart draws two bars a row."""
    result = calculate_sl_fixed_statistics(get_sample_data())
    assert 'Strategy + 1 pip' not in result.columns
    html = create_signal_strategy_chart(result)
    assert html.count('class="wr-bar"') == 2 * len(result)
    assert '>Strategy + 1 pip</span>' not in html


def test_grouping_tables_handle_an_empty_dataset():
    empty = get_empty_data()
    for fn in (calculate_weekday_statistics, calculate_setup_statistics):
        result = fn(empty)
        assert (result['Trades'] == 0).all()
        assert (result[RATE_COLUMNS] == '0.0%').all().all()


# --- stop tables -----------------------------------------------------------

def test_sl_fixed_statistics_reports_signal_and_strategy():
    result = calculate_sl_fixed_statistics(get_sample_data())
    assert list(result.columns) == ['Fixed SL', 'Trades', 'Signal', 'Strategy']
    assert result.iloc[0]['Fixed SL'] == 'Default'
    assert list(result['Fixed SL'])[1:] == [f'{p} pips' for p in SL_FIXED_PIPS]


def test_fixed_sl_signal_repeats_down_the_table():
    """Signal does not depend on the stop: it is the ceiling no size beats."""
    result = calculate_sl_fixed_statistics(get_sample_data())
    assert result['Signal'].nunique() == 1


def test_fixed_sl_default_row_keeps_the_recorded_stops():
    """Fixed SL's Default row scores the recorded stops - the same as the
    Default row of every other table."""
    sample = get_sample_data()
    fixed = calculate_sl_fixed_statistics(sample).iloc[0]
    setup = calculate_setup_statistics(sample).iloc[0]
    assert fixed['Fixed SL'] == 'Default'
    assert (fixed['Trades'], fixed['Strategy']) == (len(sample),
                                                    setup['Strategy'])


# --- TP distribution -----------------------------------------------------

def test_tp_statistics_buckets_profitable_trades_only():
    result = calculate_tp_statistics(get_sample_data())
    assert list(result['TP Range']) == [n for n, _, _ in TP_RANGES]
    profitable = int((get_sample_data()['TP'] > 0).sum())
    assert all(row.endswith(f'of {profitable}') for row in result['Trades'])


def test_tp_pip_counts_run_8_to_29_then_pool_30_plus():
    """Profitable TPs in the sample: 16, 20, 25, 3, 35, 9, 12. 3 is below the
    first column; 35 pools into 30+."""
    counts = calculate_tp_pip_counts(get_sample_data())
    assert list(counts.index) == [str(p) for p in range(8, 30)] + ['30+']
    assert counts['9'] == counts['12'] == counts['16'] == 1
    assert counts['20'] == counts['25'] == counts['30+'] == 1
    assert counts.sum() == 6


def test_tp_pip_counts_round_down_to_the_whole_pip():
    one = get_sample_data().iloc[0:1].assign(TP=[12.9])
    assert calculate_tp_pip_counts(one)['12'] == 1


def test_tp_chart_draws_a_column_per_pip_and_reports_what_it_leaves_out():
    html = create_tp_chart(get_sample_data())
    assert html.count('class="tp-bar"') == 23
    assert 'title="TP 30+ pips: 1 trade"' in html
    assert '7 profitable trades; 1 below 8 pips not shown' in html


def test_tp_chart_handles_no_profitable_trades():
    assert 'No data' in create_tp_chart(get_empty_data())
    losers = get_sample_data().assign(TP=0.0)
    assert 'No data' in create_tp_chart(losers)


def test_tp_bands_hold_each_trade_once():
    result = calculate_tp_statistics(get_sample_data())
    counted = sum(int(cell.split(' of ')[0]) for cell in result['Trades'])
    assert counted == int((get_sample_data()['TP'] > 0).sum())


# --- Strategies tables ----------------------------------------------------

def test_get_buffer_strategies_covers_all_trades_and_fixed_sl():
    """Capping moved to the Cap SL column, so there are no Max SL strategies."""
    names = [name for name, _ in get_buffer_strategies()]
    assert names[0] == 'All Trades'
    assert any(n.startswith('Fixed SL ') for n in names)
    assert not any(n.startswith('Max SL ') for n in names)


def test_buffer_statistics_scores_every_rrr():
    result = calculate_buffer_statistics(get_sample_data(), ['All Trades'])
    assert set(result['RRR']) == {f'1:{r}' for r in RRR_RATIOS}
    assert set(result['Buffer']) == {f'+{b}' for b in BUFFER_PIPS}
    assert BUFFER_PIPS == [0, 1]


def test_buffer_statistics_carry_trend_and_max_sl_after_buffer():
    """Each strategy, buffer, trend, cap and RRR appears once, and All covers
    the whole sample - a cap changes the stop, never the trades taken."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    assert list(result.columns) == ['Strategy', 'Buffer', 'Trend', 'Cap SL',
                                    'RRR', 'Trades', 'Notation', 'Win Rate']
    keys = ['Strategy', 'Buffer', 'Trend', 'Cap SL', 'RRR']
    assert not result.duplicated(keys).any()
    all_trades = result[result['Strategy'] == 'All Trades']
    assert set(all_trades['Cap SL']) == set(MAX_SL_CAPS)
    assert MAX_SL_CAPS == [0, 5, 6]
    assert set(result['Trend']) == set(TREND_FILTERS)
    assert (result[result['Trend'] == 'All']['Trades'] == len(sample)).all()


def test_buffer_statistics_carry_no_be_rule():
    """The 30MC export has no BE Rule column, so no row runs under it."""
    result = calculate_buffer_statistics(get_sample_data())
    assert 'BE Rule Applied' not in result.columns
    assert not result['Notation'].str.contains('BE').any()


def test_buffer_statistics_trend_keeps_only_its_setups():
    """Reversal is High Reversal + Low Reversal: rows 2, 4, 6, 7, 10. Of
    those, 4, 7 and 10 survive their stop and reach 1:1."""
    result = calculate_buffer_statistics(get_sample_data(), ['All Trades'])
    row = result[(result['Trend'] == 'Reversal') & (result['Buffer'] == '+0')
                 & (result['RRR'] == '1:1')].iloc[0]
    assert row['Trades'] == 5
    assert row['Notation'] == '3W – 2L'
    cont = result[(result['Trend'] == 'Continuation')
                  & (result['Buffer'] == '+0') & (result['RRR'] == '1:1')]
    assert cont.iloc[0]['Trades'] == 5
    assert cont.iloc[0]['Notation'] == '3W – 2L'


def test_fixed_sl_strategies_run_without_a_buffer_or_cap():
    """A buffer or a cap would only turn a fixed stop into another size."""
    names = [n for n, _ in get_buffer_strategies() if n != 'All Trades']
    result = calculate_buffer_statistics(get_sample_data(), names)
    assert set(result['Buffer']) == {'+0'}
    assert set(result['Cap SL']) == {0}


def test_max_sl_caps_the_buffered_stop():
    """SL 3 + 1 under a 5 pip cap is 4; SL 5 + 1 is capped back to 5."""
    trades = get_sample_data().iloc[0:2].copy()
    trades['SL'] = [3.0, 5.0]
    trades['Pullback'] = [3.5, 4.5]
    trades['TP'] = [4.5, 5.0]
    first, second = trades.iloc[0:1], trades.iloc[1:2]

    def notation(one, max_sl=0):
        return _calculate_stats_with_buffer(one, 'x', 1, 1,
                                            max_sl=max_sl)['Notation']

    # SL 3 + 1 = 4 is inside the cap: Pullback 3.5 survives, TP 4.5 reaches
    # 1:1. Had the cap been applied as 5, TP 4.5 would fall short.
    assert notation(first, max_sl=5) == '1W – 0L'
    # SL 5 + 1 = 6 survives 4.5 but needs TP 6; capped to 5 it wins on TP 5.
    assert notation(second) == '0W – 1L'
    assert notation(second, max_sl=5) == '1W – 0L'
    # A cap wider than the buffered stop changes nothing.
    assert notation(second, max_sl=9) == '0W – 1L'


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


def test_r_histogram_renders_every_level():
    html = create_r_histogram_combined(get_sample_data())
    for level in range(1, 11):
        assert f'>{level}R<' in html


def test_cumulative_histogram_never_falls_below_the_exact_one():
    sample = get_sample_data()
    raw = calculate_r_counts(sample)
    assert raw.loc[1:].sum() >= raw[1]


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
    stats = calculate_weekday_statistics(get_sample_data())
    assert 'class="sortable"' not in create_sortable_table(
        stats, 'weekday-table', sortable=False)
    assert 'class="sortable"' in create_sortable_table(stats, 'weekday-table')


def test_tables_render_no_data_when_empty():
    assert 'No data' in create_html_table(get_empty_data())
    assert 'No data' in create_sortable_table(get_empty_data(), 'x')


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

def test_streak_statistics_lists_default_then_continuation():
    """No Reversal row and no + 1 pip buffer rows."""
    result = calculate_streak_statistics(get_sample_data())
    assert list(result['Trend']) == ['Default', 'Continuation']
    assert list(result.columns) == [
        'Trend', 'Trades', 'Longest Wins', 'Longest Losses']
    assert result.iloc[0]['Trades'] == 10
    assert result.iloc[1]['Trades'] == 5


def test_streak_statistics_uses_the_strategy_win_rule():
    """Sample row 2 (Pullback 6.0 >= SL 5.0, TP 20) is a Signal win but a
    Strategy loss. With row 3 it forms a two-trade losing run; scored on
    Signal the longest losing run would be a single trade."""
    result = calculate_streak_statistics(get_sample_data()).set_index('Trend')
    assert result.loc['Default', 'Longest Losses'] == '2 (2025-02-03 - 2025-02-03)'
    assert result.loc['Default', 'Longest Wins'] == '2 (2025-02-03 - 2025-02-04)'


def test_streak_statistics_counts_longest_runs_with_their_dates():
    df = pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-03', '2025-02-04', '2025-02-05',
                 '2025-02-05', '2025-02-06', '2025-02-07'],
        'Type': ['Continuation'] * 7,
        'SL': [5.0] * 7,
        'Pullback': [1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0],
        'TP': [10.0, 10.0, 10.0, 0.0, 0.0, 10.0, 0.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Default', 'Longest Wins'] == '3 (2025-02-03 - 2025-02-04)'
    assert result.loc['Default', 'Longest Losses'] == '2 (2025-02-05 - 2025-02-05)'
    assert result.loc['Continuation', 'Longest Wins'] == '3 (2025-02-03 - 2025-02-04)'


def test_streak_statistics_reads_trades_in_date_order():
    """Two wins separated in the file but adjacent by date form one run."""
    df = pd.DataFrame({
        'Date': ['2025-02-04', '2025-02-03', '2025-02-05'],
        'Type': ['Reversal'] * 3,
        'SL': [5.0] * 3,
        'Pullback': [1.0, 5.0, 1.0],
        'TP': [10.0, 0.0, 10.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Default', 'Longest Wins'] == '2 (2025-02-04 - 2025-02-05)'


def test_streak_statistics_type_rows_skip_the_other_type():
    """A Reversal loss between two Continuation wins does not break the
    Continuation run - that type alone would have won twice running."""
    df = pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-04', '2025-02-05'],
        'Type': ['Continuation', 'Reversal', 'Continuation'],
        'SL': [5.0] * 3,
        'Pullback': [1.0, 5.0, 1.0],
        'TP': [10.0, 0.0, 10.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Default', 'Longest Wins'].startswith('1 (')
    assert result.loc['Continuation', 'Longest Wins'] == '2 (2025-02-03 - 2025-02-05)'
    assert result.loc['Continuation', 'Longest Losses'] == '0'


def test_streak_statistics_empty_dataset():
    result = calculate_streak_statistics(get_empty_data())
    assert list(result['Trend']) == ['Default', 'Continuation']
    assert list(result['Trades']) == [0] * 2
    assert set(result['Longest Wins']) == {'0'}
    assert set(result['Longest Losses']) == {'0'}


# ---------------------------------------------------------------------------
# Strategy equity curves
# ---------------------------------------------------------------------------

def test_equity_curves_cover_every_strategies_row():
    df = get_sample_data()
    curves = calculate_equity_curves(df, 1)
    table = calculate_buffer_statistics(df)
    assert len(curves) == len(table[table['RRR'] == '1:1'])
    assert all(len(c['curve']) == len(df) + 1 for c in curves)
    assert all(c['curve'][0] == 0.0 for c in curves)
    outcomes = [c['outcome'] for c in curves]
    assert outcomes == sorted(outcomes, reverse=True)


def test_equity_curve_outcome_matches_the_table_notation():
    """Final R is wins x RRR - losses, from the same rule the table scores."""
    df = get_sample_data()
    for rrr in RRR_RATIOS:
        curves = {c['label']: c for c in calculate_equity_curves(df, rrr)}
        table = calculate_buffer_statistics(df)
        row = table[(table['Strategy'] == 'All Trades') & (table['Buffer'] == '+1')
                    & (table['Trend'] == 'Reversal') & (table['Cap SL'] == 5)
                    & (table['RRR'] == f'1:{rrr}')].iloc[0]
        wins, losses = (int(v[:-1]) for v in row['Notation'].split(' \u2013 '))
        curve = curves['All Trades +1 / Reversal / cap 5']
        assert curve['outcome'] == wins * rrr - losses
        assert curve['trades'] == wins + losses == row['Trades']


def test_equity_curves_carry_no_be_rule_rows():
    df = load_data(CSV_PATH)
    curves = {c['label']: c for c in calculate_equity_curves(df, 2)}
    table = calculate_buffer_statistics(df)
    assert len(curves) == len(table[table['RRR'] == '1:2'])
    assert not any(label.endswith('/ BE') for label in curves)


def test_equity_curve_stays_flat_outside_its_trend():
    df = get_sample_data()
    curves = {c['label']: c for c in calculate_equity_curves(df, 1)}
    ordered = df.sort_values('Date', kind='stable')
    steps = pd.Series(curves['Fixed SL 5 / Reversal']['curve']).diff().dropna()
    assert (steps[(ordered['Type'] != 'Reversal').values] == 0).all()
    assert (steps[(ordered['Type'] == 'Reversal').values] != 0).all()


def test_equity_curves_walk_trades_in_date_order():
    df = pd.DataFrame({
        'Date': ['2025-02-04', '2025-02-03'],
        'Type': ['Continuation', 'Continuation'],
        'SL': [5.0, 5.0], 'Pullback': [1.0, 5.0], 'TP': [10.0, 0.0],
    })
    curves = {c['label']: c for c in calculate_equity_curves(df, 1)}
    assert curves['All Trades +0']['curve'] == [0.0, -1.0, 0.0]


def test_equity_curves_empty_dataset():
    curves = calculate_equity_curves(get_empty_data(), 1)
    assert curves and all(c['curve'] == [0.0] for c in curves)
    assert 'No data' in create_equity_chart(curves, 1)
    assert 'No data' in create_equity_chart([], 1)


def test_equity_chart_draws_every_curve_and_colours_the_best():
    curves = calculate_equity_curves(get_sample_data(), 2)
    html = create_equity_chart(curves, 2, chart_id='eq-test')
    assert 'id="eq-test"' in html
    assert html.count('class="eq-series') == len(curves)
    assert html.count('class="eq-series eq-top"') == EQUITY_HIGHLIGHT
    assert html.count('class="eq-series eq-other"') == len(curves) - EQUITY_HIGHLIGHT
    for curve in curves[:EQUITY_HIGHLIGHT]:
        assert curve['label'] in html
    assert 'data-rrr="2"' in html
    assert 'class="eq-zero"' in html
    assert '+2R per win' in html


# --- Half SL entries ------------------------------------------------------

def _half_sl(rrr):
    result = calculate_half_sl_statistics(get_sample_data())
    return result[result['RRR'] == f'1:{rrr}'].set_index('Entry')


def test_half_sl_lists_every_entry_at_every_rrr():
    result = calculate_half_sl_statistics(get_sample_data())
    assert list(result['Entry']) == HALF_SL_ENTRIES * len(RRR_RATIOS)
    assert list(result['RRR'].unique()) == [f'1:{r}' for r in RRR_RATIOS]


def test_half_sl_default_row_matches_the_other_stop_tables():
    default = _half_sl(1).loc['Default']
    sl_default = calculate_setup_statistics(get_sample_data()).iloc[0]
    wins, missed, losses = default['Notation'].split(' - ')
    assert missed == '0M'
    assert int(wins[:-1]) + int(losses[:-1]) == sl_default['Trades']
    assert default['Win Rate'] == sl_default['Strategy']


def test_half_sl_counts_match_the_sample_at_1_to_1():
    """Half SL keeps rows 1, 5, 7, 10 - rows 4 and 9 pull back exactly half
    their stop (3.5 of 7, 4 of 8) and are stopped out. The limit fills on
    those two and wins both; rows 1, 5, 7, 10 never pull back half the stop,
    so the limit misses them."""
    result = _half_sl(1)
    assert result.loc['Default', 'Notation'] == '6W - 0M - 4L'
    assert result.loc['Half SL', 'Notation'] == '4W - 0M - 6L'
    assert result.loc['Limit order on half', 'Notation'] == '2W - 4M - 4L'
    assert result.loc['Limit order on half', 'Trades'] == 6
    assert result.loc['Limit order on half', 'Win Rate'] == '33.3%'
    assert result.loc['Default', 'Outcome'] == '+2R'
    assert result.loc['Half SL', 'Outcome'] == '-2R'


def test_half_sl_limit_measures_the_run_from_the_fill_at_1_to_3():
    """Row 9 (SL 8, TP 9) cannot reach 3R from the signal, but from the
    limit its run is 9 + 4 = 13 against a 4 pip stop - over the 12 needed."""
    result = _half_sl(3)
    assert result.loc['Default', 'Notation'] == '3W - 0M - 7L'
    assert result.loc['Limit order on half', 'Notation'] == '2W - 4M - 4L'
    assert result.loc['Limit order on half', 'Outcome'] == '+2R'


def test_half_sl_limit_matches_the_worked_example():
    """SL 3.6, Pullback 2.0, TP 10: Half SL is stopped by the 2.0 pullback.
    The limit fills at 1.8 with a 1.8 stop, pulls back 0.2 more and runs
    11.8 - a win even at 1:3."""
    trade = get_sample_data().iloc[[0]].assign(SL=3.6, Pullback=2.0, TP=10.0)
    result = calculate_half_sl_statistics(trade)
    one = result[result['RRR'] == '1:1'].set_index('Entry')['Notation']
    three = result[result['RRR'] == '1:3'].set_index('Entry')['Notation']
    assert one['Half SL'] == '0W - 0M - 1L'
    assert one['Limit order on half'] == '1W - 0M - 0L'
    assert three['Limit order on half'] == '1W - 0M - 0L'


def test_half_sl_limit_needs_a_target_after_the_fill():
    """A trade that never reached a target (TP 0) is not a win from the
    limit either, even when its pullback stays inside the stop."""
    trade = get_sample_data().iloc[[0]].assign(SL=4.0, Pullback=3.0, TP=0.0)
    result = calculate_half_sl_statistics(trade)
    limit = result[result['Entry'] == 'Limit order on half']
    assert set(limit['Notation']) == {'0W - 0M - 1L'}


def test_half_sl_handles_empty_data():
    result = calculate_half_sl_statistics(get_empty_data())
    assert set(result['Notation']) == {'0W - 0M - 0L'}
    assert set(result['Win Rate']) == {'0.0%'}
