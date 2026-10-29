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
    MAPPED_SL_PIPS,
    EMA_ORDER,
    RATE_COLUMNS,
    RRR_TARGET_TICKS,
    RRR_RATIOS,
    SL_FIXED_PIPS,
    TP_RANGES,
    WEEKDAY_ORDER,
    _calculate_stats_with_buffer,
    _drop_scratch_columns,
    _normalise_header,
    EQUITY_HIGHLIGHT,
    calculate_buffer_statistics,
    calculate_equity_curves,
    calculate_r_counts,
    calculate_ema_statistics,
    calculate_sl_fixed_statistics,
    calculate_streak_statistics,
    calculate_tp_range_counts,
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
    is_mapped_sl,
    reached_r,
    target_pips,
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'trades.csv')


def get_sample_data():
    """
    Ten trades over three days, covering every case the tables must separate.

    Every stop has an RRR target mapping, and so does every stop padded by
    a pip. Row 2 is the stopped-out winner-that-wasn't: Pullback 6.0 >= SL
    5.0 with a TP of 20, so Signal counts it and Strategy does not. Both EMA
    alignments are present, five trades each.
    """
    ema = ['Aligned', 'Against', 'Aligned', 'Against', 'Aligned', 'Against',
           'Against', 'Aligned', 'Aligned', 'Against']
    direction = ['Buy', 'Sell', 'Buy', 'Sell',
                 'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell']
    opposite = {'Buy': 'Sell', 'Sell': 'Buy'}
    return pd.DataFrame({
        'Date': ['2025-02-03'] * 4 + ['2025-02-04'] * 3 + ['2025-02-05'] * 3,
        'Weekday': ['Monday'] * 4 + ['Tuesday'] * 3 + ['Wednesday'] * 3,
        'Trade': ['#1', '#2', '#3', '#4', '#1', '#2', '#3', '#1', '#2', '#3'],
        'Direction': direction,
        'EMA(50)': [d if e == 'Aligned' else opposite[d]
                    for d, e in zip(direction, ema)],
        'SL': [4.0, 5.0, 3.0, 6.0, 3.0, 6.0, 6.0, 3.0, 5.0, 5.0],
        'Pullback': [1.0, 6.0, 3.0, 3.5, 0.5, 6.0, 2.0, 3.0, 4.0, 1.5],
        'TP': [16.0, 20.0, 0.0, 25.0, 4.0, 0.0, 35.0, 0.0, 9.0, 12.0],
        'R': [3.0, -3.0, 0.0, 3.0, 1.0, 0.0, 5.0, 0.0, 1.0, 2.0],
        'EMA': ema,
    })


def get_empty_data():
    return get_sample_data().iloc[0:0].copy()


# --- load_data -------------------------------------------------------------

def test_load_data_reads_the_project_csv():
    df = load_data(CSV_PATH)
    assert len(df) > 0
    for col in ('Date', 'Weekday', 'Trade', 'Direction', 'EMA(50)',
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
                                    'EMA(50)', 'SL', 'Pullback', 'TP']


def test_load_data_fills_blank_numbers_with_zero():
    """A blank TP means the trade was not profitable, so it must read as 0 -
    the Signal rule is TP > 0 and a NaN would silently drop out of it."""
    df = load_data(CSV_PATH)
    for col in ('SL', 'Pullback', 'TP'):
        assert df[col].notna().all(), f'{col} still has NaN'
    assert (df['TP'] == 0).any(), 'expected losses to land on TP 0'


def test_load_data_derives_r_from_the_rrr_targets(tmp_path):
    """This export has no R column, so R is the whole RRR levels TP reached
    on the commission-adjusted targets. On a 5-pip stop 18 pips passes 3R
    (16.6) but not 4R (22.0); 5.7 is short of 1R (5.8). A stopped-out trade
    keeps its reach, signed negative."""
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,EMA(50),SL,Pullback,TP\n'
                   '2026-09-28,Monday,#1,Buy,Buy,5.0,1.0,18\n'
                   '2026-09-28,Monday,#2,Buy,Buy,5.0,1.0,5.7\n'
                   '2026-09-28,Monday,#3,Buy,Buy,5.0,6.0,18\n')
    assert list(load_data(str(csv))['R']) == [3.0, 0.0, -3.0]


def test_target_pips_reads_the_tick_mapping():
    """Ticks / 10: 1R on a 5-pip stop is 5.8 pips, 1:3 is 16.6, 1:10 54.5."""
    assert target_pips(5, 1) == pytest.approx(5.8)
    assert target_pips(5, 3) == pytest.approx(16.6)
    assert target_pips(5, 10) == pytest.approx(54.5)
    assert target_pips(3.0, 2) == pytest.approx(7.3)
    assert target_pips(7, 1) == pytest.approx(7.8)
    assert MAPPED_SL_PIPS == [3, 4, 5, 6, 7]
    assert all(len(ticks) == 10 for ticks in RRR_TARGET_TICKS.values())


def test_target_pips_refuses_an_unmapped_stop():
    """No silent fallback to RRR x SL."""
    with pytest.raises(ValueError):
        target_pips(8, 1)
    with pytest.raises(ValueError):
        target_pips(2.5, 1)
    with pytest.raises(ValueError):
        target_pips(5, 11)
    assert not is_mapped_sl(2.5) and is_mapped_sl(6.0)


def test_reached_r_counts_whole_levels():
    assert reached_r(5, 5.8) == 1
    assert reached_r(5, 5.7) == 0
    assert reached_r(5, 54.5) == 10
    assert reached_r(5, 80) == 10


def test_load_data_signs_r_negative_when_the_stop_was_hit():
    """A negative R records reach the recorded stop could not hold on to -
    the same convention the 15LS1CC sheet exports."""
    df = load_data(CSV_PATH)
    stopped = df[(df['Pullback'] >= df['SL']) & (df['TP'] > 0)]
    assert len(stopped) > 0, 'expected at least one stopped-out reach'
    assert (stopped['R'] < 0).all()


def test_load_data_derives_the_ema_alignment_from_the_project_csv():
    """EMA(50) holds the trend; EMA reads it against Direction."""
    df = load_data(CSV_PATH)
    assert set(df['EMA(50)']) <= {'Buy', 'Sell'}
    assert set(df['EMA']) <= set(EMA_ORDER)
    aligned = df['EMA(50)'] == df['Direction']
    assert (df.loc[aligned, 'EMA'] == 'Aligned').all()
    assert (df.loc[~aligned, 'EMA'] == 'Against').all()


@pytest.mark.parametrize('trend, direction, alignment', [
    ('Buy', 'Buy', 'Aligned'),
    ('Sell', 'Sell', 'Aligned'),
    ('Sell', 'Buy', 'Against'),
    ('Buy', 'Sell', 'Against'),
])
def test_load_data_maps_each_trend_and_direction(tmp_path, trend, direction,
                                                 alignment):
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,EMA(50),SL,Pullback,TP\n'
                   f'2025-02-03,Monday,#1,{direction},{trend},4,1,8\n')
    assert load_data(str(csv)).iloc[0]['EMA'] == alignment


def test_load_data_drops_rows_without_a_date(tmp_path):
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,EMA(50),SL,Pullback,TP\n'
                   '2025-02-03,Monday,#1,Buy,Buy,4,1,12\n'
                   ',,,,,,,\n'
                   ',,,,,,,\n')
    df = load_data(str(csv))
    assert len(df) == 1


def test_load_data_matches_the_sheets_own_headline_rates():
    """The export labels its Signal and WR columns with the figures the sheet
    computed. They are dropped at load, but they are still the trader's own
    cross-check. The sheet scores 1:1 on TP >= SL, this page on the 5.8 pip
    commission-adjusted target - no trade lands between the two, so the
    figures still agree."""
    default = calculate_ema_statistics(load_data(CSV_PATH)).iloc[0]
    assert default['Signal'] == '65.6%'
    assert default['Strategy'] == '47.8%'


def test_load_data_reads_a_blank_pullback_loser_as_stopped_out(tmp_path):
    """A loser exports Pullback and TP both blank. Its Pullback is the stop,
    so it loses on every stop scenario, never as a clean run off the entry.
    A blank Pullback beside a real TP still loads as 0."""
    csv = tmp_path / 'trades.csv'
    csv.write_text('Date,Weekday,Trade,Direction,EMA(50),SL,Pullback,TP\n'
                   '2026-09-28,Monday,#1,Buy,Buy,5.0,,\n'
                   '2026-09-28,Monday,#2,Buy,Buy,5.0,,12\n')
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
    assert html.count('class="wr-bar"') == 2 * len(WEEKDAY_ORDER)
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
    assert 'Strategy + 1 pip' not in html
    assert 'class="wr-breakeven" style="left: 50%;"' in html
    assert 'title="Monday - Strategy: 50.0%"' in html


def test_chart_draws_every_row_of_any_signal_strategy_table():
    for stats in (calculate_ema_statistics(get_sample_data()),
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
    assert row['Signal'] == '100.0%'
    assert row['Strategy'] == '0.0%'


def test_weekday_statistics_carries_only_win_rates():
    result = calculate_weekday_statistics(get_sample_data())
    for col in result.columns[2:]:
        assert result[col].str.fullmatch(r'\d+\.\d%').all(), col


def test_ema_statistics_lists_default_then_both_alignments():
    result = calculate_ema_statistics(get_sample_data())
    assert list(result['EMA']) == ['Default'] + EMA_ORDER
    assert result.iloc[0]['Trades'] == 10
    assert list(result.columns) == ['EMA', 'Trades'] + RATE_COLUMNS


def test_ema_rows_sum_to_the_default_row():
    result = calculate_ema_statistics(get_sample_data()).set_index('EMA')
    assert (result.loc[EMA_ORDER, 'Trades'].sum()
            == result.loc['Default', 'Trades'])


def test_ema_statistics_keeps_an_empty_alignment():
    """Against stays listed when no trade was taken against the trend."""
    sample = get_sample_data()
    sample['EMA'] = 'Aligned'
    result = calculate_ema_statistics(sample).set_index('EMA')
    assert result.loc['Against', 'Trades'] == 0


def test_ema_rows_score_their_own_trades():
    """Aligned is sample rows 1, 3, 5, 8, 9 - Strategy wins 1, 5 and 9
    (row 5's TP 4 clears the 3.8 pip 1R of its 3-pip stop). Against is rows
    2, 4, 6, 7, 10 - wins 4, 7, 10."""
    result = calculate_ema_statistics(get_sample_data()).set_index('EMA')
    assert result.loc['Aligned', 'Strategy'] == '60.0%'
    assert result.loc['Against', 'Strategy'] == '60.0%'


def test_rate_tables_carry_no_one_pip_column():
    """A 1 pip buffer does not apply to this strategy."""
    assert RATE_COLUMNS == ['Signal', 'Strategy']
    for fn in (calculate_weekday_statistics, calculate_ema_statistics,
               calculate_sl_fixed_statistics):
        assert 'Strategy + 1 pip' not in fn(get_sample_data()).columns


def test_every_rate_table_carries_both_readings():
    sample = get_sample_data()
    for fn in (calculate_weekday_statistics, calculate_ema_statistics):
        result = fn(sample)
        assert list(result.columns)[2:] == RATE_COLUMNS, fn.__name__
        for col in RATE_COLUMNS:
            assert result[col].str.fullmatch(r'\d+\.\d%').all(), fn.__name__


def test_fixed_sl_chart_draws_two_bars_a_row():
    result = calculate_sl_fixed_statistics(get_sample_data())
    html = create_signal_strategy_chart(result)
    assert html.count('class="wr-bar"') == 2 * len(result)


def test_grouping_tables_handle_an_empty_dataset():
    empty = get_empty_data()
    for fn in (calculate_weekday_statistics, calculate_ema_statistics):
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
    ema = calculate_ema_statistics(sample).iloc[0]
    assert fixed['Fixed SL'] == 'Default'
    assert (fixed['Trades'], fixed['Strategy']) == (len(sample),
                                                    ema['Strategy'])


# --- TP distribution -----------------------------------------------------

def test_tp_statistics_buckets_profitable_trades_only():
    result = calculate_tp_statistics(get_sample_data())
    assert list(result['TP Range']) == [n for n, _, _ in TP_RANGES]
    profitable = int((get_sample_data()['TP'] > 0).sum())
    assert all(row.endswith(f'of {profitable}') for row in result['Trades'])


def test_tp_ranges_are_the_six_bands():
    assert [n for n, _, _ in TP_RANGES] == [
        '0-10 pips', '10-15 pips', '15-20 pips', '20-25 pips', '25-30 pips',
        '30+ pips']


def test_tp_range_counts_put_each_profitable_trade_in_one_band():
    """Profitable TPs in the sample: 16, 20, 25, 3, 35, 9, 12. A band holds
    its low end, so 20 sits in 20-25, not 15-20."""
    counts = calculate_tp_range_counts(get_sample_data())
    assert list(counts.index) == [n for n, _, _ in TP_RANGES]
    assert list(counts) == [2, 1, 1, 1, 1, 1]


def test_tp_chart_draws_a_column_per_band():
    html = create_tp_chart(get_sample_data())
    assert html.count('class="tp-bar"') == len(TP_RANGES)
    assert 'title="TP 0-10 pips: 2 trades"' in html
    assert 'title="TP 30+ pips: 1 trade"' in html
    assert '7 profitable trades' in html
    assert 'not shown' not in html


def test_tp_chart_handles_no_profitable_trades():
    assert 'No data' in create_tp_chart(get_empty_data())
    losers = get_sample_data().assign(TP=0.0)
    assert 'No data' in create_tp_chart(losers)


def test_tp_bands_hold_each_trade_once():
    result = calculate_tp_statistics(get_sample_data())
    counted = sum(int(cell.split(' of ')[0]) for cell in result['Trades'])
    assert counted == int((get_sample_data()['TP'] > 0).sum())


# --- Strategies tables ----------------------------------------------------

def test_get_buffer_strategies_covers_all_trades_ema_and_fixed_sl():
    """All Trades, EMA Aligned and one Fixed SL N row per mapped stop."""
    names = [name for name, _ in get_buffer_strategies()]
    assert names == (['All Trades', 'EMA Aligned']
                     + [f'Fixed SL {n}' for n in MAPPED_SL_PIPS])


def test_ema_aligned_strategy_takes_only_aligned_trades():
    """Aligned sample rows 1, 3, 5, 8, 9: wins 1, 5 and 9 at 1:1."""
    result = calculate_buffer_statistics(get_sample_data(), ['EMA Aligned'])
    row = result[result['RRR'] == '1:1'].iloc[0]
    assert row['Trades'] == 5
    assert row['Notation'] == '3W \u2013 2L'


def test_buffer_statistics_scores_every_rrr():
    result = calculate_buffer_statistics(get_sample_data(), ['All Trades'])
    assert set(result['RRR']) == {f'1:{r}' for r in RRR_RATIOS}
    assert 'Buffer' not in result.columns


def test_buffer_statistics_carry_no_trend_or_cap():
    """Each strategy and RRR appears once, every row but EMA Aligned over
    every trade - there is no Trend and no Cap SL column, and no Reversal
    rows."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    assert list(result.columns) == ['Strategy', 'RRR', 'Trades', 'Notation',
                                    'Win Rate']
    assert not result.duplicated(['Strategy', 'RRR']).any()
    assert len(result) == (2 + len(MAPPED_SL_PIPS)) * len(RRR_RATIOS)
    every_trade = result[result['Strategy'] != 'EMA Aligned']
    assert (every_trade['Trades'] == len(sample)).all()


def test_buffer_statistics_carry_no_be_rule():
    """The 30MC export has no BE Rule column, so no row runs under it."""
    result = calculate_buffer_statistics(get_sample_data())
    assert 'BE Rule Applied' not in result.columns
    assert not result['Notation'].str.contains('BE').any()


def test_buffer_statistics_score_every_trade_on_its_stop():
    """All Trades at 1:1: rows 1, 4, 5, 7, 9, 10 survive their stop and reach
    their 1R target."""
    result = calculate_buffer_statistics(get_sample_data(), ['All Trades'])
    row = result[result['RRR'] == '1:1'].iloc[0]
    assert row['Trades'] == 10
    assert row['Notation'] == '6W – 4L'


def test_calculate_stats_with_buffer_applies_the_rrr_target():
    """Row 1: SL 4, TP 16 - past the 13.7 pip 1:3 target, short of the 18.2
    of 1:4. A TP landing exactly on the target wins, so the rule is >=."""
    one = get_sample_data().iloc[0:1]
    assert _calculate_stats_with_buffer(one, 'x', 3)['Notation'] == '1W – 0L'
    assert _calculate_stats_with_buffer(one, 'x', 4)['Notation'] == '0W – 1L'
    exact = one.assign(TP=13.7)
    assert _calculate_stats_with_buffer(exact, 'x', 3)['Notation'] == '1W – 0L'


def test_calculate_stats_with_buffer_handles_no_trades():
    stats = _calculate_stats_with_buffer(get_empty_data(), 'x', 1)
    assert stats['Trades'] == 0
    assert stats['Win Rate'] == '0.0%'


# --- R distribution -------------------------------------------------------

def test_r_counts_truncate_towards_zero_and_drop_the_sign():
    counts = calculate_r_counts(get_sample_data().assign(
        R=[3.0, -3.0, 0.0, 3.6, 1.0, 0.0, 5.0, 0.0, 1.0, 2.0]))
    # -3 counts as 3R (the reach was there either way), 3.6 truncates to 3R.
    assert counts[3] == 3
    assert counts[5] == 1
    assert counts[1] == 2


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
    stats = calculate_ema_statistics(get_sample_data())
    html = create_html_table(stats, sort_id='ema-table')
    assert 'id="ema-table"' in html
    assert "sortAnalysisTable('ema-table'" in html
    assert 'class="sortable"' in html


def test_create_html_table_without_a_sort_id_is_not_sortable():
    html = create_html_table(calculate_ema_statistics(get_sample_data()))
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

def test_streak_statistics_lists_default_then_each_alignment():
    """No + 1 pip buffer rows."""
    result = calculate_streak_statistics(get_sample_data())
    assert list(result['Trend']) == ['Default', 'EMA Aligned', 'EMA Against']
    assert list(result.columns) == [
        'Trend', 'Trades', 'Longest Wins', 'Longest Losses']
    assert result.iloc[0]['Trades'] == 10
    assert result.iloc[1]['Trades'] == 5
    assert result.iloc[2]['Trades'] == 5


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
        'EMA': ['Aligned'] * 7,
        'SL': [5.0] * 7,
        'Pullback': [1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 5.0],
        'TP': [10.0, 10.0, 10.0, 0.0, 0.0, 10.0, 0.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Default', 'Longest Wins'] == '3 (2025-02-03 - 2025-02-04)'
    assert result.loc['Default', 'Longest Losses'] == '2 (2025-02-05 - 2025-02-05)'
    assert result.loc['EMA Aligned', 'Longest Wins'] == '3 (2025-02-03 - 2025-02-04)'


def test_streak_statistics_reads_trades_in_date_order():
    """Two wins separated in the file but adjacent by date form one run."""
    df = pd.DataFrame({
        'Date': ['2025-02-04', '2025-02-03', '2025-02-05'],
        'EMA': ['Against'] * 3,
        'SL': [5.0] * 3,
        'Pullback': [1.0, 5.0, 1.0],
        'TP': [10.0, 0.0, 10.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Default', 'Longest Wins'] == '2 (2025-02-04 - 2025-02-05)'


def test_streak_statistics_ema_rows_skip_the_other_alignment():
    """A loss against the trend between two aligned wins does not break the
    aligned run - that filter alone would have won twice running."""
    df = pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-04', '2025-02-05'],
        'EMA': ['Aligned', 'Against', 'Aligned'],
        'SL': [5.0] * 3,
        'Pullback': [1.0, 5.0, 1.0],
        'TP': [10.0, 0.0, 10.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Default', 'Longest Wins'].startswith('1 (')
    assert result.loc['EMA Aligned', 'Longest Wins'] == '2 (2025-02-03 - 2025-02-05)'
    assert result.loc['EMA Aligned', 'Longest Losses'] == '0'
    assert result.loc['EMA Against', 'Longest Losses'] == '1 (2025-02-04 - 2025-02-04)'


def test_streak_statistics_empty_dataset():
    result = calculate_streak_statistics(get_empty_data())
    assert list(result['Trend']) == ['Default', 'EMA Aligned', 'EMA Against']
    assert list(result['Trades']) == [0] * 3
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
        row = table[(table['Strategy'] == 'Fixed SL 5')
                    & (table['RRR'] == f'1:{rrr}')].iloc[0]
        wins, losses = (int(v[:-1]) for v in row['Notation'].split(' \u2013 '))
        curve = curves['Fixed SL 5']
        assert curve['outcome'] == wins * rrr - losses
        assert curve['trades'] == wins + losses == row['Trades']


def test_ema_aligned_equity_curve_stays_flat_on_trades_against():
    """Sample row 2 is Against: the EMA Aligned curve does not move on it.
    The curve still spans every trade so it lines up with the others."""
    df = get_sample_data()
    curve = {c['label']: c
             for c in calculate_equity_curves(df, 1)}['EMA Aligned']
    assert len(curve['curve']) == len(df) + 1
    assert curve['trades'] == 5
    assert curve['curve'][1:3] == [1.0, 1.0]
    assert curve['outcome'] == 3 - 2


def test_equity_curves_carry_no_be_rule_rows():
    df = load_data(CSV_PATH)
    curves = {c['label']: c for c in calculate_equity_curves(df, 2)}
    table = calculate_buffer_statistics(df)
    assert len(curves) == len(table[table['RRR'] == '1:2'])
    assert not any(label.endswith('/ BE') for label in curves)


def test_equity_curves_walk_trades_in_date_order():
    df = pd.DataFrame({
        'Date': ['2025-02-04', '2025-02-03'],
        'EMA': ['Aligned', 'Aligned'],
        'SL': [5.0, 5.0], 'Pullback': [1.0, 5.0], 'TP': [10.0, 0.0],
    })
    curves = {c['label']: c for c in calculate_equity_curves(df, 1)}
    assert curves['All Trades']['curve'] == [0.0, -1.0, 0.0]


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
