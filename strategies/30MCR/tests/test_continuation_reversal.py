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
    BE_RULE_COLUMN,
    BE_RULE_PIPS,
    break_even_outcomes,
    BUFFER_PIPS,
    MAX_SL_CAPS,
    PULLBACK_BUFFER_PIPS,
    TREND_FILTERS,
    PULLBACK_ENTRY_PIPS,
    RRR_RATIOS,
    SETUP_ORDER,
    SL_BUFFER_PIPS,
    SL_FIXED_PIPS,
    SL_MAX_PIPS,
    SL_RANGES,
    TP_RANGES,
    TYPE_BUFFER_PIPS,
    TYPE_ORDER,
    WEEKDAY_ORDER,
    _calculate_stats_with_buffer,
    _drop_scratch_columns,
    _format_wl,
    _normalise_header,
    EQUITY_HIGHLIGHT,
    HALF_SL_ENTRIES,
    calculate_buffer_statistics,
    calculate_half_sl_statistics,
    calculate_equity_curves,
    calculate_pullback_statistics,
    calculate_r_counts,
    calculate_setup_statistics,
    calculate_sl_buffer_statistics,
    calculate_sl_fixed_statistics,
    calculate_sl_max_statistics,
    calculate_sl_statistics,
    calculate_streak_statistics,
    calculate_tp_pip_counts,
    calculate_tp_statistics,
    calculate_weekday_statistics,
    create_equity_chart,
    create_html_table,
    create_pullback_chart,
    create_r_histogram_combined,
    create_r_histogram_exact,
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
    assert list(df.columns)[:9] == ['Date', 'Weekday', 'Trade', 'Direction',
                                    '30M', 'BE Rule (10 pips)', 'SL',
                                    'Pullback', 'TP']


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
    csv.write_text('Date,Weekday,Trade,Direction,30M,BE Rule (10 pips),'
                   'SL,Pullback,TP\n'
                   '2025-02-03,Monday,#1,Buy,High,TRUE,4,1,12\n'
                   ',,,,,FALSE,,,\n'
                   ',,,,,FALSE,,,\n')
    df = load_data(str(csv))
    assert len(df) == 1
    assert bool(df.iloc[0]['BE Rule (10 pips)']) is True


def test_load_data_matches_the_sheets_own_headline_rates():
    """The export labels its Signal and WR columns with the figures the sheet
    computed. They are dropped at load, but they are still the trader's own
    cross-check that this module scores the same trades the same way."""
    df = load_data(CSV_PATH)
    total = len(df)
    signal = (df['TP'] > 0).sum() / total * 100
    strategy = ((df['Pullback'] < df['SL'])
                & (df['TP'] >= df['SL'])).sum() / total * 100
    assert round(signal, 1) == 50.6
    assert round(strategy, 1) == 33.5


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
    assert 'class="wr-breakeven" style="left: 50%;"' in html
    assert 'title="Monday - Strategy: 50.0%"' in html


def test_chart_draws_every_row_of_any_signal_strategy_table():
    for stats in (calculate_setup_statistics(get_sample_data()),
                  calculate_sl_statistics(get_sample_data()),
                  calculate_sl_fixed_statistics(get_sample_data())):
        html = create_signal_strategy_chart(stats)
        assert html.count('class="wr-row"') == len(stats)
        assert f'by {stats.columns[0]}"' in html


def test_buffer_signal_shape_scores_the_same_trades_as_the_table():
    """The chart's Strategy column must say what the table's Notation and Win
    Rate say, row for row; Signal repeats as the ceiling."""
    sample = get_sample_data()
    table = calculate_sl_buffer_statistics(sample)
    chart = calculate_sl_buffer_statistics(sample, with_signal=True)
    assert list(chart.columns) == ['SL Buffer', 'Trades', 'Signal', 'Strategy']
    assert list(chart['SL Buffer']) == list(table['SL Buffer'])
    for (_, t), (_, c) in zip(table.iterrows(), chart.iterrows()):
        assert c['Strategy'] == f"{t['Notation']} ({t['Win Rate']})"
    assert chart['Signal'].nunique() == 1


def test_chart_handles_an_empty_table():
    empty = calculate_weekday_statistics(get_sample_data()).iloc[0:0]
    assert 'No data' in create_signal_strategy_chart(empty)



def test_format_wl_shape():
    assert _format_wl(3, 7, 10) == '3W - 7L (30.0%)'
    assert _format_wl(0, 0, 0) == '0W - 0L (0.0%)'


def test_weekday_statistics_has_a_row_per_weekday():
    result = calculate_weekday_statistics(get_sample_data())
    assert list(result['Day']) == WEEKDAY_ORDER
    assert list(result.columns) == ['Day', 'Trades', 'Signal', 'Strategy',
                                    'Strategy + 1 pip', 'Strategy + BE Rule']


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


def _be_trades(rows):
    return pd.DataFrame(rows, columns=['SL', 'Pullback', 'TP',
                                       BE_RULE_COLUMN])


def test_break_even_checks_the_stop_first():
    """2026-09-23 #1: checked, but Pullback 4.9 took out the 3.0 stop
    before the rule could act. A 2 pip buffer lets it survive into it."""
    trades = _be_trades([(3.0, 4.9, 50.0, True)])
    wins, bes, losses = break_even_outcomes(trades, rrr=4)
    assert losses.iloc[0] and not bes.iloc[0]
    wins, bes, losses = break_even_outcomes(trades, rrr=4, buffer=2)
    assert bes.iloc[0]


def test_break_even_scores_a_checked_trade_as_break_even():
    trades = _be_trades([(14.0, 10.8, 24.0, True)])
    wins, bes, losses = break_even_outcomes(trades)
    assert bes.iloc[0] and not wins.iloc[0]


def test_break_even_leaves_a_target_inside_the_trigger_alone():
    """A 1.9 pip target is hit long before price runs 10 pips, so the
    checkbox cannot turn it into a break-even trade."""
    trades = _be_trades([(1.9, 0.5, 12.0, True)])
    wins, bes, losses = break_even_outcomes(trades)
    assert wins.iloc[0]
    wins, bes, losses = break_even_outcomes(trades, rrr=6)
    assert bes.iloc[0]


def test_break_even_saves_an_unchecked_trade_past_the_trigger():
    """Unchecked, TP 15 > 10: 1:3 on a 6 pip stop (18) is missed, so the
    stop at entry makes it break-even rather than a loss. At 1:2 (12) it
    wins."""
    trades = _be_trades([(6.0, 2.0, 15.0, False)])
    assert break_even_outcomes(trades, rrr=3)[1].iloc[0]
    assert break_even_outcomes(trades, rrr=2)[0].iloc[0]


def test_break_even_does_not_trigger_at_or_below_ten_pips():
    trades = _be_trades([(6.0, 2.0, 10.0, False)])
    wins, bes, losses = break_even_outcomes(trades, rrr=2)
    assert losses.iloc[0]


def test_break_even_puts_every_trade_in_exactly_one_outcome():
    df = load_data(CSV_PATH)
    for rrr in RRR_RATIOS:
        wins, bes, losses = break_even_outcomes(df, rrr=rrr)
        assert (wins.astype(int) + bes.astype(int)
                + losses.astype(int) == 1).all()


def test_break_even_reads_the_checkbox_as_text_too():
    trades = _be_trades([(14.0, 1.0, 24.0, 'TRUE'), (14.0, 1.0, 24.0, 'FALSE')])
    wins, bes, _ = break_even_outcomes(trades)
    assert list(bes) == [True, False]
    assert list(wins) == [False, True]


def test_break_even_without_the_column_reads_every_trade_unchecked():
    wins, bes, _ = break_even_outcomes(get_sample_data(), rrr=3)
    assert not bes[get_sample_data()['TP'] <= BE_RULE_PIPS].any()


def test_weekday_be_column_leaves_break_evens_out_of_the_rate():
    """Two Monday trades: one wins at 1:1, one checked trade with a 14 pip
    stop is break-even. 1 win over 1 decided trade, not 1 over 2."""
    trades = pd.DataFrame({
        'Weekday': ['Monday', 'Monday'], 'SL': [4.0, 14.0],
        'Pullback': [1.0, 2.0], 'TP': [16.0, 24.0],
        BE_RULE_COLUMN: [False, True]})
    result = calculate_weekday_statistics(trades)
    monday = result[result['Day'] == 'Monday'].iloc[0]
    assert monday['Strategy'] == '100.0%'
    assert monday['Strategy + BE Rule'] == '100.0%'


BUFFERED_TYPES = [f'{t} + 1 pip buffer' for t in TYPE_ORDER]


def test_setup_statistics_lists_default_then_types_only():
    """Every trade, then Continuation and Reversal, then both again with a
    buffered stop - no per-level setup rows, as the level alone says
    nothing."""
    result = calculate_setup_statistics(get_sample_data())
    assert list(result['Setup']) == ['Default'] + TYPE_ORDER + BUFFERED_TYPES
    assert result.iloc[0]['Trades'] == 10
    assert list(result.columns) == ['Setup', 'Trades', 'Signal', 'Strategy']
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
    assert labels == (['Default'] + TYPE_ORDER + ['Sweep'] + BUFFERED_TYPES
                      + ['Sweep + 1 pip buffer'])


def test_setup_buffer_rows_pad_the_stop_for_strategy_only():
    """Continuation is sample rows 1, 3, 5, 8, 9 - Strategy wins 1, 5, 9.
    At SL + 1 row 5 drops out (TP 3 < SL 3 + 1). Reversal is rows 2, 4, 6,
    7, 10 - wins 4, 7, 10 - and SL + 1 rescues none and drops none. Signal
    does not read the stop, so each buffered row keeps its type's Signal."""
    assert TYPE_BUFFER_PIPS == 1
    result = calculate_setup_statistics(get_sample_data()).set_index('Setup')
    assert result.loc['Continuation', 'Strategy'] == '3W - 2L (60.0%)'
    assert (result.loc['Continuation + 1 pip buffer', 'Strategy']
            == '2W - 3L (40.0%)')
    assert (result.loc['Reversal + 1 pip buffer', 'Strategy']
            == result.loc['Reversal', 'Strategy'])
    for t in TYPE_ORDER:
        assert result.loc[f'{t} + 1 pip buffer', 'Signal'] == result.loc[t, 'Signal']
        assert result.loc[f'{t} + 1 pip buffer', 'Trades'] == result.loc[t, 'Trades']


def test_grouping_tables_handle_an_empty_dataset():
    empty = get_empty_data()
    result = calculate_setup_statistics(empty)
    assert (result['Trades'] == 0).all()
    assert (result['Signal'] == '0W - 0L (0.0%)').all()
    result = calculate_weekday_statistics(empty)
    assert (result['Trades'] == 0).all()
    assert (result.iloc[:, 2:] == '0.0%').all().all()


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


def test_sl_max_statistics_lists_every_cap():
    result = calculate_sl_max_statistics(get_sample_data())
    assert list(result.columns) == ['Cap SL', 'Trades', 'Signal', 'Strategy']
    assert list(result['Cap SL']) == ['Default'] + [f'{p} pips'
                                                     for p in SL_MAX_PIPS]
    assert SL_MAX_PIPS == [4, 5, 6, 7, 8, 9]
    assert result['Signal'].nunique() == 1


def test_sl_max_caps_only_wider_stops():
    """SL 8, Pullback 6, TP 20: a 9 or 7 pip cap still survives the 6 pip
    pullback and wins; a 6 or 5 pip cap is taken out by it. SL 3 sits inside
    every cap, so it scores the same on every row."""
    trades = get_sample_data().iloc[0:2].copy()
    trades['SL'] = [8.0, 3.0]
    trades['Pullback'] = [6.0, 1.0]
    trades['TP'] = [20.0, 3.0]
    result = calculate_sl_max_statistics(trades).set_index('Cap SL')
    assert result.loc['Default', 'Strategy'].startswith('2W - 0L')
    assert result.loc['9 pips', 'Strategy'].startswith('2W - 0L')
    assert result.loc['7 pips', 'Strategy'].startswith('2W - 0L')
    assert result.loc['6 pips', 'Strategy'].startswith('1W - 1L')
    assert result.loc['5 pips', 'Strategy'].startswith('1W - 1L')


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
        'Cap SL': (calculate_sl_max_statistics(sample), 'Cap SL'),
        'Pullback': (calculate_pullback_statistics(sample), 'Pullback'),
    }

    seen = {}
    for name, (result, column) in tables.items():
        first = result.iloc[0]
        assert first[column] == 'Default', f'{name} does not open with Default'
        # SL Range and Fixed SL report Signal/Strategy in one cell each;
        # Adding Buffer keeps Notation and Win Rate apart; Pullback only
        # carries a W - M - L split. Compare on W/L counts so the shared rule
        # stays pinned across every shape.
        if 'Strategy' in result.columns:
            seen[name] = (first['Trades'], first['Strategy'].split(' (')[0])
        elif 'Notation' in result.columns:
            seen[name] = (first['Trades'], first['Notation'])
        else:
            # Nothing is missed when every trade is taken at the signal.
            wins, missed, losses = first['Trades'].split(' - ')
            assert missed == '0M'
            seen[name] = (int(wins[:-1]) + int(losses[:-1]),
                          f'{wins} - {losses}')

    assert len(set(seen.values())) == 1, f'Default rows disagree: {seen}'
    assert seen['SL Range'][0] == len(sample)


# --- TP distribution and pullback entries ---------------------------------

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


def _wml(cell):
    """Split '1W - 2M - 3L' into (1, 2, 3)."""
    return tuple(int(part[:-1]) for part in cell.split(' - '))


def test_pullback_statistics_lists_every_level():
    result = calculate_pullback_statistics(get_sample_data())
    assert result.iloc[0]['Pullback'] == 'Default'
    assert list(result['Pullback'])[4:] == [
        'Half', '1 pip + 1 pip buffer', '2 pips + 1 pip buffer',
        '3 pips + 1 pip buffer', 'Half + 1 pip buffer']
    assert len(result) == 2 * len(PULLBACK_ENTRY_PIPS) + 1
    assert PULLBACK_BUFFER_PIPS == 1
    assert list(result.columns) == ['Pullback', 'Trades']


def test_pullback_rows_split_every_trade_into_tapped_missed_and_lost():
    """Each row covers the whole dataset once, and losers do not depend on
    the limit - they lose whatever the pullback, for a given buffer."""
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    rows = [_wml(c) for c in result['Trades']]
    assert all(sum(row) == len(sample) for row in rows)
    buffered = result['Pullback'].str.endswith('buffer')
    assert len({row[2] for row, b in zip(rows, buffered) if not b}) == 1
    assert len({row[2] for row, b in zip(rows, buffered) if b}) == 1


def test_pullback_counts_match_the_sample():
    """Winners are rows 1, 4, 5, 7, 9, 10 with pullbacks 1.0, 3.5, 0.5, 2.0,
    4.0, 1.5; the other four lose. A 2 pip limit is tapped by 3.5, 2.0 and
    4.0 and misses 1.0, 0.5 and 1.5. Half the stop is tapped exactly by 3.5
    (SL 7) and 4.0 (SL 8)."""
    result = calculate_pullback_statistics(
        get_sample_data()).set_index('Pullback')['Trades']
    assert result['Default'] == '6W - 0M - 4L'
    assert result['1 pip'] == '5W - 1M - 4L'
    assert result['2 pips'] == '3W - 3M - 4L'
    assert result['3 pips'] == '2W - 4M - 4L'
    assert result['Half'] == '2W - 4M - 4L'


def test_pullback_chart_stacks_w_m_l_from_the_table_cells():
    stats = calculate_pullback_statistics(get_sample_data())
    html = create_pullback_chart(stats)
    assert html.count('class="pb-stack"') == len(stats)
    # '2 pips' is 3W - 3M - 4L in the sample: three segments, sized by count.
    assert 'title="2 pips - Tapped winners: 3 (30.0% of trades)"' in html
    assert 'title="2 pips - Missed winners: 3 (30.0% of trades)"' in html
    assert 'title="2 pips - Losers: 4 (40.0% of trades)"' in html


def test_pullback_chart_skips_empty_segments():
    """Default misses nothing, so it draws no M segment at all."""
    html = create_pullback_chart(calculate_pullback_statistics(get_sample_data()))
    assert 'Default - Missed winners' not in html


def test_pullback_chart_starts_a_group_at_the_buffered_rows():
    html = create_pullback_chart(calculate_pullback_statistics(get_sample_data()))
    assert html.count('pb-group') == 2  # the CSS rule and one row
    assert html.index('pb-row pb-group') < html.index('1 pip + 1 pip buffer')
    assert html.index('>Half<') < html.index('pb-row pb-group')


def test_pullback_chart_handles_an_empty_table():
    empty = calculate_pullback_statistics(get_sample_data()).iloc[0:0]
    assert 'No data' in create_pullback_chart(empty)


def test_pullback_buffer_rows_pad_the_stop_but_not_the_limit():
    """At SL + 1 the winners are rows 1, 4, 7, 9, 10 with pullbacks 1.0,
    3.5, 2.0, 4.0, 1.5; row 5 drops out because TP 3 < SL 3 + 1. Half still
    reads the recorded SL, so only 3.5 (SL 7) and 4.0 (SL 8) tap it."""
    result = calculate_pullback_statistics(
        get_sample_data()).set_index('Pullback')['Trades']
    assert result['1 pip + 1 pip buffer'] == '5W - 0M - 5L'
    assert result['2 pips + 1 pip buffer'] == '3W - 2M - 5L'
    assert result['3 pips + 1 pip buffer'] == '2W - 3M - 5L'
    assert result['Half + 1 pip buffer'] == '2W - 3M - 5L'


# --- Strategies tables ----------------------------------------------------

def test_get_buffer_strategies_covers_all_trades_and_fixed_sl():
    """Capping moved to the Max SL column, so there are no Max SL strategies."""
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
    assert list(result.columns) == ['Strategy', 'Buffer', 'Trend', 'Max SL',
                                    'RRR', 'Trades', 'Notation', 'Win Rate']
    keys = ['Strategy', 'Buffer', 'Trend', 'Max SL', 'RRR']
    assert not result.duplicated(keys).any()
    all_trades = result[result['Strategy'] == 'All Trades']
    assert set(all_trades['Max SL']) == set(MAX_SL_CAPS)
    assert MAX_SL_CAPS == [0, 5, 6]
    assert set(result['Trend']) == set(TREND_FILTERS)
    assert (result[result['Trend'] == 'All']['Trades'] == len(sample)).all()


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
    assert set(result['Max SL']) == {0}


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


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------

def test_streak_statistics_lists_default_then_types():
    result = calculate_streak_statistics(get_sample_data())
    assert list(result['Trend']) == ['Default'] + TYPE_ORDER + BUFFERED_TYPES
    assert list(result.columns) == [
        'Trend', 'Trades', 'Longest Wins', 'Longest Losses']
    assert result.iloc[0]['Trades'] == 10
    assert result.set_index('Trend').loc[TYPE_ORDER, 'Trades'].sum() == 10
    assert result.set_index('Trend').loc[BUFFERED_TYPES, 'Trades'].sum() == 10


def test_streak_buffer_rows_score_runs_on_the_padded_stop():
    """Two Continuation trades with SL 5: the first pulls back exactly 5 and
    is stopped at SL but survives SL + 1; the second wins either way. The
    buffer turns a 1-win run into a 2-win run."""
    df = pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-04'],
        'Type': ['Continuation'] * 2,
        'SL': [5.0] * 2,
        'Pullback': [5.0, 1.0],
        'TP': [10.0, 10.0],
    })
    result = calculate_streak_statistics(df).set_index('Trend')
    assert result.loc['Continuation', 'Longest Wins'] == '1 (2025-02-04 - 2025-02-04)'
    assert (result.loc['Continuation + 1 pip buffer', 'Longest Wins']
            == '2 (2025-02-03 - 2025-02-04)')
    assert result.loc['Continuation + 1 pip buffer', 'Longest Losses'] == '0'


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
    assert result.loc['Reversal', 'Trades'] == 0
    assert result.loc['Reversal', 'Longest Wins'] == '0'
    assert result.loc['Reversal', 'Longest Losses'] == '0'


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
    assert result.loc['Reversal', 'Longest Losses'] == '1 (2025-02-04 - 2025-02-04)'


def test_streak_statistics_empty_dataset():
    result = calculate_streak_statistics(get_empty_data())
    assert list(result['Trend']) == ['Default'] + TYPE_ORDER + BUFFERED_TYPES
    assert list(result['Trades']) == [0] * 5
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
                    & (table['Trend'] == 'Reversal') & (table['Max SL'] == 5)
                    & (table['RRR'] == f'1:{rrr}')].iloc[0]
        wins, losses = (int(v[:-1]) for v in row['Notation'].split(' \u2013 '))
        curve = curves['All Trades +1 / Reversal / cap 5']
        assert curve['outcome'] == wins * rrr - losses
        assert curve['trades'] == wins + losses == row['Trades']


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
    buffer_default = calculate_sl_buffer_statistics(get_sample_data()).iloc[0]
    wins, missed, losses = default['Notation'].split(' - ')
    assert missed == '0M'
    assert f'{wins} - {losses}' == buffer_default['Notation']
    assert default['Win Rate'] == buffer_default['Win Rate']


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
