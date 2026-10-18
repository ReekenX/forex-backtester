"""
Tests for utils.continuation_reversal - the analysis behind the 30MCR Fixed
report.

Run with: poetry run python -m pytest strategies/30MCR-fixed/tests/ -v
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.continuation_reversal import (  # noqa: E402
    EQUITY_HIGHLIGHT,
    RATE_COLUMNS,
    RRR_RATIOS,
    TREND_FILTERS,
    TYPE_ORDER,
    WEEKDAY_ORDER,
    _drop_scratch_columns,
    _normalise_header,
    calculate_equity_curves,
    calculate_r_counts,
    calculate_setup_statistics,
    calculate_strategy_statistics,
    calculate_streak_statistics,
    calculate_weekday_statistics,
    create_equity_chart,
    create_html_table,
    create_r_histogram_combined,
    create_signal_strategy_chart,
    create_sortable_table,
    load_data,
    strategy_wins,
)

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'trades.csv')

HEADER = ('Date,Weekday,"Trade\n3",Direction,30M,SL,Pullback,Reward,'
          '"Signal \n50%","WR \n40%\n3R"\n')


def get_sample_data():
    """
    Ten trades over three days on a 5-pip stop.

    Row 2 is the stopped-out winner-that-wasn't: Pullback 6.0 >= SL 5.0 with a
    Reward of 4R, so Signal counts it and Strategy does not. Row 6 pulls back
    exactly the stop, which is a loss too.
    """
    levels = ['High', 'High', 'Low', 'Low', 'High',
              'Low', 'High', 'Low', 'High', 'Low']
    directions = ['Buy', 'Sell', 'Sell', 'Buy', 'Buy',
                  'Buy', 'Sell', 'Sell', 'Buy', 'Buy']
    types = ['Continuation' if (lvl, d) in {('High', 'Buy'), ('Low', 'Sell')}
             else 'Reversal' for lvl, d in zip(levels, directions)]
    return pd.DataFrame({
        'Date': ['2025-02-03', '2025-02-03', '2025-02-03', '2025-02-04',
                 '2025-02-04', '2025-02-04', '2025-02-05', '2025-02-05',
                 '2025-02-05', '2025-02-05'],
        'Weekday': ['Monday'] * 3 + ['Tuesday'] * 3 + ['Wednesday'] * 4,
        'Trade': ['#1', '#2', '#3', '#1', '#2', '#3', '#1', '#2', '#3', '#4'],
        'Direction': directions,
        '30M': levels,
        'SL': [5.0] * 10,
        'Pullback': [1.0, 6.0, 0.5, 2.0, 4.9, 5.0, 0.0, 3.0, 1.5, 2.5],
        'Reward': [3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 10.0, 0.0, 1.0, 0.0],
        'R': [3.0, -4.0, 0.0, 1.0, 2.0, -3.0, 10.0, 0.0, 1.0, 0.0],
        'Setup': [f'{lvl} {t}' for lvl, t in zip(levels, types)],
        'Type': types,
    })


def _write(tmp_path, rows):
    path = tmp_path / 'trades.csv'
    path.write_text(HEADER + ''.join(row + '\n' for row in rows))
    return str(path)


# --- loading --------------------------------------------------------------

def test_load_data_reads_the_project_csv():
    df = load_data(CSV_PATH)
    assert len(df) > 0
    assert (df['SL'] > 0).all()
    assert {'Reward', 'R', 'Type', 'Setup'} <= set(df.columns)


def test_normalise_header_keeps_only_the_first_line():
    assert _normalise_header('Trade\n176') == 'Trade'
    assert _normalise_header('WR \n43.8%\n301R') == 'WR'


def test_load_data_drops_everything_right_of_reward(tmp_path):
    df = load_data(_write(tmp_path, [
        '2025-02-03,Monday,#1,Buy,High,5.0,1.0,2R,2R,2R']))
    assert list(df.columns[:8]) == ['Date', 'Weekday', 'Trade', 'Direction',
                                    '30M', 'SL', 'Pullback', 'Reward']
    assert 'Signal' not in df.columns
    assert 'WR' not in df.columns


def test_drop_scratch_columns_survives_renamed_and_repeated_headers():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=['SL', 'Reward', 'X', 'X'])
    assert list(_drop_scratch_columns(df).columns) == ['SL', 'Reward']


def test_load_data_parses_reward_in_r(tmp_path):
    df = load_data(_write(tmp_path, [
        '2025-02-03,Monday,#1,Buy,High,5.0,1.0,3R,,',
        '2025-02-03,Monday,#2,Buy,High,5.0,1.0,,,',
        '2025-02-03,Monday,#3,Buy,High,5.0,1.0,10R,,']))
    assert df['Reward'].tolist() == [3.0, 0.0, 10.0]


def test_load_data_signs_r_negative_when_the_stop_was_hit(tmp_path):
    df = load_data(_write(tmp_path, [
        '2025-02-03,Monday,#1,Buy,High,5.0,1.0,3R,,',
        '2025-02-03,Monday,#2,Buy,High,5.0,5.0,2R,,',
        '2025-02-03,Monday,#3,Buy,High,5.0,5.4,,,']))
    assert df['R'].tolist() == [3.0, -2.0, 0.0]
    # A stopped-out trade with no reward reads 0, not -0.
    assert str(df['R'].iloc[2]) == '0.0'


@pytest.mark.parametrize('level, direction, setup', [
    ('High', 'Buy', 'High Continuation'),
    ('High', 'Sell', 'High Reversal'),
    ('Low', 'Sell', 'Low Continuation'),
    ('Low', 'Buy', 'Low Reversal'),
])
def test_load_data_derives_the_setup(tmp_path, level, direction, setup):
    df = load_data(_write(tmp_path, [
        f'2025-02-03,Monday,#1,{direction},{level},5.0,1.0,1R,,']))
    assert df['Setup'].iloc[0] == setup
    assert df['Type'].iloc[0] == setup.split()[-1]


def test_load_data_drops_rows_without_a_date(tmp_path):
    df = load_data(_write(tmp_path, [
        '2025-02-03,Monday,#1,Buy,High,5.0,1.0,1R,,',
        ',,,,,,,,,']))
    assert len(df) == 1


def test_load_data_matches_the_sheets_signal_rate():
    """The sheet's 'Signal 54.5%' header figure is Reward > 0 on every trade."""
    df = load_data(CSV_PATH)
    assert f"{(df['Reward'] > 0).mean() * 100:.1f}%" == '54.5%'


def test_strategy_rate_is_one_trade_below_the_sheets_wr():
    """The sheet's 'WR 43.8%' counts the one trade whose Pullback equals its
    SL as a survivor. The project rule is strict, Pullback < SL, so the page
    reads one win fewer: 76 of 176, 43.2%."""
    df = load_data(CSV_PATH)
    wins = strategy_wins(df)
    on_the_stop = (df['Pullback'] == df['SL']) & (df['Reward'] >= 1)
    assert f'{(wins | on_the_stop).mean() * 100:.1f}%' == '43.8%'
    assert f'{wins.mean() * 100:.1f}%' == '43.2%'


def test_load_data_leaves_the_csv_alone():
    before = Path(CSV_PATH).read_bytes()
    load_data(CSV_PATH)
    assert Path(CSV_PATH).read_bytes() == before


# --- win rule -------------------------------------------------------------

def test_strategy_wins_needs_both_the_stop_and_the_reward():
    df = get_sample_data()
    assert strategy_wins(df).tolist() == [
        True, False, False, True, True, False, True, False, True, False]


def test_strategy_wins_raises_the_reward_bar_with_the_rrr():
    df = get_sample_data()
    assert int(strategy_wins(df, 2).sum()) == 3
    assert int(strategy_wins(df, 3).sum()) == 2


def test_a_pullback_equal_to_the_stop_is_a_loss():
    df = get_sample_data()
    assert not strategy_wins(df).iloc[5]


# --- grouping tables ------------------------------------------------------

def test_weekday_statistics_has_a_row_per_weekday():
    stats = calculate_weekday_statistics(get_sample_data())
    assert stats['Day'].tolist() == WEEKDAY_ORDER
    assert list(stats.columns) == ['Day', 'Trades'] + RATE_COLUMNS


def test_weekday_statistics_scores_each_day():
    stats = calculate_weekday_statistics(get_sample_data()).set_index('Day')
    assert stats.loc['Monday', 'Trades'] == 3
    assert stats.loc['Monday', 'Signal'] == '66.7%'
    assert stats.loc['Monday', 'Strategy'] == '33.3%'
    assert stats.loc['Thursday', 'Strategy'] == '0.0%'


def test_signal_counts_the_stopped_out_winner_and_strategy_does_not():
    stats = calculate_setup_statistics(get_sample_data()).set_index('Setup')
    assert stats.loc['Default', 'Signal'] == '70.0%'
    assert stats.loc['Default', 'Strategy'] == '50.0%'


def test_setup_statistics_splits_default_into_the_two_types():
    stats = calculate_setup_statistics(get_sample_data())
    assert stats['Setup'].tolist() == ['Default'] + TYPE_ORDER
    assert stats['Trades'].iloc[0] == stats['Trades'].iloc[1:].sum()


# --- streaks --------------------------------------------------------------

def test_streak_statistics_rows_and_columns():
    stats = calculate_streak_statistics(get_sample_data())
    assert stats['Trend'].tolist() == ['Default'] + TYPE_ORDER
    assert list(stats.columns) == ['Trend', 'Trades', 'Longest Wins',
                                   'Longest Losses']


def test_streak_statistics_reads_trades_in_date_order():
    """W L L W W L W L W L: two wins in a row at most, two losses in a row
    at most - each naming the first such run."""
    stats = calculate_streak_statistics(get_sample_data()).set_index('Trend')
    assert stats.loc['Default', 'Longest Wins'] == '2 (2025-02-04 - 2025-02-04)'
    assert stats.loc['Default', 'Longest Losses'] == '2 (2025-02-03 - 2025-02-03)'


def test_streak_type_rows_skip_the_other_types_trades():
    df = get_sample_data()
    stats = calculate_streak_statistics(df).set_index('Trend')
    for trend in TYPE_ORDER:
        assert stats.loc[trend, 'Trades'] == int((df['Type'] == trend).sum())


def test_streak_statistics_empty_dataset():
    stats = calculate_streak_statistics(get_sample_data().iloc[0:0])
    assert (stats['Longest Wins'] == '0').all()


# --- strategies -----------------------------------------------------------

def test_strategy_statistics_has_a_row_per_trend_and_rrr():
    stats = calculate_strategy_statistics(get_sample_data())
    assert len(stats) == len(TREND_FILTERS) * len(RRR_RATIOS)
    assert list(stats.columns) == ['Trend', 'RRR', 'Trades', 'Notation',
                                   'Win Rate', 'Outcome']


def test_strategy_statistics_scores_all_trades():
    stats = calculate_strategy_statistics(get_sample_data())
    every = stats[stats['Trend'] == 'All'].set_index('RRR')
    assert every.loc['1:1', 'Notation'] == '5W – 5L'
    assert every.loc['1:1', 'Outcome'] == '+0R'
    assert every.loc['1:2', 'Win Rate'] == '30.0%'
    assert every.loc['1:2', 'Outcome'] == '-1R'
    assert every.loc['1:3', 'Outcome'] == '-2R'


def test_strategy_trend_rows_sum_to_all():
    stats = calculate_strategy_statistics(get_sample_data())
    for rrr in RRR_RATIOS:
        rows = stats[stats['RRR'] == f'1:{rrr}'].set_index('Trend')
        assert rows.loc['All', 'Trades'] == (
            rows.loc['Continuation', 'Trades'] + rows.loc['Reversal', 'Trades'])


def test_strategy_statistics_empty_dataset():
    stats = calculate_strategy_statistics(get_sample_data().iloc[0:0])
    assert (stats['Win Rate'] == '0.0%').all()


# --- equity curves --------------------------------------------------------

def test_equity_curves_end_on_the_table_outcome():
    df = get_sample_data()
    stats = calculate_strategy_statistics(df)
    for rrr in RRR_RATIOS:
        curves = {c['label']: c for c in calculate_equity_curves(df, rrr)}
        rows = stats[stats['RRR'] == f'1:{rrr}'].set_index('Trend')
        for trend in TREND_FILTERS:
            label = 'All Trades' if trend == 'All' else trend
            assert f"{curves[label]['outcome']:+.0f}R" == rows.loc[trend, 'Outcome']


def test_equity_curves_line_up_in_time():
    df = get_sample_data()
    for curve in calculate_equity_curves(df, 2):
        assert len(curve['curve']) == len(df) + 1
        assert curve['curve'][0] == 0.0


def test_equity_curves_are_ranked_best_first():
    outcomes = [c['outcome'] for c in calculate_equity_curves(get_sample_data(), 3)]
    assert outcomes == sorted(outcomes, reverse=True)


def test_equity_chart_draws_every_curve():
    curves = calculate_equity_curves(get_sample_data(), 1)
    html = create_equity_chart(curves, 1, chart_id='eq')
    assert html.count('class="eq-series') == len(curves)
    assert len(curves) <= EQUITY_HIGHLIGHT
    assert 'id="eq"' in html


def test_equity_chart_handles_no_trades():
    curves = calculate_equity_curves(get_sample_data().iloc[0:0], 1)
    assert 'No data' in create_equity_chart(curves, 1)


# --- charts and tables ----------------------------------------------------

def test_chart_bars_follow_the_table_cells():
    stats = calculate_setup_statistics(get_sample_data())
    html = create_signal_strategy_chart(stats)
    assert 'width: 70.0%' in html
    assert 'width: 50.0%' in html
    assert 'Strategy + 1 pip' not in html


def test_chart_handles_an_empty_table():
    assert 'No data' in create_signal_strategy_chart(pd.DataFrame())


def test_sortable_table_colours_rates_against_50_percent():
    stats = calculate_setup_statistics(get_sample_data())
    html = create_sortable_table(stats, 't', colour_columns=tuple(RATE_COLUMNS))
    assert 'class="positive-edge">70.0%' in html
    assert '<td>50.0%</td>' in html


def test_html_table_colours_win_rate_against_its_rrr():
    stats = pd.DataFrame({'Trend': ['All'], 'Win Rate': ['30.0%']})
    assert 'positive-edge' in create_html_table(stats, rrr='1:3')
    assert 'negative-edge' in create_html_table(stats, rrr='1:2')


# --- R distribution -------------------------------------------------------

def test_r_counts_drop_the_sign():
    counts = calculate_r_counts(get_sample_data())
    assert counts[1] == 2
    assert counts[3] == 2
    assert counts[4] == 1
    assert counts[10] == 1


def test_r_histogram_is_cumulative():
    html = create_r_histogram_combined(get_sample_data())
    # 7 trades reach 1R or more, 1 reaches 10R.
    assert '>7</span>' in html
    assert 'R Distribution' in html
