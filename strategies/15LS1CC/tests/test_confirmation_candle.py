"""
Tests for utils.confirmation_candle module

These tests use a small dataset of 10 rows to verify the analysis functionality.
Run with: poetry run python tests/test_confirmation_candle.py
"""

import os
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.confirmation_candle import (
    calculate_statistics,
    calculate_buffer_statistics,
    calculate_fixed_sl_statistics,
    calculate_fixed_sl_1h_statistics,
    calculate_weekday_statistics,
    _calculate_buffer_statistics_filtered,
    create_html_table,
    get_strategies,
    get_buffer_strategies,
    simulate_challenge,
    _calculate_stats,
    _calculate_stats_with_buffer,
    _calculate_fixed_sl_stats,
    _calculate_fixed_sl_stats_with_strategy,
    _get_1h_strategies,
    _breakeven_rate,
    RRR_RATIOS,
    BUFFER_PIPS,
    FIXED_SL_SIZES,
    WEEKDAY_ORDER,
    _format_wl,
    SL_RANGES,
    calculate_sl_statistics,
    PULLBACK_RANGES,
    calculate_pullback_statistics,
    TP_RANGES,
    calculate_tp_statistics,
    SL_PB_SL_RANGES,
    SL_PB_PB_RANGES,
    SL_PB_BUFFERS,
    calculate_sl_pb_crosstab,
    calculate_sl_pb_distribution,
    calculate_sl_pb_buffer_impact,
    calculate_direction_statistics,
    calculate_trade_number_statistics,
    calculate_prev_outcome_statistics,
    calculate_rrr_sweep_statistics,
    calculate_stop_after_loss_statistics,
    RRR_SWEEP,
    STOP_AFTER_LOSS_RULES,
    calculate_pd_alignment_statistics,
    calculate_pd_zone_statistics,
    calculate_pd_1h_combined_statistics,
    calculate_pd_direction_statistics,
    PD_ZONES,
)


def get_sample_data():
    """Create a sample dataset with 10 trades matching the CSV structure."""
    return pd.DataFrame({
        'Date': ['2026-01-12', '2026-01-12', '2026-01-12', '2026-01-13', '2026-01-13',
                 '2026-01-14', '2026-01-14', '2026-01-15', '2026-01-15', '2026-01-16'],
        'Weekday': ['Monday', 'Monday', 'Monday', 'Tuesday', 'Tuesday',
                    'Wednesday', 'Wednesday', 'Thursday', 'Thursday', 'Friday'],
        'Trade': ['#1', '#2', '#3', '#1', '#2',
                  '#1', '#2', '#1', '#2', '#1'],
        'Direction': ['Buy', 'Buy', 'Sell', 'Buy', 'Sell',
                      'Sell', 'Buy', 'Buy', 'Sell', 'Sell'],
        '1H': ['Buy', 'Buy', 'Buy', 'Sell', 'Sell',
               'Sell', 'Sell', 'Buy', 'Buy', 'Sell'],
        'PD': ['Buy', 'Buy', 'Sell', '', 'Buy',
               'Sell', '', 'Sell', 'Buy', 'Sell'],
        'SL': [3.5, 1.1, 2.0, 4.0, 3.0,
               5.0, 2.5, 6.0, 8.0, 1.5],
        'Pullback': [3.5, 0.8, 2.1, 1.5, 3.0,
                     2.0, 2.5, 3.0, 7.0, 0.5],
        'TP': [0, 12.0, 0, 10.0, 0,
               8.0, 0, 15.0, 10.0, 5.0],
        'R': [0, 10, 0, 2.5, 0,
              1.6, 0, 2.5, 1.25, 3.3],
    })


def get_empty_data():
    """Create an empty dataset."""
    return pd.DataFrame({
        'Date': [], 'Weekday': [], 'Trade': [], 'Direction': [],
        '1H': [], 'PD': [], 'SL': [], 'Pullback': [], 'TP': [], 'R': [],
    })


def test_get_strategies():
    """Test that strategies are returned as list of tuples."""
    strategies = get_strategies()
    assert len(strategies) > 0
    for name, func in strategies:
        assert isinstance(name, str)
        assert callable(func)


def test_strategy_names_include_base():
    """Test that base strategies are present."""
    strategies = get_strategies()
    names = [name for name, _ in strategies]
    assert 'All Trades' in names
    assert '1H Aligned' in names
    assert '1H Against' in names


def test_strategy_names_include_sl_combos():
    """Test that SL combination strategies are present."""
    strategies = get_strategies()
    names = [name for name, _ in strategies]
    assert '1H Aligned + SL < 5' in names
    assert '1H Against + SL > 3' in names
    assert 'All Trades + SL < 3' in names


def test_calculate_stats_all_trades():
    """Test statistics for all trades baseline."""
    sample = get_sample_data()
    stats = _calculate_stats(sample, 'All Trades')

    assert stats['Strategy'] == 'All Trades'
    assert stats['RRR'] == '1:1'
    assert stats['Trades'] == 10


def test_calculate_stats_empty():
    """Test statistics for empty dataset."""
    empty = get_empty_data()
    stats = _calculate_stats(empty, 'Empty')

    assert stats['Trades'] == 0
    assert stats['Notation'] == '0W – 0L'
    assert stats['Win Rate'] == '0.0%'
    assert stats['Edge'] == '-50.0%'
    assert stats['Trades Required'] == 'N/A'


def test_win_condition_1_1_rrr():
    """Test win condition: Pullback < SL AND TP >= SL."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        'SL': [5.0, 5.0, 5.0, 5.0],
        'Pullback': [2.0, 5.0, 4.9, 6.0],
        'TP': [5.0, 10.0, 5.0, 10.0],
    })

    stats = _calculate_stats(trades, 'Test')

    # Trade 1: Pullback(2) < SL(5) AND TP(5) >= SL(5) => WIN
    # Trade 2: Pullback(5) < SL(5) => FALSE (not less than) => LOSS
    # Trade 3: Pullback(4.9) < SL(5) AND TP(5) >= SL(5) => WIN
    # Trade 4: Pullback(6) < SL(5) => FALSE => LOSS
    assert stats['Notation'] == '2W – 2L'


def test_win_condition_tp_must_reach_sl():
    """Test that TP must be >= SL to win at 1:1."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'SL': [5.0, 5.0],
        'Pullback': [2.0, 2.0],
        'TP': [4.9, 5.0],  # First doesn't reach 1:1, second does
    })

    stats = _calculate_stats(trades, 'Test')
    assert stats['Notation'] == '1W – 1L'


def test_edge_calculation():
    """Test edge = win_rate - 50% for 1:1 RRR."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        'SL': [5.0, 5.0, 5.0, 5.0],
        'Pullback': [2.0, 2.0, 2.0, 6.0],  # 3 wins, 1 loss
        'TP': [5.0, 5.0, 5.0, 5.0],
    })

    stats = _calculate_stats(trades, 'Test')
    # Win rate = 75%, Edge = 75% - 50% = 25%
    assert stats['Win Rate'] == '75.0%'
    assert stats['Edge'] == '25.0%'
    assert stats['edge_value'] == 25.0


def test_outcome_calculation():
    """Test outcome = wins - losses for 1:1 RRR."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [5.0, 5.0, 5.0],
        'Pullback': [2.0, 2.0, 6.0],  # 2 wins, 1 loss
        'TP': [5.0, 5.0, 5.0],
    })

    stats = _calculate_stats(trades, 'Test')
    # Outcome = 2*1 - 1 = 1R
    assert stats['Outcome'] == '1R'


def test_days_calculation():
    """Test Days and Days % calculation."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [5.0, 5.0, 5.0, 5.0],
        'Pullback': [2.0, 6.0, 2.0, 6.0],  # Win, Loss, Win, Loss
        'TP': [5.0, 5.0, 5.0, 5.0],
    })

    stats = _calculate_stats(trades, 'Test')
    # Days with wins: Jan 1 and Jan 2 = 2 days
    # Total days: Jan 1, Jan 2, Jan 3 = 3 days
    # Days % = 2/3 * 100 = 67%
    assert stats['Days'] == 2
    assert stats['Days %'] == '67%'


def test_trades_required():
    """Test Trades Required calculation."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        'SL': [5.0, 5.0, 5.0, 5.0],
        'Pullback': [2.0, 2.0, 2.0, 6.0],  # 3 wins, 1 loss
        'TP': [5.0, 5.0, 5.0, 5.0],
    })

    stats = _calculate_stats(trades, 'Test')
    # Outcome = 3 - 1 = 2R, Trades Required = 4/2 = 2.0
    assert stats['Trades Required'] == '2.0'


def test_trades_required_negative_outcome():
    """Test Trades Required is N/A when outcome is not positive."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [5.0, 5.0, 5.0],
        'Pullback': [2.0, 6.0, 6.0],  # 1 win, 2 losses
        'TP': [5.0, 5.0, 5.0],
    })

    stats = _calculate_stats(trades, 'Test')
    assert stats['Trades Required'] == 'N/A'


def test_1h_aligned_filter():
    """Test 1H Aligned filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == '1H Aligned'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] == row['1H']


def test_1h_against_filter():
    """Test 1H Against filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == '1H Against'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] != row['1H']


def test_1h_aligned_plus_against_equals_all():
    """Test that 1H Aligned + 1H Against = All Trades."""
    sample = get_sample_data()
    strategies = get_strategies()
    aligned_func = [func for name, func in strategies if name == '1H Aligned'][0]
    against_func = [func for name, func in strategies if name == '1H Against'][0]

    aligned = aligned_func(sample)
    against = against_func(sample)

    assert len(aligned) + len(against) == len(sample)


def test_sl_filter_combination():
    """Test SL filter in combination strategy."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == '1H Aligned + SL > 3'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] == row['1H']
        assert row['SL'] > 3


def test_calculate_statistics_returns_positive_edge_only():
    """Test that calculate_statistics only returns positive edge strategies."""
    sample = get_sample_data()
    result = calculate_statistics(sample)

    if len(result) > 0:
        for edge_str in result['Edge']:
            edge_val = float(edge_str.replace('%', ''))
            assert edge_val > 0


def test_calculate_statistics_sorted_by_edge():
    """Test that results are sorted by edge descending."""
    sample = get_sample_data()
    result = calculate_statistics(sample)

    if len(result) > 1:
        edges = [float(e.replace('%', '')) for e in result['Edge']]
        assert edges == sorted(edges, reverse=True)


def test_calculate_statistics_columns():
    """Test that result has expected columns with totals in headers."""
    sample = get_sample_data()
    result = calculate_statistics(sample)

    if len(result) > 0:
        columns = list(result.columns)
        assert columns[0] == 'Strategy'
        assert columns[1] == 'RRR'
        assert columns[2].startswith('Trades (')
        assert columns[3] == 'Notation'
        assert columns[4] == 'Win Rate'
        assert columns[5] == 'Outcome'
        assert columns[6] == 'Edge'
        assert columns[7].startswith('Days (')
        assert columns[8] == 'Days %'
        assert columns[9] == 'Trades Required'


def test_create_html_table_basic():
    """Test HTML table creation."""
    sample = get_sample_data()
    stats = calculate_statistics(sample)
    html = create_html_table(stats)

    assert '<table' in html
    assert 'analysis-table' in html


def test_create_html_table_empty():
    """Test HTML table with empty data."""
    html = create_html_table(pd.DataFrame())
    assert 'No profitable strategies found' in html


def test_buffer_saves_losing_trade():
    """Test that adding buffer pips can save a trade that would otherwise lose.

    Example from user: SL 3.1, Pullback 4.0, TP 10.
    Without buffer: Pullback(4.0) >= SL(3.1) => LOSS
    With +1.0 buffer: effective SL = 4.1, Pullback(4.0) < 4.1 AND TP(10) >= 4.1 => WIN
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [3.1],
        'Pullback': [4.0],
        'TP': [10.0],
    })

    # Without buffer: loss
    stats_no_buffer = _calculate_stats(trades, 'Test')
    assert stats_no_buffer['Notation'] == '0W – 1L'

    # With +0.5 buffer: effective SL = 3.6, Pullback(4.0) >= 3.6 => still LOSS
    stats_05 = _calculate_stats_with_buffer(trades, 'Test', 0.5)
    assert stats_05['Notation'] == '0W – 1L'

    # With +1.0 buffer: effective SL = 4.1, Pullback(4.0) < 4.1 AND TP(10) >= 4.1 => WIN
    stats_10 = _calculate_stats_with_buffer(trades, 'Test', 1.0)
    assert stats_10['Notation'] == '1W – 0L'


def test_buffer_tp_must_reach_effective_sl():
    """Test that TP must reach the effective SL (SL + buffer) to win."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [3.0],
        'Pullback': [1.0],
        'TP': [5.0],
    })

    # With +3.0 buffer: effective SL = 6.0, TP(5.0) < 6.0 => LOSS (TP doesn't reach target)
    stats = _calculate_stats_with_buffer(trades, 'Test', 3.0)
    assert stats['Notation'] == '0W – 1L'

    # With +2.0 buffer: effective SL = 5.0, TP(5.0) >= 5.0 => WIN
    stats = _calculate_stats_with_buffer(trades, 'Test', 2.0)
    assert stats['Notation'] == '1W – 0L'


def test_buffer_stats_has_buffer_column():
    """Test that buffer stats include the Buffer column."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [5.0],
        'Pullback': [2.0],
        'TP': [10.0],
    })

    stats = _calculate_stats_with_buffer(trades, 'Test', 1.5)
    assert stats['Buffer'] == '+1.5'


def test_buffer_stats_empty():
    """Test buffer stats with empty dataset."""
    empty = get_empty_data()
    stats = _calculate_stats_with_buffer(empty, 'Empty', 1.0)

    assert stats['Buffer'] == '+1.0'
    assert stats['Notation'] == '0W – 0L'
    assert stats['Pass'] == 0
    assert stats['Fail'] == 0


def test_get_buffer_strategies():
    """Test that buffer strategies include key strategies."""
    strategies = get_buffer_strategies()
    names = [name for name, _ in strategies]

    assert 'All Trades' in names
    assert '1H Aligned' in names
    assert '1H Against' in names


# def test_get_buffer_strategies_includes_sl_caps():
#     """Test that buffer strategies include SL cap variations."""
#     strategies = get_buffer_strategies()
#     names = [name for name, _ in strategies]
#
#     assert 'All Trades + SL < 3' in names
#     assert 'All Trades + SL < 4' in names
#     assert 'All Trades + SL < 5' in names
#     assert '1H Aligned + SL < 3' in names
#     assert '1H Against + SL < 5' in names
#
#
# def test_buffer_sl_cap_filter():
#     """Test that SL cap filter correctly excludes trades with SL >= cap."""
#     sample = get_sample_data()
#     strategies = get_buffer_strategies()
#     strategy = [func for name, func in strategies if name == 'All Trades + SL < 3'][0]
#     filtered = strategy(sample)
#
#     for _, row in filtered.iterrows():
#         assert row['SL'] < 3
#
#
# def test_buffer_sl_cap_with_1h_filter():
#     """Test SL cap combined with 1H filter."""
#     sample = get_sample_data()
#     strategies = get_buffer_strategies()
#     strategy = [func for name, func in strategies if name == '1H Aligned + SL < 4'][0]
#     filtered = strategy(sample)
#
#     for _, row in filtered.iterrows():
#         assert row['SL'] < 4
#         assert row['Direction'] == row['1H']


def test_calculate_buffer_statistics():
    """Test that buffer statistics returns expected DataFrame."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)

    assert isinstance(result, pd.DataFrame)


def test_calculate_buffer_statistics_has_pass_column():
    """Test that buffer statistics includes Pass and Drawdown columns."""
    # Need enough trades to produce passes
    trades = pd.DataFrame({
        'Date': [f'2026-01-{i+1:02d}' for i in range(20)],
        'Weekday': ['Monday'] * 20,
        'Trade': [f'#{i+1}' for i in range(20)],
        'Direction': ['Buy'] * 20,
        '1H': ['Buy'] * 20,
        'SL': [2.0] * 20,
        'Pullback': [1.0] * 20,
        'TP': [10.0] * 20,
        'R': [5.0] * 20,
    })
    result = calculate_buffer_statistics(trades)

    assert 'Pass' in result.columns
    assert 'Fail' in result.columns
    # All rows should have Pass > 0 (filtered)
    for _, row in result.iterrows():
        assert row['Pass'] > 0


def test_buffer_pips_constant():
    """Test that BUFFER_PIPS has expected values."""
    assert BUFFER_PIPS == [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def test_rrr_ratios_constant():
    """Test that RRR_RATIOS has expected values."""
    assert RRR_RATIOS == [1, 2]


def test_breakeven_rate():
    """Test breakeven rate calculation for different RRR ratios."""
    assert _breakeven_rate(1) == 50.0
    assert abs(_breakeven_rate(2) - 33.333) < 0.01


def test_1_2_rrr_win_condition():
    """Test win condition at 1:2 RRR: Pullback < SL AND TP >= 2 * SL."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [5.0, 5.0, 5.0],
        'Pullback': [2.0, 2.0, 2.0],
        'TP': [9.9, 10.0, 15.0],
    })

    stats = _calculate_stats(trades, 'Test', rrr_ratio=2)

    # Trade 1: TP(9.9) < 2*SL(10) => LOSS
    # Trade 2: TP(10.0) >= 2*SL(10) => WIN
    # Trade 3: TP(15.0) >= 2*SL(10) => WIN
    assert stats['Notation'] == '2W – 1L'
    assert stats['RRR'] == '1:2'


def test_1_2_rrr_edge_calculation():
    """Test edge calculation at 1:2 RRR: edge = win_rate - 33.3%."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [5.0, 5.0, 5.0],
        'Pullback': [2.0, 2.0, 6.0],
        'TP': [10.0, 10.0, 10.0],
    })

    stats = _calculate_stats(trades, 'Test', rrr_ratio=2)

    # 2 wins out of 3 = 66.7% win rate
    # Breakeven at 1:2 = 33.3%
    # Edge = 66.7% - 33.3% = 33.3%
    assert stats['Win Rate'] == '66.7%'
    assert abs(stats['edge_value'] - 33.3) < 0.1


def test_1_2_rrr_outcome():
    """Test outcome at 1:2 RRR: wins * 2 - losses."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [5.0, 5.0, 5.0],
        'Pullback': [2.0, 2.0, 6.0],
        'TP': [10.0, 10.0, 10.0],
    })

    stats = _calculate_stats(trades, 'Test', rrr_ratio=2)

    # 2 wins * 2R - 1 loss = 3R
    assert stats['Outcome'] == '3R'


def test_1_2_rrr_buffer():
    """Test 1:2 RRR with buffer: TP must reach 2 * effective_sl."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [3.0],
        'Pullback': [1.0],
        'TP': [8.0],
    })

    # Buffer +1.0: effective SL = 4.0, target = 2 * 4.0 = 8.0, TP(8.0) >= 8.0 => WIN
    stats = _calculate_stats_with_buffer(trades, 'Test', 1.0, rrr_ratio=2)
    assert stats['Notation'] == '1W – 0L'
    assert stats['RRR'] == '1:2'

    # Buffer +1.5: effective SL = 4.5, target = 2 * 4.5 = 9.0, TP(8.0) < 9.0 => LOSS
    stats = _calculate_stats_with_buffer(trades, 'Test', 1.5, rrr_ratio=2)
    assert stats['Notation'] == '0W – 1L'


def test_1_2_rrr_empty():
    """Test 1:2 RRR with empty dataset."""
    empty = get_empty_data()
    stats = _calculate_stats(empty, 'Empty', rrr_ratio=2)

    assert stats['Trades'] == 0
    assert stats['RRR'] == '1:2'
    assert stats['Edge'] == '-33.3%'


def test_fixed_sl_sizes_constant():
    """Test that FIXED_SL_SIZES has expected values."""
    assert FIXED_SL_SIZES == [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def test_fixed_sl_trade_survives():
    """Test fixed SL: trade survives when Pullback < fixed SL.

    SL=3, Pullback=1, TP=10. With fixed SL=1.5: Pullback(1) < 1.5 AND TP(10) >= 1.5 => WIN.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [3.0],
        'Pullback': [1.0],
        'TP': [10.0],
    })

    stats = _calculate_fixed_sl_stats(trades, 1.5)
    assert stats['Notation'] == '1W – 0L'


def test_fixed_sl_trade_loses_deep_pullback():
    """Test fixed SL: trade loses when Pullback >= fixed SL.

    SL=3, Pullback=2.5, TP=10. With fixed SL=2.0: Pullback(2.5) >= 2.0 => LOSS.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [3.0],
        'Pullback': [2.5],
        'TP': [10.0],
    })

    stats = _calculate_fixed_sl_stats(trades, 2.0)
    assert stats['Notation'] == '0W – 1L'


def test_fixed_sl_ignores_original_sl():
    """Test that fixed SL ignores the original SL value entirely.

    Original SL=10 (large), but fixed SL=2.0 is used.
    Pullback=1.5 < 2.0 => survives. TP=5 >= 2.0 => WIN.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [10.0],
        'Pullback': [1.5],
        'TP': [5.0],
    })

    stats = _calculate_fixed_sl_stats(trades, 2.0)
    assert stats['Notation'] == '1W – 0L'


def test_fixed_sl_tp_must_reach_target():
    """Test that TP must reach RRR * fixed_sl to win.

    Pullback=1.0 < fixed_sl=3.0 => survives. TP=2.5 < 3.0 => LOSS.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [5.0],
        'Pullback': [1.0],
        'TP': [2.5],
    })

    stats = _calculate_fixed_sl_stats(trades, 3.0)
    assert stats['Notation'] == '0W – 1L'


def test_fixed_sl_1_2_rrr():
    """Test fixed SL at 1:2 RRR: TP must reach 2 * fixed_sl.

    Fixed SL=2.0, Pullback=1.0, TP=4.0.
    At 1:2: TP(4.0) >= 2*2.0=4.0 => WIN.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [5.0],
        'Pullback': [1.0],
        'TP': [4.0],
    })

    stats = _calculate_fixed_sl_stats(trades, 2.0, rrr_ratio=2)
    assert stats['RRR'] == '1:2'
    assert stats['Notation'] == '1W – 0L'

    # TP=3.9 < 4.0 => LOSS
    trades2 = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [5.0],
        'Pullback': [1.0],
        'TP': [3.9],
    })

    stats2 = _calculate_fixed_sl_stats(trades2, 2.0, rrr_ratio=2)
    assert stats2['Notation'] == '0W – 1L'


def test_fixed_sl_empty():
    """Test fixed SL with empty dataset."""
    empty = get_empty_data()
    stats = _calculate_fixed_sl_stats(empty, 2.0)
    assert stats['Trades'] == 0
    assert stats['Notation'] == '0W – 0L'
    assert stats['Fixed SL'] == '2.0'


def test_fixed_sl_has_fixed_sl_column():
    """Test that fixed SL stats include the Fixed SL column."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [5.0],
        'Pullback': [2.0],
        'TP': [10.0],
    })

    stats = _calculate_fixed_sl_stats(trades, 3.5)
    assert stats['Fixed SL'] == '3.5'


def test_fixed_sl_mixed_trades():
    """Test fixed SL with mix of wins and losses.

    Fixed SL=2.0:
    - Trade 1: PB=1.0 < 2.0, TP=10 >= 2.0 => WIN
    - Trade 2: PB=2.5 >= 2.0 => LOSS
    - Trade 3: PB=0.5 < 2.0, TP=5 >= 2.0 => WIN
    - Trade 4: PB=1.8 < 2.0, TP=1.5 < 2.0 => LOSS (TP too low)
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
        'SL': [3.0, 3.0, 4.0, 5.0],
        'Pullback': [1.0, 2.5, 0.5, 1.8],
        'TP': [10.0, 10.0, 5.0, 1.5],
    })

    stats = _calculate_fixed_sl_stats(trades, 2.0)
    assert stats['Notation'] == '2W – 2L'


def test_calculate_fixed_sl_statistics():
    """Test that calculate_fixed_sl_statistics returns a DataFrame."""
    sample = get_sample_data()
    result = calculate_fixed_sl_statistics(sample)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_calculate_fixed_sl_statistics_has_both_rrr():
    """Test that fixed SL statistics only includes RRR values with positive edge."""
    sample = get_sample_data()
    result = calculate_fixed_sl_statistics(sample)
    rrr_values = result['RRR'].unique()
    for rrr in rrr_values:
        assert rrr in ['1:1', '1:2'], f"Unexpected RRR: {rrr}"


def test_calculate_fixed_sl_statistics_total_rows():
    """Test that fixed SL statistics only contains positive edge rows."""
    sample = get_sample_data()
    result = calculate_fixed_sl_statistics(sample)
    # All rows must have positive edge
    for _, row in result.iterrows():
        edge_val = float(str(row['Edge']).replace('%', ''))
        assert edge_val > 0, f"Non-positive edge found: {row['Edge']}"


def test_calculate_fixed_sl_statistics_columns():
    """Test that fixed SL result has expected columns."""
    sample = get_sample_data()
    result = calculate_fixed_sl_statistics(sample)
    columns = list(result.columns)
    assert columns[0] == 'Fixed SL'
    assert columns[1] == 'RRR'
    assert columns[2].startswith('Trades (')


def test_fixed_sl_with_strategy_has_strategy_column():
    """Test that fixed SL with strategy includes the Strategy column."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [5.0],
        'Pullback': [2.0],
        'TP': [10.0],
    })

    stats = _calculate_fixed_sl_stats_with_strategy(trades, 'Test Strategy', 3.0)
    assert stats['Strategy'] == 'Test Strategy'
    assert stats['Fixed SL'] == '3.0'


def test_fixed_sl_with_strategy_empty():
    """Test fixed SL with strategy on empty data."""
    empty = get_empty_data()
    stats = _calculate_fixed_sl_stats_with_strategy(empty, 'Empty', 2.0)
    assert stats['Trades'] == 0
    assert stats['Strategy'] == 'Empty'


def test_get_1h_strategies():
    """Test 1H strategy list."""
    strategies = _get_1h_strategies()
    names = [name for name, _ in strategies]
    assert 'All Trades' in names
    assert '1H Aligned' in names
    assert '1H Against' in names
    assert len(strategies) == 3


def test_1h_strategy_filters_correctly():
    """Test that 1H strategy filters produce correct subsets."""
    sample = get_sample_data()
    strategies = _get_1h_strategies()

    aligned_func = [func for name, func in strategies if name == '1H Aligned'][0]
    against_func = [func for name, func in strategies if name == '1H Against'][0]

    aligned = aligned_func(sample)
    against = against_func(sample)

    for _, row in aligned.iterrows():
        assert row['Direction'] == row['1H']

    for _, row in against.iterrows():
        assert row['Direction'] != row['1H']

    assert len(aligned) + len(against) == len(sample)


def test_calculate_fixed_sl_1h_statistics():
    """Test that 1H fixed SL statistics returns a DataFrame."""
    sample = get_sample_data()
    result = calculate_fixed_sl_1h_statistics(sample)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_fixed_sl_1h_total_rows():
    """Test that only positive edge rows are returned."""
    sample = get_sample_data()
    result = calculate_fixed_sl_1h_statistics(sample)
    for _, row in result.iterrows():
        edge_val = float(str(row['Edge']).replace('%', ''))
        assert edge_val > 0, f"Non-positive edge found: {row['Edge']}"


def test_fixed_sl_1h_has_strategy_column():
    """Test that 1H fixed SL result has Strategy column."""
    sample = get_sample_data()
    result = calculate_fixed_sl_1h_statistics(sample)
    columns = list(result.columns)
    assert columns[0] == 'Strategy'
    assert columns[1] == 'Fixed SL'
    assert columns[2] == 'RRR'


def test_fixed_sl_1h_strategies_present():
    """Test that only strategies with positive edge appear in results."""
    sample = get_sample_data()
    result = calculate_fixed_sl_1h_statistics(sample)
    strategies = result['Strategy'].unique()
    for s in strategies:
        assert s in ['All Trades', '1H Aligned', '1H Against'], f"Unexpected strategy: {s}"


def test_fixed_sl_1h_aligned_wins_more():
    """Test fixed SL with 1H alignment on controlled data.

    All trades are Buy with 1H=Buy, so Aligned = all trades, Against = 0 trades.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Direction': ['Buy', 'Buy'],
        '1H': ['Buy', 'Buy'],
        'SL': [5.0, 5.0],
        'Pullback': [1.0, 1.0],
        'TP': [10.0, 10.0],
    })

    result = calculate_fixed_sl_1h_statistics(trades)

    # 1H Aligned at fixed SL=2.0, 1:1 should have 2 trades (positive edge)
    aligned_rows = result[(result['Strategy'] == '1H Aligned') & (result['Fixed SL'] == '2.0') & (result['RRR'] == '1:1')]
    assert len(aligned_rows) == 1
    assert aligned_rows.iloc[0]['Notation'] == '2W – 0L'

    # 1H Against with 0 trades has negative edge, so it's filtered out
    against_rows = result[(result['Strategy'] == '1H Against') & (result['Fixed SL'] == '2.0') & (result['RRR'] == '1:1')]
    assert len(against_rows) == 0


def test_simulate_challenge_all_wins_1_1():
    """Test simulate_challenge: 10 consecutive wins at 1:1 = 1 pass."""
    mask = [True] * 10
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 1
    assert drawdowns == 0


def test_simulate_challenge_all_losses():
    """Test simulate_challenge: 10 consecutive losses = 1 drawdown."""
    mask = [False] * 10
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 0
    assert drawdowns == 1


def test_simulate_challenge_20_wins():
    """Test simulate_challenge: 20 wins at 1:1 = 2 passes."""
    mask = [True] * 20
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 2
    assert drawdowns == 0


def test_simulate_challenge_20_losses():
    """Test simulate_challenge: 20 losses = 2 drawdowns."""
    mask = [False] * 20
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 0
    assert drawdowns == 2


def test_simulate_challenge_reset_after_pass():
    """Test that sum resets to 0 after a pass, then losses start fresh."""
    # 10 wins = pass (reset to 0), then 10 losses = drawdown
    mask = [True] * 10 + [False] * 10
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 1
    assert drawdowns == 1


def test_simulate_challenge_reset_after_drawdown():
    """Test that sum resets to 0 after a drawdown, then wins start fresh."""
    # 10 losses = drawdown (reset to 0), then 10 wins = pass
    mask = [False] * 10 + [True] * 10
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 1
    assert drawdowns == 1


def test_simulate_challenge_1_2_rrr():
    """Test simulate_challenge at 1:2 RRR: each win = +2R, so 5 wins = pass."""
    mask = [True] * 5
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=2)
    assert passes == 1
    assert drawdowns == 0


def test_simulate_challenge_1_2_rrr_10_wins():
    """Test simulate_challenge at 1:2 RRR: 10 wins = +20R = 2 passes."""
    mask = [True] * 10
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=2)
    assert passes == 2
    assert drawdowns == 0


def test_simulate_challenge_empty():
    """Test simulate_challenge with no trades."""
    passes, drawdowns = simulate_challenge([], rrr_ratio=1)
    assert passes == 0
    assert drawdowns == 0


def test_simulate_challenge_mixed_no_threshold():
    """Test simulate_challenge: alternating wins/losses never reaches threshold."""
    # Win, Loss, Win, Loss... at 1:1 → sum oscillates between 0 and 1
    mask = [True, False] * 5
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 0
    assert drawdowns == 0


def test_simulate_challenge_near_threshold():
    """Test that reaching exactly the threshold triggers pass/drawdown."""
    # 9 wins at 1:1 = 9R (not enough), 10th win = 10R = pass
    mask = [True] * 9
    passes, _ = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 0

    mask = [True] * 10
    passes, _ = simulate_challenge(mask, rrr_ratio=1)
    assert passes == 1


def test_buffer_stats_has_challenge_columns():
    """Test that _calculate_stats_with_buffer returns Pass and Drawdown."""
    trades = pd.DataFrame({
        'Date': [f'2026-01-{i+1:02d}' for i in range(10)],
        'SL': [2.0] * 10,
        'Pullback': [1.0] * 10,
        'TP': [10.0] * 10,
    })

    stats = _calculate_stats_with_buffer(trades, 'Test', 0.0)
    assert 'Pass' in stats
    assert 'Fail' in stats
    assert stats['Pass'] == 1  # 10 wins at 1:1 = 1 pass
    assert stats['Fail'] == 0


def test_buffer_stats_has_trades_column():
    """Test that _calculate_stats_with_buffer returns Trades count."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02', '2026-01-03'],
        'SL': [2.0, 2.0, 2.0],
        'Pullback': [1.0, 1.0, 3.0],
        'TP': [10.0, 10.0, 10.0],
    })

    stats = _calculate_stats_with_buffer(trades, 'Test', 0.0)
    assert stats['Trades'] == 3


def test_buffer_stats_empty_has_trades_zero():
    """Test that empty _calculate_stats_with_buffer returns Trades 0."""
    empty = get_empty_data()
    stats = _calculate_stats_with_buffer(empty, 'Empty', 1.0)
    assert stats['Trades'] == 0


def test_simulate_challenge_float_rrr():
    """Test simulate_challenge with float RRR (e.g., 3.5)."""
    # 3 wins at 1:3.5 = +10.5R => 1 pass
    mask = [True] * 3
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=3.5)
    assert passes == 1
    assert drawdowns == 0


def test_simulate_challenge_float_rrr_not_enough():
    """Test simulate_challenge with float RRR: 2 wins at 1:4.5 = +9R, not enough."""
    mask = [True] * 2
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=4.5)
    assert passes == 0
    assert drawdowns == 0


def test_simulate_challenge_float_rrr_10():
    """Test simulate_challenge at 1:10 RRR: 1 win = +10R = 1 pass."""
    mask = [True]
    passes, drawdowns = simulate_challenge(mask, rrr_ratio=10.0)
    assert passes == 1
    assert drawdowns == 0


def test_buffer_stats_with_float_rrr():
    """Test _calculate_stats_with_buffer with float RRR (3.5)."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [2.0],
        'Pullback': [1.0],
        'TP': [10.0],
    })

    stats = _calculate_stats_with_buffer(trades, 'Test', 0.0, rrr_ratio=3.5)
    assert stats['RRR'] == '1:3.5'
    assert stats['Notation'] == '1W – 0L'


def test_buffer_stats_with_float_rrr_loss():
    """Test float RRR where TP doesn't reach target: SL=2, TP=6, RRR=3.5 => need 7."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'SL': [2.0],
        'Pullback': [1.0],
        'TP': [6.0],
    })

    stats = _calculate_stats_with_buffer(trades, 'Test', 0.0, rrr_ratio=3.5)
    assert stats['Notation'] == '0W – 1L'


def test_format_wl():
    """Test _format_wl output format."""
    assert _format_wl(3, 1, 4) == '3W - 1L (75.0%)'
    assert _format_wl(0, 0, 0) == '0W - 0L (0.0%)'
    assert _format_wl(1, 2, 3) == '1W - 2L (33.3%)'


def test_weekday_order_constant():
    """Test that WEEKDAY_ORDER has Monday-Friday."""
    assert WEEKDAY_ORDER == ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def test_weekday_statistics_columns():
    """Test that weekday statistics has expected columns."""
    sample = get_sample_data()
    result = calculate_weekday_statistics(sample)
    assert list(result.columns) == ['Day', 'Trades', 'Regular', 'With 2 pips buffer']


def test_weekday_statistics_all_days_present():
    """Test that all 5 weekdays appear in results."""
    sample = get_sample_data()
    result = calculate_weekday_statistics(sample)
    assert len(result) == 5
    assert list(result['Day']) == ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def test_weekday_statistics_trade_counts():
    """Test trade counts per weekday from sample data."""
    sample = get_sample_data()
    result = calculate_weekday_statistics(sample)
    # Sample: Monday=3, Tuesday=2, Wednesday=2, Thursday=2, Friday=1
    counts = dict(zip(result['Day'], result['Trades']))
    assert counts['Monday'] == 3
    assert counts['Tuesday'] == 2
    assert counts['Wednesday'] == 2
    assert counts['Thursday'] == 2
    assert counts['Friday'] == 1


def test_weekday_statistics_regular():
    """Test Regular column from sample data.

    Win condition: Pullback < SL AND TP > 0.
    Monday: #1 PB=3.5 SL=3.5 TP=0 => L, #2 PB=0.8 SL=1.1 TP=12 => W, #3 PB=2.1 SL=2.0 TP=0 => L
    Tuesday: #1 PB=1.5 SL=4.0 TP=10 => W, #2 PB=3.0 SL=3.0 TP=0 => L
    Thursday: #1 PB=3.0 SL=6.0 TP=15 => W, #2 PB=7.0 SL=8.0 TP=10 => W
    Friday: #1 PB=0.5 SL=1.5 TP=5 => W
    """
    sample = get_sample_data()
    result = calculate_weekday_statistics(sample)
    rows = {row['Day']: row for _, row in result.iterrows()}

    assert rows['Monday']['Regular'] == '1W - 2L (33.3%)'
    assert rows['Tuesday']['Regular'] == '1W - 1L (50.0%)'
    assert rows['Thursday']['Regular'] == '2W - 0L (100.0%)'
    assert rows['Friday']['Regular'] == '1W - 0L (100.0%)'


def test_weekday_statistics_buffer():
    """Test With 2 pips buffer column from sample data.

    Buffer win: Pullback < SL + 2 AND TP > 0.
    Monday: #1 PB=3.5 SL=3.5 TP=0 => L (TP=0), #2 PB=0.8 SL=1.1 TP=12 => W (0.8<3.1),
            #3 PB=2.1 SL=2.0 TP=0 => L (TP=0)
    Tuesday: #1 PB=1.5 SL=4.0 TP=10 => W, #2 PB=3.0 SL=3.0 TP=0 => L (TP=0)
    Wednesday: #1 PB=2.0 SL=5.0 TP=8 => W, #2 PB=2.5 SL=2.5 TP=0 => L (TP=0)
    Thursday: #1 PB=3.0 SL=6.0 TP=15 => W, #2 PB=7.0 SL=8.0 TP=10 => W
    Friday: #1 PB=0.5 SL=1.5 TP=5 => W
    """
    sample = get_sample_data()
    result = calculate_weekday_statistics(sample)
    rows = {row['Day']: row for _, row in result.iterrows()}

    # Monday: same as regular since losses are due to TP=0, not pullback
    assert rows['Monday']['With 2 pips buffer'] == '1W - 2L (33.3%)'
    assert rows['Thursday']['With 2 pips buffer'] == '2W - 0L (100.0%)'


def test_weekday_statistics_buffer_saves_trade():
    """Test that buffer can save a trade that would otherwise lose.

    SL=3.0, Pullback=4.0, TP=10.0.
    Regular: PB(4) >= SL(3) => L.
    Buffer: PB(4) < SL+2(5) AND TP(10) > 0 => W.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-12'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        '1H': ['Buy'],
        'SL': [3.0],
        'Pullback': [4.0],
        'TP': [10.0],
        'R': [0],
    })

    result = calculate_weekday_statistics(trades)
    rows = {row['Day']: row for _, row in result.iterrows()}

    assert rows['Monday']['Regular'] == '0W - 1L (0.0%)'
    assert rows['Monday']['With 2 pips buffer'] == '1W - 0L (100.0%)'


def test_weekday_statistics_empty():
    """Test weekday statistics with empty dataset."""
    empty = get_empty_data()
    result = calculate_weekday_statistics(empty)
    assert len(result) == 5
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Regular'] == '0W - 0L (0.0%)'
        assert row['With 2 pips buffer'] == '0W - 0L (0.0%)'


def test_weekday_statistics_single_day():
    """Test weekday statistics when only one day has trades."""
    trades = pd.DataFrame({
        'Date': ['2026-01-12', '2026-01-12'],
        'Weekday': ['Monday', 'Monday'],
        'Trade': ['#1', '#2'],
        'Direction': ['Buy', 'Buy'],
        '1H': ['Buy', 'Buy'],
        'SL': [3.0, 3.0],
        'Pullback': [1.0, 4.0],
        'TP': [5.0, 5.0],
        'R': [1.7, 0],
    })

    result = calculate_weekday_statistics(trades)
    rows = {row['Day']: row for _, row in result.iterrows()}

    assert rows['Monday']['Trades'] == 2
    assert rows['Monday']['Regular'] == '1W - 1L (50.0%)'
    # PB=4 < SL+2=5 => buffer saves it
    assert rows['Monday']['With 2 pips buffer'] == '2W - 0L (100.0%)'
    assert rows['Tuesday']['Trades'] == 0


def test_sl_ranges_constant():
    """Test that SL_RANGES has expected ranges."""
    assert len(SL_RANGES) == 5
    assert SL_RANGES[0] == ("0-3", 0, 3)
    assert SL_RANGES[1] == ("3-5", 3, 5)
    assert SL_RANGES[2] == ("< 5", 0, 5)
    assert SL_RANGES[3] == ("5-10", 5, 10)
    assert SL_RANGES[4][0] == "10+"
    assert SL_RANGES[4][1] == 10


def test_sl_statistics_columns():
    """Test that SL statistics has expected columns."""
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    assert list(result.columns) == ['SL Range', 'Trades', 'Regular', 'With 2 pips buffer']


def test_sl_statistics_all_ranges_present():
    """Test that all 4 SL ranges appear in results."""
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    assert len(result) == 5
    assert list(result['SL Range']) == ['0-3', '3-5', '< 5', '5-10', '10+']


def test_sl_statistics_trade_counts():
    """Test trade counts per SL range from sample data.

    Sample SL values: 3.5, 1.1, 2.0, 4.0, 3.0, 5.0, 2.5, 6.0, 8.0, 1.5
    0-3 (SL>=0, SL<3): 1.1, 2.0, 2.5, 1.5 = 4
    3-5 (SL>=3, SL<5): 3.5, 4.0, 3.0 = 3
    5-10 (SL>=5, SL<10): 5.0, 6.0, 8.0 = 3
    10+: 0
    """
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    counts = dict(zip(result['SL Range'], result['Trades']))
    assert counts['0-3'] == 4
    assert counts['3-5'] == 3
    assert counts['5-10'] == 3
    assert counts['10+'] == 0


def test_sl_statistics_regular():
    """Test Regular column per SL range from sample data.

    Win condition: Pullback < SL AND TP > 0.
    0-3: SL=1.1 PB=0.8 TP=12 => W, SL=2.0 PB=2.1 TP=0 => L, SL=2.5 PB=2.5 TP=0 => L, SL=1.5 PB=0.5 TP=5 => W
    3-5: SL=3.5 PB=3.5 TP=0 => L, SL=4.0 PB=1.5 TP=10 => W, SL=3.0 PB=3.0 TP=0 => L
    5-10: SL=5.0 PB=2.0 TP=8 => W, SL=6.0 PB=3.0 TP=15 => W, SL=8.0 PB=7.0 TP=10 => W
    """
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    assert rows['0-3']['Regular'] == '2W - 2L (50.0%)'
    assert rows['3-5']['Regular'] == '1W - 2L (33.3%)'
    assert rows['5-10']['Regular'] == '3W - 0L (100.0%)'


def test_sl_statistics_buffer():
    """Test With 2 pips buffer column per SL range from sample data.

    Buffer win: Pullback < SL + 2 AND TP > 0.
    0-3: SL=1.1 PB=0.8 TP=12 => W, SL=2.0 PB=2.1 TP=0 => L(TP=0), SL=2.5 PB=2.5 TP=0 => L(TP=0), SL=1.5 PB=0.5 TP=5 => W
    3-5: SL=3.5 PB=3.5 TP=0 => L(TP=0), SL=4.0 PB=1.5 TP=10 => W, SL=3.0 PB=3.0 TP=0 => L(TP=0)
    5-10: SL=5.0 PB=2.0 TP=8 => W, SL=6.0 PB=3.0 TP=15 => W, SL=8.0 PB=7.0 TP=10 => W
    """
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    # 0-3: same wins since losses are TP=0
    assert rows['0-3']['With 2 pips buffer'] == '2W - 2L (50.0%)'
    # 3-5: same since losses are TP=0
    assert rows['3-5']['With 2 pips buffer'] == '1W - 2L (33.3%)'
    assert rows['5-10']['With 2 pips buffer'] == '3W - 0L (100.0%)'


def test_sl_statistics_empty():
    """Test SL statistics with empty dataset."""
    empty = get_empty_data()
    result = calculate_sl_statistics(empty)
    assert len(result) == 5
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Regular'] == '0W - 0L (0.0%)'
        assert row['With 2 pips buffer'] == '0W - 0L (0.0%)'


def test_sl_statistics_large_sl():
    """Test SL statistics with trades in the 10+ range."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Monday'],
        'Trade': ['#1', '#2'],
        'Direction': ['Buy', 'Buy'],
        '1H': ['Buy', 'Buy'],
        'SL': [12.0, 15.0],
        'Pullback': [5.0, 16.0],
        'TP': [20.0, 20.0],
        'R': [1.7, 0],
    })

    result = calculate_sl_statistics(trades)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    assert rows['10+']['Trades'] == 2
    assert rows['10+']['Regular'] == '1W - 1L (50.0%)'
    # PB=16 < SL+2=17 => buffer saves it, PB=5 < SL+2=14 => still win
    assert rows['10+']['With 2 pips buffer'] == '2W - 0L (100.0%)'
    assert rows['0-3']['Trades'] == 0


def test_sl_statistics_buffer_saves_trade():
    """Test that buffer can save a trade in SL range analysis.

    SL=2.0, PB=3.0, TP=10. Regular: PB(3) >= SL(2) => L. Buffer: PB(3) < SL+2(4) => W.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        '1H': ['Buy'],
        'SL': [2.0],
        'Pullback': [3.0],
        'TP': [10.0],
        'R': [0],
    })

    result = calculate_sl_statistics(trades)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    assert rows['0-3']['Regular'] == '0W - 1L (0.0%)'
    assert rows['0-3']['With 2 pips buffer'] == '1W - 0L (100.0%)'


def test_pullback_ranges_constant():
    """Test that PULLBACK_RANGES has expected ranges."""
    assert len(PULLBACK_RANGES) == 4
    assert PULLBACK_RANGES[0] == ("0-3", 0, 3)
    assert PULLBACK_RANGES[1] == ("3-5", 3, 5)
    assert PULLBACK_RANGES[2] == ("5-10", 5, 10)
    assert PULLBACK_RANGES[3][0] == "10+"
    assert PULLBACK_RANGES[3][1] == 10


def test_pullback_statistics_columns():
    """Test that Pullback statistics has expected columns."""
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    assert list(result.columns) == ['Pullback Range', 'Trades', 'Regular', 'With 2 pips buffer']


def test_pullback_statistics_all_ranges_present():
    """Test that all 4 Pullback ranges appear in results."""
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    assert len(result) == 4
    assert list(result['Pullback Range']) == ['0-3', '3-5', '5-10', '10+']


def test_pullback_statistics_trade_counts():
    """Test trade counts per Pullback range from sample data.

    Sample Pullback values: 3.5, 0.8, 2.1, 1.5, 3.0, 2.0, 2.5, 3.0, 7.0, 0.5
    0-3 (PB>=0, PB<3): 0.8, 2.1, 1.5, 2.0, 2.5, 0.5 = 6
    3-5 (PB>=3, PB<5): 3.5, 3.0, 3.0 = 3
    5-10 (PB>=5, PB<10): 7.0 = 1
    10+: 0
    """
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    counts = dict(zip(result['Pullback Range'], result['Trades']))
    assert counts['0-3'] == 6
    assert counts['3-5'] == 3
    assert counts['5-10'] == 1
    assert counts['10+'] == 0


def test_pullback_statistics_regular():
    """Test Regular column per Pullback range from sample data.

    Win condition: Pullback < SL AND TP > 0.
    0-3: PB=0.8 SL=1.1 TP=12 => W, PB=2.1 SL=2.0 TP=0 => L, PB=1.5 SL=4.0 TP=10 => W,
         PB=2.0 SL=5.0 TP=8 => W, PB=2.5 SL=2.5 TP=0 => L, PB=0.5 SL=1.5 TP=5 => W
    3-5: PB=3.5 SL=3.5 TP=0 => L, PB=3.0 SL=3.0 TP=0 => L, PB=3.0 SL=6.0 TP=15 => W
    5-10: PB=7.0 SL=8.0 TP=10 => W
    """
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    rows = {row['Pullback Range']: row for _, row in result.iterrows()}

    assert rows['0-3']['Regular'] == '4W - 2L (66.7%)'
    assert rows['3-5']['Regular'] == '1W - 2L (33.3%)'
    assert rows['5-10']['Regular'] == '1W - 0L (100.0%)'


def test_pullback_statistics_buffer():
    """Test With 2 pips buffer column per Pullback range from sample data.

    Buffer win: Pullback < SL + 2 AND TP > 0.
    0-3: PB=0.8 SL=1.1 TP=12 => W, PB=2.1 SL=2.0 TP=0 => L(TP=0), PB=1.5 SL=4.0 TP=10 => W,
         PB=2.0 SL=5.0 TP=8 => W, PB=2.5 SL=2.5 TP=0 => L(TP=0), PB=0.5 SL=1.5 TP=5 => W
    3-5: PB=3.5 SL=3.5 TP=0 => L(TP=0), PB=3.0 SL=3.0 TP=0 => L(TP=0), PB=3.0 SL=6.0 TP=15 => W
    5-10: PB=7.0 SL=8.0 TP=10 => W
    """
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    rows = {row['Pullback Range']: row for _, row in result.iterrows()}

    # 0-3: same wins since losses are TP=0
    assert rows['0-3']['With 2 pips buffer'] == '4W - 2L (66.7%)'
    # 3-5: same since losses are TP=0
    assert rows['3-5']['With 2 pips buffer'] == '1W - 2L (33.3%)'
    assert rows['5-10']['With 2 pips buffer'] == '1W - 0L (100.0%)'


def test_pullback_statistics_empty():
    """Test Pullback statistics with empty dataset."""
    empty = get_empty_data()
    result = calculate_pullback_statistics(empty)
    assert len(result) == 4
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Regular'] == '0W - 0L (0.0%)'
        assert row['With 2 pips buffer'] == '0W - 0L (0.0%)'


def test_pullback_statistics_large_pullback():
    """Test Pullback statistics with trades in the 10+ range."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Monday'],
        'Trade': ['#1', '#2'],
        'Direction': ['Buy', 'Buy'],
        '1H': ['Buy', 'Buy'],
        'SL': [15.0, 12.0],
        'Pullback': [12.0, 14.0],
        'TP': [20.0, 20.0],
        'R': [1.3, 0],
    })

    result = calculate_pullback_statistics(trades)
    rows = {row['Pullback Range']: row for _, row in result.iterrows()}

    assert rows['10+']['Trades'] == 2
    # PB=12 < SL=15 TP=20 => W, PB=14 >= SL=12 TP=20 => L
    assert rows['10+']['Regular'] == '1W - 1L (50.0%)'
    # PB=12 < SL+2=17 => W, PB=14 >= SL+2=14 => L (not strictly less)
    assert rows['10+']['With 2 pips buffer'] == '1W - 1L (50.0%)'
    assert rows['0-3']['Trades'] == 0


def test_pullback_statistics_buffer_saves_trade():
    """Test that buffer can save a trade in Pullback range analysis.

    SL=3.0, PB=4.0, TP=10. Regular: PB(4) >= SL(3) => L. Buffer: PB(4) < SL+2(5) => W.
    """
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        '1H': ['Buy'],
        'SL': [3.0],
        'Pullback': [4.0],
        'TP': [10.0],
        'R': [0],
    })

    result = calculate_pullback_statistics(trades)
    rows = {row['Pullback Range']: row for _, row in result.iterrows()}

    assert rows['3-5']['Regular'] == '0W - 1L (0.0%)'
    assert rows['3-5']['With 2 pips buffer'] == '1W - 0L (100.0%)'


def test_tp_ranges_constant():
    """Test that TP_RANGES has expected ranges."""
    assert len(TP_RANGES) == 2
    assert TP_RANGES[0] == ("0-35", 0, 35)
    assert TP_RANGES[1][0] == "35+"
    assert TP_RANGES[1][1] == 35


def test_tp_statistics_columns():
    """Test that TP statistics has expected columns."""
    sample = get_sample_data()
    result = calculate_tp_statistics(sample)
    assert list(result.columns) == ['TP Range', 'Trades']


def test_tp_statistics_all_ranges_present():
    """Test that all TP ranges appear in results."""
    sample = get_sample_data()
    result = calculate_tp_statistics(sample)
    assert len(result) == 2
    assert list(result['TP Range']) == ['0-35', '35+']


def test_tp_statistics_trade_counts():
    """Test 'X of Y' formatting per TP range from sample data.

    Sample TP values: 0, 12.0, 0, 10.0, 0, 8.0, 0, 15.0, 10.0, 5.0
    Profitable overall (TP > 0): 6
    0-35 (TP>0, TP<35): 12, 10, 8, 15, 10, 5 = 6
    """
    sample = get_sample_data()
    result = calculate_tp_statistics(sample)
    counts = dict(zip(result['TP Range'], result['Trades']))
    assert counts['0-35'] == '6 of 6'
    assert counts['35+'] == '0 of 6'


def test_tp_statistics_empty():
    """Test TP statistics with empty dataset."""
    empty = get_empty_data()
    result = calculate_tp_statistics(empty)
    assert len(result) == 2
    for _, row in result.iterrows():
        assert row['Trades'] == '0 of 0'


def test_tp_statistics_large_tp():
    """Test TP statistics with trades in the 35+ range."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Monday'],
        'Trade': ['#1', '#2'],
        'Direction': ['Buy', 'Buy'],
        '1H': ['Buy', 'Buy'],
        'SL': [3.0, 3.0],
        'Pullback': [1.0, 4.0],
        'TP': [60.0, 55.0],
        'R': [20.0, 0],
    })

    result = calculate_tp_statistics(trades)
    rows = {row['TP Range']: row for _, row in result.iterrows()}

    assert rows['35+']['Trades'] == '2 of 2'


def test_sl_pb_crosstab_columns():
    """Cross-tab has SL Range, one column per PB range, and Total."""
    sample = get_sample_data()
    result = calculate_sl_pb_crosstab(sample)
    pb_labels = [label for label, _, _ in SL_PB_PB_RANGES]
    assert list(result.columns) == ['SL Range', *pb_labels, 'Total']
    assert list(result['SL Range']) == [label for label, _, _ in SL_PB_SL_RANGES]


def test_sl_pb_crosstab_counts():
    """Verify cell counts against hand-computed sample distribution."""
    sample = get_sample_data()
    result = calculate_sl_pb_crosstab(sample)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    # 0-2 bucket: PBs 0.8, 0.5 → both in 0-1
    assert rows['0-2']['0-1'] == '2 (100%)'
    assert rows['0-2']['Total'] == 2

    # 2-3 bucket: PBs 2.1, 2.5 → both in 2-3
    assert rows['2-3']['2-3'] == '2 (100%)'
    assert rows['2-3']['Total'] == 2

    # 3-5 bucket: PBs 3.5, 1.5, 3.0 → 1 in 1-2, 2 in 3-5
    assert rows['3-5']['1-2'] == '1 (33%)'
    assert rows['3-5']['3-5'] == '2 (67%)'
    assert rows['3-5']['Total'] == 3

    # 5-10 bucket: PBs 2.0, 3.0, 7.0 → 1 in 2-3, 1 in 3-5, 1 in 5-10
    assert rows['5-10']['2-3'] == '1 (33%)'
    assert rows['5-10']['3-5'] == '1 (33%)'
    assert rows['5-10']['5-10'] == '1 (33%)'
    assert rows['5-10']['Total'] == 3

    # 10+ bucket: empty
    assert rows['10+']['Total'] == 0


def test_sl_pb_crosstab_empty():
    """Empty dataset → every row has 0 totals and 0 (0%) cells."""
    empty = get_empty_data()
    result = calculate_sl_pb_crosstab(empty)
    assert len(result) == len(SL_PB_SL_RANGES)
    for _, row in result.iterrows():
        assert row['Total'] == 0
        for pb_label, _, _ in SL_PB_PB_RANGES:
            assert row[pb_label] == '0 (0%)'


def test_sl_pb_distribution_columns():
    sample = get_sample_data()
    result = calculate_sl_pb_distribution(sample)
    assert list(result.columns) == [
        'SL Range', 'Trades', 'PB Mean', 'PB Median', 'PB p75', 'PB p90',
        'PB Max', 'PB/SL Median', '% PB ≥ SL',
    ]


def test_sl_pb_distribution_values():
    """Spot-check distribution stats on sample data."""
    sample = get_sample_data()
    result = calculate_sl_pb_distribution(sample)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    # 0-2: PBs 0.8, 0.5; SLs 1.1, 1.5 → no losses
    assert rows['0-2']['Trades'] == 2
    assert rows['0-2']['PB Mean'] == '0.65'
    assert rows['0-2']['PB Median'] == '0.65'
    assert rows['0-2']['PB Max'] == '0.80'
    assert rows['0-2']['% PB ≥ SL'] == '0%'

    # 2-3: PBs 2.1, 2.5; SLs 2.0, 2.5 → both losses
    assert rows['2-3']['Trades'] == 2
    assert rows['2-3']['% PB ≥ SL'] == '100%'

    # 3-5: PBs 3.5, 1.5, 3.0; SLs 3.5, 4.0, 3.0 → 2/3 losses
    assert rows['3-5']['Trades'] == 3
    assert rows['3-5']['PB Median'] == '3.00'
    assert rows['3-5']['% PB ≥ SL'] == '67%'

    # 5-10: PBs 2.0, 3.0, 7.0; SLs 5.0, 6.0, 8.0 → no losses
    assert rows['5-10']['Trades'] == 3
    assert rows['5-10']['% PB ≥ SL'] == '0%'

    # 10+: empty
    assert rows['10+']['Trades'] == 0
    assert rows['10+']['PB Mean'] == '-'


def test_sl_pb_distribution_empty():
    empty = get_empty_data()
    result = calculate_sl_pb_distribution(empty)
    assert len(result) == len(SL_PB_SL_RANGES)
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['PB Mean'] == '-'
        assert row['% PB ≥ SL'] == '-'


def test_sl_pb_buffer_impact_columns():
    sample = get_sample_data()
    result = calculate_sl_pb_buffer_impact(sample)
    expected = ['SL Range', 'Trades', 'Loss Rate'] + [f'+{b:g} Saves' for b in SL_PB_BUFFERS]
    assert list(result.columns) == expected


def test_sl_pb_buffer_impact_values():
    sample = get_sample_data()
    result = calculate_sl_pb_buffer_impact(sample)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    # 0-2: no losses
    assert rows['0-2']['Loss Rate'] == '0%'
    assert rows['0-2']['+1 Saves'] == '0%'

    # 2-3: both losses, +1 saves both (PB-SL gap is at most 0.1)
    assert rows['2-3']['Loss Rate'] == '100%'
    assert rows['2-3']['+1 Saves'] == '100%'
    assert rows['2-3']['+2 Saves'] == '100%'

    # 3-5: 2/3 losses; +1 saves both (gaps 0 and 0)
    assert rows['3-5']['Loss Rate'] == '67%'
    assert rows['3-5']['+1 Saves'] == '67%'
    assert rows['3-5']['+3 Saves'] == '67%'

    # 5-10: no losses → no saves
    assert rows['5-10']['Loss Rate'] == '0%'
    assert rows['5-10']['+2 Saves'] == '0%'


def test_sl_pb_buffer_impact_empty():
    empty = get_empty_data()
    result = calculate_sl_pb_buffer_impact(empty)
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Loss Rate'] == '-'
        for b in SL_PB_BUFFERS:
            assert row[f'+{b:g} Saves'] == '-'


def test_buffer_statistics_filtered_all_trades():
    """Test _calculate_buffer_statistics_filtered with All Trades only."""
    sample = get_sample_data()
    result = _calculate_buffer_statistics_filtered(sample, ["All Trades"])
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        strategies = result['Strategy'].unique()
        for s in strategies:
            assert s == 'All Trades', f"Unexpected strategy: {s}"


def test_buffer_statistics_filtered_1h():
    """Test _calculate_buffer_statistics_filtered with 1H strategies only."""
    sample = get_sample_data()
    result = _calculate_buffer_statistics_filtered(sample, ["1H Aligned", "1H Against"])
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        strategies = result['Strategy'].unique()
        for s in strategies:
            assert s in ['1H Aligned', '1H Against'], f"Unexpected strategy: {s}"


def test_buffer_statistics_filtered_excludes_others():
    """Test that filtered results don't include strategies not in the list."""
    sample = get_sample_data()
    result_all = _calculate_buffer_statistics_filtered(sample, ["All Trades"])
    result_1h = _calculate_buffer_statistics_filtered(sample, ["1H Aligned", "1H Against"])
    if len(result_all) > 0:
        assert '1H Aligned' not in result_all['Strategy'].values
    if len(result_1h) > 0:
        assert 'All Trades' not in result_1h['Strategy'].values


def test_buffer_statistics_filtered_empty():
    """Test _calculate_buffer_statistics_filtered with empty data."""
    empty = get_empty_data()
    result = _calculate_buffer_statistics_filtered(empty, ["All Trades"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_direction_statistics_structure():
    """Direction stats should expose Buy + Sell rows with 1:2 and 1:3 columns."""
    result = calculate_direction_statistics(get_sample_data())
    assert list(result['Direction']) == ['Buy', 'Sell']
    assert list(result.columns) == ['Direction', 'Trades', '1:2 W/L', 'Edge 1:2', '1:3 W/L', 'Edge 1:3']


def test_direction_statistics_counts():
    """3 aligned Buys, 3 aligned Sells in the sample."""
    result = calculate_direction_statistics(get_sample_data())
    rows = {r['Direction']: r for _, r in result.iterrows()}
    assert rows['Buy']['Trades'] == 3
    assert rows['Sell']['Trades'] == 3


def test_direction_statistics_buy_winner():
    """Among aligned Buys, only Row 1 (SL=1.1 PB=0.8 TP=12) wins at both RRRs."""
    result = calculate_direction_statistics(get_sample_data())
    rows = {r['Direction']: r for _, r in result.iterrows()}
    assert rows['Buy']['1:2 W/L'].startswith('1W - 2L')
    assert rows['Buy']['1:3 W/L'].startswith('1W - 2L')
    assert rows['Sell']['1:2 W/L'].startswith('0W - 3L')


def test_direction_statistics_empty():
    """Empty input yields zero-trade rows for Buy and Sell."""
    result = calculate_direction_statistics(get_empty_data())
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Edge 1:2'] == '0.000R'


def test_trade_number_statistics_structure():
    """Trade # stats should have Trade # column and the two-RRR columns."""
    result = calculate_trade_number_statistics(get_sample_data())
    assert list(result.columns) == ['Trade #', 'Trades', '1:2 W/L', 'Edge 1:2', '1:3 W/L', 'Edge 1:3']


def test_trade_number_statistics_skips_missing():
    """Trade numbers with zero aligned trades should be omitted from the table."""
    result = calculate_trade_number_statistics(get_sample_data())
    assert set(result['Trade #']) == {'#1', '#2'}


def test_trade_number_statistics_winner():
    """#2 has 1 winner (Row 1); #1 has 0 winners."""
    result = calculate_trade_number_statistics(get_sample_data())
    rows = {r['Trade #']: r for _, r in result.iterrows()}
    assert rows['#1']['Trades'] == 4
    assert rows['#1']['1:2 W/L'].startswith('0W - 4L')
    assert rows['#2']['Trades'] == 2
    assert rows['#2']['1:2 W/L'].startswith('1W - 1L')


def test_prev_outcome_statistics_structure():
    """Prev-outcome stats should have the three context rows."""
    result = calculate_prev_outcome_statistics(get_sample_data())
    assert list(result['Context']) == ['First of Day', 'After Win', 'After Loss']


def test_prev_outcome_statistics_after_loss_winner():
    """The lone aligned winner (Row 1) follows a loss on the same day."""
    result = calculate_prev_outcome_statistics(get_sample_data())
    rows = {r['Context']: r for _, r in result.iterrows()}
    assert rows['After Loss']['Trades'] == 1
    assert rows['After Loss']['1:2 W/L'].startswith('1W - 0L')
    assert rows['After Win']['Trades'] == 0


def test_prev_outcome_statistics_first_of_day():
    """5 aligned days start with a losing trade; all 5 First-of-Day rows lose at 1:2."""
    result = calculate_prev_outcome_statistics(get_sample_data())
    rows = {r['Context']: r for _, r in result.iterrows()}
    assert rows['First of Day']['Trades'] == 5
    assert rows['First of Day']['1:2 W/L'].startswith('0W - 5L')


def test_rrr_sweep_statistics_structure():
    """RRR sweep should cover every RRR_SWEEP entry."""
    result = calculate_rrr_sweep_statistics(get_sample_data())
    assert len(result) == len(RRR_SWEEP)
    assert list(result.columns) == ['RRR', 'Trades', 'W/L', 'Edge R/Trade']


def test_rrr_sweep_statistics_counts():
    """At 1:1 the aligned sample has 4 winners; at 1:3 only 1 winner."""
    result = calculate_rrr_sweep_statistics(get_sample_data())
    rows = {r['RRR']: r for _, r in result.iterrows()}
    assert rows['1:1']['Trades'] == 6
    assert rows['1:1']['W/L'].startswith('4W - 2L')
    assert rows['1:3']['W/L'].startswith('1W - 5L')


def test_rrr_sweep_statistics_empty():
    """Empty input still yields one row per RRR with zero trades."""
    result = calculate_rrr_sweep_statistics(get_empty_data())
    assert len(result) == len(RRR_SWEEP)
    assert all(r['Trades'] == 0 for _, r in result.iterrows())


def test_stop_after_loss_statistics_structure():
    """Stop-after-loss stats should include every configured rule."""
    result = calculate_stop_after_loss_statistics(get_sample_data())
    assert list(result['Rule']) == [label for label, _ in STOP_AFTER_LOSS_RULES]


def test_stop_after_loss_first_cap_excludes_winner():
    """
    'Stop after 1st Loss' drops the aligned sample's only winner (Row 1),
    because Row 0 — the prior trade on 2026-01-12 — is a loss.
    """
    result = calculate_stop_after_loss_statistics(get_sample_data())
    rows = {r['Rule']: r for _, r in result.iterrows()}
    assert rows['Stop after 1st Loss']['Trades 1:2'] == 5
    assert rows['Stop after 1st Loss']['1:2 W/L'].startswith('0W - 5L')


def test_pd_alignment_structure():
    result = calculate_pd_alignment_statistics(get_sample_data())
    assert list(result['Filter']) == ['All PD-Known', 'PD Aligned', 'PD Against']
    assert {'1:2 W/L', '1:3 W/L', 'Edge 1:2', 'Edge 1:3', 'Trades'}.issubset(result.columns)


def test_pd_alignment_counts():
    result = calculate_pd_alignment_statistics(get_sample_data())
    rows = {r['Filter']: r for _, r in result.iterrows()}
    assert rows['All PD-Known']['Trades'] == 8
    assert rows['PD Aligned']['Trades'] == 5
    assert rows['PD Against']['Trades'] == 3


def test_pd_alignment_winner_in_aligned():
    """Trade #1 (Buy/Buy/Buy, TP=12) is the lone 1:2 win and falls under PD Aligned."""
    result = calculate_pd_alignment_statistics(get_sample_data())
    rows = {r['Filter']: r for _, r in result.iterrows()}
    assert rows['PD Aligned']['1:2 W/L'].startswith('1W - 4L')
    assert rows['PD Against']['1:2 W/L'].startswith('0W - 3L')


def test_pd_alignment_empty():
    result = calculate_pd_alignment_statistics(get_empty_data())
    rows = {r['Filter']: r for _, r in result.iterrows()}
    assert rows['All PD-Known']['Trades'] == 0
    assert rows['PD Aligned']['Trades'] == 0


def test_pd_zone_structure():
    result = calculate_pd_zone_statistics(get_sample_data())
    assert list(result['Zone']) == [label for label, _ in PD_ZONES]


def test_pd_zone_counts():
    result = calculate_pd_zone_statistics(get_sample_data())
    rows = {r['Zone']: r for _, r in result.iterrows()}
    assert rows['Discount (PD = Buy)']['Trades'] == 4
    assert rows['Premium (PD = Sell)']['Trades'] == 4
    assert rows['Discount (PD = Buy)']['1:2 W/L'].startswith('1W - 3L')
    assert rows['Premium (PD = Sell)']['1:2 W/L'].startswith('0W - 4L')


def test_pd_1h_combined_counts():
    result = calculate_pd_1h_combined_statistics(get_sample_data())
    rows = {r['Combination']: r for _, r in result.iterrows()}
    assert rows['1H Aligned + PD Aligned']['Trades'] == 4
    assert rows['1H Aligned + PD Against']['Trades'] == 2
    assert rows['1H Against + PD Aligned']['Trades'] == 1
    assert rows['1H Against + PD Against']['Trades'] == 1
    assert rows['1H Aligned + PD Aligned']['1:2 W/L'].startswith('1W - 3L')


def test_pd_direction_counts():
    """PD-Aligned trades split by Direction: Buy entries in Discount, Sell entries in Premium."""
    result = calculate_pd_direction_statistics(get_sample_data())
    rows = {r['Direction']: r for _, r in result.iterrows()}
    assert rows['Buy']['Trades'] == 2
    assert rows['Sell']['Trades'] == 3
    assert rows['Buy']['1:2 W/L'].startswith('1W - 1L')


def test_stop_after_loss_no_cap_matches_baseline():
    """'No Cap' row should equal the full aligned sample."""
    result = calculate_stop_after_loss_statistics(get_sample_data())
    rows = {r['Rule']: r for _, r in result.iterrows()}
    assert rows['No Cap']['Trades 1:2'] == 6
    assert rows['No Cap']['1:2 W/L'].startswith('1W - 5L')


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_get_strategies,
        test_strategy_names_include_base,
        test_strategy_names_include_sl_combos,
        test_calculate_stats_all_trades,
        test_calculate_stats_empty,
        test_win_condition_1_1_rrr,
        test_win_condition_tp_must_reach_sl,
        test_edge_calculation,
        test_outcome_calculation,
        test_days_calculation,
        test_trades_required,
        test_trades_required_negative_outcome,
        test_1h_aligned_filter,
        test_1h_against_filter,
        test_1h_aligned_plus_against_equals_all,
        test_sl_filter_combination,
        test_calculate_statistics_returns_positive_edge_only,
        test_calculate_statistics_sorted_by_edge,
        test_calculate_statistics_columns,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_buffer_saves_losing_trade,
        test_buffer_tp_must_reach_effective_sl,
        test_buffer_stats_has_buffer_column,
        test_buffer_stats_empty,
        test_get_buffer_strategies,
        # test_get_buffer_strategies_includes_sl_caps,
        # test_buffer_sl_cap_filter,
        # test_buffer_sl_cap_with_1h_filter,
        test_calculate_buffer_statistics,
        test_calculate_buffer_statistics_has_pass_column,
        test_buffer_pips_constant,
        test_rrr_ratios_constant,
        test_breakeven_rate,
        test_1_2_rrr_win_condition,
        test_1_2_rrr_edge_calculation,
        test_1_2_rrr_outcome,
        test_1_2_rrr_buffer,
        test_1_2_rrr_empty,
        test_fixed_sl_sizes_constant,
        test_fixed_sl_trade_survives,
        test_fixed_sl_trade_loses_deep_pullback,
        test_fixed_sl_ignores_original_sl,
        test_fixed_sl_tp_must_reach_target,
        test_fixed_sl_1_2_rrr,
        test_fixed_sl_empty,
        test_fixed_sl_has_fixed_sl_column,
        test_fixed_sl_mixed_trades,
        test_calculate_fixed_sl_statistics,
        test_calculate_fixed_sl_statistics_has_both_rrr,
        test_calculate_fixed_sl_statistics_total_rows,
        test_calculate_fixed_sl_statistics_columns,
        test_fixed_sl_with_strategy_has_strategy_column,
        test_fixed_sl_with_strategy_empty,
        test_get_1h_strategies,
        test_1h_strategy_filters_correctly,
        test_calculate_fixed_sl_1h_statistics,
        test_fixed_sl_1h_total_rows,
        test_fixed_sl_1h_has_strategy_column,
        test_fixed_sl_1h_strategies_present,
        test_fixed_sl_1h_aligned_wins_more,
        test_simulate_challenge_all_wins_1_1,
        test_simulate_challenge_all_losses,
        test_simulate_challenge_20_wins,
        test_simulate_challenge_20_losses,
        test_simulate_challenge_reset_after_pass,
        test_simulate_challenge_reset_after_drawdown,
        test_simulate_challenge_1_2_rrr,
        test_simulate_challenge_1_2_rrr_10_wins,
        test_simulate_challenge_empty,
        test_simulate_challenge_mixed_no_threshold,
        test_simulate_challenge_near_threshold,
        test_buffer_stats_has_challenge_columns,
        test_buffer_stats_has_trades_column,
        test_buffer_stats_empty_has_trades_zero,
        test_simulate_challenge_float_rrr,
        test_simulate_challenge_float_rrr_not_enough,
        test_simulate_challenge_float_rrr_10,
        test_buffer_stats_with_float_rrr,
        test_buffer_stats_with_float_rrr_loss,
        test_format_wl,
        test_weekday_order_constant,
        test_weekday_statistics_columns,
        test_weekday_statistics_all_days_present,
        test_weekday_statistics_trade_counts,
        test_weekday_statistics_regular,
        test_weekday_statistics_buffer,
        test_weekday_statistics_buffer_saves_trade,
        test_weekday_statistics_empty,
        test_weekday_statistics_single_day,
        test_sl_ranges_constant,
        test_sl_statistics_columns,
        test_sl_statistics_all_ranges_present,
        test_sl_statistics_trade_counts,
        test_sl_statistics_regular,
        test_sl_statistics_buffer,
        test_sl_statistics_empty,
        test_sl_statistics_large_sl,
        test_sl_statistics_buffer_saves_trade,
        test_pullback_ranges_constant,
        test_pullback_statistics_columns,
        test_pullback_statistics_all_ranges_present,
        test_pullback_statistics_trade_counts,
        test_pullback_statistics_regular,
        test_pullback_statistics_buffer,
        test_pullback_statistics_empty,
        test_pullback_statistics_large_pullback,
        test_pullback_statistics_buffer_saves_trade,
        test_tp_ranges_constant,
        test_tp_statistics_columns,
        test_tp_statistics_all_ranges_present,
        test_tp_statistics_trade_counts,
        test_tp_statistics_empty,
        test_tp_statistics_large_tp,
        test_buffer_statistics_filtered_all_trades,
        test_buffer_statistics_filtered_1h,
        test_buffer_statistics_filtered_excludes_others,
        test_buffer_statistics_filtered_empty,
        test_direction_statistics_structure,
        test_direction_statistics_counts,
        test_direction_statistics_buy_winner,
        test_direction_statistics_empty,
        test_trade_number_statistics_structure,
        test_trade_number_statistics_skips_missing,
        test_trade_number_statistics_winner,
        test_prev_outcome_statistics_structure,
        test_prev_outcome_statistics_after_loss_winner,
        test_prev_outcome_statistics_first_of_day,
        test_rrr_sweep_statistics_structure,
        test_rrr_sweep_statistics_counts,
        test_rrr_sweep_statistics_empty,
        test_stop_after_loss_statistics_structure,
        test_stop_after_loss_first_cap_excludes_winner,
        test_stop_after_loss_no_cap_matches_baseline,
        test_pd_alignment_structure,
        test_pd_alignment_counts,
        test_pd_alignment_winner_in_aligned,
        test_pd_alignment_empty,
        test_pd_zone_structure,
        test_pd_zone_counts,
        test_pd_1h_combined_counts,
        test_pd_direction_counts,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: ERROR - {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
