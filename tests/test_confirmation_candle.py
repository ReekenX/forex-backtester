"""
Tests for utils.confirmation_candle module

These tests use a small dataset of 10 rows to verify the analysis functionality.
Run with: poetry run python tests/test_confirmation_candle.py
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from utils.confirmation_candle import (
    calculate_statistics,
    calculate_buffer_statistics,
    create_html_table,
    get_strategies,
    get_buffer_strategies,
    _calculate_stats,
    _calculate_stats_with_buffer,
    _breakeven_rate,
    RRR_RATIOS,
    BUFFER_PIPS,
)


def get_sample_data():
    """Create a sample dataset with 10 trades matching the new CSV structure."""
    return pd.DataFrame({
        'Date': ['2026-01-12', '2026-01-12', '2026-01-12', '2026-01-13', '2026-01-13',
                 '2026-01-14', '2026-01-14', '2026-01-15', '2026-01-15', '2026-01-16'],
        'Weekday': ['Monday', 'Monday', 'Monday', 'Tuesday', 'Tuesday',
                    'Wednesday', 'Wednesday', 'Thursday', 'Thursday', 'Friday'],
        'Trade': ['#1', '#2', '#3', '#1', '#2',
                  '#1', '#2', '#1', '#2', '#1'],
        'Direction': ['Buy', 'Buy', 'Sell', 'Buy', 'Sell',
                      'Sell', 'Buy', 'Buy', 'Sell', 'Sell'],
        'EMA(50)': ['Buy', 'Buy', 'Buy', 'Sell', 'Sell',
                    'Sell', 'Sell', 'Buy', 'Buy', 'Sell'],
        'EMA(200)': ['Buy', 'Buy', 'Sell', 'Buy', 'Sell',
                     'Sell', 'Buy', 'Buy', 'Sell', 'Sell'],
        'Engulfing': ['Yes', 'No', None, 'Yes', 'Similar',
                      'No', None, 'Yes', None, 'No'],
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
        'EMA(50)': [], 'EMA(200)': [], 'Engulfing': [], 'SL': [], 'Pullback': [], 'TP': [], 'R': [],
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
    assert 'EMA(50) Aligned' in names
    assert 'EMA(200) Aligned' in names
    assert 'Both EMAs Aligned' in names
    assert 'Buy Only' not in names
    assert 'Sell Only' not in names


def test_strategy_names_include_sl_combos():
    """Test that SL combination strategies are present."""
    strategies = get_strategies()
    names = [name for name, _ in strategies]
    assert 'EMA(50) Aligned + SL < 5' in names
    assert 'Both EMAs Aligned + SL 5-10' in names
    assert 'EMA(200) Against + SL > 3' in names


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


def test_ema50_aligned_filter():
    """Test EMA(50) Aligned filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'EMA(50) Aligned'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] == row['EMA(50)']


def test_ema200_against_filter():
    """Test EMA(200) Against filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'EMA(200) Against'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] != row['EMA(200)']


def test_both_emas_aligned_filter():
    """Test Both EMAs Aligned filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Both EMAs Aligned'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] == row['EMA(50)']
        assert row['Direction'] == row['EMA(200)']


def test_sl_filter_combination():
    """Test SL filter in combination strategy."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'EMA(50) Aligned + SL > 3'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] == row['EMA(50)']
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


def test_emas_agree_filter():
    """Test EMAs Agree (trending) filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'EMAs Agree (trending)'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['EMA(50)'] == row['EMA(200)']


def test_emas_disagree_filter():
    """Test EMAs Disagree (transition) filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'EMAs Disagree (transition)'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['EMA(50)'] != row['EMA(200)']


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

    assert stats['Trades'] == 0
    assert stats['Buffer'] == '+1.0'
    assert stats['Notation'] == '0W – 0L'


def test_get_buffer_strategies():
    """Test that buffer strategies include key strategies."""
    strategies = get_buffer_strategies()
    names = [name for name, _ in strategies]

    assert 'All Trades' in names
    assert 'EMA(50) Aligned' in names
    assert 'Both EMAs Aligned' in names


def test_get_buffer_strategies_includes_sl_caps():
    """Test that buffer strategies include SL cap variations."""
    strategies = get_buffer_strategies()
    names = [name for name, _ in strategies]

    assert 'All Trades + SL < 3' in names
    assert 'All Trades + SL < 4' in names
    assert 'All Trades + SL < 5' in names
    assert 'EMA(50) Aligned + SL < 3' in names
    assert 'Both EMAs Aligned + SL < 5' in names


def test_buffer_sl_cap_filter():
    """Test that SL cap filter correctly excludes trades with SL >= cap."""
    sample = get_sample_data()
    strategies = get_buffer_strategies()
    strategy = [func for name, func in strategies if name == 'All Trades + SL < 3'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['SL'] < 3


def test_buffer_sl_cap_with_ema_filter():
    """Test SL cap combined with EMA filter."""
    sample = get_sample_data()
    strategies = get_buffer_strategies()
    strategy = [func for name, func in strategies if name == 'EMA(50) Aligned + SL < 4'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['SL'] < 4
        assert row['Direction'] == row['EMA(50)']


def test_calculate_buffer_statistics():
    """Test that buffer statistics returns expected DataFrame."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)

    assert isinstance(result, pd.DataFrame)


def test_calculate_buffer_statistics_has_both_rrr():
    """Test that buffer statistics includes both 1:1 and 1:2 RRR rows."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)

    rrr_values = result['RRR'].unique()
    assert '1:1' in rrr_values
    assert '1:2' in rrr_values


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


def test_engulfing_yes_filter():
    """Test Engulfing: Yes filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Engulfing: Yes'][0]
    filtered = strategy(sample)

    assert len(filtered) == 3  # rows 0, 3, 7 have 'Yes'
    for _, row in filtered.iterrows():
        assert row['Engulfing'] == 'Yes'


def test_engulfing_no_filter():
    """Test Engulfing: No filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Engulfing: No'][0]
    filtered = strategy(sample)

    assert len(filtered) == 3  # rows 1, 5, 9 have 'No'
    for _, row in filtered.iterrows():
        assert row['Engulfing'] == 'No'


def test_engulfing_similar_filter():
    """Test Engulfing: Similar filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Engulfing: Similar'][0]
    filtered = strategy(sample)

    assert len(filtered) == 1  # row 4 has 'Similar'
    for _, row in filtered.iterrows():
        assert row['Engulfing'] == 'Similar'


def test_engulfing_yes_or_similar_filter():
    """Test Engulfing: Yes or Similar filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Engulfing: Yes or Similar'][0]
    filtered = strategy(sample)

    assert len(filtered) == 4  # 3 Yes + 1 Similar
    for _, row in filtered.iterrows():
        assert row['Engulfing'] in ['Yes', 'Similar']


def test_has_engulfing_data_filter():
    """Test Has Engulfing Data filter (non-empty values)."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Has Engulfing Data'][0]
    filtered = strategy(sample)

    assert len(filtered) == 7  # 3 Yes + 3 No + 1 Similar
    for _, row in filtered.iterrows():
        assert pd.notna(row['Engulfing'])


def test_engulfing_yes_ema50_aligned_filter():
    """Test Engulfing: Yes + EMA(50) Aligned filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Engulfing: Yes + EMA(50) Aligned'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Engulfing'] == 'Yes'
        assert row['Direction'] == row['EMA(50)']


def test_engulfing_yes_sl_filter():
    """Test Engulfing: Yes + SL filter combination."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Engulfing: Yes + SL < 5'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Engulfing'] == 'Yes'
        assert row['SL'] < 5


def test_strategy_names_include_engulfing():
    """Test that engulfing strategies are present."""
    strategies = get_strategies()
    names = [name for name, _ in strategies]
    assert 'Engulfing: Yes' in names
    assert 'Engulfing: No' in names
    assert 'Engulfing: Similar' in names
    assert 'Engulfing: Yes or Similar' in names
    assert 'Has Engulfing Data' in names


def test_buffer_strategies_include_engulfing():
    """Test that buffer strategies include engulfing variants."""
    strategies = get_buffer_strategies()
    names = [name for name, _ in strategies]
    assert 'Engulfing: Yes' in names
    assert 'Engulfing: No' in names
    assert 'Engulfing: Yes or Similar' in names
    assert 'Has Engulfing Data' in names
    assert 'Engulfing: Yes + SL < 3' in names


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
        test_ema50_aligned_filter,
        test_ema200_against_filter,
        test_both_emas_aligned_filter,
        test_sl_filter_combination,
        test_calculate_statistics_returns_positive_edge_only,
        test_calculate_statistics_sorted_by_edge,
        test_calculate_statistics_columns,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_emas_agree_filter,
        test_emas_disagree_filter,
        test_buffer_saves_losing_trade,
        test_buffer_tp_must_reach_effective_sl,
        test_buffer_stats_has_buffer_column,
        test_buffer_stats_empty,
        test_get_buffer_strategies,
        test_get_buffer_strategies_includes_sl_caps,
        test_buffer_sl_cap_filter,
        test_buffer_sl_cap_with_ema_filter,
        test_calculate_buffer_statistics,
        test_calculate_buffer_statistics_has_both_rrr,
        test_buffer_pips_constant,
        test_rrr_ratios_constant,
        test_breakeven_rate,
        test_1_2_rrr_win_condition,
        test_1_2_rrr_edge_calculation,
        test_1_2_rrr_outcome,
        test_1_2_rrr_buffer,
        test_1_2_rrr_empty,
        test_engulfing_yes_filter,
        test_engulfing_no_filter,
        test_engulfing_similar_filter,
        test_engulfing_yes_or_similar_filter,
        test_has_engulfing_data_filter,
        test_engulfing_yes_ema50_aligned_filter,
        test_engulfing_yes_sl_filter,
        test_strategy_names_include_engulfing,
        test_buffer_strategies_include_engulfing,
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
