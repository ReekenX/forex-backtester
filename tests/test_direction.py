"""
Tests for utils.direction module

These tests use a small dataset of 10 rows to verify the direction analysis functionality.
Run with: poetry run python tests/test_direction.py
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from utils.direction import (
    calculate_statistics,
    create_html_table,
    get_strategies,
    _calculate_stats,
    RRR_RATIO,
    BREAKEVEN_RATE,
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
        'EMA(50)': [], 'EMA(200)': [], 'SL': [], 'Pullback': [], 'TP': [], 'R': [],
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
    assert 'Buy Only' in names
    assert 'Sell Only' in names
    assert 'EMA(50) Aligned' in names
    assert 'EMA(200) Aligned' in names
    assert 'Both EMAs Aligned' in names


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


def test_buy_only_filter():
    """Test Buy Only filter."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'Buy Only'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
        assert row['Direction'] == 'Buy'


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
    """Test that result has expected columns."""
    sample = get_sample_data()
    result = calculate_statistics(sample)

    if len(result) > 0:
        expected = ['Strategy', 'RRR', 'Trades', 'Notation',
                    'Win Rate', 'Outcome', 'Edge', 'Days', 'Days %', 'Trades Required']
        assert list(result.columns) == expected


def test_create_html_table_basic():
    """Test HTML table creation."""
    sample = get_sample_data()
    stats = calculate_statistics(sample)
    html = create_html_table(stats)

    assert '<table' in html
    assert 'direction-analysis-table' in html


def test_create_html_table_empty():
    """Test HTML table with empty data."""
    html = create_html_table(pd.DataFrame())
    assert 'No profitable direction strategies found' in html


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
        test_buy_only_filter,
        test_sl_filter_combination,
        test_calculate_statistics_returns_positive_edge_only,
        test_calculate_statistics_sorted_by_edge,
        test_calculate_statistics_columns,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_emas_agree_filter,
        test_emas_disagree_filter,
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
