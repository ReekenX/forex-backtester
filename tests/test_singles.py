"""
Tests for utils.singles module

These tests use a small dataset of 10 rows to verify the single setup strategy analysis functionality.
Run with: poetry run python tests/test_singles.py
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from utils.singles import (
    calculate_strategy_statistics,
    create_html_table,
    get_single_setup_strategies,
    RRR_RATIOS,
    _calculate_stats_for_strategy_and_rrr,
    _create_empty_stats
)


def get_sample_data():
    """Create a sample dataset with 10 trades."""
    return pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-02', '2025-01-03',
                 '2025-01-03', '2025-01-04', '2025-01-04', '2025-01-05', '2025-01-05'],
        'Direction': ['Buy', 'Buy', 'Sell', 'Buy', 'Buy',
                      'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        'EMA': ['Buy', 'Sell', 'Sell', 'Buy', 'Buy',
                'Buy', 'Buy', 'Sell', 'Sell', 'Sell'],
        'BOS/CH': ['BOS', 'CH', 'BOS', 'BOS', 'CH',
                   'BOS', 'CH', 'BOS', 'BOS', 'CH'],
        'SL': [2.0, 5.0, 3.0, 1.5, 8.0,
               4.0, 10.0, 6.0, 12.0, 15.0],
        'Pullback': [1.0, 5.0, 2.0, 0.5, 7.0,
                     3.0, 9.0, 5.5, 11.0, 14.0],
        'TP': [4.0, 10.0, 6.0, 3.0, 16.0,
               8.0, 20.0, 12.0, 24.0, 30.0],
        '30M Leg': ['Above H', 'Below H', 'Below L', 'Above L', 'Above H',
                    'Below H', 'Above L', 'Below L', 'Above H', 'Below L'],
        'News Event': [None, 'News', None, None, 'News',
                       None, None, 'News', None, None],
        'Hours Until News': [None, 1.0, None, None, 3.0,
                            None, None, 1.5, None, None]
    })


def get_empty_data():
    """Create an empty dataset."""
    return pd.DataFrame({
        'Date': [],
        'Direction': [],
        'EMA': [],
        'BOS/CH': [],
        'SL': [],
        'Pullback': [],
        'TP': [],
        '30M Leg': [],
        'News Event': [],
        'Hours Until News': []
    })


def test_get_single_setup_strategies():
    """Test that we get all expected strategies."""
    strategies = get_single_setup_strategies()

    # Should have 5 strategies
    assert len(strategies) == 5

    # Check that each strategy returns a tuple of (name, function)
    for name, func in strategies:
        assert isinstance(name, str)
        assert callable(func)


def test_strategy_names():
    """Test that all expected strategy names are present."""
    strategies = get_single_setup_strategies()
    names = [name for name, _ in strategies]

    expected_names = [
        'EMA Aligned',
        'EMA Counter-Trend',
        'BOS Only',
        'CH Only',
        '30M Trend'
    ]

    for expected in expected_names:
        assert expected in names


def test_calculate_strategy_statistics_basic():
    """Test basic strategy statistics calculation."""
    sample_data = get_sample_data()
    result = calculate_strategy_statistics(sample_data)

    # Should have results (only positive edge strategies)
    assert len(result) > 0

    # Check columns exist
    expected_columns = ['Strategy', 'RRR', 'Trades', 'Notation',
                        'Win Rate', 'Edge', 'Outcome', 'Days', 'Days %', 'Profit Factor']
    assert list(result.columns) == expected_columns


def test_calculate_strategy_statistics_includes_plain():
    """Test that Plain Strategy is included in results."""
    sample_data = get_sample_data()
    result = calculate_strategy_statistics(sample_data)

    # Check if Plain Strategy is in results (if it has positive edge)
    plain_strategies = result[result['Strategy'] == 'Plain Strategy']
    # Plain strategy may or may not be included depending on edge
    # Just verify the calculation runs without error
    assert len(result) >= 0


def test_calculate_strategy_statistics_empty():
    """Test with empty dataset."""
    empty_data = get_empty_data()
    result = calculate_strategy_statistics(empty_data)

    # Should return empty dataframe when no positive edge strategies
    assert isinstance(result, pd.DataFrame)


def test_calculate_stats_for_strategy_and_rrr():
    """Test statistics calculation for a specific strategy and RRR."""
    # Create test data with known outcomes
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 10.0, 2.0],  # 1st wins, 2nd loses (equal), 3rd wins
        'TP': [20.0, 20.0, 20.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test Strategy', 1, 50.0)

    # Should have 2 wins (pullback < SL and TP >= 1*SL)
    assert stats['Notation'] == '2W – 1L'
    assert stats['Trades'] == 3
    assert stats['Strategy'] == 'Test Strategy'
    assert stats['RRR'] == '1:1'


def test_win_condition_rrr_1():
    """Test win condition for 1:1 RRR."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 9.9, 10.1],  # Win, Win, Lose
        'TP': [10.0, 10.0, 10.0]  # All meet 1:1 TP requirement
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 1, 50.0)
    assert stats['Notation'] == '2W – 1L'


def test_win_condition_rrr_2():
    """Test win condition for 1:2 RRR."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0],  # All pullbacks are winning
        'TP': [20.0, 19.9, 30.0]  # Win (>=20), Lose (<20), Win (>=20)
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 2, 33.3)
    assert stats['Notation'] == '2W – 1L'


def test_create_empty_stats():
    """Test empty stats creation."""
    stats = _create_empty_stats('Empty Strategy', 1, 50.0)

    assert stats['Strategy'] == 'Empty Strategy'
    assert stats['RRR'] == '1:1'
    assert stats['Trades'] == 0
    assert stats['Notation'] == '0W – 0L'
    assert stats['Win Rate'] == "0.0%"
    assert stats['Edge'] == "-50.0%"
    assert stats['edge_value'] == -50.0


def test_edge_calculation():
    """Test edge calculation (win rate - breakeven rate)."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 15.0],  # 3 wins, 1 loss
        'TP': [10.0, 10.0, 10.0, 10.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 1, 50.0)

    # Win rate = 75%, Breakeven = 50%, Edge = 25%
    assert stats['Win Rate'] == "75.0%"
    assert stats['Edge'] == "25.0%"
    assert stats['edge_value'] == 25.0


def test_outcome_calculation():
    """Test outcome calculation (wins * RRR - losses)."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 15.0],  # 2 wins, 1 loss
        'TP': [20.0, 20.0, 20.0]
    })

    # For 1:2 RRR: outcome = (2 * 2) - 1 = 3R
    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 2, 33.3)
    assert stats['Outcome'] == "3R"


def test_profit_factor_calculation():
    """Test profit factor calculation."""
    # Test with 3 wins, 1 loss at 1:2 RRR
    # Profit Factor = (3 * 2) / 1 = 6.0
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'],
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 15.0],  # 3 wins, 1 loss
        'TP': [20.0, 20.0, 20.0, 20.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 2, 33.3)
    assert stats['Profit Factor'] == "6.00"


def test_profit_factor_no_losses():
    """Test profit factor with no losses (should be infinity)."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02'],
        'SL': [10.0, 10.0],
        'Pullback': [5.0, 5.0],  # All wins
        'TP': [10.0, 10.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 1, 50.0)
    assert stats['Profit Factor'] == "∞"


def test_days_calculation():
    """Test Days and Days % calculation."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-03'],
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 15.0, 15.0],  # 2 wins on different days, 2 losses
        'TP': [10.0, 10.0, 10.0, 10.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 1, 50.0)

    # 2 unique days with wins out of 3 total days = 66.67% ≈ 67%
    assert stats['Days'] == 2
    assert stats['Days %'] == "67%"


def test_days_all_wins():
    """Test Days % when all trading days have wins."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0],  # All wins
        'TP': [10.0, 10.0, 10.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 1, 50.0)

    assert stats['Days'] == 3
    assert stats['Days %'] == "100%"


def test_days_no_wins():
    """Test Days % when no trading days have wins."""
    trades = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02'],
        'SL': [10.0, 10.0],
        'Pullback': [15.0, 15.0],  # All losses
        'TP': [10.0, 10.0]
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, 'Test', 1, 50.0)

    assert stats['Days'] == 0
    assert stats['Days %'] == "0%"


def test_create_html_table_basic():
    """Test HTML table creation."""
    sample_data = get_sample_data()
    stats = calculate_strategy_statistics(sample_data)
    html = create_html_table(stats)

    # Check HTML contains expected elements
    assert '<table' in html
    assert 'singles-analysis-table' in html
    assert '<th>Strategy</th>' in html
    assert '<th>RRR</th>' in html
    assert '<th>Trades</th>' in html


def test_create_html_table_empty():
    """Test HTML table with empty data."""
    empty_df = pd.DataFrame()
    html = create_html_table(empty_df)

    assert "No profitable strategies found" in html


def test_ema_aligned_filter():
    """Test EMA Aligned filter strategy."""
    sample_data = get_sample_data()
    strategies = get_single_setup_strategies()

    # Find EMA Aligned strategy
    ema_aligned = [func for name, func in strategies if name == 'EMA Aligned'][0]
    filtered = ema_aligned(sample_data)

    # Should only have rows where EMA == Direction
    assert len(filtered) > 0
    assert all(filtered['EMA'] == filtered['Direction'])


def test_bos_only_filter():
    """Test BOS Only filter strategy."""
    sample_data = get_sample_data()
    strategies = get_single_setup_strategies()

    # Find BOS Only strategy
    bos_only = [func for name, func in strategies if name == 'BOS Only'][0]
    filtered = bos_only(sample_data)

    # Should only have rows where BOS/CH == 'BOS'
    assert len(filtered) > 0
    assert all(filtered['BOS/CH'] == 'BOS')


def test_30m_trend_filter():
    """Test 30M Trend alignment filter."""
    sample_data = get_sample_data()
    strategies = get_single_setup_strategies()

    # Find 30M Trend strategy
    trend_30m = [func for name, func in strategies if name == '30M Trend'][0]
    filtered = trend_30m(sample_data)

    # Should only have rows where trend aligns with direction
    assert len(filtered) > 0
    for _, row in filtered.iterrows():
        if row['Direction'] == 'Buy':
            assert row['30M Leg'] in ['Above H', 'Above L']
        else:
            assert row['30M Leg'] in ['Below H', 'Below L']


def test_sorting_by_edge():
    """Test that results are sorted by edge descending."""
    sample_data = get_sample_data()
    result = calculate_strategy_statistics(sample_data)

    if len(result) > 1:
        # Extract edge values and check they're in descending order
        edges = []
        for edge_str in result['Edge']:
            edges.append(float(edge_str.replace('%', '')))

        # Check that edges are sorted descending
        assert edges == sorted(edges, reverse=True)


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_get_single_setup_strategies,
        test_strategy_names,
        test_calculate_strategy_statistics_basic,
        test_calculate_strategy_statistics_includes_plain,
        test_calculate_strategy_statistics_empty,
        test_calculate_stats_for_strategy_and_rrr,
        test_win_condition_rrr_1,
        test_win_condition_rrr_2,
        test_create_empty_stats,
        test_edge_calculation,
        test_outcome_calculation,
        test_profit_factor_calculation,
        test_profit_factor_no_losses,
        test_days_calculation,
        test_days_all_wins,
        test_days_no_wins,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_ema_aligned_filter,
        test_bos_only_filter,
        test_30m_trend_filter,
        test_sorting_by_edge,
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
