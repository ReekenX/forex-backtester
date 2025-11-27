"""
Tests for utils.hours module

These tests use a small dataset of 10 rows to verify the hour analysis functionality.
Run with: python tests/hours.py
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from utils.hours import (
    calculate_hour_statistics,
    create_html_table,
    RRR_RATIOS,
    _calculate_stats_for_hour_and_rrr,
    _create_empty_stats
)


def get_sample_data():
    """Create a sample dataset with 10 trades."""
    return pd.DataFrame({
        'Hour': [10, 10, 10, 11, 11, 12, 12, 12, 13, 13],
        'SL': [10.0, 5.0, 20.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 6.0],
        'Pullback': [5.0, 5.0, 1.0, 0.5, 7.0, 4.0, 8.0, 11.0, 2.0, 3.0],
        'TP': [20.0, 10.0, 40.0, 12.0, 14.0, 16.0, 18.0, 20.0, 10.0, 12.0]
    })


def get_empty_data():
    """Create an empty dataset."""
    return pd.DataFrame({
        'Hour': [],
        'SL': [],
        'Pullback': [],
        'TP': []
    })


def test_calculate_hour_statistics_basic():
    """Test basic hour statistics calculation."""
    sample_data = get_sample_data()
    result = calculate_hour_statistics(sample_data)

    # Should have rows for each hour × each RRR ratio
    # Hours: 10, 11, 12, 13 = 4 hours
    # RRR ratios: 1:1, 1:2, 1:3 = 3 ratios
    # Total: 4 × 3 = 12 rows
    assert len(result) == 12

    # Check columns exist
    expected_columns = ['Strategy', 'RRR', 'Trades', 'Notation',
                        'Win Rate', 'Outcome', 'Edge', 'Days', 'Days %', 'Trades Required']
    assert list(result.columns) == expected_columns


def test_calculate_hour_statistics_hour_10():
    """Test statistics for hour 10 specifically."""
    sample_data = get_sample_data()
    result = calculate_hour_statistics(sample_data)

    # Filter results for hour 10
    hour_10_rows = result[result['Strategy'] == '10h']
    assert len(hour_10_rows) == 1  # Only first RRR shows hour label

    # Hour 10 has 3 trades
    hour_10_all = result.iloc[0:3]  # First 3 rows are hour 10
    for _, row in hour_10_all.iterrows():
        assert row['Trades'] == 3


def test_calculate_hour_statistics_empty():
    """Test with empty dataset."""
    empty_data = get_empty_data()
    result = calculate_hour_statistics(empty_data)
    assert result.empty


def test_calculate_stats_for_hour_and_rrr():
    """Test statistics calculation for a specific hour and RRR."""
    # Create test data for one hour
    trades = pd.DataFrame({
        'Hour': [10, 10, 10],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 10.0, 2.0],  # 1st wins, 2nd loses (equal), 3rd wins
        'TP': [20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)

    # Should have 2 wins (pullback < SL and TP >= 1*SL)
    assert stats['Notation'] == '2W – 1L'
    assert stats['Trades'] == 3
    assert stats['Strategy'] == '10h'  # Shows hour for RRR=1
    assert stats['RRR'] == '1:1'


def test_calculate_stats_for_hour_and_rrr_second_rrr():
    """Test that second RRR ratio doesn't show hour label."""
    trades = pd.DataFrame({
        'Hour': [10, 10],
        'SL': [10.0, 10.0],
        'Pullback': [5.0, 2.0],
        'TP': [20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02']
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 2, 33.3)

    assert stats['Strategy'] == ''  # No hour label for RRR != 1
    assert stats['RRR'] == '1:2'


def test_win_condition_rrr_1():
    """Test win condition for 1:1 RRR."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 9.9, 10.1],  # Win, Win, Lose
        'TP': [10.0, 10.0, 10.0],  # All meet 1:1 TP requirement
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)
    assert stats['Notation'] == '2W – 1L'


def test_win_condition_rrr_2():
    """Test win condition for 1:2 RRR."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0],  # All pullbacks are winning
        'TP': [20.0, 19.9, 30.0],  # Win (>=20), Lose (<20), Win (>=20)
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 2, 33.3)
    assert stats['Notation'] == '2W – 1L'


def test_create_empty_stats():
    """Test empty stats creation."""
    stats = _create_empty_stats(10, 1, 50.0)

    assert stats['Strategy'] == '10h'
    assert stats['RRR'] == '1:1'
    assert stats['Trades'] == 0
    assert stats['Notation'] == '0W – 0L'
    assert stats['Win Rate'] == "0.0%"
    assert stats['Edge'] == "-50.0%"


def test_edge_calculation():
    """Test edge calculation (win rate - breakeven rate)."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10, 10],
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 15.0],  # 3 wins, 1 loss
        'TP': [10.0, 10.0, 10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)

    # Win rate = 75%, Breakeven = 50%, Edge = 25%
    assert stats['Win Rate'] == "75.0%"
    assert stats['Edge'] == "25.0%"


def test_outcome_calculation():
    """Test outcome calculation (wins * RRR - losses)."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 15.0],  # 2 wins, 1 loss
        'TP': [20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    # For 1:2 RRR: outcome = (2 * 2) - 1 = 3R
    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 2, 33.3)
    assert stats['Outcome'] == "3R"


def test_trades_required_positive_outcome():
    """Test trades required calculation with positive outcome."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10, 10],
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 15.0],  # 3 wins, 1 loss
        'TP': [20.0, 20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
    })

    # Outcome = (3 * 2) - 1 = 5R
    # Trades required = 4 / 5 = 0.8
    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 2, 33.3)
    assert stats['Trades Required'] == "0.8"


def test_trades_required_negative_outcome():
    """Test trades required with negative outcome."""
    trades = pd.DataFrame({
        'Hour': [10, 10],
        'SL': [10.0, 10.0],
        'Pullback': [15.0, 15.0],  # 0 wins, 2 losses
        'TP': [20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02']
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)
    assert stats['Trades Required'] == ""


def test_create_html_table_basic():
    """Test HTML table creation."""
    sample_data = get_sample_data()
    stats = calculate_hour_statistics(sample_data)
    html = create_html_table(stats)

    # Check HTML contains expected elements
    assert '<table' in html
    assert 'hour-analysis-table' in html
    assert '<th>Strategy</th>' in html
    assert '<th>RRR</th>' in html
    assert '<th>Trades</th>' in html


def test_create_html_table_empty():
    """Test HTML table with empty data."""
    empty_df = pd.DataFrame()
    html = create_html_table(empty_df)

    assert "No data available" in html


def test_filter_invalid_hours():
    """Test that invalid hours are filtered out."""
    data = pd.DataFrame({
        'Hour': [0, 10, None, 11],  # 0 and None should be filtered
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 5.0],
        'TP': [20.0, 20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
    })

    result = calculate_hour_statistics(data)

    # Should only have data for hours 10 and 11 (2 hours × 3 RRR = 6 rows)
    assert len(result) == 6


def test_days_calculation():
    """Test Days and Days % calculation."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10, 10],
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 15.0, 15.0],  # 2 wins on different days, 2 losses
        'TP': [10.0, 10.0, 10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-03']  # 3 unique days, 2 with wins
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)

    # 2 unique days with wins out of 3 total days = 66.67% ≈ 67%
    assert stats['Days'] == 2
    assert stats['Days %'] == "67%"


def test_days_all_wins():
    """Test Days % when all trading days have wins."""
    trades = pd.DataFrame({
        'Hour': [10, 10, 10],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0],  # All wins
        'TP': [10.0, 10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']  # 3 unique days, all with wins
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)

    assert stats['Days'] == 3
    assert stats['Days %'] == "100%"


def test_days_no_wins():
    """Test Days % when no trading days have wins."""
    trades = pd.DataFrame({
        'Hour': [10, 10],
        'SL': [10.0, 10.0],
        'Pullback': [15.0, 15.0],  # All losses
        'TP': [10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02']  # 2 unique days, 0 with wins
    })

    stats = _calculate_stats_for_hour_and_rrr(trades, 10, 1, 50.0)

    assert stats['Days'] == 0
    assert stats['Days %'] == "0%"


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_calculate_hour_statistics_basic,
        test_calculate_hour_statistics_hour_10,
        test_calculate_hour_statistics_empty,
        test_calculate_stats_for_hour_and_rrr,
        test_calculate_stats_for_hour_and_rrr_second_rrr,
        test_win_condition_rrr_1,
        test_win_condition_rrr_2,
        test_create_empty_stats,
        test_edge_calculation,
        test_outcome_calculation,
        test_trades_required_positive_outcome,
        test_trades_required_negative_outcome,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_filter_invalid_hours,
        test_days_calculation,
        test_days_all_wins,
        test_days_no_wins,
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
