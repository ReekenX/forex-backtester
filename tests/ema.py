"""
Tests for utils.ema module

These tests use a small dataset of 10 rows to verify the EMA strategy analysis functionality.
Run with: python tests/ema.py
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from utils.ema import (
    calculate_ema_statistics,
    create_html_table,
    create_ema_strategies,
    RRR_RATIOS,
    _calculate_stats_for_strategy_and_rrr,
    _create_empty_stats
)


def get_sample_data():
    """Create a sample dataset with 10 trades."""
    return pd.DataFrame({
        'EMA': ['Buy', 'Buy', 'Sell', 'Buy', 'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell'],
        'Direction': ['Buy', 'Sell', 'Sell', 'Buy', 'Buy', 'Buy', 'Buy', 'Sell', 'Buy', 'Sell'],
        'BOS/CH': ['BOS', 'BOS', 'CH', 'BOS', 'CH', 'BOS', 'BOS', 'CH', 'BOS', 'BOS'],
        '30M Leg': ['Above H', 'Below L', 'Below H', 'Above L', 'Above H', 'Above L', 'Above H', 'Below L', 'Above H', 'Below H'],
        'SL': [10.0, 5.0, 20.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 6.0],
        'Pullback': [5.0, 5.0, 1.0, 0.5, 7.0, 4.0, 8.0, 11.0, 2.0, 3.0],
        'TP': [20.0, 10.0, 40.0, 12.0, 14.0, 16.0, 18.0, 20.0, 10.0, 12.0],
        'Date': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09'],
        'News Event': [None, None, 'NFP', None, None, None, 'CPI', None, None, None],
        'Hours Until News': [0, 0, 1.5, 0, 0, 0, 3.0, 0, 0, 0]
    })


def get_empty_data():
    """Create an empty dataset."""
    return pd.DataFrame({
        'EMA': [],
        'Direction': [],
        'BOS/CH': [],
        '30M Leg': [],
        'SL': [],
        'Pullback': [],
        'TP': [],
        'Date': [],
        'News Event': [],
        'Hours Until News': []
    })


def test_create_ema_strategies():
    """Test that EMA strategies are created correctly."""
    strategies = create_ema_strategies()

    # Should have multiple strategies
    assert len(strategies) > 0

    # Each strategy should be a tuple with (name, function)
    for strategy in strategies:
        assert len(strategy) == 2
        assert isinstance(strategy[0], str)
        assert callable(strategy[1])

    # Check that key strategies exist
    strategy_names = [s[0] for s in strategies]
    assert "EMA Aligned" in strategy_names
    assert "EMA Counter-Trend" in strategy_names
    assert "EMA + BOS" in strategy_names
    assert "EMA + CH" in strategy_names


def test_calculate_ema_statistics_basic():
    """Test basic EMA statistics calculation."""
    sample_data = get_sample_data()
    result = calculate_ema_statistics(sample_data)

    # Should have rows for each strategy × each RRR ratio
    # Number of strategies × 3 RRR ratios
    strategies = create_ema_strategies()
    expected_rows = len(strategies) * 3
    assert len(result) == expected_rows

    # Check columns exist
    expected_columns = ['Strategy', 'RRR', 'Trades', 'Notation',
                        'Win Rate', 'Outcome', 'Edge', 'Days', 'Days %', 'Trades Required', 'Drawdown']
    assert list(result.columns) == expected_columns


def test_calculate_ema_statistics_empty():
    """Test with empty dataset."""
    empty_data = get_empty_data()
    result = calculate_ema_statistics(empty_data)

    # Should have entries for all strategies but with 0 trades
    strategies = create_ema_strategies()
    expected_rows = len(strategies) * 3
    assert len(result) == expected_rows

    # All trades should be 0
    for _, row in result.iterrows():
        assert row['Trades'] == 0


def test_ema_aligned_filter():
    """Test EMA Aligned strategy filter."""
    sample_data = get_sample_data()
    strategies = create_ema_strategies()

    # Find EMA Aligned strategy
    ema_aligned = [s for s in strategies if s[0] == "EMA Aligned"][0]
    filtered = ema_aligned[1](sample_data)

    # Should only include trades where EMA == Direction
    # From sample data: rows 0, 2, 3, 4, 6, 7, 8, 9
    assert len(filtered) == 8


def test_ema_counter_trend_filter():
    """Test EMA Counter-Trend strategy filter."""
    sample_data = get_sample_data()
    strategies = create_ema_strategies()

    # Find EMA Counter-Trend strategy
    ema_counter = [s for s in strategies if s[0] == "EMA Counter-Trend"][0]
    filtered = ema_counter[1](sample_data)

    # Should only include trades where EMA != Direction
    # From sample data: rows 1, 5
    assert len(filtered) == 2


def test_ema_bos_filter():
    """Test EMA + BOS strategy filter."""
    sample_data = get_sample_data()
    strategies = create_ema_strategies()

    # Find EMA + BOS strategy
    ema_bos = [s for s in strategies if s[0] == "EMA + BOS"][0]
    filtered = ema_bos[1](sample_data)

    # Should only include trades where EMA == Direction AND BOS/CH == "BOS"
    # From sample data: rows 0, 3, 6, 8, 9
    assert len(filtered) == 5


def test_calculate_stats_for_strategy_and_rrr():
    """Test statistics calculation for a specific strategy and RRR."""
    # Create test data for one strategy
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy', 'Buy'],
        'Direction': ['Buy', 'Buy', 'Buy'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 10.0, 2.0],  # 1st wins, 2nd loses (equal), 3rd wins
        'TP': [20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test Strategy", 1, 50.0)

    # Should have 2 wins (pullback < SL and TP >= 1*SL)
    assert stats['Notation'] == '2W – 1L'
    assert stats['Trades'] == 3
    assert stats['Strategy'] == 'Test Strategy'
    assert stats['RRR'] == '1:1'


def test_create_empty_stats():
    """Test empty stats creation."""
    stats = _create_empty_stats("Empty Strategy", 1, 50.0)

    assert stats['Strategy'] == 'Empty Strategy'
    assert stats['RRR'] == '1:1'
    assert stats['Trades'] == 0
    assert stats['Notation'] == '0W – 0L'
    assert stats['Win Rate'] == "0.0%"
    assert stats['Edge'] == "-50.0%"


def test_win_condition_rrr_1():
    """Test win condition for 1:1 RRR."""
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy', 'Buy'],
        'Direction': ['Buy', 'Buy', 'Buy'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 9.9, 10.1],  # Win, Win, Lose
        'TP': [10.0, 10.0, 10.0],  # All meet 1:1 TP requirement
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 1, 50.0)
    assert stats['Notation'] == '2W – 1L'


def test_win_condition_rrr_2():
    """Test win condition for 1:2 RRR."""
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy', 'Buy'],
        'Direction': ['Buy', 'Buy', 'Buy'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0],  # All pullbacks are winning
        'TP': [20.0, 19.9, 30.0],  # Win (>=20), Lose (<20), Win (>=20)
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 2, 33.3)
    assert stats['Notation'] == '2W – 1L'


def test_edge_calculation():
    """Test edge calculation (win rate - breakeven rate)."""
    trades = pd.DataFrame({
        'EMA': ['Buy'] * 4,
        'Direction': ['Buy'] * 4,
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 15.0],  # 3 wins, 1 loss
        'TP': [10.0, 10.0, 10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 1, 50.0)

    # Win rate = 75%, Breakeven = 50%, Edge = 25%
    assert stats['Win Rate'] == "75.0%"
    assert stats['Edge'] == "25.0%"


def test_outcome_calculation():
    """Test outcome calculation (wins * RRR - losses)."""
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy', 'Buy'],
        'Direction': ['Buy', 'Buy', 'Buy'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 15.0],  # 2 wins, 1 loss
        'TP': [20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']
    })

    # For 1:2 RRR: outcome = (2 * 2) - 1 = 3R
    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 2, 33.3)
    assert stats['Outcome'] == "3R"


def test_trades_required_positive_outcome():
    """Test trades required calculation with positive outcome."""
    trades = pd.DataFrame({
        'EMA': ['Buy'] * 4,
        'Direction': ['Buy'] * 4,
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0, 15.0],  # 3 wins, 1 loss
        'TP': [20.0, 20.0, 20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
    })

    # Outcome = (3 * 2) - 1 = 5R
    # Trades required = 4 / 5 = 0.8
    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 2, 33.3)
    assert stats['Trades Required'] == "0.8"


def test_trades_required_negative_outcome():
    """Test trades required with negative outcome."""
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy'],
        'Direction': ['Buy', 'Buy'],
        'SL': [10.0, 10.0],
        'Pullback': [15.0, 15.0],  # 0 wins, 2 losses
        'TP': [20.0, 20.0],
        'Date': ['2025-01-01', '2025-01-02']
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 1, 50.0)
    assert stats['Trades Required'] == "N/A"


def test_create_html_table_basic():
    """Test HTML table creation."""
    sample_data = get_sample_data()
    stats = calculate_ema_statistics(sample_data)
    html = create_html_table(stats)

    # Check HTML contains expected elements
    assert '<table' in html
    assert 'ema-analysis-table' in html
    assert 'Strategy' in html
    assert '<th>RRR</th>' in html
    assert '<th>Trades</th>' in html


def test_create_html_table_empty():
    """Test HTML table with empty data."""
    empty_df = pd.DataFrame()
    html = create_html_table(empty_df)

    assert "No data available" in html


def test_days_calculation():
    """Test Days and Days % calculation."""
    trades = pd.DataFrame({
        'EMA': ['Buy'] * 4,
        'Direction': ['Buy'] * 4,
        'SL': [10.0, 10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 15.0, 15.0],  # 2 wins on different days, 2 losses
        'TP': [10.0, 10.0, 10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-03']  # 3 unique days, 2 with wins
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 1, 50.0)

    # 2 unique days with wins out of 3 total days = 66.67% ≈ 67%
    assert stats['Days'] == 2
    assert stats['Days %'] == "67%"


def test_days_all_wins():
    """Test Days % when all trading days have wins."""
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy', 'Buy'],
        'Direction': ['Buy', 'Buy', 'Buy'],
        'SL': [10.0, 10.0, 10.0],
        'Pullback': [5.0, 5.0, 5.0],  # All wins
        'TP': [10.0, 10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03']  # 3 unique days, all with wins
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 1, 50.0)

    assert stats['Days'] == 3
    assert stats['Days %'] == "100%"


def test_days_no_wins():
    """Test Days % when no trading days have wins."""
    trades = pd.DataFrame({
        'EMA': ['Buy', 'Buy'],
        'Direction': ['Buy', 'Buy'],
        'SL': [10.0, 10.0],
        'Pullback': [15.0, 15.0],  # All losses
        'TP': [10.0, 10.0],
        'Date': ['2025-01-01', '2025-01-02']  # 2 unique days, 0 with wins
    })

    stats = _calculate_stats_for_strategy_and_rrr(trades, "Test", 1, 50.0)

    assert stats['Days'] == 0
    assert stats['Days %'] == "0%"


def test_ema_30m_trend_filter():
    """Test EMA + 30M Trend strategy filter."""
    sample_data = get_sample_data()
    strategies = create_ema_strategies()

    # Find EMA + 30M Trend strategy
    ema_30m = [s for s in strategies if s[0] == "EMA + 30M Trend"][0]
    filtered = ema_30m[1](sample_data)

    # Should only include trades where:
    # - EMA == Direction
    # - Buy trades with 30M Leg in ["Above H", "Above L"]
    # - Sell trades with 30M Leg in ["Below H", "Below L"]
    # From sample data: rows 0, 2, 3, 4, 6, 7, 8, 9
    assert len(filtered) == 8


def test_ema_sl_filter():
    """Test EMA + SL ≤ 10 strategy filter."""
    sample_data = get_sample_data()
    strategies = create_ema_strategies()

    # Find EMA + SL ≤ 10 strategy
    ema_sl = [s for s in strategies if s[0] == "EMA + SL ≤ 10"][0]
    filtered = ema_sl[1](sample_data)

    # Should only include trades where EMA == Direction AND SL <= 10
    # From sample data: rows 0, 3, 4, 6, 7, 8, 9
    assert len(filtered) == 7


def test_ema_news_filter():
    """Test EMA + No News strategy filter."""
    sample_data = get_sample_data()
    strategies = create_ema_strategies()

    # Find EMA + No News strategy
    ema_no_news = [s for s in strategies if s[0] == "EMA + No News"][0]
    filtered = ema_no_news[1](sample_data)

    # Should only include trades where EMA == Direction AND News Event is NaN
    # From sample data: rows 0, 3, 4, 7, 8, 9
    assert len(filtered) == 6


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_create_ema_strategies,
        test_calculate_ema_statistics_basic,
        test_calculate_ema_statistics_empty,
        test_ema_aligned_filter,
        test_ema_counter_trend_filter,
        test_ema_bos_filter,
        test_calculate_stats_for_strategy_and_rrr,
        test_create_empty_stats,
        test_win_condition_rrr_1,
        test_win_condition_rrr_2,
        test_edge_calculation,
        test_outcome_calculation,
        test_trades_required_positive_outcome,
        test_trades_required_negative_outcome,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_days_calculation,
        test_days_all_wins,
        test_days_no_wins,
        test_ema_30m_trend_filter,
        test_ema_sl_filter,
        test_ema_news_filter,
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
