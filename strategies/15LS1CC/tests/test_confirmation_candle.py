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
    load_data,
    calculate_statistics,
    calculate_buffer_statistics,
    calculate_fixed_sl_statistics,
    calculate_weekday_statistics,
    _calculate_buffer_statistics_filtered,
    create_html_table,
    get_strategies,
    get_buffer_strategies,
    _calculate_stats,
    _calculate_stats_with_buffer,
    _calculate_fixed_sl_stats,
    _breakeven_rate,
    RRR_RATIOS,
    BUFFER_PIPS,
    MIN_SL_VALUES,
    FIXED_SL_STRATEGY_VALUES,
    MAX_SL_STRATEGY_VALUES,
    _max_sl_filter,
    MAX_SL_VALUES,
    FIXED_SL_SIZES,
    WEEKDAY_ORDER,
    _format_wl,
    SL_RANGES,
    SL_BUFFER_PIPS,
    SL_BUFFER_SMALL_SL_THRESHOLD,
    SL_FIXED_PIPS,
    SL_REDUCTION_PIPS,
    calculate_sl_statistics,
    _create_sl_sortable_table,
    calculate_sl_buffer_small_sl_statistics,
    calculate_sl_fixed_statistics,
    calculate_sl_buffer_statistics,
    calculate_sl_reduction_statistics,
    PULLBACK_ENTRY_PIPS,
    calculate_pullback_statistics,
    TP_RANGES,
    calculate_tp_statistics,
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
        'SL': [], 'Pullback': [], 'TP': [], 'R': [],
    })


CSV_SAMPLE = """Date,Weekday,Trade,Direction,SL,Pullback,TP,47.3%
2026-07-27,Monday,#1,Sell,4.4,0.7,34,7R
2026-07-27,Monday,#2,Sell,7.1,7.1,,
2026-07-28,Tuesday,#1,Buy,2.6,2.7,31,-10R
"""


def write_sample_csv(tmp_path):
    """Write the sample CSV (spreadsheet-shaped header and R suffixes) to disk."""
    path = tmp_path / "data.csv"
    path.write_text(CSV_SAMPLE)
    return str(path)


def test_load_data_columns(tmp_path):
    """The trailing win-rate header cell is recovered as the R column."""
    df = load_data(write_sample_csv(tmp_path))
    assert list(df.columns) == [
        'Date', 'Weekday', 'Trade', 'Direction', 'SL', 'Pullback', 'TP', 'R',
    ]


def test_load_data_strips_r_suffix(tmp_path):
    """R values exported as "7R" / "-10R" become numbers."""
    df = load_data(write_sample_csv(tmp_path))
    assert df['R'].tolist() == [7.0, 0.0, -10.0]


def test_load_data_numeric_columns(tmp_path):
    """SL, Pullback and TP are numeric with blanks filled as 0."""
    df = load_data(write_sample_csv(tmp_path))
    assert df['SL'].tolist() == [4.4, 7.1, 2.6]
    assert df['Pullback'].tolist() == [0.7, 7.1, 2.7]
    assert df['TP'].tolist() == [34.0, 0.0, 31.0]


def test_load_data_keeps_named_r_column(tmp_path):
    """A CSV that already names the column R is loaded unchanged."""
    path = tmp_path / "named.csv"
    path.write_text("Date,Weekday,Trade,Direction,SL,Pullback,TP,R\n"
                    "2026-07-27,Monday,#1,Sell,4.4,0.7,34,7\n")
    df = load_data(str(path))
    assert list(df.columns)[-1] == 'R'
    assert df['R'].tolist() == [7.0]


def test_load_data_real_csv():
    """The project CSV loads with the expected columns and numeric types."""
    import os
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data.csv')
    df = load_data(csv_path)
    assert list(df.columns) == [
        'Date', 'Weekday', 'Trade', 'Direction', 'SL', 'Pullback', 'TP', 'R',
    ]
    assert len(df) > 0
    for col in ['SL', 'Pullback', 'TP', 'R']:
        assert df[col].notna().all()


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


def test_strategy_names_include_sl_combos():
    """Test that SL combination strategies are present."""
    strategies = get_strategies()
    names = [name for name, _ in strategies]
    assert 'All Trades + SL < 5' in names
    assert 'All Trades + SL > 3' in names
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


def test_sl_filter_combination():
    """Test SL filter in combination strategy."""
    sample = get_sample_data()
    strategies = get_strategies()
    strategy = [func for name, func in strategies if name == 'All Trades + SL > 3'][0]
    filtered = strategy(sample)

    for _, row in filtered.iterrows():
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


def test_create_html_table_no_sort_by_default():
    """Without sort_id, no sort script or clickable headers are emitted."""
    df = pd.DataFrame({'Strategy': ['A'], 'Win Rate': ['40.0%']})
    html = create_html_table(df)
    assert 'sortAnalysisTable' not in html
    assert 'class="sortable"' not in html


def test_create_html_table_sortable_win_rate():
    """With sort_id, a percentage column (Win Rate) becomes click-to-sort DESC,
    while non-percentage columns stay plain."""
    df = pd.DataFrame({
        'Strategy': ['A', 'B'],
        'Trades': [10, 20],
        'Win Rate': ['40.0%', '55.5%'],
    })
    html = create_html_table(df, sort_id='strategies-table')

    assert 'id="strategies-table"' in html
    assert 'function sortAnalysisTable' in html
    assert 'return pct(b) - pct(a);' in html  # DESC only
    # Win Rate (index 2) is the only sortable column.
    assert "sortAnalysisTable('strategies-table', 2, this)" in html
    assert 'Win Rate ↓' in html
    # Strategy and Trades headers stay plain.
    assert '>Trades</th>' in html
    assert 'sortAnalysisTable(\'strategies-table\', 0' not in html
    assert 'sortAnalysisTable(\'strategies-table\', 1' not in html


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


def test_get_buffer_strategies():
    """Test that buffer strategies include key strategies."""
    strategies = get_buffer_strategies()
    names = [name for name, _ in strategies]

    assert 'All Trades' in names
    assert 'Fixed SL 2' in names
    assert 'Max SL 3' in names


def test_calculate_buffer_statistics():
    """Test that buffer statistics returns expected DataFrame."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)

    assert isinstance(result, pd.DataFrame)


def test_buffer_pips_constant():
    """Test that BUFFER_PIPS has expected values."""
    assert BUFFER_PIPS == [0, 1, 2, 3]


def test_min_sl_values_constant():
    """Min SL gating is tested at 0 (no filter), 1, 2, 3 pips."""
    assert MIN_SL_VALUES == [0]


def test_buffer_statistics_has_min_sl_column():
    """Strategies table should expose a Min SL column right after Buffer."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    cols = list(result.columns)
    assert "Min SL" in cols
    assert cols.index("Min SL") == cols.index("Buffer") + 1


def test_buffer_statistics_covers_every_min_sl_per_strategy():
    """Every configured strategy must appear at every Min SL value."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    strategy_names = [name for name, _ in get_buffer_strategies()]
    for strategy in strategy_names:
        present = set(result[result["Strategy"] == strategy]["Min SL"].unique())
        assert present == set(MIN_SL_VALUES), f"{strategy} missing Min SL values: {set(MIN_SL_VALUES) - present}"


def test_buffer_statistics_min_sl_filters_trades():
    """MIN_SL_VALUES is [0] for the v5 CSV, so no Min SL gate is applied and
    every strategy sees the full trade set."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    rows = result[
        (result["Strategy"] == "All Trades")
        & (result["Buffer"] == "+0")
        & (result["Max SL"] == 0)
        & (result["RRR"] == "1:1")
    ].set_index("Min SL")
    assert list(rows.index) == MIN_SL_VALUES
    assert rows.loc[0, "Trades"] == 10


def test_max_sl_values_constant():
    """Max SL gating values: 0 (disabled) plus 10, 15, 20 pip caps."""
    assert MAX_SL_VALUES == [0, 5]


def test_buffer_statistics_has_max_sl_column():
    """Max SL column lives directly after Min SL."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    cols = list(result.columns)
    assert "Max SL" in cols
    assert cols.index("Max SL") == cols.index("Min SL") + 1


def test_buffer_statistics_covers_every_max_sl_per_strategy():
    """Every strategy must appear at every Max SL value."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    for strategy, _ in get_buffer_strategies():
        present = set(result[result["Strategy"] == strategy]["Max SL"].unique())
        assert present == set(MAX_SL_VALUES), f"{strategy} missing Max SL values: {set(MAX_SL_VALUES) - present}"


def test_buffer_statistics_max_sl_filters_trades():
    """Max SL > 0 drops trades whose original SL exceeds it.

    Sample SL values are 3.5, 1.1, 2.0, 4.0, 3.0, 5.0, 2.5, 6.0, 8.0, 1.5:
      Max SL 0 -> 10 (no cap)
      Max SL 5 -> 8 (drops 6.0 and 8.0; 5.0 is kept since the gate is SL <= 5)
    """
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    rows = result[
        (result["Strategy"] == "All Trades")
        & (result["Buffer"] == "+0")
        & (result["Min SL"] == 0)
        & (result["RRR"] == "1:1")
    ].set_index("Max SL")
    assert list(rows.index) == MAX_SL_VALUES
    assert rows.loc[0, "Trades"] == 10
    assert rows.loc[5, "Trades"] == 8


def test_buffer_statistics_min_and_max_sl_compose():
    """Every (Min SL, Max SL) pair in the configured grid produces one row."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    for min_sl in MIN_SL_VALUES:
        for max_sl in MAX_SL_VALUES:
            row = result[
                (result["Strategy"] == "All Trades")
                & (result["Buffer"] == "+0")
                & (result["Min SL"] == min_sl)
                & (result["Max SL"] == max_sl)
                & (result["RRR"] == "1:1")
            ]
            assert len(row) == 1, f"missing row for Min {min_sl} / Max {max_sl}"


def test_max_sl_strategy_values_constant():
    """Max SL strategy caps run at 3..10 pips."""
    assert MAX_SL_STRATEGY_VALUES == [3, 4, 5, 6, 7, 8, 9, 10]


def test_max_sl_strategies_present():
    """Max SL 3..10 are configured strategies."""
    names = {n for n, _ in get_buffer_strategies()}
    for x in range(3, 11):
        assert f"Max SL {x}" in names


def test_max_sl_filter_caps_sl():
    """_max_sl_filter(x) caps SL at x (min(SL, x)); tighter stops stay unchanged."""
    sample = get_sample_data()
    capped = _max_sl_filter(3)(sample)
    assert capped['SL'].max() <= 3.0
    # A trade whose safe stop is already below the cap is unchanged (idx1 SL=1.1).
    assert capped['SL'].iloc[1] == 1.1
    # A trade wider than the cap is clipped (idx8 SL=8.0 -> 3.0).
    assert capped['SL'].iloc[8] == 3.0


def test_max_sl_strategy_win_rate():
    """Max SL 3 caps every stop at 3. Sample SL/Pullback/TP with capped SL:
      idx0 3.5->3.0 PB3.5 => stopped (loss)
      idx1 1.1      PB0.8 TP12 => win
      idx2 2.0      PB2.1 => stopped (loss)
      idx3 4.0->3.0 PB1.5 TP10 => win
      idx4 3.0      PB3.0 => stopped (loss)
      idx5 5.0->3.0 PB2.0 TP8  => win
      idx6 2.5      PB2.5 => stopped (loss)
      idx7 6.0->3.0 PB3.0 => stopped (loss)
      idx8 8.0->3.0 PB7.0 => stopped (loss)
      idx9 1.5      PB0.5 TP5  => win
    -> 4W - 6L (40.0%) over all 10 trades at 1:1.
    """
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    row = result[
        (result["Strategy"] == "Max SL 3")
        & (result["Buffer"] == "+0")
        & (result["Min SL"] == 0)
        & (result["Max SL"] == 0)
        & (result["RRR"] == "1:1")
    ]
    assert len(row) == 1
    assert row.iloc[0]["Trades"] == 10
    assert row.iloc[0]["Notation"] == "4W – 6L"
    assert row.iloc[0]["Win Rate"] == "40.0%"


def test_max_sl_strategy_buffer_zero_only():
    """Max SL strategies run at buffer +0 only (a buffer would undo the cap)."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    buffers = set(result[result["Strategy"] == "Max SL 5"]["Buffer"].unique())
    assert buffers == {"+0"}


def test_buffer_statistics_min_sl_filter_applies_before_fixed_sl():
    """Fixed-SL strategies gate on the original SL, then replace SL with the
    fixed value. Max SL 5 drops the 6.0 and 8.0 trades before Fixed SL 2 applies."""
    sample = get_sample_data()
    result = calculate_buffer_statistics(sample)
    fixed_2 = result[
        (result["Strategy"] == "Fixed SL 2")
        & (result["RRR"] == "1:1")
        & (result["Min SL"] == 0)
    ].set_index("Max SL")
    assert fixed_2.loc[0, "Trades"] == 10
    assert fixed_2.loc[5, "Trades"] == 8


def test_rrr_ratios_constant():
    """1:3 is the ceiling for this project (see CLAUDE.md)."""
    assert RRR_RATIOS == [1, 2, 3]


def test_breakeven_rate():
    """Test breakeven rate calculation for different RRR ratios."""
    assert _breakeven_rate(1) == 50.0
    assert abs(_breakeven_rate(2) - 33.333) < 0.01
    assert _breakeven_rate(3) == 25.0


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
    expected = [f'1:{r}' for r in RRR_RATIOS]
    for rrr in rrr_values:
        assert rrr in expected, f"Unexpected RRR: {rrr}"


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
    assert list(result.columns) == ['Day', 'Trades', 'Notation', 'Win Rate']


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


def test_weekday_statistics_notation_and_win_rate():
    """Win condition: Pullback < SL AND TP > 0.

    Monday: #1 PB=3.5 SL=3.5 TP=0 => L, #2 PB=0.8 SL=1.1 TP=12 => W, #3 PB=2.1 SL=2.0 TP=0 => L
    Tuesday: #1 PB=1.5 SL=4.0 TP=10 => W, #2 PB=3.0 SL=3.0 TP=0 => L
    Thursday: #1 PB=3.0 SL=6.0 TP=15 => W, #2 PB=7.0 SL=8.0 TP=10 => W
    Friday: #1 PB=0.5 SL=1.5 TP=5 => W
    """
    sample = get_sample_data()
    result = calculate_weekday_statistics(sample)
    rows = {row['Day']: row for _, row in result.iterrows()}

    assert rows['Monday']['Notation'] == '1W - 2L'
    assert rows['Monday']['Win Rate'] == '33.3%'
    assert rows['Tuesday']['Notation'] == '1W - 1L'
    assert rows['Tuesday']['Win Rate'] == '50.0%'
    assert rows['Thursday']['Notation'] == '2W - 0L'
    assert rows['Thursday']['Win Rate'] == '100.0%'
    assert rows['Friday']['Notation'] == '1W - 0L'
    assert rows['Friday']['Win Rate'] == '100.0%'


def test_weekday_statistics_stopped_out_is_a_loss():
    """A trade whose Pullback reaches its SL is a loss regardless of TP."""
    trades = pd.DataFrame({
        'Date': ['2026-01-12'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.0],
        'Pullback': [4.0],
        'TP': [10.0],
        'R': [-3.0],
    })

    rows = {r['Day']: r for _, r in calculate_weekday_statistics(trades).iterrows()}
    assert rows['Monday']['Notation'] == '0W - 1L'
    assert rows['Monday']['Win Rate'] == '0.0%'


def test_weekday_statistics_no_tp_is_a_loss():
    """A trade that survived its stop but never went profitable is a loss."""
    trades = pd.DataFrame({
        'Date': ['2026-01-12'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.0],
        'Pullback': [1.0],
        'TP': [0.0],
        'R': [0],
    })

    rows = {r['Day']: r for _, r in calculate_weekday_statistics(trades).iterrows()}
    assert rows['Monday']['Notation'] == '0W - 1L'
    assert rows['Monday']['Win Rate'] == '0.0%'


def test_weekday_statistics_empty():
    """Test weekday statistics with empty dataset."""
    empty = get_empty_data()
    result = calculate_weekday_statistics(empty)
    assert len(result) == 5
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W - 0L'
        assert row['Win Rate'] == '0.0%'


def test_weekday_statistics_single_day():
    """Test weekday statistics when only one day has trades."""
    trades = pd.DataFrame({
        'Date': ['2026-01-12', '2026-01-12'],
        'Weekday': ['Monday', 'Monday'],
        'Trade': ['#1', '#2'],
        'Direction': ['Buy', 'Buy'],
        'SL': [3.0, 3.0],
        'Pullback': [1.0, 4.0],
        'TP': [5.0, 5.0],
        'R': [1.7, 0],
    })

    result = calculate_weekday_statistics(trades)
    rows = {row['Day']: row for _, row in result.iterrows()}

    assert rows['Monday']['Trades'] == 2
    assert rows['Monday']['Notation'] == '1W - 1L'
    assert rows['Monday']['Win Rate'] == '50.0%'
    assert rows['Tuesday']['Trades'] == 0
    assert rows['Tuesday']['Notation'] == '0W - 0L'


def test_sl_ranges_constant():
    """SL_RANGES covers the cumulative 0-5 .. 0-10 bands."""
    assert SL_RANGES == [(f"0-{x}", 0, x) for x in range(5, 11)]
    assert len(SL_RANGES) == 6


def test_sl_statistics_columns():
    """Same column shape as the weekday table."""
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    assert list(result.columns) == ['SL Range', 'Trades', 'Notation', 'Win Rate']
    assert list(result.columns)[1:] == list(
        calculate_weekday_statistics(sample).columns)[1:]


def test_sl_statistics_win_needs_a_full_1_1_target():
    """TP must reach SL, not merely be positive. SL 4, TP 3 is a loss."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Tuesday'],
        'Trade': ['#1', '#1'],
        'Direction': ['Buy', 'Buy'],
        'SL': [4.0, 4.0],
        'Pullback': [1.0, 1.0],
        'TP': [3.0, 4.0],
        'R': [0.0, 1.0],
    })

    rows = {r['SL Range']: r for _, r in calculate_sl_statistics(trades).iterrows()}
    assert rows['0-5']['Notation'] == '1W - 1L'
    assert rows['0-5']['Win Rate'] == '50.0%'


def test_sl_statistics_all_ranges_present():
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    assert len(result) == len(SL_RANGES) + 1
    assert list(result['SL Range']) == ['Default'] + [f"0-{x}" for x in range(5, 11)]


def test_sl_statistics_trade_counts():
    """Sample SL values: 3.5, 1.1, 2.0, 4.0, 3.0, 5.0, 2.5, 6.0, 8.0, 1.5.

    Cumulative SL < X: 0-5: 7, 0-6: 8, 0-7: 9, 0-8: 9, 0-9: 10, 0-10: 10.
    """
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    counts = dict(zip(result['SL Range'], result['Trades']))
    assert counts['0-5'] == 7
    assert counts['0-6'] == 8
    assert counts['0-7'] == 9
    assert counts['0-8'] == 9
    assert counts['0-9'] == 10
    assert counts['0-10'] == 10


def test_sl_statistics_notation():
    """Win at 1:1 requires Pullback < SL AND TP >= SL. Sample SL/Pullback/TP:
    3.5/3.5/0, 1.1/0.8/12, 2.0/2.1/0, 4.0/1.5/10, 3.0/3.0/0, 5.0/2.0/8,
    2.5/2.5/0, 6.0/3.0/15, 8.0/7.0/10, 1.5/0.5/5.

    Stopped out (Pullback >= SL): idx 0, 2, 4, 6 - all have TP 0 anyway.
    Winners: idx 1, 3, 5, 7, 8, 9, entering the bands as the cap widens.
    """
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    assert rows['0-5']['Notation'] == '3W - 4L'
    assert rows['0-5']['Win Rate'] == '42.9%'
    assert rows['0-6']['Notation'] == '4W - 4L'
    assert rows['0-6']['Win Rate'] == '50.0%'
    assert rows['0-7']['Notation'] == '5W - 4L'
    assert rows['0-7']['Win Rate'] == '55.6%'
    assert rows['0-10']['Notation'] == '6W - 4L'
    assert rows['0-10']['Win Rate'] == '60.0%'


def test_sl_statistics_cumulative_bands():
    """Each band is cumulative (0 <= SL < X), so trade counts never shrink as
    the cap widens. Sample SL: 3.5,1.1,2.0,4.0,3.0,5.0,2.5,6.0,8.0,1.5."""
    sample = get_sample_data()
    result = calculate_sl_statistics(sample)
    counts = dict(zip(result['SL Range'], result['Trades']))

    assert counts['0-5'] == 7   # excludes 5.0, 6.0, 8.0
    assert counts['0-6'] == 8   # 5.0 joins
    assert counts['0-7'] == 9   # 6.0 joins
    assert counts['0-10'] == 10  # 8.0 joins

    widths = [counts[label] for label, _, _ in SL_RANGES]
    assert widths == sorted(widths)


def test_sl_statistics_requires_surviving_stop():
    """A trade stopped out before running to target is a loss even though its
    TP is far away. The data marks these with a negative R."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [2.0],
        'Pullback': [5.0],  # Pullback > SL: the stop was hit first
        'TP': [10.0],
        'R': [-5.0],
    })

    rows = {r['SL Range']: r for _, r in calculate_sl_statistics(trades).iterrows()}
    assert rows['0-5']['Trades'] == 1
    assert rows['0-5']['Notation'] == '0W - 1L'
    assert rows['0-5']['Win Rate'] == '0.0%'


def test_sl_statistics_survivor_wins():
    """The same trade with a Pullback inside the stop is a win."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [2.0],
        'Pullback': [1.0],
        'TP': [10.0],
        'R': [5.0],
    })

    rows = {r['SL Range']: r for _, r in calculate_sl_statistics(trades).iterrows()}
    assert rows['0-5']['Notation'] == '1W - 0L'
    assert rows['0-5']['Win Rate'] == '100.0%'


def test_sl_statistics_empty():
    empty = get_empty_data()
    result = calculate_sl_statistics(empty)
    assert len(result) == len(SL_RANGES) + 1
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W - 0L'
        assert row['Win Rate'] == '0.0%'


def test_sl_statistics_large_sl():
    """SL=12 and SL=15 fall outside every 0-X band, but Default still sees them."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Monday'],
        'Trade': ['#1', '#2'],
        'Direction': ['Buy', 'Buy'],
        'SL': [12.0, 15.0],
        'Pullback': [5.0, 16.0],
        'TP': [20.0, 20.0],
        'R': [1.7, 0],
    })

    result = calculate_sl_statistics(trades)
    rows = {row['SL Range']: row for _, row in result.iterrows()}

    assert rows['Default']['Trades'] == 2
    assert rows['Default']['Notation'] == '1W - 1L'
    for label in [f"0-{x}" for x in range(5, 11)]:
        assert rows[label]['Trades'] == 0


def test_sl_statistics_no_tp_is_loss():
    """TP=0 is a loss regardless of Pullback."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [2.0],
        'Pullback': [1.0],
        'TP': [0],
        'R': [0],
    })

    rows = {r['SL Range']: r for _, r in calculate_sl_statistics(trades).iterrows()}
    assert rows['0-5']['Notation'] == '0W - 1L'
    assert rows['0-10']['Notation'] == '0W - 1L'


def test_sl_sortable_table_notation_headers_clickable():
    """With sortable on, only the Win Rate column is click-to-sort."""
    sample = get_sample_data()
    stats = calculate_sl_statistics(sample)
    html = _create_sl_sortable_table(stats, "sl-range-stats", sortable=True)

    for idx, col in enumerate(stats.columns):
        if col == "Win Rate":
            assert f"sortSlRange('sl-range-stats', {idx}, this)" in html
            assert f"{col} ↓" in html
        else:
            assert f">{col}</th>" in html
            assert f"sortSlRange('sl-range-stats', {idx}, this)" not in html


def test_sl_sortable_table_can_be_turned_off():
    """sortable=False emits no handlers, no sort script and no arrows."""
    stats = calculate_sl_statistics(get_sample_data())
    html = _create_sl_sortable_table(stats, "sl-range-stats", sortable=False)

    assert "onclick" not in html
    assert "sortSlRange" not in html
    assert "↓" not in html
    assert 'class="sortable"' not in html
    # The data is all still there.
    for col in stats.columns:
        assert f">{col}</th>" in html


def test_sl_sortable_table_still_sorts_combined_notation_columns():
    """Tables that keep the "12W - 3L (80.0%)" form stay sortable too."""
    sample = get_sample_data()
    stats = calculate_pullback_statistics(sample)
    html = _create_sl_sortable_table(stats, "pullback-analysis")

    for idx, col in enumerate(stats.columns):
        if col in ("Pullback", "Trades"):
            assert f"sortSlRange('pullback-analysis', {idx}, this)" not in html
        else:
            assert f"sortSlRange('pullback-analysis', {idx}, this)" in html


def test_sl_sortable_table_uses_given_id():
    """The provided table_id is applied to the table element."""
    sample = get_sample_data()
    html = _create_sl_sortable_table(calculate_sl_statistics(sample), "my-custom-id")
    assert 'id="my-custom-id"' in html


def test_sl_sortable_table_sorts_descending():
    """The embedded sort script orders by win-rate percentage, descending only."""
    sample = get_sample_data()
    html = _create_sl_sortable_table(calculate_sl_statistics(sample), "sl-range-stats")
    assert "function sortSlRange" in html
    # DESC: comparator is pct(b) - pct(a).
    assert "return pct(b) - pct(a);" in html
    # Parses a percentage whether or not it is wrapped in parentheses.
    assert r"match(/([\d.]+)%/)" in html


def test_sl_sortable_table_empty():
    html = _create_sl_sortable_table(pd.DataFrame(), "sl-range-stats")
    assert "No data" in html
    assert "sortSlRange" not in html


def _pullback_rows(result):
    """Index pullback statistics rows by Pullback level."""
    return {row['Pullback']: row for _, row in result.iterrows()}


def test_sl_reduction_pips_constant():
    """Five reduction values starting at 0 (no reduction)."""
    assert SL_REDUCTION_PIPS == [0, 1, 2, 3, 4]
    assert len(SL_REDUCTION_PIPS) == 5


def test_sl_reduction_columns():
    """Same column shape as the weekday and SL range tables."""
    result = calculate_sl_reduction_statistics(get_sample_data())
    assert list(result.columns) == ['SL Reduction', 'Trades', 'Notation', 'Win Rate']


def test_sl_reduction_rows():
    result = calculate_sl_reduction_statistics(get_sample_data())
    assert list(result['SL Reduction']) == [
        'Default', '1 pip', '2 pips', '3 pips', '4 pips']
    # Every row scores the whole dataset; a reduction never drops trades.
    assert (result['Trades'] == 10).all()


def test_sl_reduction_worked_example():
    """The brief's example: SL 3.6, Pullback 1.2, TP far enough to clear the
    target at every stop. Wins at -0/-1/-2, then the 1.2 pullback takes out the
    0.6 stop at -3, and -4 leaves no stop at all."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.6],
        'Pullback': [1.2],
        'TP': [20.0],
        'R': [5.0],
    })

    rows = {r['SL Reduction']: r for _, r in
            calculate_sl_reduction_statistics(trades).iterrows()}
    assert rows['Default']['Notation'] == '1W - 0L'
    assert rows['1 pip']['Notation'] == '1W - 0L'
    assert rows['2 pips']['Notation'] == '1W - 0L'
    assert rows['3 pips']['Notation'] == '0W - 1L'   # stop 0.6 < pullback 1.2
    assert rows['4 pips']['Notation'] == '0W - 1L'   # stop -0.4, no room at all


def test_sl_reduction_target_shrinks_with_the_stop():
    """A 1:1 target is measured on the reduced stop, so a trade whose TP was
    short of the original stop can win once the stop is tightened."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [5.0],
        'Pullback': [0.5],
        'TP': [3.0],
        'R': [0.0],
    })

    rows = {r['SL Reduction']: r for _, r in
            calculate_sl_reduction_statistics(trades).iterrows()}
    assert rows['Default']['Notation'] == '0W - 1L'   # TP 3 < SL 5
    assert rows['2 pips']['Notation'] == '1W - 0L'   # TP 3 >= stop 3
    assert rows['4 pips']['Notation'] == '1W - 0L'   # TP 3 >= stop 1


def test_sl_reduction_stopped_out_trade_stays_a_loss():
    """Tightening a stop can never rescue a trade the original stop took out."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.0],
        'Pullback': [4.0],
        'TP': [30.0],
        'R': [-10.0],
    })

    result = calculate_sl_reduction_statistics(trades)
    assert (result['Notation'] == '0W - 1L').all()


def test_sl_reduction_win_rate_matches_notation():
    result = calculate_sl_reduction_statistics(get_sample_data())
    for _, row in result.iterrows():
        wins = int(row['Notation'].split('W')[0])
        assert row['Win Rate'] == f"{wins / row['Trades'] * 100:.1f}%"


def test_sl_reduction_zero_matches_the_unreduced_rule():
    """Row "0 pips" must equal the plain 1:1 rule used elsewhere."""
    sample = get_sample_data()
    rows = {r['SL Reduction']: r for _, r in
            calculate_sl_reduction_statistics(sample).iterrows()}
    wins = int((
        (sample['Pullback'] < sample['SL']) & (sample['TP'] >= sample['SL'])
    ).sum())
    assert rows['Default']['Notation'] == f"{wins}W - {len(sample) - wins}L"


def test_sl_reduction_empty():
    result = calculate_sl_reduction_statistics(get_empty_data())
    assert len(result) == len(SL_REDUCTION_PIPS)
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W - 0L'
        assert row['Win Rate'] == '0.0%'


def test_sl_reduction_is_not_sortable():
    """Row order is the point of this table, so sorting is off."""
    stats = calculate_sl_reduction_statistics(get_sample_data())
    html = _create_sl_sortable_table(stats, "sl-reduction-table", sortable=False)
    assert "onclick" not in html
    assert "sortSlRange" not in html


def test_sl_buffer_pips_constant():
    """0 to 5 pips inclusive."""
    assert SL_BUFFER_PIPS == [0, 1, 2, 3, 4, 5]


def test_sl_buffer_columns_and_rows():
    result = calculate_sl_buffer_statistics(get_sample_data())
    assert list(result.columns) == ['SL Buffer', 'Trades', 'Notation', 'Win Rate']
    assert list(result['SL Buffer']) == [
        'Default', '1 pip', '2 pips', '3 pips', '4 pips', '5 pips']
    assert (result['Trades'] == 10).all()


def test_sl_buffer_zero_row_matches_reduction_zero_row():
    """Both tables leave the stop alone at 0, so they must agree there."""
    sample = get_sample_data()
    buf = calculate_sl_buffer_statistics(sample).iloc[0]
    red = calculate_sl_reduction_statistics(sample).iloc[0]
    assert buf['Notation'] == red['Notation']
    assert buf['Win Rate'] == red['Win Rate']


def test_sl_buffer_rescues_a_stopped_out_trade():
    """SL 3.0, Pullback 3.5, TP 6. Stopped out at +0; +1 widens the stop to
    4.0 which the 3.5 pullback survives, and TP 6 clears the 4.0 target."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.0],
        'Pullback': [3.5],
        'TP': [6.0],
        'R': [-2.0],
    })

    rows = {r['SL Buffer']: r for _, r in
            calculate_sl_buffer_statistics(trades).iterrows()}
    assert rows['Default']['Notation'] == '0W - 1L'   # pullback 3.5 >= stop 3.0
    assert rows['1 pip']['Notation'] == '1W - 0L'    # stop 4.0, TP 6 >= 4.0
    assert rows['3 pips']['Notation'] == '1W - 0L'   # stop 6.0, TP 6 >= 6.0
    assert rows['4 pips']['Notation'] == '0W - 1L'   # target 7.0 now out of reach


def test_sl_buffer_target_moves_out_with_the_stop():
    """A padded stop raises the 1:1 target, so a modest winner can drop out."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [2.0],
        'Pullback': [0.5],
        'TP': [3.0],
        'R': [1.5],
    })

    rows = {r['SL Buffer']: r for _, r in
            calculate_sl_buffer_statistics(trades).iterrows()}
    assert rows['Default']['Notation'] == '1W - 0L'   # TP 3 >= SL 2
    assert rows['1 pip']['Notation'] == '1W - 0L'    # TP 3 >= 3
    assert rows['2 pips']['Notation'] == '0W - 1L'   # TP 3 < 4


def test_sl_buffer_win_rate_matches_notation():
    result = calculate_sl_buffer_statistics(get_sample_data())
    for _, row in result.iterrows():
        wins = int(row['Notation'].split('W')[0])
        assert row['Win Rate'] == f"{wins / row['Trades'] * 100:.1f}%"


def test_sl_buffer_empty():
    result = calculate_sl_buffer_statistics(get_empty_data())
    assert len(result) == len(SL_BUFFER_PIPS)
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W - 0L'
        assert row['Win Rate'] == '0.0%'


def test_sl_buffer_is_not_sortable():
    """Row order is the point of this table, so sorting is off."""
    stats = calculate_sl_buffer_statistics(get_sample_data())
    html = _create_sl_sortable_table(stats, "sl-buffer-table", sortable=False)
    assert "onclick" not in html
    assert "sortSlRange" not in html


def test_sl_buffer_small_sl_threshold_constant():
    assert SL_BUFFER_SMALL_SL_THRESHOLD == 5.0


def test_sl_buffer_small_sl_columns_and_rows():
    result = calculate_sl_buffer_small_sl_statistics(get_sample_data())
    assert list(result.columns) == ['SL Buffer', 'Trades', 'Notation', 'Win Rate']
    assert list(result['SL Buffer']) == [
        'Default', '1 pip', '2 pips', '3 pips', '4 pips', '5 pips']
    # Wide-stop trades are left alone, not dropped.
    assert (result['Trades'] == 10).all()


def test_sl_buffer_small_sl_zero_row_matches_the_other_tables():
    sample = get_sample_data()
    small = calculate_sl_buffer_small_sl_statistics(sample).iloc[0]
    plain = calculate_sl_buffer_statistics(sample).iloc[0]
    assert small['Notation'] == plain['Notation']
    assert small['Win Rate'] == plain['Win Rate']


def test_sl_buffer_small_sl_leaves_wide_stops_alone():
    """SL 6.0 (>= 5.0) keeps its stop, so padding never rescues it. The same
    trade in the unconditional table is saved at +1."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [6.0],
        'Pullback': [6.5],
        'TP': [20.0],
        'R': [-3.0],
    })

    small = {r['SL Buffer']: r for _, r in
             calculate_sl_buffer_small_sl_statistics(trades).iterrows()}
    plain = {r['SL Buffer']: r for _, r in
             calculate_sl_buffer_statistics(trades).iterrows()}

    assert small['1 pip']['Notation'] == '0W - 1L'   # stop stays 6.0
    assert small['5 pips']['Notation'] == '0W - 1L'
    assert plain['1 pip']['Notation'] == '1W - 0L'   # stop widens to 7.0


def test_sl_buffer_small_sl_pads_tight_stops():
    """SL 3.0 (< 5.0) does get the buffer, matching the unconditional table."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.0],
        'Pullback': [3.5],
        'TP': [6.0],
        'R': [-2.0],
    })

    small = {r['SL Buffer']: r for _, r in
             calculate_sl_buffer_small_sl_statistics(trades).iterrows()}
    assert small['Default']['Notation'] == '0W - 1L'
    assert small['1 pip']['Notation'] == '1W - 0L'


def test_sl_buffer_small_sl_threshold_is_exclusive():
    """A stop of exactly 5.0 is 'wide' and keeps its recorded value."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Tuesday'],
        'Trade': ['#1', '#1'],
        'Direction': ['Buy', 'Buy'],
        'SL': [5.0, 4.9],
        'Pullback': [5.2, 5.2],
        'TP': [20.0, 20.0],
        'R': [-3.0, -3.0],
    })

    rows = {r['SL Buffer']: r for _, r in
            calculate_sl_buffer_small_sl_statistics(trades).iterrows()}
    # +1: the 4.9 stop becomes 5.9 and survives the 5.2 pullback; the 5.0 stop
    # is untouched and does not.
    assert rows['1 pip']['Notation'] == '1W - 1L'


def test_sl_buffer_small_sl_win_rate_matches_notation():
    result = calculate_sl_buffer_small_sl_statistics(get_sample_data())
    for _, row in result.iterrows():
        wins = int(row['Notation'].split('W')[0])
        assert row['Win Rate'] == f"{wins / row['Trades'] * 100:.1f}%"


def test_sl_buffer_small_sl_empty():
    result = calculate_sl_buffer_small_sl_statistics(get_empty_data())
    assert len(result) == len(SL_BUFFER_PIPS)
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W - 0L'


def test_sl_buffer_small_sl_is_not_sortable():
    """Row order is the point of this table, so sorting is off."""
    stats = calculate_sl_buffer_small_sl_statistics(get_sample_data())
    html = _create_sl_sortable_table(stats, "sl-buffer-small-table", sortable=False)
    assert "onclick" not in html
    assert "sortSlRange" not in html


def test_sl_fixed_pips_constant():
    """3 to 7 pips, incremented by 1."""
    assert SL_FIXED_PIPS == [3, 4, 5, 6, 7]


def test_sl_fixed_columns_and_rows():
    result = calculate_sl_fixed_statistics(get_sample_data())
    assert list(result.columns) == ['Fixed SL', 'Trades', 'Notation', 'Win Rate']
    assert list(result['Fixed SL']) == [
        'Default', '3 pips', '4 pips', '5 pips', '6 pips', '7 pips']
    assert (result['Trades'] == 10).all()


def test_sl_fixed_default_row_matches_the_other_tables():
    """Default keeps the recorded stops, so it must equal the 0-shift rows."""
    sample = get_sample_data()
    default = calculate_sl_fixed_statistics(sample).iloc[0]
    assert default['Fixed SL'] == 'Default'
    assert default['Notation'] == calculate_sl_buffer_statistics(sample).iloc[0]['Notation']
    assert default['Notation'] == calculate_sl_reduction_statistics(sample).iloc[0]['Notation']


def test_sl_fixed_discards_the_recorded_stop():
    """A wide recorded stop is replaced, not adjusted. SL 20 / Pullback 4 / TP 6
    wins on its own stop only at Default; at a fixed 3 the 4-pip pullback takes
    it out, and at a fixed 5 it survives and TP 6 clears the 5 target."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [20.0],
        'Pullback': [4.0],
        'TP': [6.0],
        'R': [0.3],
    })

    rows = {r['Fixed SL']: r for _, r in calculate_sl_fixed_statistics(trades).iterrows()}
    assert rows['Default']['Notation'] == '0W - 1L'  # TP 6 < SL 20
    assert rows['3 pips']['Notation'] == '0W - 1L'   # pullback 4 >= stop 3
    assert rows['5 pips']['Notation'] == '1W - 0L'   # survives, TP 6 >= 5
    assert rows['7 pips']['Notation'] == '0W - 1L'   # TP 6 < target 7


def test_sl_fixed_is_the_same_stop_for_every_trade():
    """Two trades with very different recorded stops are judged identically
    once the stop is fixed."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01', '2026-01-02'],
        'Weekday': ['Monday', 'Tuesday'],
        'Trade': ['#1', '#1'],
        'Direction': ['Buy', 'Buy'],
        'SL': [2.0, 12.0],
        'Pullback': [1.0, 1.0],
        'TP': [10.0, 10.0],
        'R': [5.0, 0.8],
    })

    rows = {r['Fixed SL']: r for _, r in calculate_sl_fixed_statistics(trades).iterrows()}
    assert rows['Default']['Notation'] == '1W - 1L'  # 10>=2 wins, 10<12 loses
    assert rows['5 pips']['Notation'] == '2W - 0L'   # both survive and clear 5


def test_sl_fixed_win_rate_matches_notation():
    result = calculate_sl_fixed_statistics(get_sample_data())
    for _, row in result.iterrows():
        wins = int(row['Notation'].split('W')[0])
        assert row['Win Rate'] == f"{wins / row['Trades'] * 100:.1f}%"


def test_sl_fixed_empty():
    result = calculate_sl_fixed_statistics(get_empty_data())
    assert len(result) == len(SL_FIXED_PIPS) + 1
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W - 0L'


def test_sl_fixed_sortable_win_rate_only():
    stats = calculate_sl_fixed_statistics(get_sample_data())
    html = _create_sl_sortable_table(stats, "sl-fixed-table")
    for idx, col in enumerate(stats.columns):
        if col == 'Win Rate':
            assert f"sortSlRange('sl-fixed-table', {idx}, this)" in html
        else:
            assert f"sortSlRange('sl-fixed-table', {idx}, this)" not in html


def test_every_stop_table_opens_with_the_same_default_row():
    """SL Range, Reducing SL, Adding Buffer, Buffer (SL<5) and Fixed SL all
    start from the recorded stops, so their Default rows must agree."""
    sample = get_sample_data()
    tables = {
        'SL Range': (calculate_sl_statistics(sample), 'SL Range'),
        'Reducing SL': (calculate_sl_reduction_statistics(sample), 'SL Reduction'),
        'Adding Buffer': (calculate_sl_buffer_statistics(sample), 'SL Buffer'),
        'Buffer SL<5': (calculate_sl_buffer_small_sl_statistics(sample), 'SL Buffer'),
        'Fixed SL': (calculate_sl_fixed_statistics(sample), 'Fixed SL'),
    }

    seen = {}
    for name, (result, column) in tables.items():
        first = result.iloc[0]
        assert first[column] == 'Default', f'{name} does not open with Default'
        seen[name] = (first['Trades'], first['Notation'], first['Win Rate'])

    assert len(set(seen.values())) == 1, f'Default rows disagree: {seen}'
    assert seen['SL Range'][0] == len(sample)


def test_pullback_entry_pips_constant():
    """Pullback entry levels are 0, 1, 2 and 3 pips."""
    assert PULLBACK_ENTRY_PIPS == [0, 1, 2, 3]


def test_pullback_statistics_columns():
    """Test that Pullback statistics has expected columns."""
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    assert list(result.columns) == [
        'Pullback', 'Trades', 'Notation', '+1 pip', '+2 pips', '+3 pips',
    ]


def test_pullback_statistics_levels_present():
    """All fixed and SL-relative pullback entry levels appear in results."""
    sample = get_sample_data()
    result = calculate_pullback_statistics(sample)
    assert len(result) == 6
    assert list(result['Pullback']) == ['0 pips', '1 pip', '2 pips', '3 pips', 'Half', 'Full']


def test_pullback_statistics_values():
    """Sample SL/Pullback/TP:
    3.5/3.5/0, 1.1/0.8/12, 2.0/2.1/0, 4.0/1.5/10, 3.0/3.0/0, 5.0/2.0/8,
    2.5/2.5/0, 6.0/3.0/15, 8.0/7.0/10, 1.5/0.5/5.

    1:1 winners at buffer 0 (PB < SL AND TP >= SL): idx 1,3,5,7,8,9 = 6;
    their pullbacks are 0.8, 1.5, 2.0, 3.0, 7.0, 0.5. The same 6 win at
    +1 and +2; at +3 the SL 8 trade fails (TP 10 < 11) leaving 5.

    PB 0: all 10 entered, nothing missed.
    PB 1: 8 entered; winners with PB < 1 (0.8, 0.5) are missed.
    PB 2: 7 entered; missed winners: 0.8, 1.5, 0.5.
    PB 3: 4 entered; missed winners: 0.8, 1.5, 2.0, 0.5.
    """
    sample = get_sample_data()
    rows = _pullback_rows(calculate_pullback_statistics(sample))

    assert rows['0 pips']['Trades'] == 10
    assert rows['0 pips']['Notation'] == '6W – 4L – 0M (60.0%)'
    assert rows['0 pips']['+3 pips'] == '5W – 5L – 0M (50.0%)'

    assert rows['1 pip']['Trades'] == 8
    assert rows['1 pip']['Notation'] == '4W – 4L – 2M (50.0%)'
    assert rows['1 pip']['+3 pips'] == '3W – 5L – 2M (37.5%)'

    assert rows['2 pips']['Trades'] == 7
    assert rows['2 pips']['Notation'] == '3W – 4L – 3M (42.9%)'
    assert rows['2 pips']['+3 pips'] == '2W – 5L – 3M (28.6%)'

    assert rows['3 pips']['Trades'] == 4
    assert rows['3 pips']['Notation'] == '2W – 2L – 4M (50.0%)'
    assert rows['3 pips']['+3 pips'] == '1W – 3L – 4M (25.0%)'


def test_pullback_statistics_half_and_full_values():
    """Half fills when Pullback >= SL/2; Full when Pullback >= SL.

    Half entered: PB >= SL/2 at idx 0,1,2,4,6,7,8 -> 7 trades; buffer-0
    winners among them are 1.1/0.8/12, 6.0/3.0/15, 8.0/7.0/10 -> 3W, 4L,
    missed winners 4.0/1.5/10, 5.0/2.0/8, 1.5/0.5/5 -> 3M. At +3 the SL 8
    trade fails its higher target -> 2W.

    Full entered: PB >= SL at idx 0,2,4,6 -> 4 trades, all TP=0, so they
    lose at every buffer; every winner pulled back less than its SL -> 6M
    (5M at +3 where the SL 8 trade is no longer a winner).
    """
    sample = get_sample_data()
    rows = _pullback_rows(calculate_pullback_statistics(sample))

    assert rows['Half']['Trades'] == 7
    assert rows['Half']['Notation'] == '3W – 4L – 3M (42.9%)'
    assert rows['Half']['+3 pips'] == '2W – 5L – 3M (28.6%)'

    assert rows['Full']['Trades'] == 4
    assert rows['Full']['Notation'] == '0W – 4L – 6M (0.0%)'
    assert rows['Full']['+3 pips'] == '0W – 4L – 5M (0.0%)'


def test_pullback_statistics_full_survives_with_buffer():
    """A Full-pullback trade (SL 3.0, Pullback 4.1, TP 10) is stopped at the
    safe stop but survives and wins once a 2-pip buffer widens the stop."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [3.0],
        'Pullback': [4.1],
        'TP': [10.0],
        'R': [0],
    })

    rows = _pullback_rows(calculate_pullback_statistics(trades))

    assert rows['Full']['Trades'] == 1
    assert rows['Full']['Notation'] == '0W – 1L – 0M (0.0%)'
    # +1 pip: effective SL 4.0, the 4.1 pullback still hits the stop.
    assert rows['Full']['+1 pip'] == '0W – 1L – 0M (0.0%)'
    # +2 pips: effective SL 5.0 survives the pullback, TP 10 >= 5.
    assert rows['Full']['+2 pips'] == '1W – 0L – 0M (100.0%)'


def test_pullback_statistics_buffer_grows_target():
    """A buffer widens the stop but also raises the 1:1 target: the SL 8 trade
    (8.0/7.0/10) wins up to +2 pips (TP 10 >= 10) but loses at +3 (TP 10 < 11)."""
    sample = get_sample_data()
    rows = _pullback_rows(calculate_pullback_statistics(sample))

    assert rows['0 pips']['+2 pips'] == '6W – 4L – 0M (60.0%)'
    assert rows['0 pips']['+3 pips'] == '5W – 5L – 0M (50.0%)'


def test_pullback_statistics_stopped_trade_is_loss_not_win():
    """A trade that reaches TP but pulls back past its SL is a filled LOSS at
    buffer 0, but a buffer wide enough to survive the pullback makes it a win."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [2.0],
        'Pullback': [3.0],  # >= SL -> stopped out at the safe stop
        'TP': [10.0],       # reached target only in hindsight
        'R': [0],
    })

    rows = _pullback_rows(calculate_pullback_statistics(trades))

    assert rows['1 pip']['Trades'] == 1
    assert rows['1 pip']['Notation'] == '0W – 1L – 0M (0.0%)'
    # +1 pip: effective SL 3.0, Pullback 3.0 still hits the stop.
    assert rows['1 pip']['+1 pip'] == '0W – 1L – 0M (0.0%)'
    # +2 pips: effective SL 4.0 survives the 3-pip pullback, TP 10 >= 4.
    assert rows['1 pip']['+2 pips'] == '1W – 0L – 0M (100.0%)'


def test_pullback_statistics_missed_winner():
    """A real winner whose pullback is too shallow to fill the limit is a
    missed winner (M), not entered (Trades excludes it)."""
    trades = pd.DataFrame({
        'Date': ['2026-01-01'],
        'Weekday': ['Monday'],
        'Trade': ['#1'],
        'Direction': ['Buy'],
        'SL': [5.0],
        'Pullback': [0.5],  # never pulls back 1 pip
        'TP': [10.0],       # survives stop and TP >= SL -> 1:1 winner
        'R': [2.0],
    })

    rows = _pullback_rows(calculate_pullback_statistics(trades))

    # At 0 pips the trade is simply taken and wins.
    assert rows['0 pips']['Trades'] == 1
    assert rows['0 pips']['Notation'] == '1W – 0L – 0M (100.0%)'
    assert rows['1 pip']['Trades'] == 0
    assert rows['1 pip']['Notation'] == '0W – 0L – 1M (0.0%)'


def test_pullback_statistics_sortable_columns():
    """Notation and the +1/+2/+3 pip columns are click-to-sort DESC by win
    rate; Pullback and Trades headers stay plain."""
    sample = get_sample_data()
    stats = calculate_pullback_statistics(sample)
    html = _create_sl_sortable_table(stats, "pullback-analysis")

    columns = list(stats.columns)
    for col in ['Notation', '+1 pip', '+2 pips', '+3 pips']:
        idx = columns.index(col)
        assert f"sortSlRange('pullback-analysis', {idx}, this)" in html
        assert f"{col} ↓" in html
    assert ">Pullback</th>" in html
    assert ">Trades</th>" in html
    assert "return pct(b) - pct(a);" in html


def test_pullback_statistics_empty():
    """Test Pullback statistics with empty dataset."""
    empty = get_empty_data()
    result = calculate_pullback_statistics(empty)
    assert len(result) == 6
    for _, row in result.iterrows():
        assert row['Trades'] == 0
        assert row['Notation'] == '0W – 0L – 0M (0.0%)'
        assert row['Notation'] == '0W – 0L – 0M (0.0%)'


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
        'SL': [3.0, 3.0],
        'Pullback': [1.0, 4.0],
        'TP': [60.0, 55.0],
        'R': [20.0, 0],
    })

    result = calculate_tp_statistics(trades)
    rows = {row['TP Range']: row for _, row in result.iterrows()}

    assert rows['35+']['Trades'] == '2 of 2'


def test_buffer_statistics_filtered_all_trades():
    """Test _calculate_buffer_statistics_filtered with All Trades only."""
    sample = get_sample_data()
    result = _calculate_buffer_statistics_filtered(sample, ["All Trades"])
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        strategies = result['Strategy'].unique()
        for s in strategies:
            assert s == 'All Trades', f"Unexpected strategy: {s}"


def test_buffer_statistics_filtered_excludes_others():
    """Test that filtered results don't include strategies not in the list."""
    sample = get_sample_data()
    result_all = _calculate_buffer_statistics_filtered(sample, ["All Trades"])
    result_fixed = _calculate_buffer_statistics_filtered(sample, ["Fixed SL 2", "Fixed SL 3"])
    if len(result_all) > 0:
        assert 'Fixed SL 2' not in result_all['Strategy'].values
    if len(result_fixed) > 0:
        assert 'All Trades' not in result_fixed['Strategy'].values


def test_buffer_statistics_filtered_empty():
    """Empty input still emits a row per (buffer, RRR) with zero trades."""
    empty = get_empty_data()
    result = _calculate_buffer_statistics_filtered(empty, ["All Trades"])
    assert isinstance(result, pd.DataFrame)
    assert (result['Trades'] == 0).all()


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_load_data_real_csv,
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
        test_sl_filter_combination,
        test_calculate_statistics_returns_positive_edge_only,
        test_calculate_statistics_sorted_by_edge,
        test_calculate_statistics_columns,
        test_create_html_table_basic,
        test_create_html_table_empty,
        test_create_html_table_no_sort_by_default,
        test_create_html_table_sortable_win_rate,
        test_buffer_saves_losing_trade,
        test_buffer_tp_must_reach_effective_sl,
        test_buffer_stats_has_buffer_column,
        test_buffer_stats_empty,
        test_get_buffer_strategies,
        test_calculate_buffer_statistics,
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
        test_buffer_stats_has_trades_column,
        test_buffer_stats_empty_has_trades_zero,
        test_buffer_stats_with_float_rrr,
        test_buffer_stats_with_float_rrr_loss,
        test_format_wl,
        test_weekday_order_constant,
        test_weekday_statistics_columns,
        test_weekday_statistics_all_days_present,
        test_weekday_statistics_trade_counts,
        test_weekday_statistics_notation_and_win_rate,
        test_weekday_statistics_stopped_out_is_a_loss,
        test_weekday_statistics_no_tp_is_a_loss,
        test_weekday_statistics_empty,
        test_weekday_statistics_single_day,
        test_sl_ranges_constant,
        test_sl_statistics_columns,
        test_sl_statistics_win_needs_a_full_1_1_target,
        test_sl_statistics_all_ranges_present,
        test_sl_statistics_trade_counts,
        test_sl_statistics_notation,
        test_sl_statistics_cumulative_bands,
        test_sl_statistics_requires_surviving_stop,
        test_sl_statistics_survivor_wins,
        test_sl_statistics_empty,
        test_sl_statistics_large_sl,
        test_sl_statistics_no_tp_is_loss,
        test_sl_sortable_table_notation_headers_clickable,
        test_sl_sortable_table_can_be_turned_off,
        test_sl_sortable_table_still_sorts_combined_notation_columns,
        test_sl_sortable_table_uses_given_id,
        test_sl_sortable_table_sorts_descending,
        test_sl_sortable_table_empty,
        test_sl_reduction_pips_constant,
        test_sl_reduction_columns,
        test_sl_reduction_rows,
        test_sl_reduction_worked_example,
        test_sl_reduction_target_shrinks_with_the_stop,
        test_sl_reduction_stopped_out_trade_stays_a_loss,
        test_sl_reduction_win_rate_matches_notation,
        test_sl_reduction_zero_matches_the_unreduced_rule,
        test_sl_reduction_empty,
        test_sl_reduction_is_not_sortable,
        test_sl_buffer_pips_constant,
        test_sl_buffer_columns_and_rows,
        test_sl_buffer_zero_row_matches_reduction_zero_row,
        test_sl_buffer_rescues_a_stopped_out_trade,
        test_sl_buffer_target_moves_out_with_the_stop,
        test_sl_buffer_win_rate_matches_notation,
        test_sl_buffer_empty,
        test_sl_buffer_is_not_sortable,
        test_sl_buffer_small_sl_threshold_constant,
        test_sl_buffer_small_sl_columns_and_rows,
        test_sl_buffer_small_sl_zero_row_matches_the_other_tables,
        test_sl_buffer_small_sl_leaves_wide_stops_alone,
        test_sl_buffer_small_sl_pads_tight_stops,
        test_sl_buffer_small_sl_threshold_is_exclusive,
        test_sl_buffer_small_sl_win_rate_matches_notation,
        test_sl_buffer_small_sl_empty,
        test_sl_buffer_small_sl_is_not_sortable,
        test_sl_fixed_pips_constant,
        test_sl_fixed_columns_and_rows,
        test_sl_fixed_default_row_matches_the_other_tables,
        test_sl_fixed_discards_the_recorded_stop,
        test_sl_fixed_is_the_same_stop_for_every_trade,
        test_sl_fixed_win_rate_matches_notation,
        test_sl_fixed_empty,
        test_sl_fixed_sortable_win_rate_only,
        test_every_stop_table_opens_with_the_same_default_row,
        test_pullback_entry_pips_constant,
        test_pullback_statistics_columns,
        test_pullback_statistics_levels_present,
        test_pullback_statistics_values,
        test_pullback_statistics_half_and_full_values,
        test_pullback_statistics_full_survives_with_buffer,
        test_pullback_statistics_buffer_grows_target,
        test_pullback_statistics_stopped_trade_is_loss_not_win,
        test_pullback_statistics_missed_winner,
        test_pullback_statistics_sortable_columns,
        test_pullback_statistics_empty,
        test_tp_ranges_constant,
        test_tp_statistics_columns,
        test_tp_statistics_all_ranges_present,
        test_tp_statistics_trade_counts,
        test_tp_statistics_empty,
        test_tp_statistics_large_tp,
        test_buffer_statistics_filtered_all_trades,
        test_buffer_statistics_filtered_excludes_others,
        test_buffer_statistics_filtered_empty,
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
