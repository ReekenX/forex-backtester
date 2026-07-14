"""
Tests for the session analysis in utils.tables

These tests use a small dataset of 10 rows to verify the session analysis functionality.
Run with: poetry run python -m pytest strategies/5OB1CC/tests/test_sessions.py -v
"""

import os
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.tables import analyze_session_profitability


def get_sample_data():
    """Create a sample dataset with 10 trades across both sessions."""
    return pd.DataFrame({
        'Hour': [10, 11, 12, 13, 14, 15, 15, 16, 17, 18],
        'SL': [10.0, 5.0, 20.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 6.0],
        'Pullback': [5.0, 5.0, 1.0, 0.5, 7.0, 4.0, 8.0, 11.0, 2.0, 3.0],
        'TP': [20.0, 10.0, 40.0, 12.0, 14.0, 16.0, 18.0, 20.0, 10.0, 12.0]
    })


def test_returns_two_session_rows():
    """Analysis returns exactly two rows: London and New York."""
    result = analyze_session_profitability(get_sample_data())
    table = result['Session Analysis']

    assert list(table['Session']) == ['London', 'New York']


def test_session_columns():
    """Table has the expected columns."""
    result = analyze_session_profitability(get_sample_data())
    table = result['Session Analysis']

    assert list(table.columns) == ['Session', 'Trades', 'Wins', 'Losses', 'Win %']


def test_session_split():
    """Hours 10-14 count as London, hours 15+ count as New York."""
    result = analyze_session_profitability(get_sample_data())
    table = result['Session Analysis']

    london = table[table['Session'] == 'London'].iloc[0]
    new_york = table[table['Session'] == 'New York'].iloc[0]

    assert london['Trades'] == 5
    assert new_york['Trades'] == 5


def test_hour_15_is_new_york():
    """Hour 15 belongs to the New York session, not London."""
    df = pd.DataFrame({
        'Hour': [15],
        'SL': [5.0],
        'Pullback': [2.0],
        'TP': [10.0]
    })
    result = analyze_session_profitability(df)
    table = result['Session Analysis']

    london = table[table['Session'] == 'London'].iloc[0]
    new_york = table[table['Session'] == 'New York'].iloc[0]

    assert london['Trades'] == 0
    assert new_york['Trades'] == 1


def test_win_and_loss_counts():
    """Wins require Pullback < SL and TP >= SL; losses are the rest."""
    result = analyze_session_profitability(get_sample_data())
    table = result['Session Analysis']

    # London: hours 10-14 -> wins at rows with Pullback < SL and TP >= SL
    # Row 10: win, Row 11: Pullback == SL loss, Row 12: win, Row 13: win, Row 14: Pullback == SL loss
    london = table[table['Session'] == 'London'].iloc[0]
    assert london['Wins'] == 3
    assert london['Losses'] == 2
    assert london['Win %'] == '60.0%'

    # New York: hours 15-18
    # Row 15: win, Row 15: win, Row 16: Pullback > SL loss, Row 17: win, Row 18: win
    new_york = table[table['Session'] == 'New York'].iloc[0]
    assert new_york['Wins'] == 4
    assert new_york['Losses'] == 1
    assert new_york['Win %'] == '80.0%'


def test_empty_session_shows_zeros():
    """A session with no trades is reported with zero counts."""
    df = pd.DataFrame({
        'Hour': [10, 11],
        'SL': [5.0, 5.0],
        'Pullback': [2.0, 5.0],
        'TP': [10.0, 10.0]
    })
    result = analyze_session_profitability(df)
    table = result['Session Analysis']

    new_york = table[table['Session'] == 'New York'].iloc[0]
    assert new_york['Trades'] == 0
    assert new_york['Wins'] == 0
    assert new_york['Losses'] == 0
    assert new_york['Win %'] == '0.0%'


def test_missing_hour_column_raises():
    """Data without an Hour column raises a ValueError."""
    df = pd.DataFrame({
        'SL': [5.0],
        'Pullback': [2.0],
        'TP': [10.0]
    })
    try:
        analyze_session_profitability(df)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Hour" in str(e)


def test_input_dataframe_not_modified():
    """The input DataFrame is left untouched."""
    df = get_sample_data()
    original = df.copy()
    analyze_session_profitability(df)

    pd.testing.assert_frame_equal(df, original)
