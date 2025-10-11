"""
Export utilities for forex trading strategy analysis.

This module provides functions for exporting strategy trade data to various formats.
"""

import pandas as pd
from utils.tables import (
    create_single_setup_strategy_library,
    create_double_setup_strategy_library,
    create_triple_setup_strategy_library
)


def export_strategy_trades_to_csv(
    df: pd.DataFrame,
    strategy_name: str,
    rrr_ratio: int,
    output_file: str = "strategy_trades.csv"
) -> str:
    """
    Export trades for a specific strategy and RRR ratio to CSV.
    Args:
        df: Trading data with all columns
        strategy_name: Name of the strategy to export (e.g., "EMA + BOS")
        rrr_ratio: RRR ratio to use for profitability calculation (1, 2, or 3)
        output_file: Path to the output CSV file
    Returns:
        Path to the exported CSV file
    """
    # Find the strategy
    all_strategies = []
    all_strategies.extend(create_single_setup_strategy_library())
    all_strategies.extend(create_double_setup_strategy_library())
    all_strategies.extend(create_triple_setup_strategy_library())

    # Find matching strategy
    strategy = None
    for s in all_strategies:
        if s.name == strategy_name:
            strategy = s
            break

    if strategy is None:
        raise ValueError(f"Strategy '{strategy_name}' not found. Please check the strategy name.")

    # Apply strategy filter
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        raise ValueError(f"No trades found for strategy '{strategy_name}'")

    # Prepare trade details
    trade_details = []
    for idx, trade in filtered_df.iterrows():
        # Determine if profitable based on selected RRR
        # Win condition: all three must be true
        is_win = (
            (trade['SL'] != trade['Pullback']) and
            (trade['Pullback'] < trade['SL']) and
            (trade['TP'] >= rrr_ratio * trade['SL'])
        )

        if is_win:
            profitable = 'Yes'
        else:
            profitable = 'No'

        trade_details.append({
            'Profitable': profitable,
            'Date': trade.get('Date', ''),
            'Trade': trade.get('Trade', ''),
            'Weekday': trade.get('Weekday', ''),
            'Hour': int(trade.get('Hour', 0)) if trade.get('Hour', 0) != 0 else '',
            'Direction': trade.get('Direction', ''),
            'EMA': trade.get('EMA', ''),
            'SL': trade.get('SL', ''),
            'Pullback': trade.get('Pullback', ''),
            'TP': trade.get('TP', ''),
            'Extra': trade.get('Extra', ''),
            'BOS/CH': trade.get('BOS/CH', ''),
            '30M Leg': trade.get('30M Leg', ''),
            'Hours Until News': trade.get('Hours Until News', ''),
            'News Event': trade.get('News Event', '')
        })

    # Create DataFrame and export to CSV
    trades_df = pd.DataFrame(trade_details)
    trades_df.to_csv(output_file, index=False)

    return output_file
