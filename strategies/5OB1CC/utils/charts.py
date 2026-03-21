"""
Forex Trading Strategy Chart Utilities

This module provides visualization tools for displaying cumulative performance
charts for trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Import from other utils modules
from utils import RRR_CONFIGS, _win_condition_normal
from utils.tables import (
    Strategy,
    _calculate_wins_and_losses,
    create_single_setup_strategy_library,
    create_double_setup_strategy_library,
    create_triple_setup_strategy_library,
)


def display_strategy_cumulative_chart(df: pd.DataFrame, strategy: Strategy):
    """
    Display cumulative performance chart for a specific strategy with all three RRR ratios.

    Args:
        df: Trading data
        strategy: Strategy object to analyze
    """
    from IPython.display import display, HTML

    # Apply the strategy filter
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        display(HTML(f"<h2>{strategy.name}</h2>"))
        display(HTML("<p>No trades found for this strategy.</p>"))
        return

    # Calculate outcomes for each RRR ratio
    rrr_ratios = [1, 2, 3]
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    cumulative_outcomes = {}

    for ratio in rrr_ratios:
        outcomes = []
        for _, trade in filtered_df.iterrows():
            # Win condition: all three must be true
            is_win = (
                (trade['SL'] != trade['Pullback']) and
                (trade['Pullback'] < trade['SL']) and
                (trade['TP'] >= ratio * trade['SL'])
            )
            if is_win:
                outcomes.append(ratio)  # Win = +ratio R
            else:
                outcomes.append(-1)  # Loss = -1R

        cumulative_outcomes[ratio] = np.cumsum(outcomes)

    # Create the chart
    plt.figure(figsize=(14, 7))
    trade_numbers = range(1, len(filtered_df) + 1)

    for i, ratio in enumerate(rrr_ratios):
        cumulative = cumulative_outcomes[ratio]
        plt.plot(trade_numbers, cumulative, linewidth=2, color=colors[i],
                label=f'1:{ratio} RRR (Final: {cumulative[-1]:.1f}R)', alpha=0.85)

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Cumulative Outcome (R)', fontsize=12)
    plt.title(strategy.name, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.show()


def _is_strategy_profitable(df: pd.DataFrame, strategy: Strategy) -> bool:
    """
    Check if a strategy has positive edge for at least one RRR ratio.

    Args:
        df: Trading data
        strategy: Strategy to evaluate

    Returns:
        True if strategy has positive edge for any RRR ratio
    """
    filtered_df = strategy.apply(df)

    if len(filtered_df) == 0:
        return False

    # Check each RRR ratio for positive edge
    for ratio, breakeven_rate in RRR_CONFIGS:
        total_trades = len(filtered_df)
        wins, losses = _calculate_wins_and_losses(
            filtered_df, ratio, 'SL', _win_condition_normal
        )

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            edge = win_rate - breakeven_rate

            if edge > 0:
                return True

    return False


def display_single_setup_strategy_charts(df: pd.DataFrame):
    """Display cumulative performance charts for profitable single setup strategies only."""
    from IPython.display import display, HTML

    display(HTML("<h1>Single Setup Strategies - Performance Charts</h1>"))

    strategies = create_single_setup_strategy_library()

    # Filter to only profitable strategies
    profitable_strategies = [s for s in strategies if _is_strategy_profitable(df, s)]

    display(HTML(f"<p>Showing {len(profitable_strategies)} profitable strategies out of {len(strategies)} total strategies.</p>"))

    for strategy in profitable_strategies:
        display_strategy_cumulative_chart(df, strategy)


def display_double_setup_strategy_charts(df: pd.DataFrame):
    """Display cumulative performance charts for profitable double setup strategies only."""
    from IPython.display import display, HTML

    display(HTML("<h1>Double Setup Strategies - Performance Charts</h1>"))

    strategies = create_double_setup_strategy_library()

    # Filter to only profitable strategies
    profitable_strategies = [s for s in strategies if _is_strategy_profitable(df, s)]

    display(HTML(f"<p>Showing {len(profitable_strategies)} profitable strategies out of {len(strategies)} total strategies.</p>"))

    for strategy in profitable_strategies:
        display_strategy_cumulative_chart(df, strategy)


def display_triple_setup_strategy_charts(df: pd.DataFrame):
    """Display cumulative performance charts for profitable triple setup strategies only."""
    from IPython.display import display, HTML

    display(HTML("<h1>Triple Setup Strategies - Performance Charts</h1>"))

    strategies = create_triple_setup_strategy_library()

    # Filter to only profitable strategies
    profitable_strategies = [s for s in strategies if _is_strategy_profitable(df, s)]

    display(HTML(f"<p>Showing {len(profitable_strategies)} profitable strategies out of {len(strategies)} total strategies.</p>"))

    for strategy in profitable_strategies:
        display_strategy_cumulative_chart(df, strategy)
