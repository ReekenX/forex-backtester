"""
Forex Trading Strategy Analysis Utilities

This module provides comprehensive tools for backtesting and analyzing forex trading strategies.
It includes functions for data loading, strategy evaluation, and performance visualization.

Main components:
- Data loading and cleaning
- Strategy definition and evaluation
- Risk-Reward Ratio (RRR) analysis
- Entry timing analysis
- Performance visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from IPython.display import display, HTML

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# Risk-Reward Ratio configurations with breakeven win rates
RRR_CONFIGS: List[Tuple[int, float]] = [
    (1, 50.0),   # 1:1 RRR - need 50% to break even
    (2, 33.3),   # 1:2 RRR - need 33.3% to break even
    (3, 25.0),   # 1:3 RRR - need 25% to break even
]

# Entry method mapping for display
ENTRY_TYPE_NAMES: Dict[str, str] = {
    'SL': '1M CC',
    'SL 5M CC': '5M CC',
    'SL 5M Stop': '5M Stop'
}

# Table styling defaults
DEFAULT_COLUMN_WIDTH = '250px'
DEFAULT_HIGHLIGHT_COLOR = 'green'

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_clean_data(filepath: str = './eurusd.csv') -> pd.DataFrame:
    """
    Load EUR/USD data from CSV and clean NaN values.

    Args:
        filepath: Path to the CSV file containing trading data

    Returns:
        Cleaned dataframe with trading data
    """
    df = pd.read_csv(filepath)

    # Define columns that should have NaN replaced with 0
    columns_to_fillna = [
        'SL', 'TP', 'SL 5M CC', 'SL 5M Stop',
        'Hours Until News', 'Extra'
    ]

    # Fill NaN values to prevent calculation errors
    for col in columns_to_fillna:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

# ============================================================================
# STRATEGY DEFINITION AND MANAGEMENT
# ============================================================================

class Strategy:
    """
    Encapsulates a trading strategy with its filter logic and metadata.

    Attributes:
        name: Strategy identifier
        filter_func: Function that filters trades based on strategy rules
        description: Human-readable description of the strategy
    """

    def __init__(self, name: str, filter_func: Callable, description: str = ""):
        """
        Initialize a trading strategy.

        Args:
            name: Strategy name
            filter_func: Lambda or function that takes df and returns filtered df
            description: Optional description of the strategy
        """
        self.name = name
        self.filter_func = filter_func
        self.description = description

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the strategy filter to a dataframe of trades."""
        return self.filter_func(df)

# ============================================================================
# HELPER FUNCTIONS FOR CALCULATIONS
# ============================================================================

def _calculate_wins_and_losses(
    df: pd.DataFrame,
    ratio: int,
    sl_column: str,
    win_condition_func: Callable
) -> Tuple[int, int]:
    """
    Calculate wins and losses for a given RRR ratio.

    Args:
        df: DataFrame with trade data
        ratio: Risk-reward ratio (1, 2, or 3)
        sl_column: Stop loss column to use
        win_condition_func: Function that determines winning trades

    Returns:
        Tuple of (wins, losses)
    """
    wins = len(win_condition_func(df, ratio, sl_column))
    losses = len(df) - wins
    return wins, losses

def _format_rrr_metrics(
    total_trades: int,
    wins: int,
    losses: int,
    ratio: int,
    breakeven_rate: float,
    entry_type: str
) -> List:
    """
    Format RRR metrics for display.

    Returns:
        List of formatted metrics [total, wins, losses, win_rate, edge, outcome, entry]
    """
    if total_trades == 0:
        return [0, 0, 0, '0.0%', '0.0%', '0R', entry_type]

    win_rate = (wins / total_trades) * 100
    edge = win_rate - breakeven_rate
    outcome = (wins * ratio) - losses

    return [
        total_trades,
        wins,
        losses,
        f"{win_rate:.1f}%",
        f"{edge:.1f}%",
        f"{outcome}R",
        entry_type
    ]

def _create_empty_rrr_summary(strategy_name: str, entry_type: str) -> pd.DataFrame:
    """Create an empty RRR summary DataFrame for strategies with no trades."""
    summary_data = {
        strategy_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
    }
    for ratio, _ in RRR_CONFIGS:
        summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R', entry_type]
    return pd.DataFrame(summary_data)

# ============================================================================
# WIN CONDITION FILTERS
# ============================================================================

def _win_condition_normal(df: pd.DataFrame, ratio: int, sl_column: str) -> pd.DataFrame:
    """Standard win condition: TP > ratio * SL."""
    return df[df['TP'] > ratio * df[sl_column]]


# ============================================================================
# RRR CALCULATION FUNCTIONS
# ============================================================================

def calculate_rrr_stats(
    data_df: pd.DataFrame,
    strategy_name: str,
    sl_column: str = 'SL'
) -> pd.DataFrame:
    """
    Calculate comprehensive Risk-Reward Ratio statistics for a trading strategy.

    Args:
        data_df: Filtered DataFrame containing trades for this strategy
        strategy_name: Name of the strategy for labeling
        sl_column: Column to use for stop loss calculations

    Returns:
        Statistics table with metrics for each RRR level
    """
    total_trades = len(data_df)
    entry_str = ENTRY_TYPE_NAMES[sl_column]

    if total_trades == 0:
        return _create_empty_rrr_summary(strategy_name, entry_str)

    summary_data = {
        strategy_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
    }

    for ratio, breakeven_rate in RRR_CONFIGS:
        wins, losses = _calculate_wins_and_losses(
            data_df, ratio, sl_column, _win_condition_normal
        )

        summary_data[f'1:{ratio} RRR'] = _format_rrr_metrics(
            total_trades, wins, losses, ratio, breakeven_rate, entry_str
        )

    return pd.DataFrame(summary_data)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_entry_timing_detailed(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze different entry timing strategies with detailed statistics including RRR analysis.

    Returns 3 DataFrames, one for each entry method:
    - 5M Stop: Entry with 5-minute stop loss level
    - 5M Confirmation Candle: Entry on 5-minute candle confirmation
    - 1M Confirmation Candle: Entry on 1-minute candle confirmation

    Args:
        df: Trading data with entry signals

    Returns:
        Dictionary containing DataFrames for each entry method
    """
    entry_methods = {
        '5M Stop': {
            'filter': lambda d: d['SL 5M Stop'] != 0,
            'sl_col': 'SL 5M Stop'
        },
        '5M Confirmation Candle': {
            'filter': lambda d: d['SL 5M CC'] != 0,
            'sl_col': 'SL 5M CC'
        },
        '1M Confirmation Candle': {
            'filter': lambda d: d['SL'] != 0,
            'sl_col': 'SL'
        }
    }

    entry_tables = {}

    for method_name, method_config in entry_methods.items():
        relevant_trades = df[method_config['filter'](df)]
        sl_col = method_config['sl_col']
        total_trades = len(relevant_trades)

        summary_data = {
            method_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome']
        }

        for ratio, breakeven_rate in RRR_CONFIGS:
            if total_trades > 0:
                valid_wins = relevant_trades[
                    (relevant_trades['SL'] != relevant_trades['Pullback']) &
                    (relevant_trades['TP'] > (ratio * relevant_trades[sl_col]))
                ]
                wins = len(valid_wins)
                losses = total_trades - wins
                win_rate = (wins / total_trades * 100)
                edge = win_rate - breakeven_rate
                outcome = (wins * ratio) - losses

                summary_data[f'1:{ratio} RRR'] = [
                    total_trades,
                    wins,
                    losses,
                    f'{win_rate:.1f}%',
                    f'{edge:.1f}%',
                    f'{outcome}R'
                ]
            else:
                summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R']

        entry_tables[method_name] = pd.DataFrame(summary_data)

    return entry_tables

def analyze_pullback_profitability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze how pullback size affects trade profitability.

    Creates multiple tables showing profitability statistics for different pullback sizes.

    Args:
        df: Trading data with Pullback, TP, and SL columns

    Returns:
        Dictionary of DataFrames, one for each pullback threshold
    """
    pullback_configs = [
        ('No Pullback', lambda d: d),
        ('Pullback >= 0.5 pips', lambda d: d[d['Pullback'] >= 0.5]),
        ('Pullback >= 1.0 pip', lambda d: d[d['Pullback'] >= 1.0]),
        ('Pullback >= 2.0 pips', lambda d: d[d['Pullback'] >= 2.0]),
    ]

    pullback_tables = {}

    for pullback_name, filter_func, _ in [
        (name, func, '') for name, func in pullback_configs
    ]:
        filtered_df = filter_func(df)
        total_trades = len(filtered_df)

        summary_data = {
            pullback_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome']
        }

        for ratio, breakeven_rate in RRR_CONFIGS:
            if total_trades > 0:
                profitable = filtered_df[
                    (filtered_df['SL'] != filtered_df['Pullback']) &
                    (filtered_df['TP'] >= (ratio * filtered_df['SL']))
                ]
                wins = len(profitable)
                losses = total_trades - wins
                win_rate = (wins / total_trades * 100)
                edge = win_rate - breakeven_rate
                outcome = (wins * ratio) - losses

                summary_data[f'1:{ratio} RRR'] = [
                    total_trades,
                    wins,
                    losses,
                    f'{win_rate:.1f}%',
                    f'{edge:.1f}%',
                    f'{outcome}R'
                ]
            else:
                summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R']

        pullback_tables[pullback_name] = pd.DataFrame(summary_data)

    return pullback_tables

def analyze_sl_reduction_profitability(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze how reducing stop loss size affects trade profitability.

    Tests the impact of tightening stop losses on trade outcomes.

    Args:
        df: Trading data with SL, TP, and Pullback columns

    Returns:
        Dictionary of DataFrames showing profitability for each SL reduction strategy
    """
    sl_reduction_configs = [
        ('No adjustment', lambda sl: sl),
        ('1 pip reduction', lambda sl: sl - 1),
        ('2 pips reduction', lambda sl: sl - 2),
        ('1 pip reduction or max 4 pip SL', lambda sl: np.where(sl > 4, 4, sl - 1)),
        ('1 pip reduction or max 5 pip SL', lambda sl: np.where(sl > 5, 5, sl - 1)),
        ('1 pip reduction or max 6 pip SL', lambda sl: np.where(sl > 6, 6, sl - 1)),
        ('1 pip reduction or max 7 pip SL', lambda sl: np.where(sl > 7, 7, sl - 1)),
        ('1 pip reduction or max 8 pip SL', lambda sl: np.where(sl > 8, 8, sl - 1)),
    ]

    sl_reduction_tables = {}

    for config_name, sl_adjust_func, _, _ in [
        (name, func, 0, '') for name, func in sl_reduction_configs
    ]:
        working_df = df.copy()
        adjusted_sl = sl_adjust_func(working_df['SL'])
        adjusted_sl = np.maximum(adjusted_sl, 1.1)  # Minimum 1.1 pip (broker limit)

        total_trades = len(working_df)

        summary_data = {
            config_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome']
        }

        for ratio, breakeven_rate in RRR_CONFIGS:
            if total_trades > 0:
                profitable = working_df[
                    (working_df['SL'] != working_df['Pullback']) &
                    (working_df['Pullback'] < adjusted_sl) &
                    (working_df['TP'] >= (ratio * adjusted_sl))
                ]
                wins = len(profitable)
                losses = total_trades - wins
                win_rate = (wins / total_trades * 100)
                edge = win_rate - breakeven_rate
                outcome = (wins * ratio) - losses

                summary_data[f'1:{ratio} RRR'] = [
                    total_trades,
                    wins,
                    losses,
                    f'{win_rate:.1f}%',
                    f'{edge:.1f}%',
                    f'{outcome}R'
                ]
            else:
                summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R']

        sl_reduction_tables[config_name] = pd.DataFrame(summary_data)

    return sl_reduction_tables

# ============================================================================
# STRATEGY CREATION AND EVALUATION
# ============================================================================

def _create_technical_strategies() -> List[Tuple[str, Callable, str]]:
    """Create technical indicator based strategies."""
    return [
        ("EMA + Trade Direction",
         lambda df: df[df['EMA'] == df['Direction']],
         "Trade with EMA trend"),
        ("EMA + Opposite Trade Direction",
         lambda df: df[df['EMA'] != df['Direction']],
         "Counter-trend trades"),
        ("EMA + BOS",
         lambda df: df[(df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS')],
         "Trend + Break of Structure"),
        ("EMA + CH",
         lambda df: df[(df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'CH')],
         "Trend + Change of Character"),
        ("BOS",
         lambda df: df[df['BOS/CH'] == 'BOS'],
         "Break of Structure trades only"),
        ("CH",
         lambda df: df[df['BOS/CH'] == 'CH'],
         "Change of Character trades only"),
    ]

def _create_risk_management_strategies() -> List[Tuple[str, Callable, str]]:
    """Create risk management based strategies."""
    return [
        ("Conservative: SL <= 2 pips",
         lambda df: df[df['SL'] <= 2],
         "Very tight stop losses"),
        ("Moderate Risk: SL 3-6 pips",
         lambda df: df[(df['SL'] >= 3) & (df['SL'] <= 6)],
         "Medium stop losses"),
        ("Aggressive: SL >= 7 pips",
         lambda df: df[df['SL'] >= 7],
         "Wide stop losses"),
        ("BOS + Conservative SL <= 2 pips",
         lambda df: df[(df['BOS/CH'] == 'BOS') & (df['SL'] <= 2)],
         "BOS with tight stops"),
        ("BOS + Moderate SL 3-6 pips",
         lambda df: df[(df['BOS/CH'] == 'BOS') & (df['SL'] >= 3) & (df['SL'] <= 6)],
         "BOS with medium stops"),
    ]

def _create_30m_trend_strategies() -> List[Tuple[str, Callable, str]]:
    """Create 30-minute timeframe trend alignment strategies."""
    strategies = []

    # Basic 30M trend
    strategies.append((
        "With 30M Trend",
        lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                      (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell')))],
        "Higher timeframe trend alignment"
    ))

    # 30M + Technical indicators
    strategies.extend([
        ("30M Trend + EMA",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['EMA'] == df['Direction'])],
         "30M trend with EMA confirmation"),
        ("30M Trend + BOS",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['BOS/CH'] == 'BOS')],
         "30M trend with Break of Structure"),
        ("30M Trend + CH",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['BOS/CH'] == 'CH')],
         "30M trend with Change of Character"),
        ("30M Trend + EMA + BOS",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS')],
         "Triple confirmation: 30M + EMA + BOS"),
        ("30M Trend + EMA + CH",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'CH')],
         "Triple confirmation: 30M + EMA + CH"),
    ])

    # 30M + Risk management
    strategies.extend([
        ("30M Trend + SL < 10 pips",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['SL'] < 10)],
         "30M trend excluding large stops"),
        ("30M Trend + SL < 15 pips",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['SL'] < 15)],
         "30M trend excluding very large stops"),
        ("30M Trend + SL > 3 pips",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['SL'] > 3)],
         "30M trend excluding tiny stops"),
        ("30M Trend + SL > 5 pips",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['SL'] > 5)],
         "30M trend excluding small stops"),
        ("30M Trend + 3 < SL < 10",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['SL'] > 3) & (df['SL'] < 10)],
         "30M trend with medium stops only"),
        ("30M Trend + 5 < SL < 15",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['SL'] > 5) & (df['SL'] < 15)],
         "30M trend with moderate stops"),
    ])

    # Complex multi-factor with 30M
    strategies.extend([
        ("30M Trend + BOS + SL < 10",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['BOS/CH'] == 'BOS') & (df['SL'] < 10)],
         "30M + BOS with risk control"),
        ("30M Trend + CH + SL < 10",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['BOS/CH'] == 'CH') & (df['SL'] < 10)],
         "30M + CH with risk control"),
        ("30M Trend + EMA + SL < 10",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['EMA'] == df['Direction']) & (df['SL'] < 10)],
         "30M + EMA with risk control"),
        ("30M Trend + EMA + BOS + SL < 10",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS') & (df['SL'] < 10)],
         "Full confluence with risk limit"),
    ])

    # 30M + Pullback analysis
    strategies.extend([
        ("30M Trend + Pullback > 2",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['Pullback'] > 2)],
         "30M trend with decent pullback"),
        ("30M Trend + Pullback > 3",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['Pullback'] > 3)],
         "30M trend with strong pullback"),
    ])

    # 30M + News filters
    strategies.extend([
        ("30M Trend + No News",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      df['News Event'].isna()],
         "30M trend avoiding news"),
        ("30M Trend + News > 2hrs",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (~df['News Event'].isna()) & (df['Hours Until News'] >= 2)],
         "30M trend with safe news distance"),
    ])

    # Additional combinations
    strategies.extend([
        ("30M Trend + EMA + 3 < SL < 10",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['EMA'] == df['Direction']) & (df['SL'] > 3) & (df['SL'] < 10)],
         "30M + EMA with optimal stops"),
        ("30M Trend + BOS + Pullback > 2",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['BOS/CH'] == 'BOS') & (df['Pullback'] > 2)],
         "30M + BOS with pullback filter"),
        ("30M Trend + CH + No News",
         lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) |
                       (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) &
                      (df['BOS/CH'] == 'CH') & df['News Event'].isna()],
         "30M + CH in clean conditions"),
    ])

    return strategies

def _create_news_strategies() -> List[Tuple[str, Callable, str]]:
    """Create news event based strategies."""
    return [
        ("No News Events",
         lambda df: df[df['News Event'].isna()],
         "Avoid news volatility"),
        ("News Event Present",
         lambda df: df[~df['News Event'].isna()],
         "Trade during news"),
        ("News in 2+ Hours",
         lambda df: df[(~df['News Event'].isna()) & (df['Hours Until News'] >= 2)],
         "Trade before news impact"),
    ]

def create_strategy_library() -> List[Strategy]:
    """
    Create a comprehensive library of trading strategies for backtesting.

    Generates strategies across multiple categories:
    1. Technical Indicators (EMA, BOS/CH)
    2. Risk Management (Stop Loss levels)
    3. Market Structure (Trend alignment)
    4. Time-based (News events)
    5. Combined filters (Multi-factor strategies)

    Returns:
        List of Strategy objects ready for backtesting
    """
    all_strategies = []

    # Combine all strategy categories
    all_strategies.extend(_create_technical_strategies())
    all_strategies.extend(_create_risk_management_strategies())
    all_strategies.extend(_create_30m_trend_strategies())
    all_strategies.extend(_create_news_strategies())

    # Convert to Strategy objects
    return [Strategy(name, func, desc) for name, func, desc in all_strategies]

def evaluate_all_strategies(
    df: pd.DataFrame,
    strategies: List[Strategy]
) -> Dict[str, pd.DataFrame]:
    """
    Run backtesting on all strategies and compile results.

    Args:
        df: Trading data
        strategies: List of Strategy objects

    Returns:
        Dictionary mapping strategy names to their performance DataFrames
    """
    strategy_results = {}
    sl_columns = ['SL', 'SL 5M CC', 'SL 5M Stop']

    for sl_column in sl_columns:
        for strategy in strategies:
            # Apply strategy filter
            filtered_df = strategy.apply(df)

            # Calculate normal RRR statistics
            summary_df = calculate_rrr_stats(filtered_df, strategy.name, sl_column)
            strategy_results[f"{strategy.name}[{ENTRY_TYPE_NAMES[sl_column]}]"] = summary_df

    return strategy_results

def get_top_strategies_by_edge(
    strategy_results: Dict[str, pd.DataFrame],
    rrr_column: str
) -> pd.DataFrame:
    """
    Extract top performing strategies for a specific RRR, sorted by Edge.

    Args:
        strategy_results: Dictionary of strategy results
        rrr_column: Column name for RRR (e.g., '1:2 RRR')

    Returns:
        Top strategies ranked by edge
    """
    strategy_performance = []

    for strategy_name, df in strategy_results.items():
        # Skip if column doesn't exist
        if rrr_column not in df.columns:
            continue

        # Extract performance metrics
        total_trades = df[rrr_column].iloc[0]
        wins = df[rrr_column].iloc[1]
        losses = df[rrr_column].iloc[2]
        win_rate = df[rrr_column].iloc[3]
        edge = df[rrr_column].iloc[4]
        outcome_str = df[rrr_column].iloc[5]
        entry_str = df[rrr_column].iloc[6]

        # Parse edge value for sorting - handle both string and numeric
        try:
            if isinstance(edge, str):
                # Remove % sign and convert to float
                edge_str = edge.strip()
                if edge_str.endswith('%'):
                    edge_value = float(edge_str[:-1])
                else:
                    edge_value = float(edge_str)
            else:
                edge_value = float(edge) if edge else 0.0
        except (ValueError, TypeError, AttributeError):
            # If parsing fails, set to 0
            edge_value = 0.0

        # Clean up display name
        display_name = strategy_name.split('[')[0].strip()

        strategy_performance.append({
            'Strategy': display_name,
            'Entry': entry_str,
            'Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Win Rate': win_rate,
            'Edge': edge,
            'Outcome': outcome_str,
            'edge_value': edge_value
        })

    # Filter positive edge strategies and sort
    filtered_strategies = [s for s in strategy_performance if s['edge_value'] > 0]

    # Sort by edge_value in descending order
    top_strategies = sorted(filtered_strategies, key=lambda x: x['edge_value'], reverse=True)

    # Remove sorting key from display
    for strat in top_strategies:
        del strat['edge_value']

    return pd.DataFrame(top_strategies)

# ============================================================================
# VISUALIZATION AND DISPLAY FUNCTIONS
# ============================================================================

def style_table(
    table_df: pd.DataFrame,
    first_column_width: str = DEFAULT_COLUMN_WIDTH,
    highlight_column: Optional[str] = None,
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR
) -> pd.io.formats.style.Styler:
    """
    Apply consistent styling to a DataFrame for display.

    Args:
        table_df: DataFrame to style
        first_column_width: Width for the first column
        highlight_column: Optional column to highlight
        highlight_color: Color for highlighted column

    Returns:
        Styled DataFrame ready for display
    """
    first_column = table_df.columns[0]
    styled_df = table_df.style.set_properties(
        subset=[first_column],
        **{'width': first_column_width, 'font-weight': 'bold'}
    )

    if highlight_column and highlight_column in table_df.columns:
        styled_df = styled_df.set_properties(
            subset=[highlight_column],
            **{'color': highlight_color}
        )

    return styled_df