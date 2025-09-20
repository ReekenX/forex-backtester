"""
Utility functions for forex trading strategy analysis.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath='./eurusd.csv'):
    """
    Load EUR/USD data from CSV and clean NaN values.
    
    Args:
        filepath (str): Path to the CSV file containing trading data
    
    Returns:
        pd.DataFrame: Cleaned dataframe with trading data
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


def display_profitable_strategies(strategy_results):
    """
    Display detailed results for all or selected strategies.
    
    Args:
        strategy_results (dict): Dictionary of strategy performance DataFrames
    """
    from IPython.display import display, HTML
    
    # Filter for profitable strategies (positive outcome at 1:1 RRR)
    profitable_strategies = []
    for name, df in strategy_results.items():
        outcome_str = df['1:1 RRR'].iloc[5]
        outcome = int(outcome_str.replace('R', ''))
        if outcome > 0:
            profitable_strategies.append((name, df))
    
    display(HTML(f"<h2>💰 Profitable Strategies ({len(profitable_strategies)} of {len(strategy_results)})</h2>"))
    strategies_to_display = profitable_strategies
    
    # Display each strategy's results
    for _, summary_df in strategies_to_display:
        # Style the DataFrame for better readability
        # Use the first column of the DataFrame (which contains the strategy name)
        first_column = summary_df.columns[0]
        styled_df = summary_df.style.set_properties(
            subset=[first_column], 
            **{'width': '250px', 'font-weight': 'bold'}
        )
        
        # Highlight positive outcomes in green, negative in red
        for col in ['1:1 RRR', '1:2 RRR', '1:3 RRR']:
            outcome_str = summary_df[col].iloc[5]
            outcome = int(outcome_str.replace('R', ''))
            color = 'green' if outcome > 0 else 'red' if outcome < 0 else 'gray'
            styled_df = styled_df.set_properties(
                subset=[col],
                **{'color': color if summary_df.index[5] == 'Outcome' else 'white'}
            )
        
        display(styled_df)
        print()  # Add spacing


def analyze_pullback_profitability(df):
    """
    Analyze how pullback size affects trade profitability.

    Creates multiple tables showing profitability statistics for different pullback sizes,
    similar to the "What type of entry is the best?" table format.

    Args:
        df (pd.DataFrame): Trading data with Pullback, TP, and SL columns

    Returns:
        dict: Dictionary of DataFrames, one for each pullback threshold
    """
    import pandas as pd

    # Define pullback thresholds to analyze
    pullback_configs = [
        ('No Pullback', lambda d: d, 'No pullback filter'),
        ('Pullback >= 0.5 pips', lambda d: d[d['Pullback'] >= 0.5], 'Minimum 0.5 pip pullback'),
        ('Pullback >= 1.0 pip', lambda d: d[d['Pullback'] >= 1.0], 'Minimum 1.0 pip pullback'),
        ('Pullback >= 1.5 pips', lambda d: d[d['Pullback'] >= 1.5], 'Minimum 1.5 pip pullback'),
        ('Pullback >= 2.0 pips', lambda d: d[d['Pullback'] >= 2.0], 'Minimum 2.0 pip pullback'),
    ]

    # RRR configurations with their breakeven win rates
    rrr_configs = [
        (1, 50.0),   # 1:1 RRR - need 50% to break even
        (2, 33.3),   # 1:2 RRR - need 33.3% to break even
        (3, 25.0),   # 1:3 RRR - need 25% to break even
    ]

    # Create a DataFrame for each pullback threshold
    pullback_tables = {}

    for pullback_name, filter_func, _ in pullback_configs:
        # Filter trades for this pullback threshold
        filtered_df = filter_func(df)

        # Total trades includes ALL trades (including where SL == Pullback)
        total_trades = len(filtered_df)

        # Initialize the summary data structure
        summary_data = {
            pullback_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome']
        }

        # Calculate statistics for each RRR
        for ratio, breakeven_rate in rrr_configs:
            rrr_label = f'1:{ratio} RRR'

            if total_trades > 0:
                # Calculate wins for this RRR ratio
                # Wins must have: SL != Pullback (valid entry) AND TP >= ratio * SL (profitable)
                profitable = filtered_df[
                    (filtered_df['SL'] != filtered_df['Pullback']) &
                    (filtered_df['TP'] >= (ratio * filtered_df['SL']))
                ]
                wins = len(profitable)
                losses = total_trades - wins  # This includes trades where SL == Pullback as losses
                win_rate = (wins / total_trades * 100)

                # Calculate edge (win rate - breakeven rate)
                edge = win_rate - breakeven_rate

                # Calculate outcome in R-multiples
                outcome = (wins * ratio) - losses

                summary_data[rrr_label] = [
                    total_trades,
                    wins,
                    losses,
                    f'{win_rate:.1f}%',
                    f'{edge:.1f}%',
                    f'{outcome}R'
                ]
            else:
                # Empty strategy
                summary_data[rrr_label] = [
                    0,
                    0,
                    0,
                    '0.0%',
                    '0.0%',
                    '0R'
                ]

        pullback_tables[pullback_name] = pd.DataFrame(summary_data)

    return pullback_tables


def analyze_sl_reduction_profitability(df):
    """
    Analyze how reducing stop loss size affects trade profitability.

    This function tests the impact of tightening stop losses by 1-2 pips on trade outcomes.
    For example, if SL=3 and TP=5, the trade would normally be a loss with 1:1 RRR (TP < SL).
    But with a 1 pip reduction (SL becomes 2), the trade becomes profitable (TP=5 >= 2*1).

    Args:
        df (pd.DataFrame): Trading data with SL, TP, and Pullback columns

    Returns:
        dict: Dictionary of DataFrames showing profitability for each SL reduction strategy
    """
    import pandas as pd

    # Define SL reduction configurations
    sl_reduction_configs = [
        ('No adjustment', lambda sl: sl, 0, 'Original stop loss values'),
        ('1 pip reduction', lambda sl: sl - 1, 1, 'Stop loss reduced by 1 pip'),
        ('2 pips reduction', lambda sl: sl - 2, 2, 'Stop loss reduced by 2 pips'),
        ('1 pip reduction or max 4 pip SL', lambda sl: np.where(sl > 4, 4, sl - 1), 1, 'Stop loss reduced by 1 pip but max 4 pip'),
        ('1 pip reduction or max 5 pip SL', lambda sl: np.where(sl > 5, 5, sl - 1), 1, 'Stop loss reduced by 1 pip but max 5 pip'),
        ('1 pip reduction or max 6 pip SL', lambda sl: np.where(sl > 6, 6, sl - 1), 1, 'Stop loss reduced by 1 pip but max 6 pip'),
        ('1 pip reduction or max 7 pip SL', lambda sl: np.where(sl > 7, 7, sl - 1), 1, 'Stop loss reduced by 1 pip but max 7 pip'),
        ('1 pip reduction or max 8 pip SL', lambda sl: np.where(sl > 8, 8, sl - 1), 1, 'Stop loss reduced by 1 pip but max 8 pip'),
    ]

    # RRR configurations with their breakeven win rates
    rrr_configs = [
        (1, 50.0),   # 1:1 RRR - need 50% to break even
        (2, 33.3),   # 1:2 RRR - need 33.3% to break even
        (3, 25.0),   # 1:3 RRR - need 25% to break even
    ]

    # Create a DataFrame for each SL reduction strategy
    sl_reduction_tables = {}

    for config_name, sl_adjust_func, _, _ in sl_reduction_configs:
        # Work with all trades
        working_df = df.copy()

        # Apply SL adjustment (but ensure SL doesn't go below 0)
        adjusted_sl = sl_adjust_func(working_df['SL'])
        adjusted_sl = np.maximum(adjusted_sl, 1.1)  # Minimum 1.1 pip as it is broker limit

        # Total trades for this configuration
        total_trades = len(working_df)

        # Initialize the summary data structure
        summary_data = {
            config_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome']
        }

        # Calculate statistics for each RRR
        for ratio, breakeven_rate in rrr_configs:
            rrr_label = f'1:{ratio} RRR'

            if total_trades > 0:
                # Calculate wins for this RRR ratio with adjusted SL
                # A trade wins if:
                # 1. SL != Pullback (valid entry)
                # 2. Pullback < adjusted_sl (pullback is smaller than the new stop loss)
                # 3. TP >= ratio * adjusted_sl (profitable with the adjusted stop loss)
                profitable = working_df[
                    (working_df['SL'] != working_df['Pullback']) &
                    (working_df['Pullback'] < adjusted_sl) &
                    (working_df['TP'] >= (ratio * adjusted_sl))
                ]
                wins = len(profitable)
                losses = total_trades - wins
                win_rate = (wins / total_trades * 100)

                # Calculate edge (win rate - breakeven rate)
                edge = win_rate - breakeven_rate

                # Calculate outcome in R-multiples
                outcome = (wins * ratio) - losses

                summary_data[rrr_label] = [
                    total_trades,
                    wins,
                    losses,
                    f'{win_rate:.1f}%',
                    f'{edge:.1f}%',
                    f'{outcome}R'
                ]
            else:
                # Empty strategy
                summary_data[rrr_label] = [
                    0,
                    0,
                    0,
                    '0.0%',
                    '0.0%',
                    '0R'
                ]

        sl_reduction_tables[config_name] = pd.DataFrame(summary_data)

    return sl_reduction_tables


def analyze_entry_timing_detailed(df):
    """
    Analyze different entry timing strategies with detailed statistics including RRR analysis.

    Returns 4 separate DataFrames, one for each entry method, showing:
    - Total trades
    - Wins
    - Losses
    - Win Rate
    - Edge (for each RRR)
    - Outcome (for each RRR)
    - Entry method

    Args:
        df (pd.DataFrame): Trading data with entry signals

    Returns:
        dict: Dictionary containing 4 DataFrames, one for each entry method
    """
    import pandas as pd

    # Entry methods configuration with entry type descriptions
    entry_methods = {
        '5M Stop': {
            'filter': lambda d: d['SL 5M Stop'] != 0,
            'sl_col': 'SL 5M Stop',
            'entry_type': '5M Stop Entry'
        },
        '5M Confirmation Candle': {
            'filter': lambda d: d['SL 5M CC'] != 0,
            'sl_col': 'SL 5M CC',
            'entry_type': '5M CC Entry'
        },
        '1M Confirmation Candle': {
            'filter': lambda d: d['SL'] != 0,
            'sl_col': 'SL',
            'entry_type': '1M CC Entry'
        }
    }

    # RRR configurations with their breakeven win rates
    rrr_configs = [
        (1, 50.0),   # 1:1 RRR - need 50% to break even
        (2, 33.3),   # 1:2 RRR - need 33.3% to break even
        (3, 25.0),   # 1:3 RRR - need 25% to break even
    ]

    # Create a DataFrame for each entry method
    entry_tables = {}

    for method_name, method_config in entry_methods.items():
        # Get relevant trades for this method (same as original analyze_entry_timing)
        relevant_trades = df[method_config['filter'](df)]
        sl_col = method_config['sl_col']

        # Use all relevant trades for total count (matching original logic)
        total_trades = len(relevant_trades)

        # Initialize the summary data structure
        summary_data = {
            method_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome']
        }

        # Calculate statistics for each RRR
        for ratio, breakeven_rate in rrr_configs:
            rrr_label = f'1:{ratio} RRR'

            if total_trades > 0:
                # Calculate wins for this RRR ratio
                # Apply the same filters as original analyze_entry_timing:
                # 1. SL != Pullback (valid entry)
                # 2. TP > ratio * SL (profitable for this RRR)

                valid_wins = relevant_trades[
                    (relevant_trades['SL'] != relevant_trades['Pullback']) &
                    (relevant_trades['TP'] > (ratio * relevant_trades[sl_col]))
                ]
                wins = len(valid_wins)
                losses = total_trades - wins
                win_rate = (wins / total_trades * 100)

                # Calculate edge (win rate - breakeven rate)
                edge = win_rate - breakeven_rate

                # Calculate outcome in R-multiples
                # For each win, you gain 'ratio' R
                # For each loss, you lose 1 R
                outcome = (wins * ratio) - losses

                summary_data[rrr_label] = [
                    total_trades,
                    wins,
                    losses,
                    f'{win_rate:.1f}%',
                    f'{edge:.1f}%',
                    f'{outcome}R'
                ]
            else:
                # Empty strategy
                summary_data[rrr_label] = [
                    0,
                    0,
                    0,
                    '0.0%',
                    '0.0%',
                    '0R'
                ]

        entry_tables[method_name] = pd.DataFrame(summary_data)

    return entry_tables


def calculate_rrr_stats_with_extra(data_df, strategy_name, sl_column='SL', extra_pips=1):
    """
    Calculate RRR statistics with Extra pip adjustment for trades.

    This function applies the Extra pip logic: trades with SL == Pullback are included
    if their Extra value is less than the specified extra_pips threshold.

    Args:
        data_df (pd.DataFrame): Filtered DataFrame containing trades for this strategy
        strategy_name (str): Name of the strategy for labeling
        sl_column (str): Column to use for stop loss calculations
        extra_pips (int): Number of extra pips to consider (default 1)

    Returns:
        pd.DataFrame: Statistics table with Extra pip adjustments
    """
    # Keep the same total trades as the original strategy
    # but adjust the win calculation based on Extra pip logic
    total_trades = len(data_df)

    # RRR configurations with their breakeven win rates
    rrr_configs = [
        (1, 50.0),   # 1:1 RRR
        (2, 33.3),   # 1:2 RRR
        (3, 25.0),   # 1:3 RRR
    ]

    # Handle empty strategy results
    if total_trades == 0:
        summary_data = {
            strategy_name + f' (+{extra_pips}p)': ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
        }
        for ratio, _ in rrr_configs:
            summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R']
        return pd.DataFrame(summary_data)

    # Calculate statistics for each RRR level
    summary_data = {
        strategy_name + f' (+{extra_pips}p)': ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
    }
    entry_names = {
        'SL': '1M CC',
        'SL 5M CC': '5M CC',
        'SL 5M Stop': '5M Stop'
    }
    entry_str = entry_names[sl_column]

    for ratio, breakeven_rate in rrr_configs:
        # Calculate wins including Extra pip logic:
        # A trade wins if either:
        # 1. SL != Pullback AND TP >= ratio * SL (normal win)
        # 2. SL == Pullback AND Extra < extra_pips AND TP >= ratio * (SL + extra_pips) (win with extra pip)

        normal_wins = data_df[
            (data_df['SL'] != data_df['Pullback']) &
            (data_df['TP'] >= ratio * data_df[sl_column])
        ]

        extra_wins = data_df[
            (data_df['SL'] == data_df['Pullback']) &
            (data_df['Extra'] < extra_pips) &
            (data_df['TP'] >= ratio * (data_df[sl_column] + extra_pips))
        ]

        # Combine wins (use index to avoid duplicates)
        all_win_indices = set(normal_wins.index) | set(extra_wins.index)
        wins = len(all_win_indices)
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100

        # Edge is how much better we perform than breakeven
        edge = win_rate - breakeven_rate

        # Calculate expected outcome in R-multiples
        outcome = (wins * ratio) - losses

        summary_data[f'1:{ratio} RRR'] = [
            total_trades,
            wins,
            losses,
            f"{win_rate:.1f}%",
            f"{edge:.1f}%",
            f"{outcome}R",
            entry_str
        ]

    return pd.DataFrame(summary_data)


def calculate_rrr_stats_max_5_pips(data_df, strategy_name, sl_column='SL'):
    """
    Calculate RRR statistics with max 5 pip stop loss adjustment for trades.

    This function applies the max 5 pip SL logic: stop losses are capped at 5 pips,
    and trades are only profitable if Pullback < adjusted_sl.

    Args:
        data_df (pd.DataFrame): Filtered DataFrame containing trades for this strategy
        strategy_name (str): Name of the strategy for labeling
        sl_column (str): Column to use for stop loss calculations

    Returns:
        pd.DataFrame: Statistics table with max 5 pip SL adjustments
    """
    import numpy as np

    # Keep the same total trades as the original strategy
    total_trades = len(data_df)

    # RRR configurations with their breakeven win rates
    rrr_configs = [
        (1, 50.0),   # 1:1 RRR
        (2, 33.3),   # 1:2 RRR
        (3, 25.0),   # 1:3 RRR
    ]

    # Handle empty strategy results
    if total_trades == 0:
        summary_data = {
            strategy_name + ' (max 5p SL)': ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
        }
        for ratio, _ in rrr_configs:
            summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R']
        return pd.DataFrame(summary_data)

    # Calculate statistics for each RRR level
    summary_data = {
        strategy_name + ' (max 5p SL)': ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
    }
    entry_names = {
        'SL': '1M CC',
        'SL 5M CC': '5M CC',
        'SL 5M Stop': '5M Stop'
    }
    entry_str = entry_names[sl_column]

    for ratio, breakeven_rate in rrr_configs:
        # Apply max 5 pip SL logic (similar to analyze_sl_reduction_profitability)
        # Cap SL at 5 pips maximum
        adjusted_sl = np.where(data_df[sl_column] > 5, 5, data_df[sl_column])

        # Calculate wins with max 5 pip SL logic:
        # A trade wins if:
        # 1. SL != Pullback (valid entry)
        # 2. Pullback < adjusted_sl (pullback doesn't exceed adjusted stop)
        # 3. TP >= ratio * adjusted_sl (profitable for this RRR)

        wins_mask = (
            (data_df['SL'] != data_df['Pullback']) &
            (data_df['Pullback'] < adjusted_sl) &
            (data_df['TP'] >= ratio * adjusted_sl)
        )

        wins = wins_mask.sum()
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100

        # Edge is how much better we perform than breakeven
        edge = win_rate - breakeven_rate

        # Calculate expected outcome in R-multiples
        outcome = (wins * ratio) - losses

        summary_data[f'1:{ratio} RRR'] = [
            total_trades,
            wins,
            losses,
            f"{win_rate:.1f}%",
            f"{edge:.1f}%",
            f"{outcome}R",
            entry_str
        ]

    return pd.DataFrame(summary_data)


def calculate_rrr_stats(data_df, strategy_name, sl_column='SL'):
    """
    Calculate comprehensive Risk-Reward Ratio statistics for a trading strategy.
    
    This function evaluates a strategy's performance across different RRR targets
    and calculates key metrics including win rate, edge over breakeven, and
    expected outcome in R-multiples.
    
    Args:
        data_df (pd.DataFrame): Filtered DataFrame containing trades for this strategy
        strategy_name (str): Name of the strategy for labeling
        sl_column (str): Column to use for stop loss calculations. Options:
            - 'SL': Default 1M confirmation candle stop loss
            - 'SL 5M CC': 5M confirmation candle stop loss
            - 'SL 5M Stop': 5M stop entry stop loss
    
    Returns:
        pd.DataFrame: Statistics table with the following metrics:
            - Total trades: Number of trades in the strategy
            - Wins/Losses: Count of profitable vs unprofitable trades
            - Win Rate: Percentage of winning trades
            - Edge: Win rate minus breakeven rate for the RRR
            - Outcome: Net result in R-multiples (profit factor)
            - Entry: Entry method used for the strategy
    """
    total_trades = len(data_df)
    
    # RRR configurations with their breakeven win rates
    # For 1:1 RRR, you need 50% wins to break even
    # For 1:2 RRR, you need 33.3% wins to break even
    # For 1:3 RRR, you need 25% wins to break even
    rrr_configs = [
        (1, 50.0),   # 1:1 RRR
        (2, 33.3),   # 1:2 RRR
        (3, 25.0),   # 1:3 RRR
    ]
    
    # Handle empty strategy results
    if total_trades == 0:
        summary_data = {
            strategy_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
        }
        for ratio, _ in rrr_configs:
            summary_data[f'1:{ratio} RRR'] = [0, 0, 0, '0.0%', '0.0%', '0R']
        return pd.DataFrame(summary_data)
    
    # Calculate statistics for each RRR level
    summary_data = {
        strategy_name: ['Total trades', 'Wins', 'Losses', 'Win Rate', 'Edge', 'Outcome', 'Entry']
    }
    entry_names = {
        'SL': '1M CC',
        'SL 5M CC': '5M CC',
        'SL 5M Stop': '5M Stop'
    }
    entry_str = entry_names[sl_column]
    
    for ratio, breakeven_rate in rrr_configs:
        # Find profitable trades for this RRR
        profitable = data_df[data_df['TP'] > ratio * data_df[sl_column]]
        wins = len(profitable)
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        
        # Edge is how much better we perform than breakeven
        edge = win_rate - breakeven_rate
        
        # Calculate expected outcome in R-multiples
        # Winners make 'ratio' R, losers lose 1R
        outcome = (wins * ratio) - losses
        
        summary_data[f'1:{ratio} RRR'] = [
            total_trades,
            wins,
            losses,
            f"{win_rate:.1f}%",
            f"{edge:.1f}%",
            f"{outcome}R",
            entry_str
        ]
    
    return pd.DataFrame(summary_data)


class Strategy:
    """
    Encapsulates a trading strategy with its filter logic and metadata.
    
    Attributes:
        name (str): Strategy identifier
        filter_func (callable): Function that filters trades based on strategy rules
        description (str): Human-readable description of the strategy
    """
    
    def __init__(self, name, filter_func, description=""):
        """
        Initialize a trading strategy.
        
        Args:
            name (str): Strategy name
            filter_func (callable): Lambda or function that takes df and returns filtered df
            description (str): Optional description of the strategy
        """
        self.name = name
        self.filter_func = filter_func
        self.description = description
    
    def apply(self, df):
        """
        Apply the strategy filter to a dataframe of trades.
        
        Args:
            df (pd.DataFrame): Trading data
            
        Returns:
            pd.DataFrame: Filtered trades matching strategy criteria
        """
        return self.filter_func(df)


def create_strategy_library():
    """
    Create a comprehensive library of trading strategies for backtesting.
    
    This function generates strategies across multiple categories:
    1. Technical Indicators (EMA, BOS/CH)
    2. Risk Management (Stop Loss levels)
    3. Market Structure (Trend alignment)
    4. Time-based (Weekdays, News events)
    5. Combined filters (Multi-factor strategies)
    
    Returns:
        list: List of Strategy objects ready for backtesting
    """
    strategy_configs = []
    
    # ========== TECHNICAL INDICATOR STRATEGIES ==========
    # EMA (Exponential Moving Average) based strategies
    strategy_configs.extend([
        ("EMA + Trade Direction", lambda df: df[df['EMA'] == df['Direction']], "Trade with EMA trend"),
        ("EMA + Opposite Trade Direction", lambda df: df[df['EMA'] != df['Direction']], "Counter-trend trades"),
        ("EMA + BOS", lambda df: df[(df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS')], "Trend + Break of Structure"),
        ("EMA + CH", lambda df: df[(df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'CH')], "Trend + Change of Character"),
    ])
    
    # Market Structure strategies
    strategy_configs.extend([
        ("BOS", lambda df: df[df['BOS/CH'] == 'BOS'], "Break of Structure trades only"),
        ("CH", lambda df: df[df['BOS/CH'] == 'CH'], "Change of Character trades only"),
    ])
    
    # ========== RISK MANAGEMENT STRATEGIES ==========
    # Stop Loss size filtering
    strategy_configs.extend([
        ("Conservative: SL <= 2 pips", lambda df: df[df['SL'] <= 2], "Very tight stop losses"),
        ("Moderate Risk: SL 3-6 pips", lambda df: df[(df['SL'] >= 3) & (df['SL'] <= 6)], "Medium stop losses"),
        ("Aggressive: SL >= 7 pips", lambda df: df[df['SL'] >= 7], "Wide stop losses"),
    ])
    
    # Risk-adjusted market structure combinations
    strategy_configs.extend([
        ("BOS + Conservative SL", lambda df: df[(df['BOS/CH'] == 'BOS') & (df['SL'] <= 2)], "BOS with tight stops"),
        ("BOS + Moderate SL", lambda df: df[(df['BOS/CH'] == 'BOS') & (df['SL'] >= 3) & (df['SL'] <= 6)], "BOS with medium stops"),
    ])
    
    # ========== TIME-BASED STRATEGIES ==========
    # Day of week analysis
    # weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # for day in weekdays:
        # strategy_configs.append(
        #     (f"{day} Only", lambda df, d=day: df[df['Weekday'] == d], f"Trades on {day}")
        # )
    
    # ========== HIGHER TIMEFRAME TREND ==========
    # 30-minute timeframe trend alignment
    strategy_configs.extend([
        ("With 30M Trend", lambda df: df[(df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))], "Higher timeframe uptrend"),
        
        # 30M Trend + Technical Indicators
        ("30M Trend + EMA", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['EMA'] == df['Direction'])], "30M trend with EMA confirmation"),
        ("30M Trend + BOS", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['BOS/CH'] == 'BOS')], "30M trend with Break of Structure"),
        ("30M Trend + CH", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['BOS/CH'] == 'CH')], "30M trend with Change of Character"),
        ("30M Trend + EMA + BOS", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS')], "Triple confirmation: 30M + EMA + BOS"),
        ("30M Trend + EMA + CH", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'CH')], "Triple confirmation: 30M + EMA + CH"),
        
        # 30M Trend + Risk Management (Stop Loss Filters)
        ("30M Trend + SL < 10 pips", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['SL'] < 10)], "30M trend excluding large stops"),
        ("30M Trend + SL < 15 pips", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['SL'] < 15)], "30M trend excluding very large stops"),
        ("30M Trend + SL > 3 pips", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['SL'] > 3)], "30M trend excluding tiny stops"),
        ("30M Trend + SL > 5 pips", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['SL'] > 5)], "30M trend excluding small stops"),
        ("30M Trend + 3 < SL < 10", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['SL'] > 3) & (df['SL'] < 10)], "30M trend with medium stops only"),
        ("30M Trend + 5 < SL < 15", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['SL'] > 5) & (df['SL'] < 15)], "30M trend with moderate stops"),
        
        # Complex Multi-Factor with 30M Trend
        ("30M Trend + BOS + SL < 10", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['BOS/CH'] == 'BOS') & (df['SL'] < 10)], "30M + BOS with risk control"),
        ("30M Trend + CH + SL < 10", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['BOS/CH'] == 'CH') & (df['SL'] < 10)], "30M + CH with risk control"),
        ("30M Trend + EMA + SL < 10", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['EMA'] == df['Direction']) & (df['SL'] < 10)], "30M + EMA with risk control"),
        ("30M Trend + EMA + BOS + SL < 10", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS') & (df['SL'] < 10)], "Full confluence with risk limit"),
        
        # 30M Trend + Pullback Analysis
        ("30M Trend + Pullback > 2", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['Pullback'] > 2)], "30M trend with decent pullback"),
        ("30M Trend + Pullback > 3", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['Pullback'] > 3)], "30M trend with strong pullback"),
        
        # 30M Trend + News Filters
        ("30M Trend + No News", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & df['News Event'].isna()], "30M trend avoiding news"),
        ("30M Trend + News > 2hrs", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (~df['News Event'].isna()) & (df['Hours Until News'] >= 2)], "30M trend with safe news distance"),
        
        # Additional Combined Strategies
        ("30M Trend + EMA + 3 < SL < 10", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['EMA'] == df['Direction']) & (df['SL'] > 3) & (df['SL'] < 10)], "30M + EMA with optimal stops"),
        ("30M Trend + BOS + Pullback > 2", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['BOS/CH'] == 'BOS') & (df['Pullback'] > 2)], "30M + BOS with pullback filter"),
        ("30M Trend + CH + No News", lambda df: df[((df['30M Leg'].isin(['Above H', 'Above L']) & (df['Direction'] == 'Buy')) | (df['30M Leg'].isin(['Below H', 'Below L']) & (df['Direction'] == 'Sell'))) & (df['BOS/CH'] == 'CH') & df['News Event'].isna()], "30M + CH in clean conditions")
    ])
    
    # ========== NEWS EVENT STRATEGIES ==========
    strategy_configs.extend([
        ("No News Events", lambda df: df[df['News Event'].isna()], "Avoid news volatility"),
        ("News Event Present", lambda df: df[~df['News Event'].isna()], "Trade during news"),
        ("News in 2+ Hours", lambda df: df[(~df['News Event'].isna()) & (df['Hours Until News'] >= 2)], "Trade before news impact"),
    ])
    
    # ========== COMBINED MULTI-FACTOR STRATEGIES ==========
    # These combine multiple indicators for potentially higher probability setups
    # strategy_configs.extend([
    #     ("EMA + BOS + Low Risk", lambda df: df[(df['EMA'] == df['Direction']) & (df['BOS/CH'] == 'BOS') & (df['SL'] <= 3)], "Triple confirmation setup"),
    #     ("BOS + Bullish 30M + No News", lambda df: df[(df['BOS/CH'] == 'BOS') & (df['30M Leg'].isin(['Above H', 'Above L'])) & (df['News Event'].isna())], "Clean trend continuation"),
    #     ("Strong Alignment + EMA", lambda df: df[((df['30M Leg'] == 'Above H') & (df['Direction'] == 'Buy') & (df['EMA'] == 'Buy')) | ((df['30M Leg'] == 'Below L') & (df['Direction'] == 'Sell') & (df['EMA'] == 'Sell'))], "Full trend confluence"),
    # ])
    
    # Convert configurations to Strategy objects
    return [Strategy(name, func, desc) for name, func, desc in strategy_configs]


def evaluate_all_strategies(df, strategies, include_extra_pip=False, include_max_5_pip=False):
    """
    Run backtesting on all strategies and compile results.

    Args:
        df (pd.DataFrame): Trading data
        strategies (list): List of Strategy objects
        include_extra_pip (bool): Whether to include Extra 1 pip analysis
        include_max_5_pip (bool): Whether to include max 5 pip SL analysis

    Returns:
        dict: Dictionary mapping strategy names to their performance DataFrames
    """
    strategy_results = {}
    sl_columns = ['SL', 'SL 5M CC', 'SL 5M Stop']
    entry_names = {
        'SL': '1M CC',
        'SL 5M CC': '5M CC',
        'SL 5M Stop': '5M Stop'
    }

    for sl_column in sl_columns:
        for strategy in strategies:
            # Apply strategy filter
            filtered_df = strategy.apply(df)

            # Calculate normal RRR statistics
            summary_df = calculate_rrr_stats(filtered_df, strategy.name, sl_column)
            strategy_results[strategy.name + '[' + entry_names[sl_column] + ']'] = summary_df

            # If requested, also calculate with Extra 1 pip
            if include_extra_pip:
                summary_df_extra = calculate_rrr_stats_with_extra(filtered_df, strategy.name, sl_column, extra_pips=1)
                strategy_results[strategy.name + ' [Extra 1 pip][' + entry_names[sl_column] + ']'] = summary_df_extra

            # If requested, also calculate with max 5 pip SL
            if include_max_5_pip:
                summary_df_max5 = calculate_rrr_stats_max_5_pips(filtered_df, strategy.name, sl_column)
                strategy_results[strategy.name + ' [Max 5p SL][' + entry_names[sl_column] + ']'] = summary_df_max5

    return strategy_results


def get_top_strategies_by_edge(strategy_results, rrr_column):
    """
    Extract top performing strategies for a specific RRR, sorted by Edge.

    Args:
        strategy_results (dict): Dictionary of strategy results
        rrr_column (str): Column name for RRR (e.g., '1:2 RRR')

    Returns:
        pd.DataFrame: Top strategies ranked by edge
    """
    strategy_performance = []

    for strategy_name, df in strategy_results.items():
        # Extract performance metrics
        total_trades = df[rrr_column].iloc[0]
        wins = df[rrr_column].iloc[1]
        losses = df[rrr_column].iloc[2]
        win_rate = df[rrr_column].iloc[3]
        edge = df[rrr_column].iloc[4]
        outcome_str = df[rrr_column].iloc[5]
        entry_str = df[rrr_column].iloc[6]

        # Parse edge value for sorting
        edge_value = float(edge.replace('%', ''))

        # Check if this is an Extra 1 pip strategy
        if '[Extra 1 pip]' in strategy_name:
            # Extract the base strategy name (before [Extra 1 pip])
            base_name = strategy_name.split('[Extra 1 pip]')[0].strip()
            display_name = base_name + ' [Extra 1 pip]'
        elif '[Max 5p SL]' in strategy_name:
            # Extract the base strategy name (before [Max 5p SL])
            base_name = strategy_name.split('[Max 5p SL]')[0].strip()
            display_name = base_name + ' [Max 5p SL]'
        else:
            # Regular strategy - remove entry type bracket
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

    # Filter out strategies with negative Edge
    filtered_strategies = [
        strat for strat in strategy_performance
        if strat['edge_value'] > 0
    ]

    # Sort by edge and get top strategies
    top_strategies = sorted(
        filtered_strategies,
        key=lambda x: x['edge_value'],
        reverse=True
    )

    # Remove sorting key from display
    for strat in top_strategies:
        del strat['edge_value']

    return pd.DataFrame(top_strategies)


def style_table(table_df, first_column_width='250px', highlight_column=None, highlight_color='green'):
    """
    Apply consistent styling to a DataFrame for display.

    Args:
        table_df (pd.DataFrame): DataFrame to style
        first_column_width (str): Width for the first column (default '250px')
        highlight_column (str): Optional column to highlight
        highlight_color (str): Color for highlighted column (default 'green')

    Returns:
        pandas.io.formats.style.Styler: Styled DataFrame ready for display
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


def display_tables_with_insights(tables_dict, insights_html):
    """
    Display multiple tables with consistent styling and insights.

    Args:
        tables_dict (dict): Dictionary of table names to DataFrames
        insights_html (str): HTML string with insights to display after tables
    """
    from IPython.display import display, HTML

    for _, table_df in tables_dict.items():
        display(style_table(table_df))
        print()  # Add spacing between tables

    if insights_html:
        display(HTML(insights_html))