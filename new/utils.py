"""
Utility functions for forex trading strategy analysis.
"""

import pandas as pd


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
        'SL Breakout', 'Hours Until News', 'Extra'
    ]
    
    # Fill NaN values to prevent calculation errors
    for col in columns_to_fillna:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def percentage(value, total):
    """
    Calculate percentage of value over total.
    
    Args:
        value (int/float): The numerator value
        total (int/float): The denominator value
    
    Returns:
        str: Formatted percentage string (e.g., "45.5%")
    """
    if total == 0:
        return "0.0%"
    return f"{value / total * 100:.1f}%"


def calculate_profitable_trades(df):
    """
    Calculate profitability statistics for different trading strategies.
    
    Args:
        df (pd.DataFrame): Trading data
    
    Returns:
        pd.DataFrame: Profitability statistics for various strategies
    """
    # Define trading strategies with their filters
    strategies = {
        'Total': lambda d: d[(d['SL'] != d['Pullback'])],
        'With Extra 1 pip': lambda d: d[((d['SL'] != d['Pullback']) | ((d['SL'] == d['Pullback']) & (d['Extra'] < 1)))],
        'With Extra 2 pips': lambda d: d[((d['SL'] != d['Pullback']) | ((d['SL'] == d['Pullback']) & (d['Extra'] < 2)))],
        'With Extra 3 pips': lambda d: d[((d['SL'] != d['Pullback']) | ((d['SL'] == d['Pullback']) & (d['Extra'] < 3)))],
        'With EMA Direction': lambda d: d[(d['SL'] != d['Pullback']) & (d['EMA'] == d['Direction'])],
        'Against EMA Direction': lambda d: d[(d['SL'] != d['Pullback']) & (d['EMA'] != d['Direction'])],
        'Just BOS Trades': lambda d: d[(d['SL'] != d['Pullback']) & (d['BOS/CH'] == 'BOS')],
        'Just CH Trades': lambda d: d[(d['SL'] != d['Pullback']) & (d['BOS/CH'] == 'CH')],
        'With EMA + BOS': lambda d: d[(d['SL'] != d['Pullback']) & (d['EMA'] == d['Direction']) & (d['BOS/CH'] == 'BOS')],
        'With EMA + CH': lambda d: d[(d['SL'] != d['Pullback']) & (d['EMA'] == d['Direction']) & (d['BOS/CH'] == 'CH')],
    }
    
    results = {'Data': list(strategies.keys())}
    
    # Calculate for each Risk-Reward Ratio
    for rrr in [1, 2, 3]:
        rrr_results = []
        for strategy_name, filter_func in strategies.items():
            filtered = filter_func(df)
            
            if strategy_name == 'With Extra 1 pip':
                profitable = filtered[filtered['TP'] >= (filtered['SL'] + 1) * rrr]
            elif strategy_name == 'With Extra 2 pips':
                profitable = filtered[filtered['TP'] >= (filtered['SL'] + 2) * rrr]
            elif strategy_name == 'With Extra 3 pips':
                profitable = filtered[filtered['TP'] >= (filtered['SL'] + 3) * rrr]
            else:
                profitable = filtered[filtered['TP'] >= filtered['SL'] * rrr]
            
            rrr_results.append(percentage(len(profitable), len(df)))
        
        results[f'1:{rrr} RRR'] = rrr_results
    
    # Create DataFrame and sort by best 1:1 RRR performance
    df_results = pd.DataFrame(results)
    df_results['Value_num'] = df_results['1:1 RRR'].str.rstrip('%').astype(float)
    return df_results.sort_values('Value_num', ascending=False).drop(columns='Value_num').reset_index(drop=True)


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
    for strategy_name, summary_df in strategies_to_display:
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
    
    Args:
        df (pd.DataFrame): Trading data with Pullback, TP, and SL columns
    
    Returns:
        pd.DataFrame: Table showing profitability statistics for different pullback sizes
    """
    import pandas as pd
    
    # Define pullback thresholds to analyze
    pullback_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = []
    
    for threshold in pullback_thresholds:
        # Filter trades with pullback >= threshold
        filtered_df = df[df['Pullback'] >= threshold]
        
        # Count total trades
        total_trades = len(filtered_df)
        
        # Count profitable trades (TP > SL)
        profitable_trades = len(filtered_df[filtered_df['TP'] > filtered_df['SL']])
        
        # Calculate percentage
        if total_trades > 0:
            percentage = (profitable_trades / total_trades) * 100
        else:
            percentage = 0
        
        results.append({
            'Pullback': f'{threshold} pip{"s" if threshold != 1 else ""}',
            'All Trades': total_trades,
            'Profitable Trades': profitable_trades,
            'Percentage': f'{percentage:.1f}%'
        })
    
    # Also add a row for all trades (no pullback filter)
    all_trades = len(df)
    all_profitable = len(df[df['TP'] > df['SL']])
    all_percentage = (all_profitable / all_trades * 100) if all_trades > 0 else 0
    
    results.insert(0, {
        'Pullback': 'All (No Filter)',
        'All Trades': all_trades,
        'Profitable Trades': all_profitable,
        'Percentage': f'{all_percentage:.1f}%'
    })
    
    return pd.DataFrame(results)


def analyze_entry_timing(df):
    """
    Analyze different entry timing strategies and their success rates.
    
    This function compares various entry methods:
    - 1M Confirmation Candle: Entry on 1-minute candle confirmation
    - 5M Confirmation Candle: Entry on 5-minute candle confirmation  
    - 5M Stop: Entry with 5-minute stop loss level
    - 5M Breakout: Entry on 5-minute confirmation candle that I built indicator for
    
    Args:
        df (pd.DataFrame): Trading data with entry signals
    
    Returns:
        pd.DataFrame: Entry timing statistics sorted by success rate
    """
    # Calculate winning trades for each entry method
    entry_methods = {
        '1M Confirmation Candle': {
            'filter': lambda d: d['SL'] != 0,
            'profitable': lambda d: d[(d['SL'] != d['Pullback']) & (d['TP'] > d['SL'])],
            'sl_col': 'SL'
        },
        '5M Confirmation Candle': {
            'filter': lambda d: d['SL 5M CC'] != 0,
            'profitable': lambda d: d[(d['SL'] != d['Pullback']) & (d['TP'] > d['SL 5M CC'])],
            'sl_col': 'SL 5M CC'
        },
        '5M Stop': {
            'filter': lambda d: d['SL 5M Stop'] != 0,
            'profitable': lambda d: d[(d['SL'] != d['Pullback']) & (d['TP'] > d['SL 5M Stop'])],
            'sl_col': 'SL 5M Stop'
        },
        '5M Breakout': {
            'filter': lambda d: d['SL Breakout'] != 0,
            'profitable': lambda d: d[(d['SL'] != d['Pullback']) & (d['TP'] > d['SL Breakout'])],
            'sl_col': 'SL Breakout'
        }
    }
    
    results = []
    for method_name, method_config in entry_methods.items():
        # Get relevant trades for this method
        relevant_trades = df[method_config['filter'](df)]
        profitable_trades = method_config['profitable'](df)
        total_trades = len(relevant_trades)
        wins = len(profitable_trades)
        losses = total_trades - wins
        
        # Calculate metrics for different scenarios
        sl_col = method_config['sl_col']
        
        # With Extra calculation
        with_extra_filter = ((df['SL'] != df['Pullback']) | ((df['Extra'] != 0) & (df['Extra'] < 1)))
        with_extra_profitable = df[with_extra_filter & (df['TP'] > df[sl_col])]
        
        # 1:3 RRR with Extra
        rrr3_with_extra = df[with_extra_filter & (df['TP'] > df[sl_col] * 3)]
        
        results.append({
            'Idea': method_name,
            'Notation': f"{wins}W - {losses}L",
            'Win Rate': percentage(wins, total_trades),
            'With Extra': percentage(len(with_extra_profitable), total_trades),
            'With Extra & 1:3 RRR': percentage(len(rrr3_with_extra), total_trades)
        })
    
    # Convert to DataFrame and sort by win percentage
    df_results = pd.DataFrame(results)
    df_results['Value_num'] = df_results['Win Rate'].str.rstrip('%').astype(float)
    return df_results.sort_values('Value_num', ascending=False).drop(columns='Value_num').reset_index(drop=True)


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
            - 'SL Breakout': 5M breakout stop loss
    
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
        'SL 5M Stop': '5M Stop',
        'SL Breakout': '5M Breakout'
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


def evaluate_all_strategies(df, strategies):
    """
    Run backtesting on all strategies and compile results.
    
    Args:
        df (pd.DataFrame): Trading data
        strategies (list): List of Strategy objects
        
    Returns:
        dict: Dictionary mapping strategy names to their performance DataFrames
    """
    strategy_results = {}
    sl_columns = ['SL', 'SL 5M CC', 'SL 5M Stop', 'SL Breakout']
    entry_names = {
        'SL': '1M CC',
        'SL 5M CC': '5M CC',
        'SL 5M Stop': '5M Stop',
        'SL Breakout': '5M Breakout'
    }
    
    for sl_column in sl_columns:
        for strategy in strategies:
            # Apply strategy filter
            filtered_df = strategy.apply(df)
            
            # Calculate RRR statistics
            summary_df = calculate_rrr_stats(filtered_df, strategy.name, sl_column)
            
            # Store results
            strategy_results[strategy.name + '[' + entry_names[sl_column] + ']'] = summary_df
    
    return strategy_results


def get_top_strategies(strategy_results, rrr_column):
    """
    Extract top performing strategies for a specific RRR.
    
    Args:
        strategy_results (dict): Dictionary of strategy results
        rrr_column (str): Column name for RRR (e.g., '1:2 RRR')
        top_n (int): Number of top strategies to return
        
    Returns:
        pd.DataFrame: Top strategies ranked by outcome
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
        
        # Parse outcome value for sorting
        outcome = int(outcome_str.replace('R', ''))
        
        strategy_performance.append({
            'Strategy': strategy_name.split('[')[0].strip(),
            'Entry': entry_str,
            'Trades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Win Rate': win_rate,
            'Edge': edge,
            'Outcome': outcome_str,
            'outcome_value': outcome
        })
    
    # Filter out strategies with negative Edge
    filtered_strategies = [
        strat for strat in strategy_performance 
        if not strat['Edge'].startswith('-')
    ]
    
    # Sort by outcome and get top N
    top_strategies = sorted(
        filtered_strategies, 
        key=lambda x: x['outcome_value'], 
        reverse=True
    )
    
    # Remove sorting key from display
    for strat in top_strategies:
        del strat['outcome_value']
    
    return pd.DataFrame(top_strategies)