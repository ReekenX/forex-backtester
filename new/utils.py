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
        with_extra_filter = ((df['SL'] != df['Pullback']) | (df['Extra'] != 0))
        with_extra_profitable = df[with_extra_filter & (df['TP'] > df[sl_col])]
        
        # 1:3 RRR with Extra
        rrr3_with_extra = df[with_extra_filter & (df['TP'] > df[sl_col] * 3)]
        
        results.append({
            'Idea': method_name,
            'Count': f"{wins}W - {losses}L",
            'Percentage': percentage(wins, total_trades),
            'With Extra': percentage(len(with_extra_profitable), total_trades),
            'Extra & 1:3 RRR': percentage(len(rrr3_with_extra), total_trades)
        })
    
    # Convert to DataFrame and sort by win percentage
    df_results = pd.DataFrame(results)
    df_results['Value_num'] = df_results['Percentage'].str.rstrip('%').astype(float)
    return df_results.sort_values('Value_num', ascending=False).drop(columns='Value_num').reset_index(drop=True)