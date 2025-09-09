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