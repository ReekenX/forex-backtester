"""
Correlation Analysis Functions for Forex Trading Strategy

This module provides correlation analysis tools to understand relationships between
trading parameters and profitability metrics.

Main components:
- SL size vs win rate correlation
- Pullback depth vs outcome correlation
- Hour/weekday vs win rate correlation
- Technical indicators vs profitability correlation
- Range/Strength vs win rate correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Import shared constants from parent utils module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RRR_CONFIGS, _win_condition_normal


# ============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# ============================================================================


def analyze_sl_vs_winrate_correlation(
    df: pd.DataFrame,
    sl_column: str = "SL",
    bin_size: float = 1.0
) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlation between SL size and win rate across different RRR levels.

    Args:
        df: DataFrame with trade data
        sl_column: Column name for stop loss values
        bin_size: Size of SL bins in pips (default: 1.0)

    Returns:
        Dictionary with DataFrames for each RRR level containing SL ranges and win rates
    """
    # Filter out trades with SL == 0 or invalid SL, and SL > 20 pips
    valid_df = df[(df[sl_column] > 0) & (df[sl_column] <= 20)].copy()

    if len(valid_df) == 0:
        return {}

    # Create SL bins
    max_sl = valid_df[sl_column].max()
    bins = np.arange(0, max_sl + bin_size, bin_size)
    valid_df['SL_Range'] = pd.cut(valid_df[sl_column], bins=bins)

    results = {}

    for ratio, breakeven_rate in RRR_CONFIGS:
        correlation_data = []

        for sl_range in valid_df['SL_Range'].cat.categories:
            range_df = valid_df[valid_df['SL_Range'] == sl_range]
            total = len(range_df)

            if total == 0:
                continue

            # Calculate wins using the standard win condition
            wins = len(_win_condition_normal(range_df, ratio, sl_column))
            win_rate = (wins / total) * 100 if total > 0 else 0

            # Get midpoint of SL range for plotting
            sl_midpoint = (sl_range.left + sl_range.right) / 2

            correlation_data.append({
                'SL Range': f"{sl_range.left:.1f}-{sl_range.right:.1f}",
                'SL Midpoint': sl_midpoint,
                'Total Trades': total,
                'Wins': wins,
                'Losses': total - wins,
                'Win Rate (%)': round(win_rate, 2),
                'Breakeven (%)': breakeven_rate,
                'Edge (%)': round(win_rate - breakeven_rate, 2)
            })

        results[f'1:{ratio}'] = pd.DataFrame(correlation_data)

    return results


def display_sl_vs_winrate_correlation(
    df: pd.DataFrame,
    sl_column: str = "SL",
    bin_size: float = 1.0
):
    """
    Display correlation charts between SL size and win rate for different RRR levels.

    Args:
        df: DataFrame with trade data
        sl_column: Column name for stop loss values
        bin_size: Size of SL bins in pips (default: 1.0)
    """
    from IPython.display import display, HTML

    display(HTML("<h2>SL Size vs Win Rate Correlation Analysis</h2>"))

    # Get correlation data
    correlation_results = analyze_sl_vs_winrate_correlation(df, sl_column, bin_size)

    if not correlation_results:
        display(HTML("<p>No valid data found for analysis.</p>"))
        return

    # Create figure with subplots for each RRR level
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Stop Loss Size vs Win Rate Correlation', fontsize=16, fontweight='bold')

    for idx, (rrr_label, corr_df) in enumerate(correlation_results.items()):
        ax = axes[idx]

        if corr_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{rrr_label} RRR')
            continue

        # Plot win rate vs SL size
        ax.plot(corr_df['SL Midpoint'], corr_df['Win Rate (%)'],
                marker='o', linewidth=2, markersize=8, label='Win Rate', color='#2E86AB')

        # Plot breakeven line
        breakeven = corr_df['Breakeven (%)'].iloc[0]
        ax.axhline(y=breakeven, color='red', linestyle='--', linewidth=2,
                   label=f'Breakeven ({breakeven}%)', alpha=0.7)

        # Add win count as text on points
        for _, row in corr_df.iterrows():
            ax.annotate(f"{int(row['Wins'])}",
                       (row['SL Midpoint'], row['Win Rate (%)']),
                       textcoords="offset points", xytext=(0,10),
                       ha='center', fontsize=8, color='gray')

        # Styling
        ax.set_xlabel('Stop Loss Size (pips)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{rrr_label} RRR', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def display_pullback_vs_outcome_correlation(df: pd.DataFrame, bin_size: float = 1.0):
    """
    Display correlation charts between pullback depth and trade outcome.

    Args:
        df: DataFrame with trade data
        bin_size: Size of pullback bins in pips (default: 1.0)
    """
    from IPython.display import display, HTML

    display(HTML("<h2>Pullback Depth vs Outcome Correlation</h2>"))

    valid_df = df[df['Pullback'] > 0].copy()

    if len(valid_df) == 0:
        display(HTML("<p>No valid pullback data found.</p>"))
        return

    max_pullback = valid_df['Pullback'].max()
    bins = np.arange(0, max_pullback + bin_size, bin_size)
    valid_df['Pullback_Range'] = pd.cut(valid_df['Pullback'], bins=bins)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Pullback Depth vs Win Rate Correlation', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        data_points = []

        for pb_range in valid_df['Pullback_Range'].cat.categories:
            range_df = valid_df[valid_df['Pullback_Range'] == pb_range]
            total = len(range_df)

            if total < 5:  # Skip ranges with too few trades
                continue

            wins = len(_win_condition_normal(range_df, ratio, 'SL'))
            win_rate = (wins / total) * 100
            pb_midpoint = (pb_range.left + pb_range.right) / 2

            data_points.append({
                'midpoint': pb_midpoint,
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if data_points:
            midpoints = [d['midpoint'] for d in data_points]
            win_rates = [d['win_rate'] for d in data_points]
            wins_counts = [d['wins'] for d in data_points]
            totals = [d['total'] for d in data_points]

            ax.plot(midpoints, win_rates, marker='o', linewidth=2, markersize=8, color='#2E86AB')
            ax.axhline(y=breakeven_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)

            for i, (mp, wr, w, tot) in enumerate(zip(midpoints, win_rates, wins_counts, totals)):
                ax.annotate(f"{w}", (mp, wr), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8, color='gray')

            ax.set_xlabel('Pullback Depth (pips)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def display_hour_vs_winrate_correlation(df: pd.DataFrame):
    """Display correlation between hour of day and win rate."""
    from IPython.display import display, HTML

    display(HTML("<h2>Hour of Day vs Win Rate Correlation</h2>"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Trading Hour vs Win Rate Correlation', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        hour_data = []

        for hour in sorted(df['Hour'].unique()):
            if hour == 0:
                continue

            hour_df = df[df['Hour'] == hour]
            total = len(hour_df)

            if total < 3:
                continue

            wins = len(_win_condition_normal(hour_df, ratio, 'SL'))
            win_rate = (wins / total) * 100

            hour_data.append({
                'hour': int(hour),
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if hour_data:
            hours = [d['hour'] for d in hour_data]
            win_rates = [d['win_rate'] for d in hour_data]
            wins_counts = [d['wins'] for d in hour_data]
            totals = [d['total'] for d in hour_data]

            ax.bar(hours, win_rates, color='#2E86AB', alpha=0.7, width=0.8)
            ax.axhline(y=breakeven_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)

            for h, wr, w, tot in zip(hours, win_rates, wins_counts, totals):
                ax.text(h, wr + 2, f"{w}", ha='center', fontsize=8, color='gray')

            ax.set_xlabel('Hour of Day (UTC)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim(0, 100)
            ax.set_xticks(hours)

    plt.tight_layout()
    plt.show()


def display_weekday_vs_winrate_correlation(df: pd.DataFrame):
    """Display correlation between day of week and win rate."""
    from IPython.display import display, HTML

    display(HTML("<h2>Day of Week vs Win Rate Correlation</h2>"))

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Weekday vs Win Rate Correlation', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        weekday_data = []

        for weekday in weekday_order:
            weekday_df = df[df['Weekday'] == weekday]
            total = len(weekday_df)

            if total == 0:
                continue

            wins = len(_win_condition_normal(weekday_df, ratio, 'SL'))
            win_rate = (wins / total) * 100

            weekday_data.append({
                'weekday': weekday[:3],
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if weekday_data:
            weekdays = [d['weekday'] for d in weekday_data]
            win_rates = [d['win_rate'] for d in weekday_data]
            wins_counts = [d['wins'] for d in weekday_data]
            totals = [d['total'] for d in weekday_data]

            ax.bar(weekdays, win_rates, color='#2E86AB', alpha=0.7)
            ax.axhline(y=breakeven_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)

            for i, (wd, wr, w, tot) in enumerate(zip(weekdays, win_rates, wins_counts, totals)):
                ax.text(i, wr + 2, f"{w}", ha='center', fontsize=8, color='gray')

            ax.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def display_sl_vs_pullback_correlation(df: pd.DataFrame):
    """Display correlation between initial SL size and maximum pullback."""
    from IPython.display import display, HTML

    display(HTML("<h2>Initial SL Size vs Pullback Correlation</h2>"))

    valid_df = df[(df['SL'] > 0) & (df['Pullback'] > 0)].copy()

    if len(valid_df) == 0:
        display(HTML("<p>No valid data found.</p>"))
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(valid_df['SL'], valid_df['Pullback'], alpha=0.5, color='#2E86AB', s=50)

    # Add trend line
    z = np.polyfit(valid_df['SL'], valid_df['Pullback'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_df['SL'].min(), valid_df['SL'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

    # Calculate correlation
    corr = valid_df['SL'].corr(valid_df['Pullback'])

    ax.set_xlabel('Stop Loss Size (pips)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pullback Depth (pips)', fontsize=12, fontweight='bold')
    ax.set_title(f'SL Size vs Pullback Correlation (r={corr:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    plt.tight_layout()
    plt.show()


def display_ema_vs_profitability_correlation(df: pd.DataFrame):
    """Display correlation between EMA alignment and profitability."""
    from IPython.display import display, HTML

    display(HTML("<h2>EMA Alignment vs Win Rate Correlation</h2>"))

    ema_categories = ['Aligned', 'Not Aligned', 'Opposite']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('EMA Alignment vs Win Rate', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        ema_data = []

        for ema_state in ema_categories:
            ema_df = df[df['EMA'] == ema_state]
            total = len(ema_df)

            if total == 0:
                continue

            wins = len(_win_condition_normal(ema_df, ratio, 'SL'))
            win_rate = (wins / total) * 100

            ema_data.append({
                'ema': ema_state,
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if ema_data:
            emas = [d['ema'] for d in ema_data]
            win_rates = [d['win_rate'] for d in ema_data]
            wins_counts = [d['wins'] for d in ema_data]
            totals = [d['total'] for d in ema_data]

            colors = ['#2E86AB' if wr >= breakeven_rate else '#D65F5F' for wr in win_rates]
            ax.bar(emas, win_rates, color=colors, alpha=0.7)
            ax.axhline(y=breakeven_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)

            for i, (e, wr, w, tot) in enumerate(zip(emas, win_rates, wins_counts, totals)):
                ax.text(i, wr + 2, f"{w}", ha='center', fontsize=8, color='gray')

            ax.set_xlabel('EMA State', fontsize=11, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def display_bosch_vs_success_correlation(df: pd.DataFrame):
    """Display correlation between BOS/CH patterns and RRR success."""
    from IPython.display import display, HTML

    display(HTML("<h2>BOS/CH Pattern vs Win Rate Correlation</h2>"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('BOS/CH Pattern vs Win Rate', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        pattern_data = []

        for pattern in df['BOS/CH'].unique():
            if pd.isna(pattern) or pattern == '':
                continue

            pattern_df = df[df['BOS/CH'] == pattern]
            total = len(pattern_df)

            if total < 3:
                continue

            wins = len(_win_condition_normal(pattern_df, ratio, 'SL'))
            win_rate = (wins / total) * 100

            pattern_data.append({
                'pattern': pattern,
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if pattern_data:
            # Sort by win rate
            pattern_data = sorted(pattern_data, key=lambda x: x['win_rate'], reverse=True)

            patterns = [d['pattern'] for d in pattern_data]
            win_rates = [d['win_rate'] for d in pattern_data]
            wins_counts = [d['wins'] for d in pattern_data]
            totals = [d['total'] for d in pattern_data]

            colors = ['#2E86AB' if wr >= breakeven_rate else '#D65F5F' for wr in win_rates]
            ax.barh(patterns, win_rates, color=colors, alpha=0.7)
            ax.axvline(x=breakeven_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)

            for i, (p, wr, w, tot) in enumerate(zip(patterns, win_rates, wins_counts, totals)):
                ax.text(wr + 2, i, f"{w}", va='center', fontsize=8, color='gray')

            ax.set_ylabel('BOS/CH Pattern', fontsize=11, fontweight='bold')
            ax.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='x')
            ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.show()


def display_30m_trend_vs_success_correlation(df: pd.DataFrame):
    """Display correlation between 30M trend and trade success."""
    from IPython.display import display, HTML

    display(HTML("<h2>30M Trend vs Win Rate Correlation</h2>"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('30M Trend Alignment vs Win Rate', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        trend_data = []

        for trend in df['30M Leg'].unique():
            if pd.isna(trend) or trend == '':
                continue

            trend_df = df[df['30M Leg'] == trend]
            total = len(trend_df)

            if total < 3:
                continue

            wins = len(_win_condition_normal(trend_df, ratio, 'SL'))
            win_rate = (wins / total) * 100

            trend_data.append({
                'trend': trend,
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if trend_data:
            # Sort by win rate
            trend_data = sorted(trend_data, key=lambda x: x['win_rate'], reverse=True)

            trends = [d['trend'] for d in trend_data]
            win_rates = [d['win_rate'] for d in trend_data]
            wins_counts = [d['wins'] for d in trend_data]
            totals = [d['total'] for d in trend_data]

            colors = ['#2E86AB' if wr >= breakeven_rate else '#D65F5F' for wr in win_rates]
            ax.barh(trends, win_rates, color=colors, alpha=0.7)
            ax.axvline(x=breakeven_rate, color='red', linestyle='--', linewidth=2, alpha=0.7)

            for i, (t, wr, w, tot) in enumerate(zip(trends, win_rates, wins_counts, totals)):
                ax.text(wr + 2, i, f"{w}", va='center', fontsize=8, color='gray')

            ax.set_ylabel('30M Trend', fontsize=11, fontweight='bold')
            ax.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='x')
            ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.show()


def display_range_vs_winrate_correlation(df: pd.DataFrame, bin_size: float = 5.0):
    """
    Display correlation between Range and win rate for different RRR levels.

    Args:
        df: DataFrame with trade data
        bin_size: Size of Range bins in pips (default: 5.0)
    """
    from IPython.display import display, HTML

    display(HTML("<h2>Range vs Win Rate Correlation Analysis</h2>"))

    # Filter out rows with missing Range data
    valid_df = df[df['Range'].notna() & (df['Range'] != '') & (df['Range'] > 0)].copy()

    if len(valid_df) == 0:
        display(HTML("<p>No valid Range data found for analysis.</p>"))
        return

    # Create bins for Range
    max_range = valid_df['Range'].max()
    bins = np.arange(0, max_range + bin_size, bin_size)
    valid_df['Range_Bin'] = pd.cut(valid_df['Range'], bins=bins)

    # Create figure with subplots for each RRR level
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Range vs Win Rate Correlation', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        data_points = []

        for range_bin in valid_df['Range_Bin'].cat.categories:
            bin_df = valid_df[valid_df['Range_Bin'] == range_bin]
            total = len(bin_df)

            if total < 5:  # Skip bins with too few trades
                continue

            wins = len(_win_condition_normal(bin_df, ratio, 'SL'))
            win_rate = (wins / total) * 100
            range_midpoint = (range_bin.left + range_bin.right) / 2

            data_points.append({
                'midpoint': range_midpoint,
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if data_points:
            midpoints = [d['midpoint'] for d in data_points]
            win_rates = [d['win_rate'] for d in data_points]
            wins_counts = [d['wins'] for d in data_points]

            # Plot win rate vs Range
            ax.plot(midpoints, win_rates, marker='o', linewidth=2, markersize=8,
                   label='Win Rate', color='#2E86AB')

            # Plot breakeven line
            ax.axhline(y=breakeven_rate, color='red', linestyle='--', linewidth=2,
                      label=f'Breakeven ({breakeven_rate}%)', alpha=0.7)

            # Add win count as text on points
            for mp, wr, w in zip(midpoints, win_rates, wins_counts):
                ax.annotate(f"{int(w)}", (mp, wr), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8, color='gray')

            # Styling
            ax.set_xlabel('Range (pips)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def display_strength_vs_winrate_correlation(df: pd.DataFrame, bin_size: float = 5.0):
    """
    Display correlation between Strength and win rate for different RRR levels.

    Args:
        df: DataFrame with trade data
        bin_size: Size of Strength bins (default: 5.0)
    """
    from IPython.display import display, HTML

    display(HTML("<h2>Strength vs Win Rate Correlation Analysis</h2>"))

    # Filter out rows with missing Strength data
    valid_df = df[df['Strength'].notna() & (df['Strength'] != '') & (df['Strength'] > 0)].copy()

    if len(valid_df) == 0:
        display(HTML("<p>No valid Strength data found for analysis.</p>"))
        return

    # Create bins for Strength
    max_strength = valid_df['Strength'].max()
    bins = np.arange(0, max_strength + bin_size, bin_size)
    valid_df['Strength_Bin'] = pd.cut(valid_df['Strength'], bins=bins)

    # Create figure with subplots for each RRR level
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Strength vs Win Rate Correlation', fontsize=16, fontweight='bold')

    for idx, (ratio, breakeven_rate) in enumerate(RRR_CONFIGS):
        ax = axes[idx]
        data_points = []

        for strength_bin in valid_df['Strength_Bin'].cat.categories:
            bin_df = valid_df[valid_df['Strength_Bin'] == strength_bin]
            total = len(bin_df)

            if total < 5:  # Skip bins with too few trades
                continue

            wins = len(_win_condition_normal(bin_df, ratio, 'SL'))
            win_rate = (wins / total) * 100
            strength_midpoint = (strength_bin.left + strength_bin.right) / 2

            data_points.append({
                'midpoint': strength_midpoint,
                'win_rate': win_rate,
                'wins': wins,
                'total': total
            })

        if data_points:
            midpoints = [d['midpoint'] for d in data_points]
            win_rates = [d['win_rate'] for d in data_points]
            wins_counts = [d['wins'] for d in data_points]

            # Plot win rate vs Strength
            ax.plot(midpoints, win_rates, marker='o', linewidth=2, markersize=8,
                   label='Win Rate', color='#2E86AB')

            # Plot breakeven line
            ax.axhline(y=breakeven_rate, color='red', linestyle='--', linewidth=2,
                      label=f'Breakeven ({breakeven_rate}%)', alpha=0.7)

            # Add win count as text on points
            for mp, wr, w in zip(midpoints, win_rates, wins_counts):
                ax.annotate(f"{int(w)}", (mp, wr), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8, color='gray')

            # Styling
            ax.set_xlabel('Strength', fontsize=11, fontweight='bold')
            ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best')
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'1:{ratio} RRR', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def display_all_correlations(df: pd.DataFrame):
    """Display all correlation analyses in one call."""
    display_sl_vs_winrate_correlation(df)
    display_pullback_vs_outcome_correlation(df)
    display_hour_vs_winrate_correlation(df)
    display_weekday_vs_winrate_correlation(df)
    display_sl_vs_pullback_correlation(df)
    display_bosch_vs_success_correlation(df)
    display_30m_trend_vs_success_correlation(df)
    display_range_vs_winrate_correlation(df)
    display_strength_vs_winrate_correlation(df)
