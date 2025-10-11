"""
Forex Trading Strategy Optimizer

This module provides combinatorial strategy testing similar to Meta Trader's Strategy Tester.
It generates all possible combinations of filters from CSV columns and backtests them exhaustively.

Main components:
- FilterDimension: Defines a single filter dimension with multiple options
- generate_all_combinations: Creates all possible strategy combinations
- optimize_strategies: Runs exhaustive backtesting on all combinations
- display_optimization_results: Visualizes results in sortable tables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from itertools import product, combinations
from IPython.display import display, HTML

from utils import _win_condition_normal
from utils.tables import (
    Strategy,
    create_sortable_table,
)


# ============================================================================
# FILTER DIMENSION DEFINITIONS
# ============================================================================


class FilterDimension:
    """
    Represents a single filter dimension with multiple options.

    Attributes:
        name: Dimension name (e.g., "EMA", "SL", "News")
        options: Dictionary mapping option names to filter functions
    """

    def __init__(self, name: str, options: Dict[str, Callable]):
        """
        Initialize a filter dimension.

        Args:
            name: Name of the dimension
            options: Dictionary of {option_name: filter_function}
        """
        self.name = name
        self.options = options

    def get_option_names(self) -> List[str]:
        """Get list of all option names."""
        return list(self.options.keys())

    def apply_filter(self, df: pd.DataFrame, option_name: str) -> pd.DataFrame:
        """Apply a specific filter option to the dataframe."""
        if option_name not in self.options:
            raise ValueError(f"Option '{option_name}' not found in dimension '{self.name}'")
        return self.options[option_name](df)


# ============================================================================
# FILTER DIMENSION LIBRARY
# ============================================================================


def create_filter_dimensions() -> List[FilterDimension]:
    """
    Create all available filter dimensions for strategy optimization.

    Returns:
        List of FilterDimension objects representing all available filters
    """
    dimensions = []

    # EMA Dimension
    dimensions.append(FilterDimension(
        name="EMA",
        options={
            "Any": lambda df: df,
            "Aligned": lambda df: df[df["EMA"] == df["Direction"]],
            "Counter": lambda df: df[df["EMA"] != df["Direction"]],
        }
    ))

    # BOS/CH Dimension
    dimensions.append(FilterDimension(
        name="BOS/CH",
        options={
            "Any": lambda df: df,
            "BOS": lambda df: df[df["BOS/CH"] == "BOS"],
            "CH": lambda df: df[df["BOS/CH"] == "CH"],
        }
    ))

    # 30M Leg (Trend) Dimension
    dimensions.append(FilterDimension(
        name="30M Trend",
        options={
            "Any": lambda df: df,
            "Aligned": lambda df: df[
                (df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Buy")) |
                (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Sell"))
            ],
            "Counter": lambda df: df[
                (df["30M Leg"].isin(["Above H", "Above L"]) & (df["Direction"] == "Sell")) |
                (df["30M Leg"].isin(["Below H", "Below L"]) & (df["Direction"] == "Buy"))
            ],
        }
    ))

    # Stop Loss Dimension
    dimensions.append(FilterDimension(
        name="SL",
        options={
            "Any": lambda df: df,
            "≤2": lambda df: df[df["SL"] <= 2],
            "≤5": lambda df: df[df["SL"] <= 5],
            "≤10": lambda df: df[df["SL"] <= 10],
            "≤15": lambda df: df[df["SL"] <= 15],
            "2-5": lambda df: df[(df["SL"] > 2) & (df["SL"] <= 5)],
            "5-10": lambda df: df[(df["SL"] > 5) & (df["SL"] <= 10)],
            "10-15": lambda df: df[(df["SL"] > 10) & (df["SL"] <= 15)],
            ">15": lambda df: df[df["SL"] > 15],
        }
    ))

    # News Event Dimension
    dimensions.append(FilterDimension(
        name="News",
        options={
            "Any": lambda df: df,
            "No News": lambda df: df[df["News Event"].isna()],
            "With News": lambda df: df[~df["News Event"].isna()],
            "News >2h": lambda df: df[(~df["News Event"].isna()) & (df["Hours Until News"] >= 2)],
        }
    ))

    # Hour Dimension
    dimensions.append(FilterDimension(
        name="Hour",
        options={
            "Any": lambda df: df,
            "10-12": lambda df: df[(df["Hour"] >= 10) & (df["Hour"] < 12)],
            "12-15": lambda df: df[(df["Hour"] >= 12) & (df["Hour"] < 15)],
            "15-18": lambda df: df[(df["Hour"] >= 15) & (df["Hour"] <= 18)],
            "10": lambda df: df[df["Hour"] == 10],
            "11": lambda df: df[df["Hour"] == 11],
            "12": lambda df: df[df["Hour"] == 12],
            "13": lambda df: df[df["Hour"] == 13],
            "14": lambda df: df[df["Hour"] == 14],
            "15": lambda df: df[df["Hour"] == 15],
            "16": lambda df: df[df["Hour"] == 16],
            "17": lambda df: df[df["Hour"] == 17],
            "18": lambda df: df[df["Hour"] == 18],
        }
    ))

    # Weekday Dimension
    dimensions.append(FilterDimension(
        name="Weekday",
        options={
            "Any": lambda df: df,
            "Monday": lambda df: df[df["Weekday"] == "Monday"],
            "Tuesday": lambda df: df[df["Weekday"] == "Tuesday"],
            "Wednesday": lambda df: df[df["Weekday"] == "Wednesday"],
            "Thursday": lambda df: df[df["Weekday"] == "Thursday"],
            "Friday": lambda df: df[df["Weekday"] == "Friday"],
        }
    ))

    # Direction Dimension
    dimensions.append(FilterDimension(
        name="Direction",
        options={
            "Any": lambda df: df,
            "Buy": lambda df: df[df["Direction"] == "Buy"],
            "Sell": lambda df: df[df["Direction"] == "Sell"],
        }
    ))

    return dimensions


# ============================================================================
# COMBINATION GENERATION
# ============================================================================


def generate_all_combinations(
    dimensions: List[FilterDimension],
    max_filters: int = 3,
    min_filters: int = 1,
    exclude_all_any: bool = True
) -> List[Tuple[str, Callable]]:
    """
    Generate all possible strategy combinations from filter dimensions.

    Args:
        dimensions: List of FilterDimension objects
        max_filters: Maximum number of filters per strategy (e.g., 3)
        min_filters: Minimum number of filters per strategy (e.g., 1)
        exclude_all_any: If True, exclude strategies where all filters are "Any"

    Returns:
        List of tuples: [(strategy_name, combined_filter_function), ...]
    """
    all_strategies = []

    # Generate combinations for each size (1 filter, 2 filters, 3 filters, etc.)
    for num_dimensions in range(min_filters, max_filters + 1):
        # Get all combinations of dimensions
        for dimension_combo in combinations(dimensions, num_dimensions):
            # Get all option combinations for these dimensions
            option_lists = [dim.get_option_names() for dim in dimension_combo]

            # Generate Cartesian product of all options
            for option_combo in product(*option_lists):
                # Build strategy name
                filters = []
                for dim, option in zip(dimension_combo, option_combo):
                    if option != "Any":
                        filters.append(f"{dim.name}={option}")

                # Skip if excluding "all Any" and no non-Any filters
                if exclude_all_any and len(filters) == 0:
                    continue

                # Create strategy name
                if len(filters) == 0:
                    strategy_name = "Plain Strategy"
                else:
                    strategy_name = " + ".join(filters)

                # Create combined filter function
                def create_combined_filter(dims, opts):
                    """Create a combined filter function (closure)."""
                    def combined_filter(df):
                        result = df.copy()
                        for dim, opt in zip(dims, opts):
                            result = dim.apply_filter(result, opt)
                        return result
                    return combined_filter

                filter_func = create_combined_filter(dimension_combo, option_combo)

                all_strategies.append((strategy_name, filter_func))

    return all_strategies


# ============================================================================
# STRATEGY OPTIMIZATION
# ============================================================================


def optimize_strategies(
    df: pd.DataFrame,
    max_filters: int = 3,
    min_filters: int = 1,
    min_trades: int = 10,
    min_edge: float = 0.0,
    rrr_ratios: List[int] = [1, 2, 3],
    sl_column: str = "SL",
    custom_dimensions: Optional[List[FilterDimension]] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run exhaustive strategy optimization across all filter combinations.

    Args:
        df: Trading data DataFrame
        max_filters: Maximum number of filters per strategy
        min_filters: Minimum number of filters per strategy
        min_trades: Minimum trade count required for a strategy
        min_edge: Minimum edge percentage required (e.g., 5.0 for 5%)
        rrr_ratios: List of RRR ratios to test (e.g., [1, 2, 3])
        sl_column: Stop loss column name
        custom_dimensions: Optional custom filter dimensions (uses default if None)
        top_n: If specified, return only top N strategies by edge

    Returns:
        DataFrame with optimization results sorted by edge (descending)
    """

    # Create filter dimensions
    dimensions = custom_dimensions if custom_dimensions else create_filter_dimensions()

    # Generate all combinations
    print(f"Generating strategy combinations (max {max_filters} filters)...")
    all_combinations = generate_all_combinations(
        dimensions,
        max_filters=max_filters,
        min_filters=min_filters,
        exclude_all_any=True
    )
    print(f"Total combinations to test: {len(all_combinations)}")

    # Convert to Strategy objects
    strategies = [Strategy(name, func, "") for name, func in all_combinations]

    # Run backtesting
    print(f"\nBacktesting {len(strategies)} strategies...")
    all_results = []

    for idx, strategy in enumerate(strategies):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(strategies)} strategies...")

        # Apply strategy filter
        filtered_df = strategy.apply(df)

        # Skip if not enough trades
        if len(filtered_df) < min_trades:
            continue

        # Calculate stats for each RRR ratio
        for rrr_ratio in rrr_ratios:
            # Calculate wins and losses
            winning_trades = _win_condition_normal(filtered_df, rrr_ratio, sl_column)
            wins = len(winning_trades)
            losses = len(filtered_df) - wins

            total_trades = len(filtered_df)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

            # Calculate edge (win rate - breakeven rate)
            breakeven_rates = {1: 50.0, 2: 33.3, 3: 25.0}
            breakeven_rate = breakeven_rates.get(rrr_ratio, 50.0)
            edge_value = win_rate - breakeven_rate

            # Skip if below minimum edge
            if edge_value < min_edge:
                continue

            # Calculate profit factor
            profit_factor = (wins * rrr_ratio) / losses if losses > 0 else float('inf')

            # Calculate expected payoff (per trade in R multiples)
            expected_payoff = (win_rate / 100 * rrr_ratio) - ((100 - win_rate) / 100)

            # Calculate drawdown percentage
            drawdown_pct = (losses / total_trades * 100) if total_trades > 0 else 0.0

            # Store result
            all_results.append({
                "Strategy": strategy.name,
                "Trades": total_trades,
                "RRR": f"1:{rrr_ratio}",
                "Win %": f"{win_rate:.1f}%",
                "Edge": f"{edge_value:.1f}%",
                "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                "Expected Payoff": f"{expected_payoff:.2f}R",
                "Drawdown %": f"{drawdown_pct:.1f}%",
            })

    print(f"\nOptimization complete! Found {len(all_results)} strategies meeting criteria.")

    # Convert to DataFrame
    if len(all_results) == 0:
        print("No strategies met the minimum criteria.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Sort by edge (extract numeric value for sorting)
    results_df['edge_value'] = results_df['Edge'].apply(
        lambda x: float(x.strip('%')) if isinstance(x, str) and x.strip().endswith('%') else 0.0
    )
    results_df = results_df.sort_values('edge_value', ascending=False)
    results_df = results_df.drop('edge_value', axis=1)

    # Take top N if specified
    if top_n:
        results_df = results_df.head(top_n)

    return results_df


# ============================================================================
# RESULTS DISPLAY
# ============================================================================


def display_optimization_results(
    results_df: pd.DataFrame,
    title: str = "Strategy Optimization Results"
):
    """
    Display optimization results in a sortable HTML table.

    Args:
        results_df: DataFrame with optimization results
        title: Title for the results table
    """
    if results_df.empty:
        print("No results to display.")
        return

    display(HTML(f"<h2>{title}</h2>"))
    display(HTML(f"<p>Total strategies found: <strong>{len(results_df)}</strong></p>"))

    # Create sortable table
    html_table = create_sortable_table(
        results_df,
        first_column_width='500px',
        highlight_column='Edge',
        highlight_color='green'
    )

    display(HTML(html_table))


def export_optimization_results(
    results_df: pd.DataFrame,
    filepath: str = "data/optimization_results.csv"
):
    """
    Export optimization results to CSV file (Meta Trader style).

    Args:
        results_df: DataFrame with optimization results
        filepath: Path to save CSV file
    """
    if results_df.empty:
        print("No results to export.")
        return

    results_df.to_csv(filepath, index=False)
    print(f"Results exported to: {filepath}")
