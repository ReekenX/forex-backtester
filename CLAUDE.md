# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Read README.md First

Always read the README.md file for complete project information including installation, usage, data format specifications, and commit conventions.

## Project Focus

When questions are asked about this project, assume they refer to:
- **main.ipynb** - The primary Jupyter notebook for analysis
- **utils.py** - The core utility module containing all analysis functions

## Architecture and Key Components

### Core Module: utils.py

The utils module contains all backtesting logic organized into sections:

1. **Data Loading** (`load_and_clean_data`): Handles CSV loading and NaN cleanup

2. **Strategy Framework**:
   - `Strategy` class: Encapsulates filtering logic and metadata
   - `create_strategy_library()`: Generates 50+ predefined strategies across categories:
     - Technical indicators (EMA, BOS/CH combinations)
     - Risk management (SL-based filters)
     - 30M timeframe trend alignment
     - News event filters
     - Multi-factor combinations

3. **Risk-Reward Analysis**:
   - `calculate_rrr_stats()`: Computes metrics for 1:1, 1:2, 1:3 RRR
   - Win rate, edge (above breakeven), and outcome calculations
   - Support for three entry methods: 1M CC, 5M CC, 5M Stop

4. **Specialized Analyzers**:
   - `analyze_entry_timing_detailed()`: Compares entry method effectiveness
   - `analyze_pullback_profitability()`: Evaluates pullback impact
   - `analyze_sl_reduction_profitability()`: Tests stop loss optimization

5. **Performance Evaluation**:
   - `evaluate_all_strategies()`: Batch processes all strategies
   - `get_top_strategies_by_edge()`: Ranks strategies by profitability edge


### Trading Logic Constraints

1. Pullback == SL means instant loss
2. Minimum broker SL is 1.1 pips
3. Win condition: TP > (RRR ratio × SL) and Pullback < SL
4. 30M trend alignment:
   - Buy trend: "Above H" or "Above L"
   - Sell trend: "Below H" or "Below L"

## Notebook Structure Guidelines

### main.ipynb Requirements
- Keep the notebook **extremely simple** with minimal code
- Use only `display_*()` function calls from utils.py
- Avoid complex logic or calculations in notebook cells
- All analysis logic should be implemented in utils.py functions
- Example structure:
  ```python
  display_hour_analysis(df)
  display_weekday_analysis(df)
  display_strategy_analysis(df)
  ```

