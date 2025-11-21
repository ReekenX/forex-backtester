# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Read README.md First

Always read the README.md file for complete project information including installation, usage, data format specifications, and commit conventions.

## Project Focus

This project uses a modular structure with specialized Jupyter notebooks and utility modules:

**Jupyter Notebooks (labs/ directory):**
- **tables.ipynb** - Deep strategy analysis and customization testing
- **charts.ipynb** - Visualization of profitable strategies
- **export.ipynb** - CSV data export functionality
- **correlations.ipynb** - Correlation analysis (e.g., SL size vs Win Rate)
- **optimizer.ipynb** - Meta Trader-style exhaustive strategy optimization
- **hours.ipynb** - Hour-by-hour trading performance analysis
- **ema.ipynb** - EMA-based strategy analysis

**Utility Modules (utils/ package):**
- **tables.py** - Analysis functions for strategy evaluation
- **charts.py** - Charting and visualization functions
- **export.py** - Data export utilities
- **correlations.py** - Correlation analysis functions
- **optimizer.py** - Combinatorial strategy optimizer (Meta Trader-style)
- **hours.py** - Hour analysis functions
- **ema.py** - EMA strategy analysis functions

**Test Modules (tests/ directory):**
- **hours.py** - Tests for hours analysis module
- **ema.py** - Tests for EMA analysis module

If you need to run any commands, like `jupyter`, then prefix it with `poetry run`. For example: `poetry run jupyter notebook labs/tables.ipynb`

## Architecture and Key Components

### Core Package: utils/

The utils package contains all backtesting logic organized into specialized modules:

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

6. **Strategy Optimizer** (utils/optimizer.py):
   - `FilterDimension`: Defines filter dimensions with multiple options
   - `create_filter_dimensions()`: Creates all available filter dimensions (EMA, BOS/CH, SL, News, Hour, Weekday, etc.)
   - `generate_all_combinations()`: Generates Cartesian product of all filter combinations
   - `optimize_strategies()`: Exhaustively backtests all strategy combinations
   - `display_optimization_results()`: Shows results in sortable HTML tables
   - `export_optimization_results()`: Exports results to CSV (Meta Trader format)


### Trading Logic Constraints

1. Pullback == SL means instant loss
2. Minimum broker SL is 1.1 pips
3. Win condition: TP > (RRR ratio × SL) and Pullback < SL
4. 30M trend alignment:
   - Buy trend: "Above H" or "Above L"
   - Sell trend: "Below H" or "Below L"

## Development Flow for New Features

When building new analysis features, always follow this three-file pattern:

### 1. Notebook (labs/*.ipynb)
- **Purpose**: Clean, minimal interface for users
- **Content**: Only imports and function calls
- **Example**: `labs/hours.ipynb`
- Keep code to minimum - just load data and call display functions

### 2. Python Module (utils/*.py)
- **Purpose**: All business logic and calculations
- **Content**: Analysis functions, data processing, HTML generation
- **Example**: `utils/hours.py`
- Build from scratch without reusing existing code
- Simple, easy-to-understand implementation
- Include comprehensive docstrings

### 3. Test Module (tests/*.py)
- **Purpose**: Verify functionality with small datasets
- **Content**: Unit tests using 10-row sample data
- **Example**: `tests/hours.py`
- Test all core functions and edge cases
- Run with: `poetry run python tests/<module>.py`

### Reference Implementation

See the "hours" implementation as the reference example:
- **labs/hours.ipynb** - Clean notebook with just imports and function calls
- **utils/hours.py** - Complete analysis logic built from scratch
- **tests/hours.py** - 18 comprehensive tests using 10-row datasets

This pattern ensures:
- Clean separation of concerns
- Easy testing and maintenance
- Simple, understandable code
- Consistent project structure

## Notebook Structure Guidelines

### labs/ Notebooks Requirements
- Keep notebooks **extremely simple** with minimal code
- Use only function calls from the utils package modules
- Avoid complex logic or calculations in notebook cells
- All analysis logic should be implemented in utils/ module functions
- Each notebook imports from its corresponding utils module:
  ```python
  # In labs/tables.ipynb
  from utils.tables import display_strategy_analysis

  # In labs/charts.ipynb
  from utils.charts import display_strategy_charts

  # In labs/correlations.ipynb
  from utils.correlations import analyze_sl_winrate_correlation

  # In labs/export.ipynb
  from utils.export import export_to_csv

  # In labs/optimizer.ipynb
  from utils.optimizer import optimize_strategies, display_optimization_results

  # In labs/hours.ipynb
  from utils.hours import display_hour_analysis

  # In labs/ema.ipynb
  from utils.ema import display_ema_analysis
  ```

## Standard Table Structure

All analysis tables should follow this standardized column format:

### Standard Columns (in order)
1. **Strategy** - Strategy name or grouping identifier (e.g., "10h" for hour 10)
2. **RRR** - Risk-reward ratio (e.g., "1:1", "1:2", "1:3")
3. **Trades** - Total number of trades
4. **Notation** - Win/Loss notation (e.g., "12W – 33L")
5. **Win Rate** - Percentage of winning trades (e.g., "65.5%")
6. **Outcome** - Net result in R multiples (e.g., "15R")
7. **Edge** - Profitability above breakeven (e.g., "15.5%")
8. **Days** - Number of unique days with at least one win
9. **Days %** - Percentage of trading days with wins (e.g., "67%")
10. **Trades Required** - Trades needed to earn 1R (e.g., "2.5" or "N/A")

### Table Styling
- Use dark mode optimized colors:
  - Background: `#1e1e1e`
  - Text: `#e0e0e0`
  - Positive Edge: `#4ade80` (green)
  - Negative Edge: `#f87171` (red)
  - Borders: `#404040`
- Strategy column width: 300px
- Apply highlighting to Edge column based on positive/negative values

### Days Calculation
- **Days**: Count unique dates where at least one trade was a win
- **Days %**: `(Days with wins / Total trading days) * 100`
- Trading days are counted from the filtered dataset, not calendar days

