# AI Rules

A Jupyter-based project for analyzing and backtesting Forex trading data in `data/eurusd_2026_1m_confirmation_candle.csv`.

## Background About Trades

This CSV file contains real trading data.

It is EURUSD currency traded during London session.

One strategy is being tested – 15 minute market structure following and execution on 1 minute confirmation candle.

### Trading Data CSV Fields

1. SL column is a number in pips that was from entry signal to "safe stop"
2. Pullback column is a number of pips that trade reached after entry and before reaching TP
3. TP column is a number in pips that trade could have reached
4. Empty TP column means that trade was not profitable
5. If Pullback equals SL, it means that trade was an immediate loss
6. If Pullback is higher than SL, it means that overall trade could have been profitable but a higher SL was needed than "safe stop"
7. R column (if not empty) is a number of how many R's this trade achieved (e.g., 10 pips for TP and 3 pips for SL would have achieved 10/3=3 R)
8. Minimum broker SL is 1.1 pips
9. Win condition: TP > (RRR ratio × SL)
10. When a trade is entered, only the SL column is known. Pullback and TP are only learned after the trade is finished. Therefore, Pullback and TP columns must not be used for strategy filtering (e.g., "take a trade when Pullback is smaller than SL" does not make sense because Pullback is unknown at entry time)

**Example 1**: SL 3.1 pips, Pullback 2.4 pips and TP 10 pips. When entering a position, safe stop loss was 3.1 pips away from entry. Then price at some point went 2.4 pips against the entry but later recovered and shot 10 pips from entry. Total reward (R) was 10/3.1=3R.

**Example 2**: SL 2.1 pips, Pullback 3.4 pips and TP 10 pips. This trade would be a loss, but only if the safe stop had been higher - it would have been a winner.

## Development Flow for New Features

When building new analysis features, always follow this three-file pattern:

### 1. Notebook (labs/*.ipynb)
- **Purpose**: Clean, minimal interface for users
- **Content**: Only imports and function calls
- **Example**: `labs/hours.ipynb`
- Keep code to a minimum - just load data and call display functions

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

### Acceptance Criteria

Before announcing that any feature or modification is complete and working:

1. **All tests must pass** - Run the corresponding test module(s) and verify all tests succeed
2. **Test execution** - Use `poetry run python tests/<module>.py` to run tests
3. **No exceptions** - Tests must complete without errors or warnings
4. **Test coverage** - All new functionality must have corresponding tests

Only after all tests pass successfully should you confirm the work is complete. If any tests fail, fix the issues before announcing completion.

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

## Previewing Lab Data

To view analysis results from a notebook without opening Jupyter, run the underlying Python functions directly:

```bash
poetry run python -c "
import pandas as pd
from utils.confirmation_candle import load_data, calculate_buffer_statistics

df = load_data('data/eurusd_2026_1m_confirmation_candle.csv')
stats = calculate_buffer_statistics(df)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
print(stats.to_string(index=False))
"
```

Each lab notebook has a corresponding utils module. To preview any lab's data:
1. Find the utils module it imports (e.g., `labs/hours.ipynb` → `utils/hours.py`)
2. Call the calculation functions directly with the CSV path relative to project root (not `../data/` as notebooks use)
3. Use pandas display options for readable terminal output

Filter results as needed:
```python
# Show only specific strategies
filtered = stats[stats['Strategy'].str.contains('EMA')]

# Show top N rows
print(stats.head(20).to_string(index=False))
```

