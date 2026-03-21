# AI Rules

A Jupyter-based project for analyzing and backtesting Forex trading data.

## Project Layout

Each strategy lives in its own directory under `strategies/`:

```
strategies/<name>/
├── data.csv       # Trading data
├── utils/         # Analysis modules
│   ├── __init__.py
│   └── *.py
└── tests/         # Test modules
    └── test_*.py
```

Notebooks live in `labs/<name>.ipynb` and import from their strategy's utils package via `sys.path.insert(0, '../strategies/<name>')`.

Current strategies:
- **5OB1CC** - 5-minute Order Block, 1-minute Confirmation Candle (`strategies/5OB1CC/`)
- **15LS1CC** - 15-minute Leg Structure, 1-minute Confirmation Candle (`strategies/15LS1CC/`)

## Background About Trades

This project contains real trading data for EURUSD currency traded during London session.

## Background About Trader

Trader's goal is to pass a prop firm challenge.

Prop firm rules are:
- Gain 8% of profit (R=0.5%) to pass the challenge
- Daily max drawdown 5%
- Max total drawdown 10%
- No minimum trading days

### Data Format

Each strategy has its own `data.csv` inside its `strategies/<name>/` directory.

#### 5OB1CC Data Format (strategies/5OB1CC/data.csv)
- **Date**: Trading date (YYYY-MM-DD format)
- **Trade**: Trade identifier (e.g., #1, #2)
- **Range**: Size in pips of the market structure leg that was broken
- **Strength**: Size in pips that price went after making new High or Low
- **Weekday**: Day of the week (from Monday till Friday)
- **Hour**: Trading hour in Lithuanian timezone (values from 10 to 18)
- **Direction**: Trade direction (Buy or Sell)
- **EMA**: EMA signal (Buy or Sell)
- **SL**: Stop Loss value (distance to safe stop when trade signal was received)
- **Pullback**: Pullback value (if equal to `SL` column - this trade was a loss)
- **TP**: Take Profit value (any value above 0 or empty means that this trade was profitable)
- **Extra**: Extra pips needed to make this trade profitable
- **BOS/CH**: Market structure type (BOS - Break of Structure; CH - Change of Character)
- **30M Leg**: 30-minute timeframe leg analysis
- **Hours Until News**: Time until news event in hours
- **News Event**: Associated news event title

#### 15LS1CC Data Format (strategies/15LS1CC/data.csv)
- **Date**: Trading date (YYYY-MM-DD format)
- **Weekday**: Day of the week
- **Trade**: Trade identifier
- **Direction**: Trade direction (Buy or Sell)
- **1H**: Higher timeframe (1-hour) alignment (Buy or Sell)
- **SL**: Stop Loss in pips
- **Pullback**: Pullback in pips
- **TP**: Take Profit in pips (empty = not profitable)
- **R**: Risk-reward achieved (e.g., 10.0)

## Trading Data CSV Fields

1. SL column is a number in pips that was from entry signal to "safe stop"
2. Pullback column is a number of pips that trade reached after entry and before reaching TP
3. TP column is a number in pips that trade could have reached
4. Empty TP column means that trade was not profitable
5. If Pullback equals SL, it means that trade was an immediate loss
6. If Pullback is higher than SL, it means that overall trade could have been profitable but a higher SL was needed than "safe stop"
7. R column (if not empty) is a number of how many R's this trade achieved (e.g., 10 pips for TP and 3 pips for SL would have achieved 10/3=3 R)
8. Minimum broker SL is 1.1 pips
9. Win condition: TP > (RRR ratio x SL)
10. When a trade is entered, only the SL column is known. Pullback and TP are only learned after the trade is finished. Therefore, Pullback and TP columns must not be used for strategy filtering (e.g., "take a trade when Pullback is smaller than SL" does not make sense because Pullback is unknown at entry time)

**Example 1**: SL 3.1 pips, Pullback 2.4 pips and TP 10 pips. When entering a position, safe stop loss was 3.1 pips away from entry. Then price at some point went 2.4 pips against the entry but later recovered and shot 10 pips from entry. Total reward (R) was 10/3.1=3R.

**Example 2**: SL 2.1 pips, Pullback 3.4 pips and TP 10 pips. This trade would be a loss, but only if the safe stop had been higher - it would have been a winner.

## Development Flow for New Features

When building new analysis features, follow this three-file pattern within the strategy directory:

### 1. Notebook (labs/<strategy>.ipynb)
- **Purpose**: Clean, minimal interface for users
- **Content**: Only imports and function calls
- Keep code to a minimum - just load data and call display functions

### 2. Python Module (strategies/<strategy>/utils/*.py)
- **Purpose**: All business logic and calculations
- **Content**: Analysis functions, data processing, HTML generation
- Build from scratch without reusing existing code
- Simple, easy-to-understand implementation

### 3. Test Module (strategies/<strategy>/tests/test_*.py)
- **Purpose**: Verify functionality with small datasets
- **Content**: Unit tests using 10-row sample data
- Test all core functions and edge cases
- Run with: `poetry run python -m pytest strategies/<strategy>/tests/ -v`

### Reference Implementations
- **5OB1CC**: `labs/5OB1CC.ipynb`, `strategies/5OB1CC/utils/hours.py`, `strategies/5OB1CC/tests/test_hours.py`
- **15LS1CC**: `labs/15LS1CC.ipynb`, `strategies/15LS1CC/utils/confirmation_candle.py`, `strategies/15LS1CC/tests/test_confirmation_candle.py`

## Acceptance Criteria

Before announcing that any feature or modification is complete and working:

1. **All tests must pass** - Run the corresponding test module(s) and verify all tests succeed
2. **Test execution** - Use `poetry run python -m pytest strategies/<strategy>/tests/ -v` to run tests
3. **No exceptions** - Tests must complete without errors or warnings
4. **Test coverage** - All new functionality must have corresponding tests

Only after all tests pass successfully should you confirm the work is complete. If any tests fail, fix the issues before announcing completion.

## Standard Table Structure

All analysis tables should follow this standardized column format:

### Standard Columns (in order)
1. **Strategy** - Strategy name or grouping identifier (e.g., "10h" for hour 10)
2. **RRR** - Risk-reward ratio (e.g., "1:1", "1:2", "1:3")
3. **Trades** - Total number of trades
4. **Notation** - Win/Loss notation (e.g., "12W - 33L")
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
# 15LS1CC strategy
poetry run python -c "
import sys; sys.path.insert(0, 'strategies/15LS1CC')
import pandas as pd
from utils.confirmation_candle import load_data, calculate_buffer_statistics

df = load_data('strategies/15LS1CC/data.csv')
stats = calculate_buffer_statistics(df)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
print(stats.to_string(index=False))
"
```

Each lab notebook has a corresponding utils module inside its strategy directory. To preview any lab's data:
1. Find the strategy directory it imports from (e.g., `labs/15LS1CC.ipynb` -> `strategies/15LS1CC/utils/`)
2. Add the strategy path to sys.path, then call the calculation functions
3. Use pandas display options for readable terminal output
