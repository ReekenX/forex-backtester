# Forex Backtester

A Jupyter-based project for analyzing and backtesting forex trading data.

## Strategy Background

Trading data is taken from EURUSD currency during London session.

Two strategies are being tested:
- **5OB1CC** - 5-minute Order Block, 1-minute Confirmation Candle
- **15LS1CC** - 15-minute Leg Structure, 1-minute Confirmation Candle

## Project Structure

```
forex-backtester/
├── strategies/
│   ├── 5OB1CC/                  # 5min Order Block 1min Confirmation Candle
│   │   ├── data.csv             # Trading data for this strategy
│   │   ├── utils/               # Analysis modules
│   │   │   ├── __init__.py      # Data loading and shared utilities
│   │   │   ├── tables.py        # Strategy analysis functions
│   │   │   ├── charts.py        # Charting and visualization
│   │   │   ├── correlations.py  # Correlation analysis
│   │   │   ├── export.py        # Data export utilities
│   │   │   ├── hours.py         # Hour analysis functions
│   │   │   ├── weekdays.py      # Weekday analysis functions
│   │   │   ├── singles.py       # Single setup analysis
│   │   │   ├── doubles.py       # Double setup analysis
│   │   │   ├── ema.py           # EMA analysis
│   │   │   └── optimizer.py     # Strategy optimizer
│   │   └── tests/               # Test modules
│   │       ├── test_hours.py
│   │       ├── test_weekdays.py
│   │       ├── test_singles.py
│   │       ├── test_doubles.py
│   │       └── test_ema.py
│   └── 15LS1CC/                 # 15min Leg Structure 1min Confirmation Candle
│       ├── data.csv             # Trading data for this strategy
│       ├── utils/               # Analysis modules
│       │   ├── __init__.py
│       │   └── confirmation_candle.py
│       └── tests/
│           └── test_confirmation_candle.py
├── labs/                        # Jupyter notebooks (one per strategy)
│   ├── 5OB1CC.ipynb             # Combined analysis for 5OB1CC strategy
│   └── 15LS1CC.ipynb            # Analysis for 15LS1CC strategy
├── pyproject.toml               # Poetry configuration and dependencies
└── Makefile                     # Build automation commands
```

## Requirements

- Python 3.11 or higher
- Poetry (for dependency management)

## Installation

1. Clone this repository
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
   or
   ```bash
   make install
   ```

## Usage

### Running Jupyter

Launch Jupyter Notebook to work with the analysis notebooks:
```bash
poetry run jupyter notebook
```

This will open Jupyter in the current directory, allowing you to navigate to the `labs/` folder and open any notebook:
- **5OB1CC.ipynb** - Combined analysis for the 5-minute Order Block strategy
- **15LS1CC.ipynb** - Analysis for the 15-minute Leg Structure strategy

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

## Dependencies

- **jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Charting and visualization
- **pytest**: Testing framework

## Makefile Commands

- `make install`: Install all Poetry dependencies
- `make clean`: Remove Python cache files and Jupyter checkpoints
- `make format`: Format code using Black
- `make lint`: Lint and auto-fix code using Ruff
- `make test`: Run all tests

## Running Tests

Tests are located inside each strategy's `tests/` directory:

```bash
# Run all tests
make test

# Run tests for a specific strategy
poetry run python -m pytest strategies/15LS1CC/tests/ -v
poetry run python -m pytest strategies/5OB1CC/tests/ -v
```

## Commit Convention

Use conventional commits with the following types:
- `feat:` - New feature
- `fix:` - Bug fix
- `chore:` - Other changes
- `docs:` - Documentation only changes
- `style:` - Code style changes (formatting, etc)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Adding or updating tests

There is a hook to check for that automatically:

```bash
ln -s $(realpath .git-hooks-commit-msg) .git/hooks/commit-msg
```
