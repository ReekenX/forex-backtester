# Forex Backtester

A Jupyter-based project for analyzing and backtesting forex trading data, starting with EUR/USD currency pairs.

## Project Structure

```
forex-backtester/
├── old/                     # Legacy backtesting implementation with my 900+ trades
│   ├── eurusd.csv           # Sample EUR/USD exchange rate data
│   └── lab.ipynb            # Previous analysis notebook
├── new/                     # Enhanced backtesting with the more detailed trading data
│   ├── eurusd.csv           # EUR/USD data with extended analysis columns
│   └── lab.ipynb            # Current analysis notebook
├── pyproject.toml           # Poetry configuration and dependencies
├── Makefile                 # Build automation commands
└── README.md                # This file
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

Launch JupyterLab to work with notebooks:
```bash
make run
```

Or use Poetry directly:
```bash
poetry run jupyter lab
```

### Data Format

The project uses two data formats for trading analysis:

#### Legacy Format (old/eurusd.csv)
- **Date**: Trading date (YYYY-MM-DD format)
- **Trade**: Trade identifier (e.g., #1, #2)
- **Direction**: Trade direction (Buy/Sell)
- **EMA**: EMA signal (Buy/Sell)
- **SL**: Stop Loss value
- **Pullback**: Pullback value
- **TP**: Take Profit value
- **BOS/CH**: Market structure type (BOS - Break of Structure / CH - Change of Character)

#### Enhanced Format (new/eurusd.csv)
Includes all legacy fields plus additional analysis columns:
- **Weekday**: Day of the week (eg. Monday)
- **Hour**: Trading hour in Lithuanian timezone (valeus from 10 to 18 to cover entire London session)
- **30M Leg**: 30-minute timeframe leg analysis ("Above H" and "Above L" – buy trend, "Below H" and "Below L" – sell trend)
- **HH Until News**: Time until news event in hours
- **News Event**: Associated news event title (eg. PMI)

#### Rules For Backtesting

- **Trade**: Trade identifier is useless, do not use it for any backtesting
- **Pullback**: Pullback value can't be filtered because it is unknown until trade has been executed. Pullback value means that once trade signal is received, SL value is recorded, but how big the pullback will be - is unknown. If Pullback matches SL value - it means that trade was instant loss. If Pullback value is less than SL value, it means that at some point in time, before reaching TP - this entry got "discount".
- **TP**: Take Profit value is unknown, do not filter strategy on that
- **News Event**: Associated news event titles list could be long, so just filter for blank value (means that there are no event near by) and any value (means that there is a news incoming) 

## Dependencies

- **jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

## Makefile Commands

- `make run`: Launch JupyterLab environment
- `make install`: Install all Poetry dependencies
- `make clean`: Remove Python cache files and Jupyter checkpoints

### Commit Convention

Use conventional commits for automatic versioning:
- `feat:` - New feature (bumps minor version)
- `fix:` - Bug fix (bumps patch version)
- `chore:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:` - Other changes

There is a hook to check for that automatically:

```bash
cp .git-pre-commit-hook .git/hooks/commit-msg
```