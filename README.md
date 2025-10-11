# Forex Backtester

A Jupyter-based project for analyzing and backtesting forex trading data.

## Strategy Background

Trading data is taken from 5 minute timeframe and only EURUSD currency.

Trades are taken during London session but hours tracked in Lithuanian timezone.

## Trader Background

Trader risking 0.5% per trade.

Not taking trades right before red (high importance) news.

## Project Structure

```
forex-backtester/
├── eurusd.csv               # EUR/USD real trading data
├── main.ipynb               # Strategy analysis notebook (table)
├── charts.ipynb             # Strategy analysis notebook (charts)
├── utils.py                 # Helper functions strategy analysis notebook
├── pyproject.toml           # Poetry configuration and dependencies
└── Makefile                 # Build automation commands
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

Launch Jupyter Notebook to work with the current analysis:
```bash
make run
```

This will open the `main.ipynb` notebook directly. Alternatively, use Poetry:
```bash
poetry run jupyter notebook main.ipynb
```

### Data Format

The project uses enhanced trading data format in `eurusd.csv`:

#### Data Format (eurusd.csv)
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
- **Extra**: Extra pips that were needed to make this trade profitable. Example: 3.0 pips SL, 3 pips Pullback and TP of 10 pips still means lost trade. But if Extra column has value like 0.1, it means that SL of 3.1 (3.0 + 0.1) would had been profitable and would had reached 2R (10 / 3.1)
- **BOS/CH**: Market structure type (BOS - Break of Structure; CH - Change of Character)
- **30M Leg**: 30-minute timeframe leg analysis ("Above H" and "Above L" – buy trend, "Below H" and "Below L" – sell trend)
- **Hours Until News**: Time until news event in hours
- **News Event**: Associated news event title (eg. PMI)

#### Rules For Backtesting (from eurusd.csv)

- **Trade**: Do not use this field for backtesting
- **Pullback**: Pullback value can't be filtered because it is unknown until trade has been executed. Pullback value means that once trade signal is received, SL value is recorded, but how big the pullback will be - is unknown. If Pullback matches SL value - it means that trade was instant loss. If Pullback value is less than SL value, it means that at some point in time, before reaching TP - this entry got "discount".
- **TP**: Take Profit value is unknown, do not filter strategy on that
- **News Event**: Associated news event titles list could be long, so just filter for blank value (means that there are no event near by) and any value (means that there is a news incoming) 

## Dependencies

- **jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

## Makefile Commands

- `make run`: Launch Jupyter Notebook
- `make install`: Install all Poetry dependencies
- `make clean`: Remove Python cache files and Jupyter checkpoints
- `make format`: Format code using Black
- `make lint`: Lint and auto-fix code using Ruff

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

Examples of good commit messages:

- fix: commit msg hook not called
- feat: add trading data with 100 trades
- refactor: cleanup codebase and add code comments
- fix: nan values in the CSV causes some formulas to crash