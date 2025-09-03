# Forex Backtester

A Jupyter-based project for analyzing and backtesting forex trading data, starting with EUR/USD currency pairs.

## Project Structure

```
forex-backtester/
├── old/                     # Legacy backtesting implementation
│   ├── eurusd.csv           # Sample EUR/USD exchange rate data
│   └── lab.ipynb            # Previous analysis notebook
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

The CSV data files follow this structure:
- **Date**: Trading date (YYYY-MM-DD format)
- **Open**: Opening price
- **High**: Highest price during the period
- **Low**: Lowest price during the period
- **Close**: Closing price
- **Volume**: Trading volume

Example:
```csv
Date,Open,High,Low,Close,Volume
2024-01-02,1.1034,1.1089,1.1028,1.1085,45230
```

## Features

The project is being restructured to provide enhanced backtesting capabilities. The legacy implementation in the `old/` folder includes:
- CSV data loading using pandas
- Data table display
- Summary statistics
- Data type information

## Dependencies

- **jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

## Makefile Commands

- `make run`: Launch JupyterLab environment
- `make install`: Install all Poetry dependencies
- `make clean`: Remove Python cache files and Jupyter checkpoints

## Future Enhancements

This project can be extended with:
- Technical indicators calculation
- Trading strategy backtesting
- Performance metrics and reporting
- Data visualization with matplotlib/plotly
- Multiple currency pair support
- Real-time data integration