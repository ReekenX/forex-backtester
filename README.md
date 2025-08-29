# Forex Backtester

A Jupyter-based project for analyzing and backtesting forex trading data, starting with EUR/USD currency pairs.

## Project Structure

```
forex-backtester/
├── data/                      # Directory containing forex data files
│   └── eurusd.csv            # Sample EUR/USD exchange rate data
├── forex_analysis.ipynb      # Main Jupyter notebook for data analysis
├── pyproject.toml            # Poetry configuration and dependencies
├── Makefile                  # Build automation commands
└── README.md                 # This file
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

### Running the Jupyter Notebook

The easiest way to start the notebook:
```bash
make run
```

Alternatively, you can use Poetry directly:
```bash
poetry run jupyter notebook forex_analysis.ipynb
```

Or launch JupyterLab for a more feature-rich environment:
```bash
poetry run jupyter lab
```

### Data Format

The CSV data files in the `data/` directory follow this structure:
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

The current notebook provides:
- CSV data loading using pandas
- Data table display
- Summary statistics
- Data type information

## Dependencies

- **jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

## Makefile Commands

- `make run`: Launch Jupyter notebook with the main analysis file
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