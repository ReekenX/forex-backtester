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
│       ├── v5_data.csv          # Trading data (15-minute timeframe)
│       ├── utils/               # Analysis modules
│       │   ├── __init__.py
│       │   ├── confirmation_candle.py
│       │   └── report.py        # Static HTML report builder
│       └── tests/
│           ├── test_confirmation_candle.py
│           └── test_report.py
├── labs/                        # Notebooks and report renderers
│   ├── 5OB1CC.ipynb             # Combined analysis for 5OB1CC strategy
│   ├── 15LS1CC.ipynb            # Analysis for 15LS1CC strategy
│   ├── render.py                # Renders 15LS1CC to a static HTML page
│   └── build/                   # Rendered output, 15C.html (gitignored)
├── pyproject.toml               # Poetry configuration and dependencies
└── Makefile                     # Build automation commands
```

## Requirements

- Python 3.11 or higher
- Poetry (for dependency management)
- [watchexec](https://github.com/watchexec/watchexec) (optional) - only needed
  for the live-reloading HTML report:
  ```bash
  brew install watchexec
  ```

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

### Live HTML report (15LS1CC)

The 15LS1CC lab also renders to a single self-contained HTML page, so you can
read the analysis in a browser without starting a Jupyter kernel. Every table
recomputes from `v5_data.csv` in about two seconds, which is why there is no
notebook state to keep warm.

Render once:

```bash
make report                 # or: poetry run python labs/render.py
open labs/build/15C.html
```

Re-render on every save (this is the normal working loop - edit `v5_data.csv` or
`confirmation_candle.py`, hit save, and the page updates itself):

```bash
make watch                  # wraps:
# watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py
```

`watchexec` is event-driven, not a timer: it asks the OS to notify it when a
watched file changes, and idles at ~0% CPU otherwise.

**The page reloads itself only when the data actually changes** - never on a
timer - and it restores your scroll position and any column sort afterwards, so
a rebuild does not throw away what you were looking at. Change detection uses
`fetch()` on `build-id.txt` when the page is served over HTTP; opened as a
`file://` URL that fetch is blocked by the browser, so it falls back to loading
`build-id.js` through a `<script>` tag, which `file://` does allow. If neither
works the page says so in its status line instead of reloading blindly.

Serving the folder gives change-triggered reloads on both paths:

```bash
make serve                  # or: python3 -m http.server -d labs/build 8000
# then open http://localhost:8000/15C.html
```

If the CSV is half-written when the watcher fires, the render fails gracefully:
the page shows a red **Build failed** panel with the traceback, the watcher
stays alive, and the page recovers on its own once the CSV parses again.

For a frozen snapshot to share, print or screenshot (a self-reloading page never
settles for a headless browser):

```bash
poetry run python labs/render.py out.html --no-reload
```

## Dependencies

- **jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Charting and visualization
- **pytest**: Testing framework

External tools:

- **watchexec** (optional): re-runs `labs/render.py` when a watched file changes

## Makefile Commands

- `make install`: Install all Poetry dependencies
- `make clean`: Remove Python cache files and Jupyter checkpoints
- `make format`: Format code using Black
- `make lint`: Lint and auto-fix code using Ruff
- `make test`: Run all tests
- `make report`: Render the 15LS1CC HTML report once into `labs/build/`
- `make watch`: Re-render the report on every save (requires watchexec)
- `make serve`: Serve `labs/build/` on http://localhost:8000

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
