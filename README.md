# Forex Backtester

A Jupyter-based project for analyzing and backtesting forex trading data.

Trading data is 5 minute timeframe, only EURUSD currency traded only during London session.

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
- **Extra**: Extra pips that was needed to make this trade profitable. Example: 3.0 pips SL, 3 pips Pullback and TP of 10 pips still means lost trade. But if Extra column has value like 0.1, it means that SL of 3.1 (3.0 + 0.1) would had been profitable and would had reached 2R (10 / 3.1)
- **Hour**: Trading hour in Lithuanian timezone (values from 10 to 18 to cover entire London session)
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

## AI Overview / Opinion

### What a 7% Edge with 1:3 RRR Means

With a 1:3 risk-reward ratio, you need a 25% win rate to break even. A 7% edge means your win rate is ~32% (25% + 7%), which theoretically gives you:
- **Expected return per trade**: 0.32 × 3R - 0.68 × 1R = 0.96R - 0.68R = **+0.28R per trade**
- Over 100 trades, you'd expect to make ~28R profit

This is mathematically significant, but the gap between backtest and live performance suggests execution issues.

### Why Live Results Differ from Backtests - Critical Factors

#### 1. Psychological Execution Degradation
- **Entry hesitation**: Missing optimal entries due to fear, waiting for "confirmation" that wasn't in backtest
- **Premature exits**: Taking partial profits or closing at breakeven when seeing profit
- **Revenge trading**: Taking non-setup trades after losses
- **Size variation**: Reducing size after losses, increasing after wins

#### 2. Spread & Slippage Impact
Your 7% edge = 0.28R per trade. With EUR/USD:
- 1-2 pip spread costs can eat 10-20% of your SL
- On a 10-pip SL with 1.5 pip spread, you're already down 15% per trade
- This alone could reduce your edge from 7% to 4-5%

#### 3. Sample Size & Variance
- With 32% win rate, you can have long losing streaks (10-15 losses in a row is normal)
- Need 200-300+ trades to see edge materialize statistically
- Short-term results will fluctuate wildly

### Actionable Improvements

#### 1. Tighten Your Best Setup
From the data, "30M Trend + BOS + SL < 10" shows the highest edge. Focus ONLY on this setup initially.

#### 2. Execution Checklist (No discretion allowed)
```
[ ] 30M trend confirmed
[ ] BOS structure break present
[ ] SL < 10 pips
[ ] No news in next 2 hours
[ ] Entry at exact trigger point (no hesitation)
[ ] Full position size (no variation)
[ ] Exit ONLY at TP or SL (no management)
```

#### 3. Track Execution Quality
Create two columns in your journal:
- **Setup Grade** (A/B/C based on how perfectly it matched rules)
- **Execution Grade** (A = followed plan exactly, B = minor deviation, C = major deviation)

Only A/A trades should match backtest results.

#### 4. Reduce Spread Impact
- Trade during London/NY overlap (tightest spreads)
- Use limit orders for entries when possible
- Consider ECN broker with commission vs spread

#### 5. Statistical Confidence
- Trade minimum position size until you have 100 live trades matching backtest win rate
- Expect 15-20 trade drawdowns even with positive edge
- Your 7% edge needs ~200 trades to be statistically significant

#### 6. The "Boring" Fix
Most traders fail because they can't execute mechanically. Consider:
- Setting alerts and walking away until triggered
- Using stop/limit orders to remove decision-making
- Trading smaller size to reduce emotional impact

### Should You Improve the 7% Edge?

**Focus on execution, NOT edge improvement.** Here's why:

#### Your 7% Edge is Already Professional-Level
With 1:3 RRR, a 7% edge gives you **0.28R per trade**. This means:
- **100 trades/month**: 28R monthly return
- **Compounded annually**: Can double account multiple times
- Most hedge funds would kill for this edge

#### The Math Reality Check
Even if you optimized to 10% edge (very difficult):
- Current: 32% win rate → 0.28R per trade
- Optimized: 35% win rate → 0.40R per trade
- Improvement: +0.12R per trade

But you're currently capturing ~0R (breakeven) instead of 0.28R. **The 0.28R gap from poor execution is 2.3x bigger than any realistic optimization.**

#### Why Strategy Tinkering is a Trap
1. **Over-optimization reduces frequency**: Adding filters for 10% edge might cut trades by 50%, giving you same or worse total return
2. **Complexity kills execution**: More rules = more decisions = more mistakes
3. **Curve fitting illusion**: That "better" edge might not survive forward testing
4. **Psychological escape**: Tweaking strategy feels productive but avoids the real problem (discipline)

#### The Professional Approach
1. **Trade current strategy for 200 trades with A-grade execution**
2. **Only after proving you can capture the 7% edge consistently**, consider optimizations
3. **Remember**: Billion-dollar funds operate on 2-3% edges executed perfectly

Your edge is already in the top 20% of viable strategies. Master executing it before seeking perfection.

**Bottom line**: 7% edge executed well > 15% edge executed poorly. Your edge exists, but it's fragile. Perfect execution is the difference between profit and breakeven.

### Additional Setup Data to Track for Edge Improvement

Looking at the current data fields, these additional setup characteristics could reveal hidden edge improvements:

#### 1. Multiple Timeframe Structure
- **1H Trend Direction** - Aligns with higher timeframe bias
- **4H Structure** - BOS/CH on 4H (major structure shifts)
- **Daily Candle Context** - Inside day / Engulfing / Doji / Trend day
- **Weekly Level Distance** - Pips to nearest weekly S/R

#### 2. Momentum & Strength Metrics
- **Move Strength to BOS** - How many pips was the move that created the BOS?
- **Pullback Depth %** - What % did price retrace before entry signal?
- **Candle Close Strength** - How far into the range did signal candle close? (0-100%)
- **Consecutive Candles** - How many same-direction candles before signal?

#### 3. Price Action Quality
- **Wick Ratio** - Signal candle wick vs body ratio
- **Pre-BOS Consolidation** - How many candles consolidated before breakout?
- **Clean vs Messy** - Was the BOS clean or through multiple wicks?
- **Touches Before Break** - How many times was level tested before BOS?

#### 4. Market Phase
- **Range-bound vs Trending** - Last 50 candles ATR/Range ratio
- **Time Since Last Major High/Low** - Hours since significant pivot
- **Friday/Monday** - These days often behave differently
- **Month-End/Start** - First/last 3 days of month (flows affect price)

#### 5. Relative Extremes
- **RSI (14) on 30M** - Overbought/oversold context
- **Distance from 50 EMA** - In pips (mean reversion potential)
- **Day's Range Position** - Entry in bottom/middle/top third of day's range
- **20-Day Range %** - Where is current price in 20-day range?

#### Most Likely to Improve Your Edge

Based on the current 7% edge with "30M Trend + BOS + SL < 10", prioritize:

1. **1H Trend Direction** - Could filter out counter-trend trades
2. **Move Strength to BOS** - Stronger moves might have better follow-through
3. **Pullback Depth %** - Sweet spot between too deep (momentum lost) and too shallow (no discount)
4. **Time Since Last Major High/Low** - Fresh breaks vs late entries

Start logging these 4 fields. After 100+ trades, patterns might emerge like:
- "When 1H trend aligns, edge jumps to 12%"
- "BOS moves > 15 pips have 40% win rate vs 32%"
- "30-50% pullbacks perform best, 70%+ pullbacks are losers"

This data-driven approach will reveal which market conditions your setup thrives in.