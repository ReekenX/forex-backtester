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

Every strategy additionally renders to a static HTML page, from the same
template. `strategies/<name>/utils/report.py` holds the page-building logic and
`labs/render.py` (15LS1CC) / `labs/render_5OB.py` (5OB1CC) /
`labs/render_30MCR.py` (30MCR) are thin entry points, mirroring the
notebook/module split. Output goes to `labs/build/` (gitignored). See
"Rendering the HTML Report" below. 30MCR is report-only - it has no notebook.

Current strategies:
- **5OB1CC** - 5-minute Order Block, 1-minute Confirmation Candle (`strategies/5OB1CC/`)
- **15LS1CC** - 15-minute Leg Structure, 1-minute Confirmation Candle (`strategies/15LS1CC/`)
- **30MCR** - 30-minute Continuation/Reversal (`strategies/30MCR/`)
- **30MCR Fixed** - 30MCR traded on a fixed 5-pip stop, scored on a `Reward` column (`strategies/30MCR-fixed/`). Report-only, see "30MCR Fixed" below
- **30MC** - a copy of 30MCR on a new export with real `Pullback` and `TP` (`strategies/30MC/`). Report-only, see "30MC" below

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

Each strategy keeps its data inside its `strategies/<name>/` directory: `data.csv` for 5OB1CC, `v5_data.csv` for 15LS1CC, `trades.csv` for 30MCR. The 15LS1CC file is a 15-minute timeframe export; a one-minute export will land beside it under its own name.

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
- **TP**: Take Profit value in pips. A blank cell and a 0 both mean the trade was not profitable, so `TP > 0` is the Signal rule; `load_data` fills the blanks with 0 so both read the same way
- **Extra**: Extra pips needed to make this trade profitable
- **BOS/CH**: Market structure type (BOS - Break of Structure; CH - Change of Character)
- **30M Leg**: 30-minute timeframe leg analysis
- **Hours Until News**: Time until news event in hours
- **News Event**: Associated news event title

#### 15LS1CC Data Format (strategies/15LS1CC/v5_data.csv)
- **Date**: Trading date (YYYY-MM-DD format)
- **Weekday**: Day of the week
- **Trade**: Trade identifier
- **Direction**: Trade direction (Buy or Sell)
- **SL**: Stop Loss in pips
- **Pullback**: Pullback in pips
- **TP**: Take Profit in pips (empty = not profitable)
- **R**: Risk-reward achieved, exported with an "R" suffix (e.g. `7R`). Negative when the trade was stopped out before reaching that target (e.g. `-6R` means 6R was available but Pullback exceeded SL). The spreadsheet export labels this column with a computed win-rate cell (e.g. `47.3%`) instead of `R`, so `load_data` renames the trailing column.

#### 30MCR Data Format (strategies/30MCR/trades.csv)
- **Date**: Trading date (YYYY-MM-DD format)
- **Weekday**: Day of the week
- **Trade**: Trade identifier. The export wraps a computed figure under the column name in the same header cell (`Trade\n69`), so `load_data` normalises every header to its first line
- **Direction**: Trade direction (Buy or Sell)
- **30M**: The 30-minute level the signal formed at - `High` or `Low`. `load_data` reads it against `Direction` to derive `Type`: `High` + `Buy` and `Low` + `Sell` are `Continuation`, the other two pairings are `Reversal`. `Setup` joins the two (`High Continuation`, `High Reversal`, `Low Continuation`, `Low Reversal`). All are known at entry, so all are tradeable filters
- **BE Rule (10 pips)**: Checkbox exported as `TRUE` / `FALSE`. Feeds the break-even scenario below. It sits left of `TP`, so `load_data` keeps it
- **SL**: Stop Loss in pips
- **Pullback**: Pullback in pips
- **TP**: Take Profit in pips (empty = not profitable)

**Break-even rule.** An optional management rule: once price runs 10 pips in
the trade's favour, the stop moves to entry. The three numbers are learned in
order - `SL` at entry, then `Pullback`, then `TP` - so the stop is checked
first. A trade with `Pullback >= SL` (against the effective stop, buffer
included) is a loss under the rule exactly as without it; the rule only
reads trades that survived the stop, and only when their target sits beyond
10 pips. A target inside 10 pips is hit before the rule can trigger, so the
normal win rule applies there, checkbox or not (2026-08-24 #1, checked with
`SL 1.9`, still wins at 1:1). Past that:

- **Checked (`TRUE`)** - a break-even trade, 0R. Price reached 10 pips, came
  back to entry and took the trade out before the target. Its recorded `TP` is
  where price went afterwards, so it must not score as a win.
- **Unchecked, `TP` above 10 pips** - the trade can no longer lose. It wins
  if `TP` reaches the target for the RRR (`TP >= RRR x SL`), otherwise it is
  break-even, 0R.
- **Unchecked, `TP` of 10 pips or less** - the rule never triggered, so the
  normal win rule applies.

So a stop buffer can change which branch a trade lands in. 2026-09-23 #1 is
checked but recorded `SL 3.0, Pullback 4.9`: a loss on the recorded stop, a
break-even trade with a 2-pip buffer.

With the rule off, the column is ignored and every trade scores under the
normal win rule. The flag is learned after the trade, like `Pullback` and
`TP`, so it describes outcomes and must never be used as an entry filter.

`break_even_outcomes(df, rrr, buffer, stop)` is the one place this rule lives
- the 1:2 and 1:3 Strategies tables read it; it
returns wins, break-evens and losses, each trade in exactly one. A
break-even returns 0R, so a win rate under the rule is wins over wins plus
losses, which keeps 50% the 1:1 breakeven.

The checkbox exports `FALSE` on every empty sheet row, so those rows no longer
read as blank lines. `load_data` drops any row without a `Date`; without that
cut 817 empty rows scored as losses and pulled every win rate down to ~9%.

Everything right of `TP` (currently `All`, `WR`, `+1pip`, `+1pip CAP(6)`
and a run of `#DIV/0!` cells) is the spreadsheet's own scratch work - precomputed win rates and
R totals for stop scenarios the page recomputes itself - and `load_data` drops
it. Those headers are renamed and repeated between exports, so the cut is
anchored on `TP`, never on a scratch name. The header figures are still a
useful cross-check: the sheet's `All 50.6%` and `WR 33.5%` are exactly
what `calculate_setup_statistics`'s Default row reports. The place the sheet
disagrees is its `+1pip 40.9%` against this page's 40.3%: one trade recorded
`SL 4.7, Pullback 5.7` survives a 5.7-pip stop in the sheet and not here, because the project's win rule is
`Pullback < SL`, strictly.

There is no R column, so `load_data` derives it as `TP / SL`, signed negative
when `Pullback >= SL` - the convention the 15LS1CC sheet exports.

## Trading Data CSV Fields

1. SL column is a number in pips that was from entry signal to "safe stop"
2. Pullback column is a number of pips that trade reached after entry and before reaching TP
3. TP column is a number in pips that trade could have reached
4. Empty TP column means that trade was not profitable
5. If Pullback equals SL, it means that trade was an immediate loss
6. If Pullback is higher than SL, it means that overall trade could have been profitable but a higher SL was needed than "safe stop"
7. R column (if not empty) is a number of how many R's this trade achieved (e.g., 10 pips for TP and 3 pips for SL would have achieved 10/3=3 R)
8. Minimum broker SL is 1.1 pips
9. Win condition: a trade must BOTH survive its stop and reach the target - `Pullback < SL AND TP >= RRR x SL`. Checking only the TP leg scores trades that were stopped out before running to target as wins (the data marks those with a negative R) and inflates every win rate. When a stop is adjusted, both halves use the adjusted value: `Pullback < effective SL AND TP >= RRR x effective SL`. The one deliberate exception is the `Signal` column, which asks a different question - see "Signal vs Strategy" below
10. When a trade is entered, only the SL column is known. Pullback and TP are only learned after the trade is finished. Therefore, Pullback and TP columns must not be used for strategy filtering (e.g., "take a trade when Pullback is smaller than SL" does not make sense because Pullback is unknown at entry time)

**Example 1**: SL 3.1 pips, Pullback 2.4 pips and TP 10 pips. When entering a position, safe stop loss was 3.1 pips away from entry. Then price at some point went 2.4 pips against the entry but later recovered and shot 10 pips from entry. Total reward (R) was 10/3.1=3R.

**Example 2**: SL 2.1 pips, Pullback 3.4 pips and TP 10 pips. This trade would be a loss, but only if the safe stop had been higher - it would have been a winner.

## Signal vs Strategy

Two different questions can be asked of the same trade, and they give different
win rates. Keep them apart:

- **Signal** - `TP > 0`. Was the trade idea right? Price reached a target, so
  the direction was correct. The stop is deliberately ignored, which means this
  counts trades whose `Pullback` exceeded `SL`: the idea worked, but the entry
  was too early to survive the move against it. Example 2 above is one of these.
- **Strategy** - `Pullback < SL AND TP >= SL` (at 1:1). Would trading it have
  made money? The trade had to survive its stop *and* reach the target. This is
  the tradeable number.

Strategy is always a subset of Signal, so its win rate is always the lower of
the two. The gap between them is the cost of entry timing - directionally
correct calls that the recorded stop could not hold onto. A wide gap points at
the entry, not at the idea.

Report tables state which rule they use. `Weekday Analysis`, `4H Alignment
Analysis`, `SL Range Analysis`, `Fixed SL Analysis` and `Cap SL Analysis` show
both side by side; the remaining SL tables use Strategy only, named in each
note.

`TP Range Analysis` deliberately keeps neither column: it buckets trades that
already have `TP > 0`, so a Signal column would read 100% on every row. It is a
distribution, not a win rate.

In the stop-scenario tables Signal does not depend on the stop, so it repeats
down every row. That is the point: it is the ceiling no stop size can beat, and
each row's Strategy says how much of it that stop captures.

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
- **15LS1CC HTML report**: `labs/render.py`, `strategies/15LS1CC/utils/report.py`, `strategies/15LS1CC/tests/test_report.py`
- **5OB1CC HTML report**: `labs/render_5OB.py`, `strategies/5OB1CC/utils/report.py`, `strategies/5OB1CC/utils/order_block.py`, `strategies/5OB1CC/tests/test_report.py`, `strategies/5OB1CC/tests/test_order_block.py`
- **30MCR HTML report**: `labs/render_30MCR.py`, `strategies/30MCR/utils/report.py`, `strategies/30MCR/utils/continuation_reversal.py`, `strategies/30MCR/tests/test_report.py`, `strategies/30MCR/tests/test_continuation_reversal.py`

## Acceptance Criteria

Before announcing that any feature or modification is complete and working:

1. **All tests must pass** - Run the corresponding test module(s) and verify all tests succeed
2. **Test execution** - Use `poetry run python -m pytest strategies/<strategy>/tests/ -v` to run tests
3. **No exceptions** - Tests must complete without errors or warnings
4. **Test coverage** - All new functionality must have corresponding tests

Only after all tests pass successfully should you confirm the work is complete. If any tests fail, fix the issues before announcing completion.

Commit changes you made.

### Commit Convention

Use conventional commits for versioning:
- `feat:` - New feature/strategy/experiment
- `fix:` - Bug fix
- `chore:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:` - for other tasks
- Subject line only — no body
- Never use a scope (e.g., write `feat: ...`, not `feat(api): ...`)

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

## Rendering the HTML Report

Three pages, one template, one output directory:

| Strategy | Entry point | Page | Source CSV |
| --- | --- | --- | --- |
| 30MCR | `labs/render_30MCR.py` | `labs/build/30MCR.html` | `trades.csv` |
| 15LS1CC | `labs/render.py` | `labs/build/15C.html` | `v5_data.csv` |
| 5OB1CC | `labs/render_5OB.py` | `labs/build/5OB.html` | `data.csv` |
| 30MCR Fixed | `labs/render_30MCR_fixed.py` | `labs/build/30MCR-fixed.html` | `trades.csv` |
| 30MC | `labs/render_30MC.py` | `labs/build/30MC.html` | `trades.csv` |

Pair any of them with a file watcher for the normal working loop:

```bash
brew install watchexec   # one-off

poetry run python labs/render_30MCR.py                              # or: make report
watchexec -w strategies/30MCR -e py,csv -- poetry run python labs/render_30MCR.py
poetry run python labs/render_30MCR.py out.html --no-reload         # frozen snapshot

poetry run python labs/render.py                                    # or: make report-15c
watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

poetry run python labs/render_5OB.py                                # or: make report-5ob
watchexec -w strategies/5OB1CC -e py,csv -- poetry run python labs/render_5OB.py
```

The bare `make report` / `open` / `watch` targets follow the strategy under
active work - 30MC - and every page keeps an explicit target of its own:
`report-30mcr` / `open-30mcr` / `watch-30mcr`, `report-30mcr-fixed` /
`open-30mcr-fixed` / `watch-30mcr-fixed`, `report-15c` / `open-15c` /
`watch-15c` and `report-5ob` / `open-5ob` / `watch-5ob`. `make serve` serves all three pages out of `labs/build`.

`watchexec` is event-driven (OS filesystem notifications), not a polling timer.

Because all three pages land in `labs/build`, their live-reload sidecars are
named apart - `build-id.txt` / `build-id.js` for 15C, `build-id-5ob.txt` /
`build-id-5ob.js` for 5OB, `build-id-30mcr.txt` / `build-id-30mcr.js` for
30MCR. A shared name would have each page reload on another's data. Their
sessionStorage sort and scroll keys are namespaced the same way.
`test_build_id_sidecars_do_not_collide_with_the_15c_page` (5OB) and
`test_build_id_sidecars_do_not_collide_with_the_other_pages` (30MCR) pin
this.

### How the pages differ

Same shell, same tables, same win rule; the data dictates the rest. The 30MCR
page was built last and became the template, so the charts and the stop-table
shapes below are on all three pages - compare them side by side with
`make serve`:

- **Every Signal/Strategy table up to `Cap SL Analysis` carries a chart
  under it** - `Weekday`, the grouping tables (`4H Alignment` on 15C; `Hour`,
  `EMA Alignment`, `Structure` and `30M Leg` on 5OB; `Setup` on 30MCR),
  `SL Range`, `Fixed SL` and `Cap SL`.
  `create_signal_strategy_chart` draws Signal and Strategy as paired bars per
  row on one 0-100% axis, with a dashed line at the 50% breakeven of 1:1. It
  reads the win rate back out of the table's own cells - a bare `"52.7%"` on
  30MCR, `"12W - 3L (52.7%)"` on 15C and 5OB - so the two cannot disagree; the chart keeps the table's original order when a column is
  click-sorted. Series colours are categorical slots 1 and 2 of the dark
  palette (`#3987e5`, `#d95926`); text stays in the page's greys.
- **`Cap SL Analysis` sits right below `Fixed SL`** on every page: the
  recorded stop kept but capped at 9, 8, 7, 6, 5 and 4 pips,
  `min(SL, cap)`, through `calculate_sl_max_statistics`. Rows are stop sizes
  to compare, so the table stays click-to-sort. 30MCR lists them tightest
  first (4 up to 9), like `Fixed SL`; 15C and 5OB still go 9 down to 4.
- **`Adding Buffer` carries the same chart**, with its table left in the
  `Notation, Win Rate` shape. The chart reads
  `calculate_sl_buffer_statistics(df, with_signal=True)` - the same scenarios
  through the same `_sl_scenario_statistics`, so Signal repeats as the
  ceiling exactly as it does under `Fixed SL`.
- **`Pullback Analysis` carries a 100% stacked bar per row** under its
  table. `create_pullback_chart` reads the `W - M - L` cells back and draws
  tapped winners (`#3987e5`), missed winners (`#d95926`) and losers (recessive
  grey, since L is constant per stop). The `+ 1 pip buffer` rows start a
  second group, as they score against a wider stop. 30MCR drops the table
  and keeps only the chart, cut to two series: the missed trades in orange
  (`#d95926`) from the left of each bar, then the rest in grey (`#4a4a4a`).
  Each bar spans the winning trades - `TP > 0`, whatever the pullback, the
  Signal count - and the grey is labelled with that count. A missed trade
  reached its target, so it is always one of them. `calculate_pullback_statistics` still
  produces the `W - M - L` cells it reads.
- **`TP Range Analysis` keeps its four-band table and adds a pip chart
  under it.** `create_tp_chart` gives every whole TP pip from 8 to 29 its own
  column (TP rounded down) and pools the rest into `30+`, set slightly apart.
  Counts sit on the caps, so there is no y-axis. A TP below 8 gets no column;
  the caption counts those instead of dropping them silently.
- **The 15C notebook mirrors the page's tables**, so `Cap SL` has a
  `display_analysis_sl_max` cell there too. 5OB's notebook predates the report
  and does not; 30MCR has no notebook.

Where the pages still differ:

- **5OB1CC has no `R` column**, so `order_block.load_data` derives it as
  `TP / SL`, signed negative when `Pullback >= SL` - the convention the
  15LS1CC sheet exports. The R Distribution charts read that derived column.
- **5OB1CC has no `4H` column.** Its `EMA Alignment Analysis` is the
  counterpart: `EMA == Direction` is "Aligned", the rest is "Against".
- **5OB1CC carries four extra entry-time filters**, so each gets a
  Signal/Strategy table in the grouping block below Weekday: `Hour`
  (click-to-sort, Lithuanian time), `EMA Alignment`, `Structure` (`BOS/CH`) and
  `30M Leg`. All four are known before the trade is taken, so they are
  legitimate filters - unlike `Pullback` or `TP`.
- **`Range`, `Strength` and `Extra` are read by nothing.** They are populated
  on roughly 25 of 1102 rows, which is too thin to bucket.
- **A blank `Pullback` loads as 0**, same as a blank `TP` - so those trades
  score as having run clean off the entry. Five rows in the current export are
  like that.
- **The Three Setups trade log carries an `Hour` column** that 15C has no data
  for, and runs to one row per trade (1102 today, against 15C's 90).

30MCR, against the same 15C baseline:

- **It derives `R` the same way 5OB does**, `TP / SL` signed negative when
  `Pullback >= SL`, because its export has no `R` column either.
- **Its grouping block is one table, `Setup`**: `Default`, then
  `Continuation` and `Reversal`, which sum to it. The type is known at entry,
  so it is a tradeable filter, and the table is click-to-sort. There is no
  High/Low table and no Direction table - neither split said anything on its
  own.
- **It has no `4H`, `EMA` or `Hour` column**, so none of those tables exist.
- **Its Signal/Strategy tables carry win rates only** - `Weekday`, `Setup`,
  `SL Range` and `Cap SL` are `<label>, Trades, Signal, Strategy, Strategy
  + 1 pip` (`RATE_COLUMNS`), each a bare `"52.7%"`, all at 1:1 on the row's
  own stop. `+ 1 pip` pads that stop by `TYPE_BUFFER_PIPS` (on `Cap SL` it
  is `min(SL, cap) + 1`). `Fixed SL` drops the `+ 1 pip` column: a fixed
  stop plus a pip is the next row down, so its chart draws two bars a row. `_rate_columns` builds all three, so every table reads
  them the same way. Every rate is coloured against 50% - green above, red
  below, plain on it - through `colour_columns`. The charts under them - and
  the `Adding Buffer` chart - draw all three as grouped bars, `Strategy + 1
  pip` in categorical slot 3 (`#199e70`). There is no BE Rule column: these tables
  are 1:1, where almost every target sits inside the 10-pip trigger, so the
  rule barely moves them. `break_even_outcomes` stays for higher-RRR use.
- **A `Streaks` table sits between the R Distribution chart and the
  Strategies tables.** `calculate_streak_statistics` reads the trades in
  date order and reports the longest unbroken run of wins and of losses
  under the family's 1:1 win rule, with the run's first and last trade
  dates. Rows are `Default`, `Continuation` and `Reversal`; the type rows
  walk only their own trades, so the other type's losses do not break them.
  Both type rows repeat as `<type> + 1 pip buffer`, scored on SL + 1.
- **A `Half SL` table sits between `Streaks` and the Strategies tables.**
  `calculate_half_sl_statistics` compares, at every RRR, `Default`, `Half SL`
  (stop at SL / 2 from the signal) and `Limit order on half` (a limit SL / 2
  into the pullback, stop left at the recorded level, so SL / 2 from the
  fill; scored on Pullback - SL / 2 and TP + SL / 2). `Notation` is
  `W - M - L` like `Pullback`, with M = limit never filled; `Win Rate` is
  W / (W + L) and `Outcome` W x RRR - L.
- **It has one R Distribution chart, the cumulative one.** The `R
  Distribution (Exact)` chart 15C and 5OB still carry is gone.
- **It has no Three Setups trade log.** The page ends on the Strategies
  tables, and those test buffers of `+0` and `+1` only. Their `Trend`
  column (`All`, `Continuation`, `Reversal`) filters on the setup's `Type`
  before the stop rule runs, so `Reversal` is High + Low Reversal only.
  Their `Cap SL` column (`0` = no cap, 5, 6) caps the stop after the
  buffer, `min(SL + buffer, cap)`, and replaces the old `Max SL X` strategy
  rows; it is named like the `Cap SL Analysis` table, which caps the same
  way. 15C and 5OB still call their gate column `Max SL`. There is no `RRR` column: each table holds one RRR, named in its
  heading, and `create_html_table(rrr=...)` still colours `Win Rate` against
  that RRR's breakeven.
  The 1:2 and 1:3 tables (`BE_RULE_RRRS`) run every row twice, with and
  without the break-even rule, and say which in a `BE Rule Applied`
  column: a green ticked checkbox when applied, an empty cell otherwise. The
  box is not `disabled` (browsers grey that out); it ignores clicks instead. A BE row scores through `break_even_outcomes` on the row's own
  buffered, capped stop: `Notation` is `W – BE – L` and `Win Rate` is
  W / (W + L). Its equity curve takes 0R per break-even and is labelled
  `... / BE`. 1:1 has no such column - nearly every 1:1 target sits inside
  the 10-pip trigger. The short columns are narrowed through
  `STRATEGIES_COLUMN_WIDTHS` so a BE Notation cell stays on one line. `Fixed SL X` rows run uncapped and unbuffered only. 15C and 5OB
  still carry the older Strategies shape - `Min SL` / `Max SL` gate columns,
  buffers `+0` to `+3`, and `Max SL X` rows - because `Trend` has no
  data-independent meaning there.
- **Each Strategies table carries an equity chart under it.**
  `calculate_equity_curves(df, rrr)` walks every row of that table trade
  by trade in date order, +RRR per win and -1 per loss under the same
  rule, with trades outside the row's trend left flat so the curves line
  up in time. `create_equity_chart` draws them as one SVG: the five best
  final outcomes in categorical slots 1-5 with direct labels and a legend,
  the rest in a recessive grey that comes forward on hover with a tooltip
  naming the row and its R at that trade.
- **Everything right of `TP` is dropped at load** - see
  the data format above, including the places the sheet's own figures and
  this page's disagree.
- **Headers are normalised to their first line** before anything reads them:
  the export wraps a computed figure under the column name in the same cell.

### 30MCR Fixed

`make report-30mcr-fixed` / `open-30mcr-fixed` / `watch-30mcr-fixed`. A copy
of the 30MCR page whose export differs in three ways:

- **`SL` is 5 pips on every trade.** There is no recorded safe stop to vary.
- **`TP` is replaced by `Reward`**: the most the trade could have made, in R,
  after commissions, exported as `3R`. Blank loads as 0. The win rule is
  `Pullback < SL AND Reward >= RRR`; Signal is `Reward > 0`. `R` is `Reward`,
  signed negative when `Pullback >= SL`. Everything right of `Reward` is
  sheet scratch and is dropped.
- **There is no `BE Rule` column.**
- **`Scaling` follows `Reward`**: a second trade opened when the first
  reached 1R, exported as `3R` (how far it ran) or `-1R` (stopped out). A
  blank means no second trade and loads as NaN, not 0. The scratch cut keeps
  it and drops everything to its right.
  A trade with `Pullback` higher than `SL` was stopped out before it
  reached 1R, so its scaling trade never happened - whatever the `Scaling`
  cell says. 2026-10-21 #1 (`Pullback 5.1`, `Scaling -1R`) is one of these.

Because costs are already inside `Reward`, it cannot be turned back into pips
to re-score a trade on another stop. So every stop-scenario table (`SL Range`,
`Fixed SL`, `Cap SL`, `TP Range`, `Pullback`, `Adding Buffer`, `Half SL`, the
`+ 1 pip` readings and the BE rule) is left off. The page is `Weekday`,
`Setup`, `R Distribution`, `Streaks`, `Scaling` and one Strategies table per RRR - rows
are the three trends, with an `Outcome` column and an equity chart. The
sheet's `WR 43.9%` header is one trade above the page's 43.4%: it counts a
trade with `Pullback == SL` as surviving.

`Scaling` is two tables. `calculate_scaling_statistics` gives each trend's
second-trade count, `W - L`, 1:1 win rate, average R, average win and best
run. The averages read winners at the most they reached, so they are a
ceiling. `calculate_scaling_rrr_statistics` scores the second trade on a
fixed target (a run short of it is -1R) beside `Base`, the first trade's
Strategies Outcome, and `Combined`, the two summed.

The 1:2 and 1:3 Strategies tables (`BE_AT_1R_RRRS`) repeat every trend as
`<trend> + BE at 1R`: the stop moves to entry once the trade reaches 1R,
scored by `be_at_1r_results`. A trade stopped out (`Pullback >= SL`) or
short of 1R loses as usual. Past 1R, `Scaling -1R` means price came back to
entry first, 0R. Any other `Scaling` means it never did: a win if `Reward`
reached the target, 0R otherwise. A blank `Scaling` keeps the normal rule.
Notation is `W – BE – L`, Win Rate W / (W + L), and the equity curve takes
0R per break-even. 1:1 has no such rows - its target is the trigger.

The same tables then repeat every trend once more as `<trend> + Half SL at
1R` (`HALF_SL_AT_1R_RRRS`, scored by `half_sl_at_1r_results`): once the
trade reaches 1R the stop is pulled to SL / 2. Past 1R it wins if `Reward`
reached the target and `Pullback < SL / 2`; anything else loses 0.5R.
`Pullback` is the trade's deepest point overall, not only after 1R, so a dip
past SL / 2 before 1R counts against it - the rule reads conservatively.
Notation is `W – HL – L`, Win Rate W / Trades, and `Outcome` can carry a
half (`-24.5R`).

`<trend> + Quarter SL at 1R` rows (`QUARTER_SL_AT_1R_RRRS`,
`quarter_sl_at_1r_results`) follow the same rule with the stop cut by a
quarter, to SL x 0.75: `Pullback < SL x 0.75` to win, -0.75R otherwise past
1R, Notation `W – QL – L`. Both share `_tighter_sl_at_1r_results`.

Every Strategies table, 1:1 included, also repeats each trend as `<trend> +
Scaling` (`SCALING_RRRS`, `scaling_results`): the trade plus its second
trade, each scored on the same RRR - the second wins when `Scaling >= RRR`,
else -1R, so a trade with both legs lost costs 2R. `second_trades` is the one
place that decides a second trade exists: `Scaling` not blank and the first
trade not stopped out (`Pullback >= SL`). The `Scaling` tables read it too,
so their `Combined` equals the `+ Scaling` row's Outcome. Notation and Win
Rate count positions (Trades plus second trades).

`Scaled Trades Only` closes the page, below `Strategies (1:3 RRR)`: the
first trades skipped, only the second trades taken, at every RRR
(`calculate_scaled_only_statistics`). Columns are `Trend, Trades, Notation,
Win Rate, Outcome`; the RRR sits in the Trend label (`All (1:2)`), and
`create_html_table(row_rrrs=...)` colours each Win Rate against its own
RRR. Its figures equal the `Scaling` at-each-RRR table's. The equity chart
under it (`calculate_scaled_only_equity_curves`) walks all trades, flat
where no second trade opened; `create_equity_chart(rrr=None, caption=...)`
handles a chart that mixes RRRs.

### 30MC

A copy of the 30MCR page and module (`strategies/30MC/`), fed from a new
export. It differs from 30MCR in the data only:

- **`SL` is 5 pips on every trade.** It is a placeholder for the real safe
  stop; until that lands, 5 pips is read as the correct stop.
- **`Pullback` and `TP` are real.** A loser exports both blank, so
  `load_data` reads that `Pullback` as the stop (`Pullback = SL`), not as a
  clean 0-pip run - otherwise `Half SL`'s limit would score losers as never
  filled. A blank `Pullback` beside a real `TP` still loads as 0.
- **There is no `BE Rule (10 pips)` column**, so the break-even rule is
  gone: no `break_even_outcomes`, no `BE Rule Applied` column or `/ BE`
  equity curves. The 1:2 and 1:3 Strategies tables run each row once.
- **It has no `Setup Analysis`.** Every trade is a Continuation, so the
  table only repeated Default.
- **It has no `SL Range Analysis`.** Every stop is 5 pips, so every trade sat
  in one band. Weekday leads straight into `Fixed SL`.
- **It has no `Cap SL Analysis`.** With every stop at 5 pips a cap only
  repeats `Fixed SL`. The Strategies tables keep their `Cap SL` column.
- **It has no `Pullback Analysis`.** `calculate_pullback_statistics` and
  `create_pullback_chart` are gone; `Half SL` keeps its own limit scoring.
- **It has no `Adding Buffer`.** `calculate_sl_buffer_statistics` is gone;
  the `+ 1 pip` columns and the Strategies `Buffer` column still pad the stop.
- **`Streaks` lists `Default` and `Continuation` only** - no `Reversal`
  row and no `+ 1 pip buffer` rows.
- **`Scaling` follows `TP`** and is empty so far. The scratch cut drops it
  with the sheet's `Signal` / `WR` columns, which the Default row matches
  (68.5% / 49.3%).

Sidecars are `build-id-30mc.txt` / `build-id-30mc.js`, storage keys
`30mc-scroll` / `30mc-sorts`.

### Conventions for report sections

Sections are declared in `report.py`'s `SECTIONS` list as
`(anchor, nav_label, heading, note, builder)`. When adding or changing one:

- **A section anchor and a table's `sort_id` must differ.** Both become DOM ids;
  a collision makes `getElementById(tableId)` in the sort script return the
  `<section>` and click-to-sort throws. Convention: anchor `foo`, table
  `foo-table`. `test_no_duplicate_dom_ids` guards this.
- **Headings and nav labels are escaped once** by `html.escape`, so write a
  literal `<`, not `&lt;`. The `note` is inserted raw, so that one does use
  entities. `test_headings_are_not_double_escaped` guards this.
- **Verify rendered output in a browser**, not just the HTML string - column
  widths, overflow and click-to-sort only fail at render time. Headless Chrome
  over CDP works; a self-reloading page never settles, so screenshot the
  `--no-reload` build.

### Conventions for stop tables

The SL tables (`SL Range`, `Adding Buffer`, `Fixed SL`, `Cap SL`, `Pullback`) are
a family and must stay consistent. All three pages have this family, each in
its own module - keep a change to one in step with the others:

- **One shared win rule.** Every stop scenario goes through
  `_sl_scenario_statistics`, which takes `(label, effective SL)` pairs. Do not
  reimplement the win condition in a new table - add a scenario builder instead.
  Two tables silently drifting apart on this rule is a bug that has already
  happened once.
- **First row is labelled `Default`** and means "stops as recorded". Every
  table's Default row must agree;
  `test_every_stop_table_opens_with_the_same_default_row` pins that.
- **Columns are `<label>, Trades, Notation, Win Rate`**, with `Notation` as
  `"12W - 3L"` and `Win Rate` as `"52.7%"` in separate columns - never combined
  into one cell. `SL Range Analysis`, `Fixed SL Analysis` and
  `Cap SL Analysis` are the exceptions: they carry `<label>, Trades, Signal, Strategy` like the weekday
  and alignment tables, each cell combined as `"12W - 3L (52.7%)"` by
  `_format_wl` - on 15C and 5OB. 30MCR's carry the three bare rates of
  `RATE_COLUMNS` instead (see "How the pages differ").
  Their `Strategy` column is still the family's shared win rule, which is what
  `test_every_stop_table_opens_with_the_same_default_row` compares against.
  `_sl_scenario_statistics` emits that shape via `with_signal=True`, so the win
  rule stays in one place - do not compute it in the caller.
  `Pullback Analysis` is a third shape: `Pullback, Trades`, with
  `Trades` as `"25W - 12M - 11L"` - winners whose pullback tapped the limit,
  winners it missed, and losers, which lose whatever the pullback and so read
  the same on every row that shares a stop. Each row covers every trade once.
  Every level but Default repeats as `<level> + 1 pip buffer`: the stop is
  padded as in `Adding Buffer`, the limit (including Half's recorded-SL / 2)
  stays put.
- **Row order carries meaning**, so these tables are rendered with
  `sortable=False`. `Fixed SL Analysis` and `Cap SL Analysis` are the
  exceptions - their rows are stop sizes to compare, so they stay
  click-to-sort.
- **Label column width is `40%` everywhere except `Adding Buffer`** on 15C
  and 5OB, which still carries the older `50%`. `Weekday`, the alignment
  tables, `SL Range`, `Fixed SL`, `Cap SL`, `TP Range` and `Pullback` all pin
  `first_col_width="40%"` so they line up down the page. 30MCR pins every
  table - Strategies included - to `FIRST_COL_WIDTH` (`30%`) in `report.py`;
  `test_every_table_pins_its_first_column_to_30_percent` guards it.
- **Table ids are not shared between the pages.** 15C's SL Range table is
  `sl-range-stats` and its Pullback table `pullback-analysis`; 5OB's and
  30MCR's are `sl-range-table` and `pullback-table` (30MCR renders no
  Pullback table, only its chart). Each page's ids only have
  to be unique within that page, so do not assume a test on one applies to the
  other.

## Previewing Lab Data

To view analysis results from a notebook without opening Jupyter, run the underlying Python functions directly:

```bash
# 15LS1CC strategy
poetry run python -c "
import sys; sys.path.insert(0, 'strategies/15LS1CC')
import pandas as pd
from utils.confirmation_candle import load_data, calculate_buffer_statistics

df = load_data('strategies/15LS1CC/v5_data.csv')
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

For 15LS1CC, `poetry run python labs/render.py` renders every table at once and
is usually faster than calling functions one by one. For 5OB1CC's report tables,
`labs/render_5OB.py` does the same - and `strategies/5OB1CC/utils/order_block.py`
is the module behind them:

```bash
# 5OB1CC report tables
poetry run python -c "
import sys; sys.path.insert(0, 'strategies/5OB1CC')
import pandas as pd
from utils.order_block import load_data, calculate_hour_statistics

df = load_data('strategies/5OB1CC/data.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
print(calculate_hour_statistics(df).to_string(index=False))
"
```

The 5OB1CC notebook's other modules (`hours.py`, `ema.py`, `tables.py`, ...)
predate the report and are untouched by it.

30MCR has no notebook at all, so `labs/render_30MCR.py` is the whole loop -
and `strategies/30MCR/utils/continuation_reversal.py` is the module behind it:

```bash
# 30MCR report tables
poetry run python -c "
import sys; sys.path.insert(0, 'strategies/30MCR')
import pandas as pd
from utils.continuation_reversal import load_data, calculate_setup_statistics

df = load_data('strategies/30MCR/trades.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)
print(calculate_setup_statistics(df).to_string(index=False))
"
```

## Acceptance Criteria for Lab Changes

On top of the test rules above, a change to 15LS1CC analysis is not done until:

1. `poetry run python labs/render.py` succeeds and the affected table is correct
2. The notebook re-executes clean:
   `poetry run jupyter nbconvert --to notebook --execute --inplace labs/15LS1CC.ipynb`
   (check no cell has an `error` output)
3. Notebook cell comments citing figures are re-checked - they go stale silently
   when the data or the win rule changes
4. `git diff --stat strategies/15LS1CC/v5_data.csv` is empty, in case a test or a
   manual check wrote to it

A change to the 5OB1CC report is not done until:

1. `poetry run python labs/render_5OB.py` succeeds and the affected table is
   correct
2. `poetry run python -m pytest strategies/5OB1CC/tests/ -v` passes - that suite
   covers the older notebook modules too, so a shared-CSV change shows up there
3. `git diff --stat strategies/5OB1CC/data.csv` is empty
4. The page was looked at in a browser, not just asserted on as a string

A change to the 30MCR report is not done until:

1. `poetry run python labs/render_30MCR.py` succeeds and the affected table is
   correct
2. `poetry run python -m pytest strategies/30MCR/tests/ -v` passes
3. `git diff --stat strategies/30MCR/trades.csv` is empty
4. The page was looked at in a browser, not just asserted on as a string

The 5OB1CC notebook (`labs/5OB1CC.ipynb`) does not import the report modules, so
a report-only change does not need it re-executed. 30MCR has no notebook.
