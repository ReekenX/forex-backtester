# Autoresearch Results Log

Goal: Find strategies with 10%+ edge on 1:1 or 1:2 RRR (min 15 trades)

## Baseline (225 trades, 44 days)
- All Trades 1:1: WR=29.8%, Edge=-20.2%, Outcome=-91R
- All Trades 1:2: WR=27.6%, Edge=-5.8%, Outcome=-39R

## Iteration #0: Systematic Scan (no buffer)

### Top Strategies at 1:2 RRR (by edge x trades score)

| Strategy | Trades | W/L | WR | Edge | Outcome |
|---|---|---|---|---|---|
| EMA200 Aligned + Engulf=YesSim + SL 2-4 | 16 | 11W-5L | 68.8% | **35.4%** | 17R |
| Engulf=Yes + SL 2.0-4.0 | 33 | 17W-16L | 51.5% | **18.2%** | 18R |
| EMA200 Aligned + SL 2.0-4.0 | 35 | 17W-18L | 48.6% | **15.2%** | 16R |
| Engulf=YesSim + SL 2.0-4.0 | 37 | 18W-19L | 48.6% | **15.3%** | 17R |
| Sell + SL 2.0-4.0 | 33 | 16W-17L | 48.5% | **15.2%** | 15R |

### Top Strategies at 1:1 RRR

| Strategy | Trades | W/L | WR | Edge | Outcome |
|---|---|---|---|---|---|
| EMA200 Aligned + Engulf=YesSim + SL 2-4 | 16 | 11W-5L | 68.8% | **18.8%** | 6R |
| Engulf=Yes + SL 2-4 +3pip buffer | 33 | 21W-12L | 63.6% | **13.6%** | 9R |
| Engulf=Yes + SL 2-4 +2pip buffer | 33 | 20W-13L | 60.6% | **10.6%** | 7R |

### Key Pattern: SL 2.0-4.0 pips is the strongest SL filter

### With Buffer (extra pips on SL)

| Strategy | Trades | W/L | WR | Edge | Outcome |
|---|---|---|---|---|---|
| EMA200 Aligned + SL 2-4 +2pip | 35 | 20W-15L | 57.1% | **23.8%** | 25R |
| EMA200+EngulfYesSim+SL2-4 +2pip | 16 | 12W-4L | 75.0% | **41.7%** | 20R |
