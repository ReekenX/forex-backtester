"""
Autoresearch: Systematic edge finder for 1M confirmation candle data.

Scans all single, double, and triple filter combinations to find strategies
with 10%+ edge on 1:1 or 1:2 RRR.

Only uses information available at entry time (NOT Pullback or TP).
"""

import pandas as pd
import itertools
from typing import Dict, List, Tuple, Callable

MIN_TRADES = 15
MIN_EDGE = 10.0
RRR_RATIOS = [1, 2]


def load_data(filepath: str = "data/eurusd_2026_1m_confirmation_candle.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    for col in ["SL", "TP", "Pullback"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # Extract trade number as integer
    df["TradeNum"] = df["Trade"].str.replace("#", "").astype(int)
    return df


def breakeven_rate(rrr: int) -> float:
    return 100.0 / (1 + rrr)


def calc_edge(trades: pd.DataFrame, rrr: int) -> Dict:
    total = len(trades)
    if total == 0:
        return {"trades": 0, "wins": 0, "win_rate": 0, "edge": -breakeven_rate(rrr), "outcome": 0}

    wins_df = trades[
        (trades["Pullback"] < trades["SL"]) &
        (trades["TP"] >= rrr * trades["SL"])
    ]
    wins = len(wins_df)
    losses = total - wins
    win_rate = (wins / total) * 100
    edge = win_rate - breakeven_rate(rrr)
    outcome = (wins * rrr) - losses
    days_with_wins = wins_df["Date"].nunique() if len(wins_df) > 0 else 0
    total_days = trades["Date"].nunique()

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "edge": edge,
        "outcome": outcome,
        "days_with_wins": days_with_wins,
        "total_days": total_days,
    }


def get_atomic_filters() -> List[Tuple[str, Callable]]:
    """Return all individual filters that can be applied at entry time."""
    filters = []

    # Direction
    filters.append(("Buy", lambda df: df[df["Direction"] == "Buy"]))
    filters.append(("Sell", lambda df: df[df["Direction"] == "Sell"]))

    # EMA(50)
    filters.append(("EMA50=Aligned", lambda df: df[df["Direction"] == df["EMA(50)"]]))
    filters.append(("EMA50=Against", lambda df: df[df["Direction"] != df["EMA(50)"]]))

    # EMA(200)
    filters.append(("EMA200=Aligned", lambda df: df[df["Direction"] == df["EMA(200)"]]))
    filters.append(("EMA200=Against", lambda df: df[df["Direction"] != df["EMA(200)"]]))

    # EMAs agree/disagree
    filters.append(("EMAs=Agree", lambda df: df[df["EMA(50)"] == df["EMA(200)"]]))
    filters.append(("EMAs=Disagree", lambda df: df[df["EMA(50)"] != df["EMA(200)"]]))

    # Engulfing
    filters.append(("Engulf=Yes", lambda df: df[df["Engulfing"] == "Yes"]))
    filters.append(("Engulf=No", lambda df: df[df["Engulfing"] == "No"]))
    filters.append(("Engulf=YesSim", lambda df: df[df["Engulfing"].isin(["Yes", "Similar"])]))
    filters.append(("Engulf=NoSim", lambda df: df[df["Engulfing"].isin(["No", "Similar"])]))

    # SL ranges (known at entry)
    for threshold in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        t = threshold
        filters.append((f"SL<{t}", lambda df, t=t: df[df["SL"] < t]))
        filters.append((f"SL>={t}", lambda df, t=t: df[df["SL"] >= t]))

    # SL bands
    for lo, hi in [(1.0, 2.0), (1.0, 3.0), (1.5, 3.0), (1.5, 4.0), (2.0, 4.0), (2.0, 5.0), (3.0, 6.0), (3.0, 8.0), (5.0, 10.0)]:
        filters.append((f"SL={lo}-{hi}", lambda df, lo=lo, hi=hi: df[(df["SL"] >= lo) & (df["SL"] <= hi)]))

    # Weekday
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        filters.append((f"Day={day}", lambda df, d=day: df[df["Weekday"] == d]))

    # Weekday groups
    filters.append(("Day=Mon-Wed", lambda df: df[df["Weekday"].isin(["Monday", "Tuesday", "Wednesday"])]))
    filters.append(("Day=Tue-Thu", lambda df: df[df["Weekday"].isin(["Tuesday", "Wednesday", "Thursday"])]))
    filters.append(("Day=Wed-Fri", lambda df: df[df["Weekday"].isin(["Wednesday", "Thursday", "Friday"])]))
    filters.append(("Day=Mon-Tue", lambda df: df[df["Weekday"].isin(["Monday", "Tuesday"])]))
    filters.append(("Day=Thu-Fri", lambda df: df[df["Weekday"].isin(["Thursday", "Friday"])]))
    filters.append(("Day=NotMon", lambda df: df[df["Weekday"] != "Monday"]))
    filters.append(("Day=NotFri", lambda df: df[df["Weekday"] != "Friday"]))

    # Trade number (sequence within day)
    for n in [1, 2, 3]:
        filters.append((f"Trade={n}", lambda df, n=n: df[df["TradeNum"] == n]))
    filters.append(("Trade>=2", lambda df: df[df["TradeNum"] >= 2]))
    filters.append(("Trade>=3", lambda df: df[df["TradeNum"] >= 3]))
    filters.append(("Trade<=2", lambda df: df[df["TradeNum"] <= 2]))
    filters.append(("Trade<=3", lambda df: df[df["TradeNum"] <= 3]))
    filters.append(("Trade<=4", lambda df: df[df["TradeNum"] <= 4]))
    filters.append(("Trade=1-2", lambda df: df[df["TradeNum"].isin([1, 2])]))
    filters.append(("Trade=2-3", lambda df: df[df["TradeNum"].isin([2, 3])]))
    filters.append(("Trade=1-3", lambda df: df[df["TradeNum"].isin([1, 2, 3])]))

    return filters


def check_filter_compatibility(names: Tuple[str, ...]) -> bool:
    """Check if filter combination makes sense (no contradictions)."""
    # Don't combine Buy + Sell
    if "Buy" in names and "Sell" in names:
        return False
    # Don't combine same category opposites
    for prefix in ["EMA50=", "EMA200=", "EMAs=", "Engulf="]:
        vals = [n for n in names if n.startswith(prefix)]
        if len(vals) > 1:
            return False
    # Don't combine conflicting SL filters
    sl_filters = [n for n in names if n.startswith("SL")]
    if len(sl_filters) > 1:
        # Allow one SL< and one SL>= (band), but not two SL< or two SL>=
        lt_count = sum(1 for s in sl_filters if "<" in s and ">=" not in s)
        gte_count = sum(1 for s in sl_filters if ">=" in s)
        band_count = sum(1 for s in sl_filters if "-" in s)
        if lt_count > 1 or gte_count > 1 or band_count > 1 or len(sl_filters) > 2:
            return False
        if band_count > 0 and (lt_count > 0 or gte_count > 0):
            return False
    # Don't combine conflicting day filters
    day_filters = [n for n in names if n.startswith("Day=")]
    if len(day_filters) > 1:
        return False
    # Don't combine conflicting trade filters
    trade_filters = [n for n in names if n.startswith("Trade")]
    if len(trade_filters) > 1:
        return False
    return True


def scan_combinations(df: pd.DataFrame, max_depth: int = 3) -> pd.DataFrame:
    """Scan all filter combinations up to max_depth and return results with 10%+ edge."""
    atomic = get_atomic_filters()
    results = []

    # Single filters
    print(f"Scanning {len(atomic)} single filters...")
    for name, func in atomic:
        filtered = func(df)
        for rrr in RRR_RATIOS:
            stats = calc_edge(filtered, rrr)
            if stats["trades"] >= MIN_TRADES and stats["edge"] >= MIN_EDGE:
                results.append({
                    "Strategy": name,
                    "RRR": f"1:{rrr}",
                    **stats,
                })

    # Double combinations
    if max_depth >= 2:
        combos_2 = list(itertools.combinations(range(len(atomic)), 2))
        print(f"Scanning {len(combos_2)} double combinations...")
        for i, j in combos_2:
            n1, f1 = atomic[i]
            n2, f2 = atomic[j]
            if not check_filter_compatibility((n1, n2)):
                continue
            try:
                filtered = f2(f1(df))
            except Exception:
                continue
            for rrr in RRR_RATIOS:
                stats = calc_edge(filtered, rrr)
                if stats["trades"] >= MIN_TRADES and stats["edge"] >= MIN_EDGE:
                    results.append({
                        "Strategy": f"{n1} + {n2}",
                        "RRR": f"1:{rrr}",
                        **stats,
                    })

    # Triple combinations
    if max_depth >= 3:
        combos_3 = list(itertools.combinations(range(len(atomic)), 3))
        print(f"Scanning {len(combos_3)} triple combinations...")
        for i, j, k in combos_3:
            n1, f1 = atomic[i]
            n2, f2 = atomic[j]
            n3, f3 = atomic[k]
            if not check_filter_compatibility((n1, n2, n3)):
                continue
            try:
                filtered = f3(f2(f1(df)))
            except Exception:
                continue
            for rrr in RRR_RATIOS:
                stats = calc_edge(filtered, rrr)
                if stats["trades"] >= MIN_TRADES and stats["edge"] >= MIN_EDGE:
                    results.append({
                        "Strategy": f"{n1} + {n2} + {n3}",
                        "RRR": f"1:{rrr}",
                        **stats,
                    })

    result_df = pd.DataFrame(results)
    if len(result_df) == 0:
        print("No strategies found with 10%+ edge and 15+ trades")
        return result_df

    result_df = result_df.sort_values("edge", ascending=False).reset_index(drop=True)
    return result_df


def format_results(df: pd.DataFrame) -> str:
    """Format results for terminal display."""
    if df.empty:
        return "No strategies found."

    lines = []
    lines.append(f"\n{'='*120}")
    lines.append(f"STRATEGIES WITH {MIN_EDGE}%+ EDGE (min {MIN_TRADES} trades)")
    lines.append(f"{'='*120}")
    lines.append(f"{'Strategy':<60} {'RRR':>5} {'Trades':>7} {'W/L':>12} {'WinRate':>8} {'Edge':>8} {'Outcome':>8} {'Days':>10}")
    lines.append(f"{'-'*120}")

    for _, row in df.iterrows():
        notation = f"{row['wins']}W-{row['losses']}L"
        days_info = f"{row['days_with_wins']}/{row['total_days']}"
        lines.append(
            f"{row['Strategy']:<60} {row['RRR']:>5} {row['trades']:>7} {notation:>12} "
            f"{row['win_rate']:>7.1f}% {row['edge']:>7.1f}% {row['outcome']:>7}R {days_info:>10}"
        )

    lines.append(f"{'='*120}")
    lines.append(f"Total strategies found: {len(df)}")
    return "\n".join(lines)


def scan_with_buffers(df: pd.DataFrame) -> pd.DataFrame:
    """Scan top strategies with SL buffer (extra pips added to SL)."""
    atomic = get_atomic_filters()
    buffers = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = []

    # Test "All Trades" + buffer
    strategies_to_test = [("All Trades", lambda df: df)]

    # Add single filters
    for name, func in atomic:
        strategies_to_test.append((name, func))

    # Add top double combinations
    for i, j in itertools.combinations(range(len(atomic)), 2):
        n1, f1 = atomic[i]
        n2, f2 = atomic[j]
        if not check_filter_compatibility((n1, n2)):
            continue
        strategies_to_test.append((f"{n1} + {n2}", lambda df, f1=f1, f2=f2: f2(f1(df))))

    print(f"Scanning {len(strategies_to_test)} strategies with {len(buffers)} buffer values...")

    for name, func in strategies_to_test:
        try:
            filtered = func(df)
        except Exception:
            continue
        for buffer in buffers:
            effective_sl = filtered["SL"] + buffer
            for rrr in RRR_RATIOS:
                total = len(filtered)
                if total < MIN_TRADES:
                    continue
                wins_df = filtered[
                    (filtered["Pullback"] < effective_sl) &
                    (filtered["TP"] >= rrr * effective_sl)
                ]
                wins = len(wins_df)
                losses = total - wins
                win_rate = (wins / total) * 100
                edge = win_rate - breakeven_rate(rrr)
                outcome = (wins * rrr) - losses

                if edge >= MIN_EDGE:
                    results.append({
                        "Strategy": f"{name} +{buffer}pip",
                        "RRR": f"1:{rrr}",
                        "trades": total,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": win_rate,
                        "edge": edge,
                        "outcome": outcome,
                        "days_with_wins": wins_df["Date"].nunique() if len(wins_df) > 0 else 0,
                        "total_days": filtered["Date"].nunique(),
                    })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values("edge", ascending=False).reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)

    df = load_data()
    total = len(df)
    print(f"Loaded {total} trades across {df['Date'].nunique()} days")

    # Baseline: All trades
    for rrr in RRR_RATIOS:
        stats = calc_edge(df, rrr)
        print(f"\nBaseline 1:{rrr} — {stats['trades']} trades, {stats['wins']}W-{stats['losses']}L, "
              f"WR={stats['win_rate']:.1f}%, Edge={stats['edge']:.1f}%, Outcome={stats['outcome']}R")

    # Scan without buffer
    print("\n--- SCANNING WITHOUT BUFFER ---")
    results_no_buffer = scan_combinations(df, max_depth=3)
    print(format_results(results_no_buffer))

    # Scan with buffer
    print("\n--- SCANNING WITH SL BUFFER ---")
    results_buffer = scan_with_buffers(df)
    print(format_results(results_buffer))
