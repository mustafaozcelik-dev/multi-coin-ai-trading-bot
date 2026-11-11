import ccxt
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# V1/V2 comparison. Both strategies use the exact same downloaded candles,
# simulated balance, fee assumptions, and execution rules.
TIMEFRAME = "15m"
DAYS_BACK = 30
STARTING_BALANCE = 1000.0
RISK_PER_TRADE = 0.01
LEVERAGE = 3.0
MAINTENANCE_MARGIN_RATE = 0.005  # simplified estimate; exchange tiers vary
TAKER_FEE_RATE = 0.001           # assumed 0.1% per side

# V1 parameters (the existing EMA20/50 + RSI strategy).
V1_STOP_PCT = 0.01
V1_REWARD_RISK = 2.0

# V2 parameters. These are explicit starting assumptions for an experiment,
# not optimized or guaranteed profitable settings.
V2_SCORE_MIN = 70
BNB_LONG_SCORE_MIN = 80           # stricter BNB LONG filter
ATR_STOP_MULTIPLIER = 1.5
V2_TARGET_R_MULTIPLES = (1.0, 2.0, 3.0)
V2_TARGET_SIZE_FRACTIONS = (0.40, 0.30, 0.30)
MOVE_STOP_TO_BREAKEVEN_AFTER_TP1 = True

COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "ADA/USDT",
]

# Initial rules based on the small 54-trade sample the user supplied.
# These are configurable experiment filters, not permanent coin decisions.
V2_ALLOWED_SIDES = {
    "BTC/USDT": {"SHORT"},
    "ETH/USDT": set(),
    "SOL/USDT": set(),
    "BNB/USDT": {"LONG", "SHORT"},
    "XRP/USDT": {"LONG", "SHORT"},
    "DOGE/USDT": {"LONG"},
    "LINK/USDT": set(),
    "AVAX/USDT": {"LONG", "SHORT"},
    "ADA/USDT": set(),
}

if not 1 <= LEVERAGE <= 125:
    raise ValueError("LEVERAGE 1 ile 125 arasinda olmalidir.")
if not 0 < RISK_PER_TRADE < 1:
    raise ValueError("RISK_PER_TRADE 0 ile 1 arasinda olmalidir.")
if abs(sum(V2_TARGET_SIZE_FRACTIONS) - 1.0) > 1e-9:
    raise ValueError("TP pozisyon oranlari toplami 1.0 olmalidir.")

CACHE_PATH = Path(__file__).resolve().parent / "backtest_v2_data_cache.json"
CACHE_DATA = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else None
exchange = None
if CACHE_DATA is None:
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    exchange.load_markets()
interval_ms = 15 * 60 * 1000
now_ms = int(CACHE_DATA["_meta"]["now_ms"]) if CACHE_DATA is not None else exchange.milliseconds()
start_ms = now_ms - DAYS_BACK * 24 * 60 * 60 * 1000
# Warm-up covers EMA99 plus RSI/MACD/ATR calculation.
fetch_from_ms = start_ms - 200 * interval_ms


def fetch_history(symbol, mark_price=False):
    if CACHE_DATA is not None:
        key = "mark" if mark_price else "trade"
        rows = CACHE_DATA[symbol][key]
        unique = {int(row[0]): [int(row[0])] + [float(value) for value in row[1:6]] for row in rows if int(row[0]) + interval_ms <= now_ms}
        return [unique[key] for key in sorted(unique)]
    rows = []
    since = fetch_from_ms
    params = {"price": "mark"} if mark_price else {}
    while since < now_ms:
        batch = exchange.fetch_ohlcv(
            symbol, TIMEFRAME, since=since, limit=1000, params=params
        )
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + interval_ms
        if next_since <= since:
            break
        since = next_since
        if len(batch) < 1000:
            break
    unique = {row[0]: row for row in rows if row[0] + interval_ms <= now_ms}
    return [unique[key] for key in sorted(unique)]


def add_indicators(rows):
    frame = pd.DataFrame(
        rows, columns=["time", "open", "high", "low", "close", "volume"]
    )
    close = frame["close"]
    for span in (7, 20, 25, 50, 99):
        frame[f"EMA{span}"] = close.ewm(span=span, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    frame["RSI"] = 100 - (100 / (1 + rs))

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    frame["MACD"] = macd
    frame["MACD_SIGNAL"] = macd.ewm(span=9, adjust=False).mean()
    frame["MACD_HIST"] = frame["MACD"] - frame["MACD_SIGNAL"]

    prior_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prior_close).abs(),
            (frame["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["ATR"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["VOLUME_SMA20"] = frame["volume"].rolling(20, min_periods=20).mean()
    frame["VOLUME_RATIO"] = frame["volume"] / frame["VOLUME_SMA20"]

    # V1 signal, matching the existing script's EMA20/50 + RSI rule.
    frame["v1_signal"] = ""
    v1_long = (
        (frame["EMA20"] > frame["EMA50"])
        & (close > frame["EMA20"])
        & (frame["RSI"] > 50)
        & (frame["RSI"] < 70)
    )
    v1_short = (
        (frame["EMA20"] < frame["EMA50"])
        & (close < frame["EMA20"])
        & (frame["RSI"] > 30)
        & (frame["RSI"] < 50)
    )
    frame.loc[v1_long, "v1_signal"] = "LONG"
    frame.loc[v1_short, "v1_signal"] = "SHORT"

    frame["v2_long_score"] = 0
    frame["v2_short_score"] = 0
    for idx, row in frame.iterrows():
        for side in ("LONG", "SHORT"):
            bullish = side == "LONG"
            ema_ok = (
                row["EMA7"] > row["EMA25"] > row["EMA99"]
                if bullish
                else row["EMA7"] < row["EMA25"] < row["EMA99"]
            )
            rsi_ok = (
                50 < row["RSI"] < 70
                if bullish
                else 30 < row["RSI"] < 50
            )
            macd_ok = (
                row["MACD"] > row["MACD_SIGNAL"] and row["MACD_HIST"] > 0
                if bullish
                else row["MACD"] < row["MACD_SIGNAL"] and row["MACD_HIST"] < 0
            )
            volume_ok = pd.notna(row["VOLUME_RATIO"]) and row["VOLUME_RATIO"] >= 1.0
            # BTC direction is filled after all coin frames are prepared.
            score = (25 if ema_ok else 0) + (15 if rsi_ok else 0) + (20 if macd_ok else 0) + (15 if volume_ok else 0)
            frame.at[idx, "v2_long_score" if bullish else "v2_short_score"] = score
    return frame


def add_v2_scores_with_btc(frame, btc_trend):
    frame = frame.merge(btc_trend, on="time", how="left")
    frame["BTC_TREND"] = frame["BTC_TREND"].ffill()
    for idx, row in frame.iterrows():
        for side in ("LONG", "SHORT"):
            ema_ok = (
                row["EMA7"] > row["EMA25"] > row["EMA99"]
                if side == "LONG"
                else row["EMA7"] < row["EMA25"] < row["EMA99"]
            )
            rsi_ok = 50 < row["RSI"] < 70 if side == "LONG" else 30 < row["RSI"] < 50
            macd_ok = (
                row["MACD"] > row["MACD_SIGNAL"] and row["MACD_HIST"] > 0
                if side == "LONG"
                else row["MACD"] < row["MACD_SIGNAL"] and row["MACD_HIST"] < 0
            )
            volume_ok = pd.notna(row["VOLUME_RATIO"]) and row["VOLUME_RATIO"] >= 1.0
            btc_ok = row["BTC_TREND"] == side
            score = (25 if ema_ok else 0) + (15 if rsi_ok else 0) + (20 if macd_ok else 0) + (15 if volume_ok else 0) + (25 if btc_ok else 0)
            frame.at[idx, "v2_long_score" if side == "LONG" else "v2_short_score"] = score
    return frame


def utc_text(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def estimated_liquidation_price(entry, side):
    mmr = MAINTENANCE_MARGIN_RATE
    if side == "LONG":
        return entry * (1 - 1 / LEVERAGE) / (1 - mmr)
    return entry * (1 + 1 / LEVERAGE) / (1 + mmr)


def candidate_signal(row, strategy, coin):
    if strategy == "V1":
        side = row["v1_signal"]
        if side not in ("LONG", "SHORT"):
            return None
        return side, None, None

    allowed = V2_ALLOWED_SIDES.get(coin, set())
    possibilities = []
    for side in allowed:
        score = int(row["v2_long_score"] if side == "LONG" else row["v2_short_score"])
        threshold = BNB_LONG_SCORE_MIN if coin == "BNB/USDT" and side == "LONG" else V2_SCORE_MIN
        ema_ok = row["EMA7"] > row["EMA25"] > row["EMA99"] if side == "LONG" else row["EMA7"] < row["EMA25"] < row["EMA99"]
        rsi_ok = 50 < row["RSI"] < 70 if side == "LONG" else 30 < row["RSI"] < 50
        macd_ok = (
            row["MACD"] > row["MACD_SIGNAL"] and row["MACD_HIST"] > 0
            if side == "LONG"
            else row["MACD"] < row["MACD_SIGNAL"] and row["MACD_HIST"] < 0
        )
        volume_ok = pd.notna(row["VOLUME_RATIO"]) and row["VOLUME_RATIO"] >= 1.0
        # Require trend, RSI, MACD and volume agreement, plus minimum score.
        if score >= threshold and ema_ok and rsi_ok and macd_ok and volume_ok:
            possibilities.append((side, score, float(row["ATR"])))
    if not possibilities:
        return None
    return max(possibilities, key=lambda item: item[1])


def run_backtest(strategy, frames):
    candles_by_time = {}
    candidates_by_time = {}
    for coin, frame in frames.items():
        records = frame.to_dict("records")
        for i, row in enumerate(records):
            timestamp = int(row["time"])
            candles_by_time.setdefault(timestamp, {})[coin] = row
            if i == 0 or i >= len(records) - 1:
                continue
            entry_time = int(records[i + 1]["time"])
            if entry_time < start_ms:
                continue
            signal = candidate_signal(row, strategy, coin)
            if signal:
                side, score, atr = signal
                candidates_by_time.setdefault(entry_time, []).append((coin, side, score, atr))

    balance = STARTING_BALANCE
    peak_balance = balance
    max_drawdown = 0.0
    trades = []
    active = None

    def close_trade(position, timestamp, price, reason):
        nonlocal balance, peak_balance, max_drawdown
        direction = 1 if position["side"] == "LONG" else -1
        if reason == "LIQUIDATION_ESTIMATE":
            gross = position["realized_gross"] - position["initial_margin"] * (position["remaining_qty"] / position["quantity"])
            extra_exit_fee = 0.0
        else:
            gross = position["realized_gross"] + direction * (price - position["entry_price"]) * position["remaining_qty"]
            extra_exit_fee = price * position["remaining_qty"] * TAKER_FEE_RATE
        fees = position["entry_fee"] + position["realized_exit_fees"] + extra_exit_fee
        net = gross - fees
        balance += net
        peak_balance = max(peak_balance, balance)
        if peak_balance > 0:
            max_drawdown = max(max_drawdown, (peak_balance - balance) / peak_balance)
        trades.append({
            "strategy": strategy,
            "coin": position["coin"],
            "side": position["side"],
            "entry_time_utc": utc_text(position["entry_time"]),
            "exit_time_utc": utc_text(timestamp),
            "entry_price": position["entry_price"],
            "quantity_coin": position["quantity"],
            "position_notional_usdt": position["entry_price"] * position["quantity"],
            "leverage": LEVERAGE,
            "initial_margin_usdt": position["initial_margin"],
            "estimated_liquidation_price": position["liquidation_price"],
            "exit_price_final": price,
            "stop_loss_initial": position["initial_stop"],
            "take_profit_1": position["targets"][0],
            "take_profit_2": position["targets"][1] if len(position["targets"]) > 1 else None,
            "take_profit_3": position["targets"][2] if len(position["targets"]) > 2 else None,
            "v2_score": position["score"],
            "risk_budget_usdt": position["risk_budget"],
            "gross_pnl_usdt": gross,
            "fees_usdt": fees,
            "net_pnl_usdt": net,
            "result": "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN",
            "exit_reason": reason,
            "tp1_hit": position["tp_hits"] >= 1,
            "tp2_hit": position["tp_hits"] >= 2,
            "tp3_hit": position["tp_hits"] >= 3,
            "balance_after_usdt": balance,
        })

    def process_bar(position, trade_bar):
        """Apply conservative OHLC event handling and update partial exits."""
        side = position["side"]
        is_long = side == "LONG"
        open_price = float(trade_bar["open"])
        high = float(trade_bar["high"])
        low = float(trade_bar["low"])
        mark_low = float(trade_bar["mark_low"])
        mark_high = float(trade_bar["mark_high"])

        if is_long and mark_low <= position["liquidation_price"]:
            return position["liquidation_price"], "LIQUIDATION_ESTIMATE"
        if not is_long and mark_high >= position["liquidation_price"]:
            return position["liquidation_price"], "LIQUIDATION_ESTIMATE"

        stop = position["current_stop"]
        stop_hit = low <= stop if is_long else high >= stop
        targets_hit = high >= position["targets"][-1] if is_long else low <= position["targets"][-1]
        if stop_hit:
            fill = min(open_price, stop) if is_long else max(open_price, stop)
            return fill, "SL_AND_TP_SAME_CANDLE_STOP_FIRST" if targets_hit else ("BREAKEVEN" if position["tp_hits"] >= 1 and abs(stop - position["entry_price"]) < 1e-10 else "SL")

        while position["next_target"] < len(position["targets"]):
            index = position["next_target"]
            target = position["targets"][index]
            target_hit = high >= target if is_long else low <= target
            if not target_hit:
                break
            qty_to_close = min(
                position["quantity"] * position["target_fractions"][index],
                position["remaining_qty"],
            )
            direction = 1 if is_long else -1
            position["realized_gross"] += direction * (target - position["entry_price"]) * qty_to_close
            position["realized_exit_fees"] += target * qty_to_close * TAKER_FEE_RATE
            position["remaining_qty"] -= qty_to_close
            position["tp_hits"] += 1
            position["next_target"] += 1
            if strategy == "V2" and index == 0 and MOVE_STOP_TO_BREAKEVEN_AFTER_TP1:
                position["current_stop"] = position["entry_price"]
                # If the same candle also crossed breakeven, assume the remaining
                # quantity stopped there after TP1 (intrabar order is unknowable).
                breakeven_hit = low <= position["entry_price"] if is_long else high >= position["entry_price"]
                if breakeven_hit and position["remaining_qty"] > 1e-12:
                    return position["entry_price"], "TP1_THEN_BREAKEVEN_SAME_CANDLE"
            if position["remaining_qty"] <= 1e-12:
                return target, "TP1_TP2_TP3" if strategy == "V2" else "TP"
        if position["remaining_qty"] <= 1e-12:
            return position["targets"][-1], "TP1_TP2_TP3" if strategy == "V2" else "TP"
        return None

    for timestamp in sorted(candles_by_time):
        if timestamp < start_ms:
            continue
        bars = candles_by_time[timestamp]
        if active is not None:
            bar = bars.get(active["coin"])
            if bar is not None:
                result = process_bar(active, bar)
                if result:
                    exit_price, reason = result
                    close_trade(active, timestamp, exit_price, reason)
                    active = None
            # Exit timestamps are not reused for a new entry.
            continue

        if balance <= 0:
            break
        for coin, side, score, atr in candidates_by_time.get(timestamp, []):
            bar = bars.get(coin)
            if bar is None or pd.isna(bar["open"]) or float(bar["open"]) <= 0:
                continue
            entry = float(bar["open"])
            if strategy == "V1":
                stop_distance = entry * V1_STOP_PCT
                target_distances = (stop_distance * V1_REWARD_RISK,)
                target_fractions = (1.0,)
            else:
                if atr is None or pd.isna(atr) or atr <= 0:
                    continue
                stop_distance = float(atr) * ATR_STOP_MULTIPLIER
                target_distances = tuple(stop_distance * multiple for multiple in V2_TARGET_R_MULTIPLES)
                target_fractions = V2_TARGET_SIZE_FRACTIONS
            stop = entry - stop_distance if side == "LONG" else entry + stop_distance
            targets = tuple(
                entry + distance if side == "LONG" else entry - distance
                for distance in target_distances
            )
            risk_budget = balance * RISK_PER_TRADE
            risk_per_coin = stop_distance + (entry + stop) * TAKER_FEE_RATE
            quantity = risk_budget / risk_per_coin
            quantity = min(quantity, balance * LEVERAGE / entry)
            if quantity <= 0:
                continue
            position_notional = entry * quantity
            active = {
                "coin": coin,
                "side": side,
                "score": score,
                "entry_time": timestamp,
                "entry_price": entry,
                "quantity": quantity,
                "remaining_qty": quantity,
                "initial_margin": position_notional / LEVERAGE,
                "risk_budget": quantity * risk_per_coin,
                "liquidation_price": estimated_liquidation_price(entry, side),
                "initial_stop": stop,
                "current_stop": stop,
                "targets": targets,
                "target_fractions": target_fractions,
                "next_target": 0,
                "tp_hits": 0,
                "realized_gross": 0.0,
                "realized_exit_fees": 0.0,
                "entry_fee": position_notional * TAKER_FEE_RATE,
            }
            result = process_bar(active, bar)
            if result:
                exit_price, reason = result
                close_trade(active, timestamp, exit_price, reason)
                active = None
            break

    if active is not None:
        last_time = max(t for t, rows in candles_by_time.items() if active["coin"] in rows)
        final_bar = candles_by_time[last_time][active["coin"]]
        remaining = active["remaining_qty"]
        direction = 1 if active["side"] == "LONG" else -1
        final_close = float(final_bar["close"])
        active["realized_gross"] += direction * (final_close - active["entry_price"]) * remaining
        close_trade(active, int(final_bar["time"]), final_close, "END_OF_DATA")

    return trades, balance, max_drawdown


print("V1 / V2 AYNI VERIDE KIYASLAMA (SANAL USD-M FUTURES, 15m)")
print(f"Donem: son {DAYS_BACK} gun | Baslangic bakiye: {STARTING_BALANCE:.2f} USDT | Risk: {RISK_PER_TRADE:.1%} | Kaldirac: {LEVERAGE:.1f}x")
print("Gercek emir gonderilmez. Funding, likidasyon ceza ucreti ve kademe bazli bakim teminati tam modellenmez.\n")

frames = {}
for coin in COINS:
    try:
        trade_candles = fetch_history(coin)
        mark_candles = fetch_history(coin, mark_price=True)
        if len(trade_candles) < 150 or len(mark_candles) < 150:
            print(f"{coin}: Yetersiz futures/mark verisi, atlandi.")
            continue
        frame = add_indicators(trade_candles)
        mark_frame = pd.DataFrame(
            mark_candles,
            columns=["time", "mark_open", "mark_high", "mark_low", "mark_close", "mark_volume"],
        )
        frame = frame.merge(mark_frame, on="time", how="inner")
        if len(frame) < 150:
            print(f"{coin}: Eslesen futures/mark mumu yetersiz, atlandi.")
            continue
        frames[coin] = frame
        print(f"{coin}: {len(frame)} eslesen mum alindi")
    except Exception as error:
        print(f"{coin}: Veri alinamadi, atlandi: {error}")

btc = frames.get("BTC/USDT")
if btc is None:
    raise RuntimeError("BTC verisi yok; BTC yon filtresi olmadan V2 calistirilamaz.")
btc_trend = btc[["time", "EMA25", "EMA99"]].copy()
btc_trend["BTC_TREND"] = ""
btc_trend.loc[btc_trend["EMA25"] > btc_trend["EMA99"], "BTC_TREND"] = "LONG"
btc_trend.loc[btc_trend["EMA25"] < btc_trend["EMA99"], "BTC_TREND"] = "SHORT"
btc_trend = btc_trend[["time", "BTC_TREND"]]
for coin in list(frames):
    # BTC EMA25/99 trend is attached to every coin, including BTC itself.
    frames[coin] = add_v2_scores_with_btc(frames[coin], btc_trend)

all_trades = []
summaries = []
for strategy in ("V1", "V2"):
    trades, ending_balance, max_drawdown = run_backtest(strategy, frames)
    all_trades.extend(trades)
    wins = sum(t["result"] == "WIN" for t in trades)
    losses = sum(t["result"] == "LOSS" for t in trades)
    gross_wins = sum(max(0.0, t["net_pnl_usdt"]) for t in trades)
    gross_losses = sum(min(0.0, t["net_pnl_usdt"]) for t in trades)
    profit_factor = gross_wins / abs(gross_losses) if gross_losses else (float("inf") if gross_wins else 0.0)
    summary = {
        "strategy": strategy,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wins / len(trades) * 100 if trades else 0.0,
        "net_pnl_usdt": ending_balance - STARTING_BALANCE,
        "profit_factor": profit_factor,
        "ending_balance_usdt": ending_balance,
        "max_realized_drawdown_pct": max_drawdown * 100,
        "estimated_liquidations": sum(t["exit_reason"] == "LIQUIDATION_ESTIMATE" for t in trades),
    }
    summaries.append(summary)
    print(
        f"{strategy}: {len(trades)} islem | {wins} kazanc / {losses} zarar | "
        f"basari %{summary['win_rate_pct']:.2f} | net {summary['net_pnl_usdt']:+.2f} USDT | "
        f"PF {profit_factor:.2f} | max dusus %{max_drawdown * 100:.2f} | "
        f"tahmini likidasyon {summary['estimated_liquidations']}"
    )

output_dir = Path(__file__).resolve().parent
trades_path = output_dir / "backtest_v2_trades.csv"
summary_path = output_dir / "backtest_v2_summary.csv"
pd.DataFrame(all_trades).to_csv(trades_path, index=False)
pd.DataFrame(summaries).to_csv(summary_path, index=False)
print(f"\nIslem ayrintilari: {trades_path}")
print(f"Karsilastirma ozeti: {summary_path}")
print("V2 coin/yön filtreleri az sayida eski islemden baslangic varsayimidir; kesin sonuc kabul etmeyin.")



