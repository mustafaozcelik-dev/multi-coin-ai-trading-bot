import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Backtest settings: change these values to try other assumptions.
TIMEFRAME = "15m"
DAYS_BACK = 30
STARTING_BALANCE = 1000.0  # simulated USDT
RISK_PER_TRADE = 0.01      # risk 1% of current simulated balance
STOP_LOSS_PCT = 0.01       # stop 1% from entry
REWARD_RISK_RATIO = 2.0    # take profit at 2R
FEE_RATE = 0.001           # assumed 0.1% fee on entry and exit

coins = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "ADA/USDT",
]

exchange = ccxt.binance({"enableRateLimit": True})
exchange.load_markets()
interval_ms = exchange.parse_timeframe(TIMEFRAME) * 1000
now_ms = exchange.milliseconds()
start_ms = now_ms - DAYS_BACK * 24 * 60 * 60 * 1000
# Extra candles warm up EMA50 and RSI before the measured 30-day period.
fetch_from_ms = start_ms - 60 * interval_ms


def fetch_history(symbol):
    rows = []
    since = fetch_from_ms
    while since < now_ms:
        batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + interval_ms
        if next_since <= since:
            break
        since = next_since
        if len(batch) < 1000:
            break
    # Remove duplicate timestamps and the still-open candle.
    unique = {row[0]: row for row in rows if row[0] + interval_ms <= now_ms}
    return [unique[key] for key in sorted(unique)]


def add_indicators(rows):
    frame = pd.DataFrame(
        rows, columns=["time", "open", "high", "low", "close", "volume"]
    )
    frame["EMA20"] = frame["close"].ewm(span=20, adjust=False).mean()
    frame["EMA50"] = frame["close"].ewm(span=50, adjust=False).mean()
    delta = frame["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    frame["RSI"] = 100 - (100 / (1 + rs))
    long_condition = (
        (frame["EMA20"] > frame["EMA50"])
        & (frame["close"] > frame["EMA20"])
        & (frame["RSI"] > 50)
        & (frame["RSI"] < 70)
    )
    short_condition = (
        (frame["EMA20"] < frame["EMA50"])
        & (frame["close"] < frame["EMA20"])
        & (frame["RSI"] > 30)
        & (frame["RSI"] < 50)
    )
    frame["signal"] = ""
    frame.loc[long_condition, "signal"] = "LONG"
    frame.loc[short_condition, "signal"] = "SHORT"
    return frame


def utc_text(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


print("30 GUNLUK GECMIŞ VERI BACKTESTI (15m)")
print("Sanal bakiye: %.2f USDT | Islem riski: %.1f%% | SL: %.1f%% | TP: %.1fR" % (
    STARTING_BALANCE, RISK_PER_TRADE * 100, STOP_LOSS_PCT * 100, REWARD_RISK_RATIO
))
print("Gercek emir gonderilmez. Sonuclar ucret varsayimina ve mum ici belirsizlige tabidir.\n")

frames = {}
for coin in coins:
    candles = fetch_history(coin)
    if len(candles) < 60:
        print(coin, "icin yeterli mum verisi yok; atlandi.")
        continue
    frames[coin] = add_indicators(candles)
    print(f"{coin}: {len(candles)} mum alindi")

# Create entry candidates: use a completed candle's signal at the next candle open.
candles_by_time = {}
candidates_by_time = {}
for coin, frame in frames.items():
    records = frame.to_dict("records")
    for i, row in enumerate(records):
        candles_by_time.setdefault(int(row["time"]), {})[coin] = row
        if i == 0 or i >= len(records) - 1:
            continue
        entry_time = int(records[i + 1]["time"])
        if entry_time < start_ms:
            continue
        signal = records[i]["signal"]
        if signal in ("LONG", "SHORT"):
            candidates_by_time.setdefault(entry_time, []).append((coin, signal))

balance = STARTING_BALANCE
trades = []
active = None
peak_balance = balance
max_drawdown = 0.0


def close_trade(position, timestamp, exit_price, reason):
    global balance, peak_balance, max_drawdown
    direction = 1 if position["side"] == "LONG" else -1
    gross = direction * (exit_price - position["entry_price"]) * position["quantity"]
    fees = (position["entry_price"] + exit_price) * position["quantity"] * FEE_RATE
    net = gross - fees
    balance += net
    peak_balance = max(peak_balance, balance)
    if peak_balance > 0:
        max_drawdown = max(max_drawdown, (peak_balance - balance) / peak_balance)
    trades.append({
        "coin": position["coin"],
        "side": position["side"],
        "entry_time_utc": utc_text(position["entry_time"]),
        "exit_time_utc": utc_text(timestamp),
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "stop_loss": position["stop"],
        "take_profit": position["target"],
        "risk_usdt": position["risk_amount"],
        "gross_pnl_usdt": gross,
        "fees_usdt": fees,
        "net_pnl_usdt": net,
        "result": "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN",
        "exit_reason": reason,
        "balance_after_usdt": balance,
    })


for timestamp in sorted(candles_by_time):
    if timestamp < start_ms:
        continue

    # Check an open position against its candle. If SL and TP both occur in
    # one candle, assume SL was hit first (conservative OHLCV-only handling).
    if active is not None:
        bar = candles_by_time[timestamp].get(active["coin"])
        if bar is not None and timestamp > active["entry_time"]:
            if active["side"] == "LONG":
                hit_stop = bar["low"] <= active["stop"]
                hit_target = bar["high"] >= active["target"]
            else:
                hit_stop = bar["high"] >= active["stop"]
                hit_target = bar["low"] <= active["target"]
            if hit_stop:
                close_trade(active, timestamp, active["stop"], "SL" if not hit_target else "SL_AND_TP_SAME_CANDLE_SL_ASSUMED")
                active = None
                continue
            if hit_target:
                close_trade(active, timestamp, active["target"], "TP")
                active = None
                continue
        continue

    if balance <= 0:
        break
    choices = candidates_by_time.get(timestamp, [])
    bars = candles_by_time[timestamp]
    for coin, side in choices:
        bar = bars.get(coin)
        if bar is None or pd.isna(bar["open"]) or bar["open"] <= 0:
            continue
        entry = float(bar["open"])
        stop_distance = entry * STOP_LOSS_PCT
        risk_amount = balance * RISK_PER_TRADE
        quantity = risk_amount / stop_distance
        if side == "LONG":
            stop = entry - stop_distance
            target = entry + stop_distance * REWARD_RISK_RATIO
        else:
            stop = entry + stop_distance
            target = entry - stop_distance * REWARD_RISK_RATIO
        active = {
            "coin": coin, "side": side, "entry_time": timestamp,
            "entry_price": entry, "stop": stop, "target": target,
            "quantity": quantity, "risk_amount": risk_amount,
        }
        break

if active is not None:
    final_bar = candles_by_time[max(candles_by_time)].get(active["coin"])
    if final_bar is not None:
        close_trade(active, int(final_bar["time"]), float(final_bar["close"]), "END_OF_DATA")

output_dir = Path(__file__).resolve().parent
trades_path = output_dir / "backtest_trades.csv"
daily_path = output_dir / "backtest_daily_pnl.csv"
trades_df = pd.DataFrame(trades)
if trades_df.empty:
    trades_df = pd.DataFrame(columns=[
        "coin", "side", "entry_time_utc", "exit_time_utc", "entry_price",
        "exit_price", "stop_loss", "take_profit", "risk_usdt", "gross_pnl_usdt",
        "fees_usdt", "net_pnl_usdt", "result", "exit_reason", "balance_after_usdt"
    ])
trades_df.to_csv(trades_path, index=False)

if trades:
    trades_df["exit_date_utc"] = pd.to_datetime(trades_df["exit_time_utc"], utc=True).dt.strftime("%Y-%m-%d")
    daily = trades_df.groupby("exit_date_utc").agg(
        pnl_usdt=("net_pnl_usdt", "sum"),
        trades=("net_pnl_usdt", "size"),
        wins=("result", lambda values: (values == "WIN").sum()),
        losses=("result", lambda values: (values == "LOSS").sum()),
    ).reset_index().rename(columns={"exit_date_utc": "date_utc"})
else:
    daily = pd.DataFrame(columns=["date_utc", "pnl_usdt", "trades", "wins", "losses"])
daily.to_csv(daily_path, index=False)

wins = sum(1 for trade in trades if trade["result"] == "WIN")
losses = sum(1 for trade in trades if trade["result"] == "LOSS")
net_pnl = balance - STARTING_BALANCE
win_rate = (wins / len(trades) * 100) if trades else 0.0
print("\n===== BACKTEST OZETI =====")
print(f"Tamamlanan sanal islem: {len(trades)} | Kazanan: {wins} | Kaybeden: {losses} | Kazanma orani: {win_rate:.2f}%")
print(f"Baslangic bakiyesi: {STARTING_BALANCE:.2f} USDT | Son bakiye: {balance:.2f} USDT | Net P/L: {net_pnl:.2f} USDT")
print(f"Gerceklesen azami dusus (realized): {max_drawdown * 100:.2f}%")
print(f"Islem kaydi: {trades_path}")
print(f"Gunluk P/L raporu (UTC): {daily_path}")
