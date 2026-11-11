import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Backtest assumptions. These are estimates, not Binance account settings.
TIMEFRAME = "15m"
DAYS_BACK = 30
STARTING_BALANCE = 1000.0          # simulated USDT
RISK_PER_TRADE = 0.01              # maximum planned loss, including estimated fees
STOP_LOSS_PCT = 0.01               # stop distance from entry
REWARD_RISK_RATIO = 2.0            # TP is 2R from entry
LEVERAGE = 3.0                     # isolated-margin simulation
MAINTENANCE_MARGIN_RATE = 0.005     # simplified estimate; Binance tiers vary by symbol/notional
TAKER_FEE_RATE = 0.001              # assumed 0.1% per side; change to your fee tier

coins = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "ADA/USDT",
]

if not 1 <= LEVERAGE <= 125:
    raise ValueError("LEVERAGE 1 ile 125 arasinda olmalidir.")
if not 0 < RISK_PER_TRADE < 1 or not 0 < STOP_LOSS_PCT < 1:
    raise ValueError("Risk ve stop yuzdeleri 0 ile 1 arasinda olmalidir.")
if not 0 <= MAINTENANCE_MARGIN_RATE < 1 or not 0 <= TAKER_FEE_RATE < 1:
    raise ValueError("Bakim teminati ve komisyon oranlari gecersiz.")

# USD-M perpetual futures. Mark-price candles are used for liquidation checks.
exchange = ccxt.binanceusdm({"enableRateLimit": True})
exchange.load_markets()
interval_ms = exchange.parse_timeframe(TIMEFRAME) * 1000
now_ms = exchange.milliseconds()
start_ms = now_ms - DAYS_BACK * 24 * 60 * 60 * 1000
fetch_from_ms = start_ms - 60 * interval_ms  # indicator warm-up


def fetch_history(symbol, mark_price=False):
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
    frame["EMA20"] = frame["close"].ewm(span=20, adjust=False).mean()
    frame["EMA50"] = frame["close"].ewm(span=50, adjust=False).mean()
    delta = frame["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    frame["RSI"] = 100 - (100 / (1 + rs))
    long_signal = (
        (frame["EMA20"] > frame["EMA50"])
        & (frame["close"] > frame["EMA20"])
        & (frame["RSI"] > 50)
        & (frame["RSI"] < 70)
    )
    short_signal = (
        (frame["EMA20"] < frame["EMA50"])
        & (frame["close"] < frame["EMA20"])
        & (frame["RSI"] > 30)
        & (frame["RSI"] < 50)
    )
    frame["signal"] = ""
    frame.loc[long_signal, "signal"] = "LONG"
    frame.loc[short_signal, "signal"] = "SHORT"
    return frame


def utc_text(timestamp_ms):
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def estimated_liquidation_price(entry, side):
    # Simplified isolated USDT-M formula with a flat maintenance-margin rate
    # and zero maintenance amount. Actual Binance tiers/fees can differ.
    mmr = MAINTENANCE_MARGIN_RATE
    if side == "LONG":
        return entry * (1 - 1 / LEVERAGE) / (1 - mmr)
    return entry * (1 + 1 / LEVERAGE) / (1 + mmr)


def get_exit(position, trade_bar, mark_bar):
    """Return (price, reason) for a bar; assume the adverse event happened first."""
    if mark_bar is not None:
        if position["side"] == "LONG":
            liquidation_hit = mark_bar["mark_low"] <= position["liquidation_price"]
        else:
            liquidation_hit = mark_bar["mark_high"] >= position["liquidation_price"]
        if liquidation_hit:
            return position["liquidation_price"], "LIQUIDATION_ESTIMATE"

    if trade_bar is None:
        return None
    open_price = float(trade_bar["open"])
    if position["side"] == "LONG":
        hit_stop = trade_bar["low"] <= position["stop"]
        hit_target = trade_bar["high"] >= position["target"]
        if hit_stop:
            fill = min(open_price, position["stop"])
            return fill, "SL_AND_TP_SAME_CANDLE_SL_ASSUMED" if hit_target else "SL"
        if hit_target:
            return position["target"], "TP"
    else:
        hit_stop = trade_bar["high"] >= position["stop"]
        hit_target = trade_bar["low"] <= position["target"]
        if hit_stop:
            fill = max(open_price, position["stop"])
            return fill, "SL_AND_TP_SAME_CANDLE_SL_ASSUMED" if hit_target else "SL"
        if hit_target:
            return position["target"], "TP"
    return None


print("30 GUNLUK USD-M FUTURES BACKTESTI (15m)")
print(
    f"Sanal bakiye: {STARTING_BALANCE:.2f} USDT | Risk: {RISK_PER_TRADE:.1%} | "
    f"SL: {STOP_LOSS_PCT:.1%} | TP: {REWARD_RISK_RATIO:.1f}R | Kaldirac: {LEVERAGE:.1f}x"
)
print(
    "Gercek emir gonderilmez. Likidasyon fiyati sabit bakim teminati varsayimiyla "
    "tahmindir; funding, likidasyon ceza ucreti ve mark/son fiyat farki modellenmez.\n"
)

frames = {}
for coin in coins:
    try:
        trade_candles = fetch_history(coin)
        mark_candles = fetch_history(coin, mark_price=True)
        if len(trade_candles) < 60 or len(mark_candles) < 60:
            print(f"{coin}: Yetersiz futures/mark verisi, atlandi.")
            continue
        frame = add_indicators(trade_candles)
        mark_frame = pd.DataFrame(
            mark_candles, columns=["time", "mark_open", "mark_high", "mark_low", "mark_close", "mark_volume"]
        )
        frame = frame.merge(mark_frame, on="time", how="inner")
        if len(frame) < 60:
            print(f"{coin}: Eslesen futures/mark mum sayisi yetersiz, atlandi.")
            continue
        frames[coin] = frame
        print(f"{coin}: {len(frame)} eslesen futures ve mark mumu alindi")
    except Exception as error:
        print(f"{coin}: Veri alinamadi; atlandi: {error}")

# One shared simulated account: only one open position at a time.
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
        signal = records[i]["signal"]  # only use the already-closed candle
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
    entry_fee = position["position_notional"] * TAKER_FEE_RATE
    if reason == "LIQUIDATION_ESTIMATE":
        # Conservative isolated-margin assumption: allocated initial margin is lost.
        gross = -position["initial_margin"]
        exit_fee = 0.0
    else:
        gross = direction * (exit_price - position["entry_price"]) * position["quantity"]
        exit_fee = exit_price * position["quantity"] * TAKER_FEE_RATE
    fees = entry_fee + exit_fee
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
        "quantity_coin": position["quantity"],
        "position_notional_usdt": position["position_notional"],
        "leverage": LEVERAGE,
        "initial_margin_usdt": position["initial_margin"],
        "estimated_liquidation_price": position["liquidation_price"],
        "exit_price": exit_price,
        "stop_loss": position["stop"],
        "take_profit": position["target"],
        "risk_budget_usdt": position["risk_budget"],
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
    bars = candles_by_time[timestamp]

    if active is not None:
        coin_bar = bars.get(active["coin"])
        mark_bar = coin_bar  # mark OHLC fields are joined to the same row
        if coin_bar is not None:
            outcome = get_exit(active, coin_bar, mark_bar)
            if outcome is not None:
                exit_price, reason = outcome
                close_trade(active, timestamp, exit_price, reason)
                active = None
        if active is not None:
            continue
        # Do not re-enter at the same timestamp as an exit.
        continue

    if balance <= 0:
        break
    for coin, side in candidates_by_time.get(timestamp, []):
        bar = bars.get(coin)
        if bar is None or pd.isna(bar["open"]) or bar["open"] <= 0:
            continue
        entry = float(bar["open"])
        stop_distance = entry * STOP_LOSS_PCT
        stop = entry - stop_distance if side == "LONG" else entry + stop_distance
        target = (
            entry + stop_distance * REWARD_RISK_RATIO
            if side == "LONG"
            else entry - stop_distance * REWARD_RISK_RATIO
        )
        risk_budget = balance * RISK_PER_TRADE
        # Include estimated entry and stop-exit fees inside the risk budget.
        risk_per_coin = stop_distance + (entry + stop) * TAKER_FEE_RATE
        quantity = risk_budget / risk_per_coin
        # Never allocate more initial margin than the simulated balance.
        max_quantity = balance * LEVERAGE / entry
        quantity = min(quantity, max_quantity)
        position_notional = entry * quantity
        initial_margin = position_notional / LEVERAGE
        active = {
            "coin": coin,
            "side": side,
            "entry_time": timestamp,
            "entry_price": entry,
            "stop": stop,
            "target": target,
            "quantity": quantity,
            "position_notional": position_notional,
            "initial_margin": initial_margin,
            "risk_budget": quantity * risk_per_coin,
            "liquidation_price": estimated_liquidation_price(entry, side),
        }
        # Check the entry candle too; intrabar SL/TP/liquidation can occur after the open.
        outcome = get_exit(active, bar, bar)
        if outcome is not None:
            exit_price, reason = outcome
            close_trade(active, timestamp, exit_price, reason)
            active = None
        break

if active is not None:
    last_time = max(t for t, rows in candles_by_time.items() if active["coin"] in rows)
    final_bar = candles_by_time[last_time][active["coin"]]
    if final_bar is not None:
        close_trade(active, int(final_bar["time"]), float(final_bar["close"]), "END_OF_DATA")

output_dir = Path(__file__).resolve().parent
trades_path = output_dir / "backtest_trades.csv"
daily_path = output_dir / "backtest_daily_pnl.csv"
columns = [
    "coin", "side", "entry_time_utc", "exit_time_utc", "entry_price",
    "quantity_coin", "position_notional_usdt", "leverage", "initial_margin_usdt",
    "estimated_liquidation_price", "exit_price", "stop_loss", "take_profit",
    "risk_budget_usdt", "gross_pnl_usdt", "fees_usdt", "net_pnl_usdt",
    "result", "exit_reason", "balance_after_usdt",
]
trades_df = pd.DataFrame(trades, columns=columns)
trades_df.to_csv(trades_path, index=False)

if trades:
    trades_df["exit_date_utc"] = pd.to_datetime(trades_df["exit_time_utc"], utc=True).dt.strftime("%Y-%m-%d")
    daily = trades_df.groupby("exit_date_utc").agg(
        pnl_usdt=("net_pnl_usdt", "sum"),
        trades=("net_pnl_usdt", "size"),
        wins=("result", lambda values: (values == "WIN").sum()),
        losses=("result", lambda values: (values == "LOSS").sum()),
    )
else:
    daily = pd.DataFrame(columns=["pnl_usdt", "trades", "wins", "losses"])
all_days = pd.date_range(
    datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date(),
    datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).date(),
    freq="D",
).strftime("%Y-%m-%d")
daily = daily.reindex(all_days, fill_value=0).rename_axis("date_utc").reset_index()
daily.to_csv(daily_path, index=False)

wins = sum(trade["result"] == "WIN" for trade in trades)
losses = sum(trade["result"] == "LOSS" for trade in trades)
liquidations = sum(trade["exit_reason"] == "LIQUIDATION_ESTIMATE" for trade in trades)
net_pnl = balance - STARTING_BALANCE
win_rate = wins / len(trades) * 100 if trades else 0.0
print("\n===== BACKTEST OZETI =====")
print(f"Tamamlanan sanal islem: {len(trades)} | Kazanan: {wins} | Kaybeden: {losses} | Basari: {win_rate:.2f}%")
print(f"Baslangic: {STARTING_BALANCE:.2f} USDT | Son bakiye: {balance:.2f} USDT | Net P/L: {net_pnl:.2f} USDT")
print(f"Gerceklesen azami dusus: {max_drawdown:.2%} | Tahmini likidasyon: {liquidations}")
print(f"Islem raporu: {trades_path}")
print(f"Gunluk P/L raporu (UTC): {daily_path}")



