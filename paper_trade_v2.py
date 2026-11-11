import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import subprocess
import urllib.parse
import pandas as pd

# Paper-trading only: this program uses public market data and sends no orders.
TIMEFRAME = "15m"
POLL_SECONDS = 60
MAX_COMPLETED_TRADES = 1
STARTING_BALANCE = 1000.0
RISK_PER_TRADE = 0.01
LEVERAGE = 3.0
TAKER_FEE_RATE = 0.001
MAINTENANCE_MARGIN_RATE = 0.005

SCORE_MIN = 70
BNB_LONG_SCORE_MIN = 80
ATR_STOP_MULTIPLIER = 1.5
TARGET_R = (1.0, 2.0, 3.0)
TARGET_PARTS = (0.40, 0.30, 0.30)
MOVE_STOP_TO_BREAKEVEN_AFTER_TP1 = True

COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "LINK/USDT", "AVAX/USDT", "ADA/USDT",
]
ALLOWED_SIDES = {
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

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "paper_state_v2.json"
TRADES_PATH = BASE_DIR / "paper_trades_v2.csv"
TRADE_COLUMNS = [
    "coin", "side", "entry_time_utc", "exit_time_utc", "entry_price",
    "exit_price", "quantity_coin", "position_notional_usdt", "leverage",
    "initial_margin_usdt", "estimated_liquidation_price", "stop_loss_initial",
    "take_profit_1", "take_profit_2", "take_profit_3", "score",
    "risk_budget_usdt", "gross_pnl_usdt", "fees_usdt", "net_pnl_usdt",
    "result", "exit_reason", "tp1_hit", "tp2_hit", "tp3_hit", "balance_after_usdt",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_state(state):
    temp = STATE_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE_PATH)


def load_state():
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("completed_trades", 0)
        return state
    return {
        "balance": STARTING_BALANCE,
        "active": None,
        "last_signal_candle": {},
        "created_utc": utc_now(),
    }


def public_get(path, params):
    query = urllib.parse.urlencode(params)
    url = "https://fapi.binance.com" + path + "?" + query
    safe_url = url.replace("'", "''")
    command = (
        "$ProgressPreference='SilentlyContinue'; "
        "$ErrorActionPreference='Stop'; "
        "$result=Invoke-RestMethod -Uri '" + safe_url + "' -TimeoutSec 20; "
        "ConvertTo-Json -InputObject $result -Depth 10 -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout.lstrip("\ufeff"))


def fetch_candles(coin, limit=140):
    raw = public_get(
        "/fapi/v1/klines",
        {"symbol": coin.replace("/", ""), "interval": TIMEFRAME, "limit": limit},
    )
    if isinstance(raw, dict) and isinstance(raw.get("value"), list):
        raw = raw["value"]
    return [[int(row[0])] + [float(value) for value in row[1:6]] for row in raw]

def indicator_frame(candles):
    frame = pd.DataFrame(
        candles, columns=["time", "open", "high", "low", "close", "volume"]
    )
    close = frame["close"]
    for span in (7, 25, 99):
        frame[f"EMA{span}"] = close.ewm(span=span, adjust=False).mean()

    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    frame["RSI"] = 100 - (100 / (1 + rs))
    frame.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100

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
    return frame


def btc_trend(frame):
    row = frame.iloc[-2]  # last completed 15-minute candle
    if row["EMA25"] > row["EMA99"]:
        return "LONG"
    if row["EMA25"] < row["EMA99"]:
        return "SHORT"
    return ""


def get_signal(coin, row, btc_side):
    options = []
    for side in ALLOWED_SIDES[coin]:
        long_side = side == "LONG"
        ema_ok = row["EMA7"] > row["EMA25"] > row["EMA99"] if long_side else row["EMA7"] < row["EMA25"] < row["EMA99"]
        rsi_ok = 50 < row["RSI"] < 70 if long_side else 30 < row["RSI"] < 50
        macd_ok = (
            row["MACD"] > row["MACD_SIGNAL"] and row["MACD_HIST"] > 0
            if long_side
            else row["MACD"] < row["MACD_SIGNAL"] and row["MACD_HIST"] < 0
        )
        volume_ok = pd.notna(row["VOLUME_RATIO"]) and row["VOLUME_RATIO"] >= 1.0
        btc_ok = btc_side == side
        score = (25 if ema_ok else 0) + (15 if rsi_ok else 0) + (20 if macd_ok else 0) + (15 if volume_ok else 0) + (25 if btc_ok else 0)
        threshold = BNB_LONG_SCORE_MIN if coin == "BNB/USDT" and side == "LONG" else SCORE_MIN
        if score >= threshold and ema_ok and rsi_ok and macd_ok and volume_ok:
            options.append((side, score, float(row["ATR"])))
    return max(options, key=lambda x: x[1]) if options else None


def liquidation_price(entry, side):
    rate = MAINTENANCE_MARGIN_RATE
    if side == "LONG":
        return entry * (1 - 1 / LEVERAGE) / (1 - rate)
    return entry * (1 + 1 / LEVERAGE) / (1 + rate)


def append_completed_trade(position, state, exit_price, reason):
    direction = 1 if position["side"] == "LONG" else -1
    remaining = position["remaining_qty"]
    if reason == "LIQUIDATION_ESTIMATE":
        gross = position["realized_gross"] - position["initial_margin"] * (remaining / position["quantity"])
        extra_fee = 0.0
    else:
        gross = position["realized_gross"] + direction * (exit_price - position["entry_price"]) * remaining
        extra_fee = exit_price * remaining * TAKER_FEE_RATE
    fees = position["entry_fee"] + position["realized_exit_fees"] + extra_fee
    net = gross - fees
    state["balance"] += net
    state["completed_trades"] = state.get("completed_trades", 0) + 1
    row = {
        "coin": position["coin"],
        "side": position["side"],
        "entry_time_utc": position["entry_time_utc"],
        "exit_time_utc": utc_now(),
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "quantity_coin": position["quantity"],
        "position_notional_usdt": position["entry_price"] * position["quantity"],
        "leverage": LEVERAGE,
        "initial_margin_usdt": position["initial_margin"],
        "estimated_liquidation_price": position["liquidation_price"],
        "stop_loss_initial": position["initial_stop"],
        "take_profit_1": position["targets"][0],
        "take_profit_2": position["targets"][1],
        "take_profit_3": position["targets"][2],
        "score": position["score"],
        "risk_budget_usdt": position["risk_budget"],
        "gross_pnl_usdt": gross,
        "fees_usdt": fees,
        "net_pnl_usdt": net,
        "result": "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN",
        "exit_reason": reason,
        "tp1_hit": position["tp_hits"] >= 1,
        "tp2_hit": position["tp_hits"] >= 2,
        "tp3_hit": position["tp_hits"] >= 3,
        "balance_after_usdt": state["balance"],
    }
    write_header = not TRADES_PATH.exists()
    with TRADES_PATH.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=TRADE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"CLOSED {position['coin']} {position['side']} {reason} | "
        f"net {net:+.2f} USDT | bakiye {state['balance']:.2f} USDT"
    )
    state["active"] = None
    save_state(state)


def manage_active(state):
    position = state["active"]
    try:
        symbol = position["coin"].replace("/", "")
        ticker = public_get("/fapi/v1/ticker/price", {"symbol": symbol})
        price = float(ticker["price"])
        if not math.isfinite(price) or price <= 0:
            return
        mark_data = public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
        mark = float(mark_data["markPrice"])
    except Exception as error:
        print(f"Aktif pozisyon fiyat hatasi: {error}")
        return

    if position["side"] == "LONG" and mark <= position["liquidation_price"]:
        append_completed_trade(position, state, position["liquidation_price"], "LIQUIDATION_ESTIMATE")
        return
    if position["side"] == "SHORT" and mark >= position["liquidation_price"]:
        append_completed_trade(position, state, position["liquidation_price"], "LIQUIDATION_ESTIMATE")
        return

    is_long = position["side"] == "LONG"
    stop = position["current_stop"]
    stop_hit = price <= stop if is_long else price >= stop
    if stop_hit:
        fill = min(price, stop) if is_long else max(price, stop)
        reason = "BREAKEVEN" if position["tp_hits"] and abs(stop - position["entry_price"]) < 1e-10 else "SL"
        append_completed_trade(position, state, fill, reason)
        return

    while position["next_target"] < 3:
        index = position["next_target"]
        target = position["targets"][index]
        target_hit = price >= target if is_long else price <= target
        if not target_hit:
            break
        qty = min(position["quantity"] * TARGET_PARTS[index], position["remaining_qty"])
        direction = 1 if is_long else -1
        position["realized_gross"] += direction * (target - position["entry_price"]) * qty
        position["realized_exit_fees"] += target * qty * TAKER_FEE_RATE
        position["remaining_qty"] -= qty
        position["tp_hits"] += 1
        position["next_target"] += 1
        print(f"TP{index + 1} {position['coin']} {position['side']} | sanal kismi kapama")
        if index == 0 and MOVE_STOP_TO_BREAKEVEN_AFTER_TP1:
            position["current_stop"] = position["entry_price"]
        if position["remaining_qty"] <= 1e-12:
            append_completed_trade(position, state, target, "TP1_TP2_TP3")
            return
    save_state(state)
    print(f"Aktif: {position['coin']} {position['side']} | fiyat {price:.6g} | stop {position['current_stop']:.6g}")


def open_position(state, coin, side, score, atr, candle_time):
    ticker = public_get("/fapi/v1/ticker/price", {"symbol": coin.replace("/", "")})
    entry = float(ticker["price"])
    if not math.isfinite(entry) or entry <= 0 or not math.isfinite(atr) or atr <= 0:
        return False
    stop_distance = atr * ATR_STOP_MULTIPLIER
    stop = entry - stop_distance if side == "LONG" else entry + stop_distance
    risk_budget = state["balance"] * RISK_PER_TRADE
    risk_per_coin = stop_distance + (entry + stop) * TAKER_FEE_RATE
    quantity = min(risk_budget / risk_per_coin, state["balance"] * LEVERAGE / entry)
    if quantity <= 0:
        return False
    targets = [
        entry + stop_distance * multiple if side == "LONG"
        else entry - stop_distance * multiple
        for multiple in TARGET_R
    ]
    position_notional = entry * quantity
    state["active"] = {
        "coin": coin,
        "side": side,
        "score": score,
        "entry_time_utc": utc_now(),
        "signal_candle_time": int(candle_time),
        "entry_price": entry,
        "quantity": quantity,
        "remaining_qty": quantity,
        "initial_margin": position_notional / LEVERAGE,
        "risk_budget": quantity * risk_per_coin,
        "liquidation_price": liquidation_price(entry, side),
        "initial_stop": stop,
        "current_stop": stop,
        "targets": targets,
        "next_target": 0,
        "tp_hits": 0,
        "realized_gross": 0.0,
        "realized_exit_fees": 0.0,
        "entry_fee": position_notional * TAKER_FEE_RATE,
    }
    state["last_signal_candle"][coin] = int(candle_time)
    save_state(state)
    print(
        f"OPEN {coin} {side} | giris {entry:.6g} | miktar {quantity:.8g} | "
        f"SL {stop:.6g} | TP {targets[0]:.6g}/{targets[1]:.6g}/{targets[2]:.6g} | "
        f"puan {score}/100 | risk {risk_budget:.2f} USDT"
    )
    return True


def main():
    state = load_state()
    print("V2 SANAL ISLEM TAKIBI BASLADI - GERCEK EMIR YOK")
    print(f"Sanal bakiye: {state['balance']:.2f} USDT | risk: %{RISK_PER_TRADE * 100:.1f} | kaldirac varsayimi: {LEVERAGE:.0f}x")
    print(f"Islem kaydi: {TRADES_PATH}")
    print(f"Bu deneme tamamlaninca bot duracak (hedef: {MAX_COMPLETED_TRADES} sanal islem).")
    if state.get("completed_trades", 0) >= MAX_COMPLETED_TRADES:
        print("Tek sanal islem hedefi zaten tamamlanmis. Takip baslatilmadi.")
        return
    if state["active"]:
        print(f"Devam eden sanal pozisyon: {state['active']['coin']} {state['active']['side']}")
    print("Durdurmak icin terminalde Ctrl+C. Acik sanal pozisyon kaydedilir ve sonraki acilista devam eder.")

    while True:
        try:
            if state["active"]:
                manage_active(state)
                if state.get("completed_trades", 0) >= MAX_COMPLETED_TRADES:
                    print("Tek sanal islem tamamlandi. Takip durduruldu.")
                    return
                time.sleep(POLL_SECONDS)
                continue

            fetched = {}
            for coin in COINS:
                try:
                    candles = fetch_candles(coin, limit=140)
                    if len(candles) >= 110:
                        fetched[coin] = indicator_frame(candles)
                except Exception as error:
                    print(f"{coin} veri hatasi: {error}")

            btc = fetched.get("BTC/USDT")
            if btc is not None:
                trend = btc_trend(btc)
                for coin in COINS:
                    frame = fetched.get(coin)
                    if frame is None:
                        continue
                    row = frame.iloc[-2]
                    candle_time = int(row["time"])
                    if state["last_signal_candle"].get(coin) == candle_time:
                        continue
                    signal = get_signal(coin, row, trend)
                    if signal:
                        side, score, atr = signal
                        opened = open_position(state, coin, side, score, atr, candle_time)
                        if opened:
                            break
                print(f"Tarama tamamlandi | BTC yonu: {trend or 'YATAY'} | bakiye: {state['balance']:.2f} USDT")
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            save_state(state)
            print("\nTakip durduruldu. Sanal bakiye ve acik pozisyon kaydedildi.")
            return
        except Exception as error:
            print(f"Beklenmeyen hata: {error}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
