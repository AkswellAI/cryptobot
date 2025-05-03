import os
import requests
import pandas as pd
import numpy as np
from binance.client import Client
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone

# === 1) Переменные окружения ===
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("CHAT_ID")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
if not all([TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Missing environment variables")

# === 2) Инициализация Binance ===
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})

# === 3) Получение топ‑100 USDT‑пар ===
def get_top_symbols(limit: int = 100) -> list[str]:
    tickers = binance.get_ticker()
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    sorted_ = sorted(usdt, key=lambda t: float(t['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_[:limit]]

# === 4) Загрузка свечей ===
def get_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    kl = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        'ts','open','high','low','close','volume',
        'ct','qav','trades','tbv','tqv','ignore'
    ])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# === 5) Индикаторы ===
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def vwap(df: pd.DataFrame) -> float:
    # VWAP за весь период df
    vp = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return vp.iloc[-1]

def stoch_rsi(close: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    # RSI
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(length).mean() / down.rolling(length).mean()
    rsi = 100 - (100 / (1 + rs))
    # Stochastic of RSI
    min_rsi = rsi.rolling(length).min()
    max_rsi = rsi.rolling(length).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

# === 6) Логика сигнала ===
def check_signal(df: pd.DataFrame) -> dict | None:
    # Расчёт EMA9/21
    df['ema9']  = ema(df['close'], 9)
    df['ema21'] = ema(df['close'], 21)
    # Расчёт VWAP
    vwap_val = vwap(df)
    # Объём
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    curr_vol = df['volume'].iloc[-1]
    # Stochastic RSI
    k, d = stoch_rsi(df['close'])
    curr_k, curr_d = k.iloc[-1], d.iloc[-1]
    prev_k, prev_d = k.iloc[-2], d.iloc[-2]

    # Условия Long
    cross = (df['ema9'].iloc[-2] < df['ema21'].iloc[-2]) and (df['ema9'].iloc[-1] > df['ema21'].iloc[-1])
    price = df['close'].iloc[-1]
    above_vwap = price > vwap_val
    vol_ok = curr_vol > avg_vol
    stoch_cross = (prev_k < prev_d and curr_k > curr_d and curr_k < 20)

    if cross and above_vwap and vol_ok and stoch_cross:
        # Стоп‑лосс под VWAP или минимум low за 20 свечей
        recent_low = df['low'].rolling(20).min().iloc[-1]
        sl = min(vwap_val, recent_low)
        entry = price
        tp = entry + (entry - sl) * 2  # R:R = 1:2
        return {
            "signal": "BUY",
            "entry": round(entry, 6),
            "sl": round(sl, 6),
            "tp": round(tp, 6),
            "vwap": round(vwap_val, 6),
            "vol": curr_vol,
            "avg_vol": round(avg_vol, 6),
            "stoch_k": round(curr_k, 2),
            "stoch_d": round(curr_d, 2)
        }

    return None

# === 7) Основная задача ===
def job():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] Job start")
    symbols = get_top_symbols(100)
    for sym in symbols:
        try:
            df = get_klines(sym, Client.KLINE_INTERVAL_1MINUTE, limit=100)
            result = check_signal(df)
            if result:
                price = result['entry']
                msg = (
                    f"*{sym}* ({now})\n"
                    f"Signal: *{result['signal']}*\n"
                    f"Entry: `{price}`  SL: `{result['sl']}`  TP: `{result['tp']}`\n"
                    f"VWAP: `{result['vwap']}`\n"
                    f"Vol: `{result['vol']}` (avg `{result['avg_vol']}`)\n"
                    f"StochRSI: K={result['stoch_k']} D={result['stoch_d']}"
                )
                send_telegram_message(msg)
        except Exception as e:
            print(f"{sym} error: {e}")

# === 8) Планировщик ===
if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(job, "interval", minutes=5, next_run_time=datetime.now(timezone.utc))
    print("Bot started. Scheduler running...")
    job()
    scheduler.start()
