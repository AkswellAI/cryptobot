import os
import requests
import pandas as pd
from binance.client import Client
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone

# Загружаем переменные окружения
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("CHAT_ID")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

if not all([TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Missing required environment variables")

# Binance клиент
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})

def get_top_symbols(limit: int = 100) -> list[str]:
    tickers = binance.get_ticker()
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    sorted_ = sorted(usdt, key=lambda t: float(t['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_[:limit]]

def get_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    kl = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','qav','trades','tbv','tqv','ignore'
    ])
    df['close'] = df['close'].astype(float)
    return df

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def check_signal(df: pd.DataFrame) -> str | None:
    df['rsi'] = calculate_rsi(df['close'], 14)
    curr_rsi = df['rsi'].iloc[-1]

    if curr_rsi < 30:
        return "BUY"
    if curr_rsi > 70:
        return "SELL"
    return None

def generate_trade_details(signal: str, entry_price: float):
    if signal == "BUY":
        sl = entry_price * 0.98
        tp = entry_price * 1.03
    else:
        sl = entry_price * 1.02
        tp = entry_price * 0.97
    return round(entry_price, 2), round(sl, 2), round(tp, 2)

def job():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] Job started.")
    symbols = get_top_symbols(100)
    for symbol in symbols:
        try:
            df = get_klines(symbol, Client.KLINE_INTERVAL_1MINUTE)
            signal = check_signal(df)

            if signal:
                price = float(binance.get_symbol_ticker(symbol=symbol)['price'])
                entry, sl, tp = generate_trade_details(signal, price)
                message = (
                    f"*{symbol}* ({now})\n"
                    f"Timeframe: 1m → *{signal}*\n"
                    f"Entry: `{entry}`  SL: `{sl}`  TP: `{tp}`"
                )
                send_telegram_message(message)
        except Exception as e:
            print(f"{symbol}: error {e}")

# Планировщик
if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(job, "interval", minutes=5, next_run_time=datetime.now(timezone.utc))
    print("Bot started. Scheduler running...")
    job()
    scheduler.start()
