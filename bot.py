# bot.py


import os
import requests
import pandas as pd
from binance.client import Client
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timezone

# 1) Load environment variables
load_dotenv()
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("CHAT_ID")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
if not all([TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Please set TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY and BINANCE_API_SECRET in .env")

# 2) Initialize Binance client
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# 3) Send message to Telegram
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})

# 4) Get top‑100 USDT pairs by volume
def get_top_symbols(limit: int = 100) -> list[str]:
    tickers = binance.get_ticker()
    usdt    = [t for t in tickers if t['symbol'].endswith('USDT')]
    sorted_ = sorted(usdt, key=lambda t: float(t['quoteVolume']), reverse=True)
    return [t['symbol'] for t in sorted_[:limit]]

# 5) Load 15m and 1h klines
def get_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    kl = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','qav','trades','tbv','tqv','ignore'
    ])
    df['close'] = df['close'].astype(float)
    return df

# 6) Calculate RSI
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up   = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# 7) Generate MA+RSI signal
def check_signal(df: pd.DataFrame) -> str | None:
    df['ma_short'] = df['close'].rolling(10).mean()
    df['ma_long']  = df['close'].rolling(50).mean()
    df['rsi']      = calculate_rsi(df['close'], period=14)

    prev_s, prev_l = df['ma_short'].iloc[-2], df['ma_long'].iloc[-2]
    curr_s, curr_l = df['ma_short'].iloc[-1], df['ma_long'].iloc[-1]
    curr_rsi = df['rsi'].iloc[-1]

    if prev_s < prev_l and curr_s > curr_l and curr_rsi < 30:
        return "BUY"
    if prev_s > prev_l and curr_s < curr_l and curr_rsi > 70:
        return "SELL"
    return None

# 8) Calculate SL/TP
def generate_trade_details(signal: str, entry_price: float):
    if signal == "BUY":
        sl = entry_price * 0.98   # 2% below
        tp = entry_price * 1.03   # 3% above
    else:  # SELL
        sl = entry_price * 1.02   # 2% above
        tp = entry_price * 0.97   # 3% below
    return round(entry_price, 2), round(sl, 2), round(tp, 2)

# 9) Main job: check and send signals
def job():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[{now}] Checking top‑100 USDT pairs (15m & 1h)...")
    symbols = get_top_symbols(100)
    for symbol in symbols:
        try:
            df15 = get_klines(symbol, Client.KLINE_INTERVAL_15MINUTE)
            df60 = get_klines(symbol, Client.KLINE_INTERVAL_1HOUR)
            sig15 = check_signal(df15)
            sig60 = check_signal(df60)

            if sig15 and sig15 == sig60:
                entry, sl, tp = generate_trade_details(sig15, float(binance.get_symbol_ticker(symbol=symbol)['price']))
                message = (
                    f"*{symbol}* ({now})\n"
                    f"Timeframes: 15m & 1h → *{sig15}*\n"
                    f"Entry: `{entry}`  SL: `{sl}`  TP: `{tp}`"
                )
                print(f"{symbol}: sending {sig15} signal (entry={entry}, sl={sl}, tp={tp})")
                send_telegram_message(message)
            else:
                print(f"{symbol}: no match (15m={sig15}, 1h={sig60})")
        except Exception as e:
            print(f"{symbol}: error {e}")

# 10) Scheduler: run every 5 minutes
if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(job, "interval", minutes=5, next_run_time=datetime.now(timezone.utc))
    print("Bot started: checking top‑100 pairs every 5 minutes.")
    job()  # run once at start
    scheduler.start()
