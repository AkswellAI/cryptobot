import os
import sys
import logging
import ccxt
import pandas as pd
import ta
from telegram import Update
from telegram.error import Conflict
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackContext,
)

# === 0) Загрузка env vars ===
TOKEN              = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("CHAT_ID")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
if not all([TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Set TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET")

# === 1) Конфигурация логов ===
root = logging.getLogger()
root.setLevel(logging.INFO)
stdout_h = logging.StreamHandler(sys.stdout)
stdout_h.setLevel(logging.DEBUG)
stdout_h.addFilter(lambda r: r.levelno <= logging.INFO)
stdout_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
stderr_h = logging.StreamHandler(sys.stderr)
stderr_h.setLevel(logging.WARNING)
stderr_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
root.handlers = [stdout_h, stderr_h]
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# === 2) Инициализация Binance ===
exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
})

# === 3) Настройки тест-режима ===
TIMEFRAME    = "1m"    # 1-минутка для быстрого теста
LIMIT        = 20      # последние 20 свечей
RSI_WINDOW   = 14
TOP_LIMIT    = 200
CHECK_INTERVAL = 60    # раз в минуту

# единственная стратегия — простой RSI
STRATEGIES = ["rsi_simple"]

# подписчики
subscribers = set()

# === 4) Хэндлеры ===
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subscribers.add(update.effective_chat.id)
    await update.message.reply_text("✅ Subscribed to simple RSI (<50=BUY, >50=SELL)")

async def error_handler(
    update: object, ctx: CallbackContext
) -> None:
    if isinstance(ctx.error, Conflict):
        return
    logger.error("Unhandled exception", exc_info=ctx.error)

async def clear_state(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

# === 5) Утилиты ===
def get_top_symbols(n=TOP_LIMIT):
    ticks = exchange.fetch_tickers()
    usdt  = [s for s in ticks if s.endswith("/USDT")]
    return sorted(
        usdt,
        key=lambda s: ticks[s].get("quoteVolume", 0),
        reverse=True
    )[:n]

def fetch_ohlcv(symbol):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df   = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["close"] = df["close"].astype(float)
    return df

# === 6) Чистая RSI-стратегия ===
def detect_rsi_simple(symbol, df):
    rsi = ta.momentum.RSIIndicator(df["close"], RSI_WINDOW).rsi().iloc[-1]
    entry = df["close"].iloc[-1]
    if rsi < 50:
        return f"📈 RSI {rsi:.1f} → BUY {symbol} @ `{entry:.6f}`"
    if rsi > 50:
        return f"📉 RSI {rsi:.1f} → SELL {symbol} @ `{entry:.6f}`"
    return None

# === 7) Основной job ===
async def check_for_signals(ctx: ContextTypes.DEFAULT_TYPE):
    syms = get_top_symbols()
    logger.info(f"Scanning top {len(syms)} symbols, first={syms[0]}")
    for s in syms:
        df = fetch_ohlcv(s)
        msg = detect_rsi_simple(s, df)
        if msg:
            for cid in subscribers:
                await ctx.bot.send_message(cid, msg)

# === 8) Запуск ===
def main():
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(clear_state)
        .build()
    )
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    # сразу добавляем свой CHAT_ID (не нужно /start)
    subscribers.add(int(CHAT_ID))
    app.job_queue.run_repeating(
        check_for_signals, interval=CHECK_INTERVAL, first=5
    )
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
