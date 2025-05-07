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

# === 1) Логирование: INFO→stdout, WARNING+→stderr ===
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
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# === 2) Инициализация Binance Futures через CCXT ===
exchange = ccxt.binance({
    "apiKey":    BINANCE_API_KEY,
    "secret":    BINANCE_API_SECRET,
    "options": {
        "defaultType": "future",
    },
})

# === 3) Параметры стратегий ===
TIMEFRAME         = "5m"
LIMIT             = 100
STOP_LOSS_RATIO   = 0.99
TAKE_PROFIT_RATIO = 1.02
VOLUME_WINDOW     = 20
EMA_WINDOW        = 21
RSI_WINDOW        = 14
EMA_FAST          = 9
EMA_SLOW          = 21
STOCHRSI_LEN      = 14
STOCHRSI_K        = 3
STOCHRSI_D        = 3
TOP_LIMIT         = 200
CHECK_INTERVAL    = 300

STRATEGIES = ["breakout", "rsi_ma_volume", "ema_vwap_stochrsi"]

# подписчики
subscribers = set()

# === 4) Хэндлеры ===
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subscribers.add(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Subscribed to: " + ", ".join(STRATEGIES)
    )

async def error_handler(update: object, ctx: CallbackContext) -> None:
    if isinstance(ctx.error, Conflict):
        return
    logger.error("Unhandled exception", exc_info=ctx.error)

async def clear_state(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

# === 5) Утилиты ===
def get_top_symbols(n=TOP_LIMIT):
    tickers = exchange.fetch_tickers()
    usdt = [s for s in tickers if s.endswith("/USDT")]
    return sorted(
        usdt,
        key=lambda s: tickers[s].get("quoteVolume", 0),
        reverse=True
    )[:n]

def fetch_ohlcv(symbol):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# === 6) Стратегии ===
def detect_breakout(symbol, df):
    resistance = df["high"].rolling(VOLUME_WINDOW).max().iloc[-2]
    support    = df["low"].rolling(VOLUME_WINDOW).min().iloc[-2]
    last, prev = df.iloc[-1], df.iloc[-2]
    avg_vol    = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    entry      = last["close"]
    sl         = entry * STOP_LOSS_RATIO
    tp         = entry * TAKE_PROFIT_RATIO

    # LONG
    if prev["close"] < resistance and entry > resistance and last["volume"] > avg_vol:
        return (
            f"🚀 [Breakout LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp:.6f}`\n"
            f"SL:    `{sl:.6f}`"
        )
    # SHORT (TP ниже входа, SL выше входа)
    if prev["close"] > support and entry < support and last["volume"] > avg_vol:
        tp_s = entry * STOP_LOSS_RATIO
        sl_s = entry * TAKE_PROFIT_RATIO
        return (
            f"💥 [Breakout SHORT] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp_s:.6f}`\n"
            f"SL:    `{sl_s:.6f}`"
        )
    return None

def detect_rsi_ma_volume(symbol, df):
    df["ema"]     = ta.trend.ema_indicator(df["close"], EMA_WINDOW)
    df["rsi"]     = ta.momentum.RSIIndicator(df["close"], RSI_WINDOW).rsi()
    df["avg_vol"] = df["volume"].rolling(VOLUME_WINDOW).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    entry = last["close"]
    sl    = entry * STOP_LOSS_RATIO
    tp    = entry * TAKE_PROFIT_RATIO

    # LONG
    if last["rsi"] < 30 and prev["close"] < prev["ema"] and entry > last["ema"] and last["volume"] > last["avg_vol"]:
        return (
            f"📈 [RSI+MA+Vol LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp:.6f}`\n"
            f"SL:    `{sl:.6f}`"
        )
    # SHORT
    if last["rsi"] > 70 and prev["close"] > prev["ema"] and entry < last["ema"] and last["volume"] > last["avg_vol"]:
        tp_s = entry * STOP_LOSS_RATIO
        sl_s = entry * TAKE_PROFIT_RATIO
        return (
            f"📉 [RSI+MA+Vol SHORT] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp_s:.6f}`\n"
            f"SL:    `{sl_s:.6f}`"
        )
    return None

def detect_ema_vwap_stochrsi(symbol, df):
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    vp            = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    vwap_val      = vp.iloc[-1]
    avg_vol       = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    last, prev    = df.iloc[-1], df.iloc[-2]
    entry         = last["close"]
    sl            = entry * STOP_LOSS_RATIO
    tp            = entry * TAKE_PROFIT_RATIO

    # StochRSI
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs  = up.rolling(STOCHRSI_LEN).mean() / down.rolling(STOCHRSI_LEN).mean()
    rsi = 100 - (100 / (1 + rs))
    mn  = rsi.rolling(STOCHRSI_LEN).min()
    mx  = rsi.rolling(STOCHRSI_LEN).max()
    st  = (rsi - mn) / (mx - mn) * 100
    k   = st.rolling(STOCHRSI_K).mean().iloc[-1]
    d   = st.rolling(STOCHRSI_D).mean().iloc[-1]

    cross      = prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    above_vwap = entry > vwap_val
    vol_ok     = last["volume"] > avg_vol
    stoch_ok   = (k > d and k < 20)

    if cross and above_vwap and vol_ok and stoch_ok:
        return (
            f"🛡 [EMA/VWAP/StochRSI LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp:.6f}`\n"
            f"SL:    `{sl:.6f}`"
        )
    return None

# === 7) Основная задача ===
async def check_for_signals(ctx: ContextTypes.DEFAULT_TYPE):
    syms = get_top_symbols()
    logger.info(f"Scanning {len(syms)} symbols; first={syms[0]}")
    for s in syms:
        df = fetch_ohlcv(s)
        for strat in STRATEGIES:
            if strat == "breakout":
                msg = detect_breakout(s, df)
            elif strat == "rsi_ma_volume":
                msg = detect_rsi_ma_volume(s, df)
            else:
                msg = detect_ema_vwap_stochrsi(s, df)
            if msg:
                for cid in subscribers:
                    await ctx.bot.send_message(cid, msg)

# === 8) Запуск бота ===
def main() -> None:
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(clear_state)
        .build()
    )
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    subscribers.add(int(CHAT_ID))
    app.job_queue.run_repeating(check_for_signals, interval=CHECK_INTERVAL, first=10)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
