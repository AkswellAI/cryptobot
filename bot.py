import os
import sys
import logging
import ccxt
import pandas as pd
import ta
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.error import Conflict
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackContext,
)

# === 0) Load env vars ===
TOKEN              = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("CHAT_ID")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
if not all([TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Set TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET")

# === 1) Logging: INFO→stdout, WARNING+→stderr ===
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

# === 2) CCXT Binance Futures ===
exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "options": {"defaultType": "future"},
})
exchange.load_markets()

# === 3) Strategy params ===
TIMEFRAME      = "1h"
LIMIT          = 100
LOSS_RATIO     = 0.01    # 1%
PROFIT_RATIO   = 0.025   # 2.5%
VOLUME_WINDOW  = 20
EMA_WINDOW     = 21
RSI_WINDOW     = 14
EMA_FAST       = 9
EMA_SLOW       = 21
STOCHRSI_LEN   = 14
STOCHRSI_K     = 3
STOCHRSI_D     = 3
TOP_LIMIT      = 200
CHECK_INTERVAL = 300     # 5 minutes

STRATEGIES     = ["breakout", "rsi_ma_volume", "ema_vwap_stochrsi"]

# State
subscribers  = set()
open_trades  = []  # list of dicts: {symbol, side, entry, sl, tp, strategy, opened_at}
daily_stats  = {"total": 0, "tp": 0, "sl": 0}

# === 4) Handlers ===
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    subscribers.add(update.effective_chat.id)
    await update.message.reply_text("✅ Subscribed to: " + ", ".join(STRATEGIES))

async def error_handler(update: object, ctx: CallbackContext) -> None:
    if isinstance(ctx.error, Conflict):
        return
    logger.error("Unhandled exception", exc_info=ctx.error)

async def clear_state(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

# === 5) Utilities ===
def get_top_symbols(n=TOP_LIMIT):
    tickers = exchange.fetch_tickers()
    usdt    = [s for s in tickers if "/USDT" in s]
    return sorted(usdt, key=lambda s: tickers[s].get("quoteVolume", 0), reverse=True)[:n]

def fetch_ohlcv(symbol):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df   = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["close"] = df["close"].astype(float)
    return df

# === 6) Detect functions return dict or None ===
def detect_breakout(symbol, df):
    res = df["high"].rolling(VOLUME_WINDOW).max().iloc[-2]
    sup = df["low"].rolling(VOLUME_WINDOW).min().iloc[-2]
    last, prev = df.iloc[-1], df.iloc[-2]
    avg_vol = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    entry = last["close"]
    sl    = entry * (1 - LOSS_RATIO)
    tp    = entry * (1 + PROFIT_RATIO)

    if prev["close"] < res and entry > res and last["volume"] > avg_vol:
        msg = (
            f"🚀 [Breakout LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp:.6f}`\n"
            f"SL:    `{sl:.6f}`"
        )
        return {"symbol": symbol, "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "strategy": "breakout", "msg": msg}
    if prev["close"] > sup and entry < sup and last["volume"] > avg_vol:
        tp_s = entry * (1 - PROFIT_RATIO)
        sl_s = entry * (1 + LOSS_RATIO)
        msg = (
            f"💥 [Breakout SHORT] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp_s:.6f}`\n"
            f"SL:    `{sl_s:.6f}`"
        )
        return {"symbol": symbol, "side": "SHORT", "entry": entry, "sl": sl_s, "tp": tp_s, "strategy": "breakout", "msg": msg}
    return None

def detect_rsi_ma_volume(symbol, df):
    df["ema"]     = ta.trend.ema_indicator(df["close"], EMA_WINDOW)
    df["rsi"]     = ta.momentum.RSIIndicator(df["close"], RSI_WINDOW).rsi()
    df["avg_vol"] = df["volume"].rolling(VOLUME_WINDOW).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    entry = last["close"]
    sl    = entry * (1 - LOSS_RATIO)
    tp    = entry * (1 + PROFIT_RATIO)

    if last["rsi"] < 30 and prev["close"] < prev["ema"] and entry > last["ema"] and last["volume"] > last["avg_vol"]:
        msg = (
            f"📈 [RSI+MA+Vol LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp:.6f}`\n"
            f"SL:    `{sl:.6f}`"
        )
        return {"symbol": symbol, "side": "LONG", "entry": entry, "sl": sl, "tp": tp, "strategy": "rsi_ma_volume", "msg": msg}
    if last["rsi"] > 70 and prev["close"] > prev["ema"] and entry < last["ema"] and last["volume"] > last["avg_vol"]:
        tp_s = entry * (1 - PROFIT_RATIO)
        sl_s = entry * (1 + LOSS_RATIO)
        msg = (
            f"📉 [RSI+MA+Vol SHORT] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp_s:.6f}`\n"
            f"SL:    `{sl_s:.6f}`"
        )
        return {"symbol": symbol, "side": "SHORT", "entry": entry, "sl": sl_s, "tp": tp_s, "strategy": "rsi_ma_volume", "msg": msg}
    return None

def detect_ema_vwap_stochrsi(symbol, df):
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    vp = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    vwap_val = vp.iloc[-1]
    avg_vol  = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    last, prev = df.iloc[-1], df.iloc[-2]
    entry = last["close"]
    sl    = entry * (1 - LOSS_RATIO)
    tp    = entry * (1 + PROFIT_RATIO)

    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs  = up.rolling(STOCHRSI_LEN).mean() / down.rolling(STOCHRSI_LEN).mean()
    rsi = 100 - (100/(1+rs))
    mn  = rsi.rolling(STOCHRSI_LEN).min()
    mx  = rsi.rolling(STOCHRSI_LEN).max()
    st  = (rsi - mn)/(mx - mn)*100
    k   = st.rolling(STOCHRSI_K).mean().iloc[-1]
    d   = st.rolling(STOCHRSI_D).mean().iloc[-1]

    cross = prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
    above_vwap = entry > vwap_val
    vol_ok  = last["volume"] > avg_vol
    stoch_ok = (k > d and k < 20)

    if cross and above_vwap and vol_ok and stoch_ok:
        msg = (
            f"🛡 [EMA/VWAP/StochRSI LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`\n"
            f"TP:    `{tp:.6f}`\n"
            f"SL:    `{sl:.6f}`"
        )
        return {"symbol": symbol, "side":"LONG", "entry": entry, "sl": sl, "tp": tp, "strategy":"ema_vwap_stochrsi", "msg": msg}
    return None

# === 7) Main job: signals, managing open_trades, and daily summary ===
async def check_for_signals(ctx: ContextTypes.DEFAULT_TYPE):
    symbols = get_top_symbols()
    if not symbols:
        logger.warning("No symbols, skipping")
        return
    for s in symbols:
        df = fetch_ohlcv(s)
        for strat in STRATEGIES:
            detect = globals()[f"detect_{strat}"]
            result = detect(s, df)
            if not result:
                continue
            # send entry signal
            for cid in subscribers:
                await ctx.bot.send_message(cid, result["msg"])
            # record open trade
            open_trades.append({
                **result,
                "opened_at": datetime.now(),
            })
            daily_stats["total"] += 1

    # check open trades for SL/TP
    still_open = []
    for t in open_trades:
        df2 = fetch_ohlcv(t["symbol"])
        price = df2["close"].iloc[-1]
        hit_tp = (t["side"]=="LONG"  and price>=t["tp"]) or (t["side"]=="SHORT" and price<=t["tp"])
        hit_sl = (t["side"]=="LONG"  and price<=t["sl"]) or (t["side"]=="SHORT" and price>=t["sl"])
        if hit_tp or hit_sl:
            kind = "TP" if hit_tp else "SL"
            daily_stats["tp" if hit_tp else "sl"] += 1
            msg2 = (
                f"🛑 [Trade CLOSED – {kind}] {t['symbol']} ({t['strategy']})\n"
                f"Side: {t['side']}, Entry: {t['entry']:.6f}, "
                f"{kind}@{price:.6f}\n"
                f"Opened at: {t['opened_at'].strftime('%Y-%m-%d %H:%M')}"
            )
            for cid in subscribers:
                await ctx.bot.send_message(cid, msg2)
        else:
            still_open.append(t)
    open_trades[:] = still_open

async def send_daily_stats(ctx: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"📊 Daily Summary:\n"
        f"Total trades: {daily_stats['total']}\n"
        f"TP hit:       {daily_stats['tp']}\n"
        f"SL hit:       {daily_stats['sl']}"
    )
    for cid in subscribers:
        await ctx.bot.send_message(cid, msg)
    # reset for next day
    daily_stats.update(total=0, tp=0, sl=0)

# === 8) Start bot ===
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

    jq = app.job_queue
    jq.run_repeating(check_for_signals, interval=CHECK_INTERVAL, first=10)
    # daily summary at 09:00 Europe/Berlin
    run_time = dtime(hour=9, minute=0, tzinfo=ZoneInfo("Europe/Berlin"))
    jq.run_daily(send_daily_stats, time=run_time)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
