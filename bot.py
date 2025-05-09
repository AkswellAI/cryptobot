import os
import sys
import json
import logging
import ccxt
import pathlib
import pandas as pd
import ta
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# === 0) ENV & constants ===
TOKEN              = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("CHAT_ID")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
if not all([TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Set TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET")

DATA_FILE = pathlib.Path("trades.json")
TIMEFRAME = "1h"
LIMIT     = 100
LOSS_RATIO   = 0.01    # 1%
PROFIT_RATIO = 0.025   # 2.5%
VOLUME_WINDOW= 20
EMA_WINDOW   = 21
RSI_WINDOW   = 14
EMA_FAST     = 9
EMA_SLOW     = 21
STOCHRSI_LEN = 14
STOCHRSI_K   = 3
STOCHRSI_D   = 3
TOP_LIMIT    = 200
CHECK_INTERVAL = 300   # 5min

STRATEGIES = ["breakout", "rsi_ma_volume", "ema_vwap_stochrsi"]

# === 1) Logging setup ===
root = logging.getLogger()
root.setLevel(logging.INFO)
stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.INFO)
stderr = logging.StreamHandler(sys.stderr)
stderr.setLevel(logging.WARNING)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
stdout.setFormatter(fmt)
stderr.setFormatter(fmt)
root.handlers = [stdout, stderr]
logger = logging.getLogger(__name__)

# === 2) CCXT futures client ===
exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "options": {"defaultType": "future"},
})
exchange.load_markets()

# === 3) Persistence ===
def load_trades():
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    return []

def save_trades(trades):
    DATA_FILE.write_text(json.dumps(trades))

open_trades = load_trades()
daily_stats = {"total":0,"tp":0,"sl":0}

# === 4) Handlers ===
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    update.message.reply_text("✅ Subscribed.")
    # nothing else: we push to CHAT_ID

async def clear_webhook(app):
    await app.bot.delete_webhook(drop_pending_updates=True)

# === 5) Utils ===
def get_top_symbols(n=TOP_LIMIT):
    tickers = exchange.fetch_tickers()
    usdt = [s for s in tickers if s.endswith("/USDT")]
    byvol = sorted(usdt, key=lambda s: tickers[s].get("quoteVolume",0), reverse=True)
    return byvol[:n]

def fetch_ohlcv(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    return df

def fetch_price(symbol):
    t = exchange.fetch_ticker(symbol)
    return float(t["last"])

# === 6) Strategies ===
def detect_breakout(symbol, df):
    res = df["high"].rolling(VOLUME_WINDOW).max().iloc[-2]
    sup = df["low"].rolling(VOLUME_WINDOW).min().iloc[-2]
    last,prev = df.iloc[-1], df.iloc[-2]
    entry = last["close"]
    sl = entry*(1-LOSS_RATIO)
    tp = entry*(1+PROFIT_RATIO)
    if prev["close"]<res and entry>res and last["volume"]>df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]:
        msg = (f"🚀 [Breakout LONG] {symbol}\n"
               f"Entry: `{entry:.6f}`\nTP:    `{tp:.6f}`\nSL:    `{sl:.6f}`")
        return {"symbol":symbol,"side":"LONG","entry":entry,"sl":sl,"tp":tp,"msg":msg}
    if prev["close"]>sup and entry<sup and last["volume"]>df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]:
        sl2=entry*(1+LOSS_RATIO)
        tp2=entry*(1-PROFIT_RATIO)
        msg = (f"💥 [Breakout SHORT] {symbol}\n"
               f"Entry: `{entry:.6f}`\nTP:    `{tp2:.6f}`\nSL:    `{sl2:.6f}`")
        return {"symbol":symbol,"side":"SHORT","entry":entry,"sl":sl2,"tp":tp2,"msg":msg}
    return None

def detect_rsi_ma_volume(symbol, df):
    df["ema"] = ta.trend.ema_indicator(df["close"],EMA_WINDOW)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"],RSI_WINDOW).rsi()
    df["avg_vol"] = df["volume"].rolling(VOLUME_WINDOW).mean()
    last,prev = df.iloc[-1],df.iloc[-2]
    entry=last["close"]; sl=entry*(1-LOSS_RATIO); tp=entry*(1+PROFIT_RATIO)
    if last["rsi"]<30 and prev["close"]<prev["ema"] and entry>last["ema"] and last["volume"]>last["avg_vol"]:
        msg=(f"📈 [RSI+MA+Vol LONG] {symbol}\n"
             f"Entry: `{entry:.6f}`\nTP:    `{tp:.6f}`\nSL:    `{sl:.6f}`")
        return {"symbol":symbol,"side":"LONG","entry":entry,"sl":sl,"tp":tp,"msg":msg}
    if last["rsi"]>70 and prev["close"]>prev["ema"] and entry<last["ema"] and last["volume"]>last["avg_vol"]:
        sl2=entry*(1+LOSS_RATIO); tp2=entry*(1- PROFIT_RATIO)
        msg=(f"📉 [RSI+MA+Vol SHORT] {symbol}\n"
             f"Entry: `{entry:.6f}`\nTP:    `{tp2:.6f}`\nSL:    `{sl2:.6f}`")
        return {"symbol":symbol,"side":"SHORT","entry":entry,"sl":sl2,"tp":tp2,"msg":msg}
    return None

def detect_ema_vwap_stochrsi(symbol,df):
    df["ema_f"] = df["close"].ewm(EMA_FAST).mean()
    df["ema_s"] = df["close"].ewm(EMA_SLOW).mean()
    vp = (df["close"]*df["volume"]).cumsum()/df["volume"].cumsum()
    vwap = vp.iloc[-1]
    avgv= df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    last,prev = df.iloc[-1],df.iloc[-2]
    entry=last["close"]; sl=entry*(1-LOSS_RATIO); tp=entry*(1+PROFIT_RATIO)
    # stochRSI
    delta=df["close"].diff(); up,down=delta.clip(lower=0),-delta.clip(upper=0)
    rs=up.rolling(STOCHRSI_LEN).mean()/down.rolling(STOCHRSI_LEN).mean()
    rsi=100-100/(1+rs)
    mn, mx = rsi.rolling(STOCHRSI_LEN).min(), rsi.rolling(STOCHRSI_LEN).max()
    st = (rsi-mn)/(mx-mn)*100
    k = st.rolling(STOCHRSI_K).mean().iloc[-1]
    d = st.rolling(STOCHRSI_D).mean().iloc[-1]
    cross= prev["ema_f"]<prev["ema_s"] and last["ema_f"]>last["ema_s"]
    if cross and entry>vwap and last["volume"]>avgv and k>d and k<20:
        msg=(f"🛡 [EMA/VWAP/StochRSI LONG] {symbol}\n"
             f"Entry: `{entry:.6f}`\nTP:    `{tp:.6f}`\nSL:    `{sl:.6f}`")
        return {"symbol":symbol,"side":"LONG","entry":entry,"sl":sl,"tp":tp,"msg":msg}
    return None

# === 7) Job ===
async def check_for_signals(ctx: ContextTypes.DEFAULT_TYPE):
    syms = get_top_symbols()
    for s in syms:
        df = fetch_ohlcv(s)
        for strat in STRATEGIES:
            dfn = globals()[f"detect_{strat}"](s,df)
            if not dfn: continue
            # send entry
            await ctx.bot.send_message(int(CHAT_ID), dfn["msg"])
            open_trades.append({**dfn, "opened_at": datetime.utcnow().isoformat()})
            daily_stats["total"]+=1
            save_trades(open_trades)
    # check SL/TP on live price
    still = []
    for t in open_trades:
        price = fetch_price(t["symbol"])
        hit_tp = (t["side"]=="LONG"  and price>=t["tp"]) or \
                 (t["side"]=="SHORT" and price<=t["tp"])
        hit_sl = (t["side"]=="LONG"  and price<=t["sl"]) or \
                 (t["side"]=="SHORT" and price>=t["sl"])
        if hit_tp or hit_sl:
            kind="TP" if hit_tp else "SL"
            daily_stats["tp" if hit_tp else "sl"]+=1
            txt=(f"🛑 [Trade CLOSED – {kind}] {t['symbol']}\n"
                 f"Entry: `{t['entry']:.6f}`  {kind}@`{price:.6f}`")
            await ctx.bot.send_message(int(CHAT_ID), txt)
        else:
            still.append(t)
    open_trades[:] = still
    save_trades(open_trades)

async def send_daily(ctx):
    msg=(f"📊 Daily summary:\n"
         f"Total: {daily_stats['total']}\n"
         f"TP:    {daily_stats['tp']}\n"
         f"SL:    {daily_stats['sl']}")
    await ctx.bot.send_message(int(CHAT_ID), msg)
    daily_stats.update(total=0,tp=0,sl=0)

# === 8) Main ===
def main():
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(clear_webhook)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    jq = app.job_queue
    jq.run_repeating(check_for_signals, interval=CHECK_INTERVAL, first=10)
    run_time = dtime(9,0, tzinfo=ZoneInfo("Europe/Berlin"))
    jq.run_daily(send_daily, time=run_time)
    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
