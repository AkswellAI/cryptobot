import os
import logging
import ccxt
import pandas as pd
import ta
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# === 0) Загрузка переменных окружения  ===
TOKEN            = os.getenv("TELEGRAM_TOKEN")
CHAT_ID          = os.getenv("CHAT_ID")
BINANCE_API_KEY  = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

if not all([TOKEN, CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise RuntimeError("Please set TELEGRAM_TOKEN, CHAT_ID, BINANCE_API_KEY and BINANCE_API_SECRET env vars")

# === 1) Логирование ===
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === 2) Инициализация Binance через CCXT ===
exchange = ccxt.binance({
    'apiKey':    BINANCE_API_KEY,
    'secret':    BINANCE_API_SECRET,
})

# === 3) Параметры стратегий и общие константы ===
TIMEFRAME          = '5m'
LIMIT              = 100
STRATEGIES         = ['breakout', 'rsi_ma_volume', 'ema_vwap_stochrsi']
STOP_LOSS_RATIO    = 0.99   # 1% SL
TAKE_PROFIT_RATIO  = 1.02   # 2% TP
VOLUME_WINDOW      = 20
EMA_WINDOW         = 21
RSI_WINDOW         = 14
EMA_FAST           = 9
EMA_SLOW           = 21
STOCHRSI_LEN       = 14
STOCHRSI_K         = 3
STOCHRSI_D         = 3
VWAP_WINDOW        = 100
TOP_LIMIT          = 200

# === 4) Подписка пользователей ===
subscribers = set()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    await update.message.reply_text(
        "✅ Subscribed to signals:\n" + "\n".join(f"– {s}" for s in STRATEGIES)
    )

# === 5) Утилиты: топ-200 пар и загрузка OHLCV ===
def get_top_symbols(limit=TOP_LIMIT):
    tickers = exchange.fetch_tickers()
    usdt = [s for s in tickers if s.endswith('/USDT')]
    sorted_ = sorted(
        usdt,
        key=lambda s: tickers[s].get('quoteVolume', 0),
        reverse=True
    )
    return sorted_[:limit]

def fetch_ohlcv(symbol):
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

# === 6) Стратегии ===
def detect_breakout(symbol, df):
    res = df['high'].rolling(VOLUME_WINDOW).max().iloc[-2]
    sup = df['low' ].rolling(VOLUME_WINDOW).min().iloc[-2]
    last, prev = df.iloc[-1], df.iloc[-2]
    avg_vol = df['volume'].rolling(VOLUME_WINDOW).mean().iloc[-1]
    entry = last['close']
    sl = entry * STOP_LOSS_RATIO
    tp = entry * TAKE_PROFIT_RATIO

    # LONG
    if prev['close'] < res and entry > res and last['volume'] > avg_vol:
        return (
            f"🚀 [Breakout LONG] {symbol} ({TIMEFRAME})\n"
            f"Level: {res:.6f}, Close: {entry:.6f}\n"
            f"Vol: {last['volume']:.0f} (avg {avg_vol:.0f})\n"
            f"Entry: `{entry:.6f}`  TP: `{tp:.6f}`\n\n"
            f"SL:    `{sl:.6f}`"
        )
    # SHORT
    if prev['close'] > sup and entry < sup and last['volume'] > avg_vol:
        sl_s = entry / STOP_LOSS_RATIO
        tp_s = entry * (2 - STOP_LOSS_RATIO)
        return (
            f"💥 [Breakout SHORT] {symbol} ({TIMEFRAME})\n"
            f"Level: {sup:.6f}, Close: {entry:.6f}\n"
            f"Vol: {last['volume']:.0f} (avg {avg_vol:.0f})\n"
            f"Entry: `{entry:.6f}`  TP: `{tp_s:.6f}`\n\n"
            f"SL:    `{sl_s:.6f}`"
        )
    return None

def detect_rsi_ma_volume(symbol, df):
    df['ema'] = ta.trend.ema_indicator(df['close'], EMA_WINDOW)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], RSI_WINDOW).rsi()
    df['avg_vol'] = df['volume'].rolling(VOLUME_WINDOW).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    entry = last['close']
    sl = entry * STOP_LOSS_RATIO
    tp = entry * TAKE_PROFIT_RATIO

    # LONG
    if last['rsi'] < 30 and prev['close'] < prev['ema'] and entry > last['ema'] and last['volume'] > last['avg_vol']:
        return (
            f"📈 [RSI+MA+Vol LONG] {symbol} ({TIMEFRAME})\n"
            f"RSI: {last['rsi']:.2f}, Price: {entry:.6f}\n"
            f"Entry: `{entry:.6f}`  TP: `{tp:.6f}`\n\n"
            f"SL:    `{sl:.6f}`"
        )
    # SHORT
    if last['rsi'] > 70 and prev['close'] > prev['ema'] and entry < last['ema'] and last['volume'] > last['avg_vol']:
        sl_s = entry / STOP_LOSS_RATIO
        tp_s = entry * (2 - STOP_LOSS_RATIO)
        return (
            f"📉 [RSI+MA+Vol SHORT] {symbol} ({TIMEFRAME})\n"
            f"RSI: {last['rsi']:.2f}, Price: {entry:.6f}\n"
            f"Entry: `{entry:.6f}`  TP: `{tp_s:.6f}`\n\n"
            f"SL:    `{sl_s:.6f}`"
        )
    return None

def detect_ema_vwap_stochrsi(symbol, df):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST,  adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW,  adjust=False).mean()
    vp = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    vwap_val = vp.iloc[-1]
    avg_vol  = df['volume'].rolling(VOLUME_WINDOW).mean().iloc[-1]
    last, prev = df.iloc[-1], df.iloc[-2]
    entry = last['close']
    sl = entry * STOP_LOSS_RATIO
    tp = entry * TAKE_PROFIT_RATIO

    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(STOCHRSI_LEN).mean() / down.rolling(STOCHRSI_LEN).mean()
    rsi = 100 - (100/(1+rs))
    min_rsi = rsi.rolling(STOCHRSI_LEN).min()
    max_rsi = rsi.rolling(STOCHRSI_LEN).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    k = stoch.rolling(STOCHRSI_K).mean().iloc[-1]
    d = stoch.rolling(STOCHRSI_D).mean().iloc[-1]

    cross      = prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']
    above_vwap = entry > vwap_val
    vol_ok     = last['volume'] > avg_vol
    stoch_ok   = (k > d and k < 20)

    if cross and above_vwap and vol_ok and stoch_ok:
        return (
            f"🛡 [EMA/VWAP/StochRSI LONG] {symbol} ({TIMEFRAME})\n\n"
            f"Entry: `{entry:.6f}`  TP: `{tp:.6f}`\n\n"
            f"SL:    `{sl:.6f}`\n"
            f"VWAP:  {vwap_val:.6f}  Vol: {last['volume']:.0f}/{avg_vol:.0f}\n"
            f"StochRSI: K={k:.1f} D={d:.1f}"
        )
    return None

# === 7) Основная асинхронная задача ===
async def check_for_signals(context: ContextTypes.DEFAULT_TYPE):
    for symbol in get_top_symbols():
        df = fetch_ohlcv(symbol)
        for strat in STRATEGIES:
            if strat == 'breakout':
                msg = detect_breakout(symbol, df)
            elif strat == 'rsi_ma_volume':
                msg = detect_rsi_ma_volume(symbol, df)
            else:
                msg = detect_ema_vwap_stochrsi(symbol, df)
            if msg:
                for chat_id in subscribers:
                    await context.bot.send_message(chat_id, msg)

# === 8) Запуск ===
def main() -> None:
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(clear_state)
        .build()
    )
    app.add_handler(CommandHandler('start', start))
    app.job_queue.run_repeating(check_for_signals, interval=300, first=10)
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
