import logging
import ccxt
import pandas as pd
import numpy as np
import ta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ====== Настройки ======
TOKEN            = '7996074288:AAFO-OBnXEd0KBdddDmVLEWBwIHFLjd6Z5Q'
TIMEFRAME        = '5m'
LIMIT            = 100

# Настройки стратегий
VOLUME_MULTIPLIER = 1.2   # для breakout
WINDOW            = 20    # для breakout и среднего объёма
EMA_WINDOW        = 21    # для RSI+MA+Vol
RSI_WINDOW        = 14
VOLUME_WINDOW     = 20
EMA_FAST          = 9     # для EMA9/21+VWAP+StochRSI
EMA_SLOW          = 21
STOCHRSI_LEN      = 14
STOCHRSI_K        = 3
STOCHRSI_D        = 3

STRATEGIES = ['breakout', 'rsi_ma_volume', 'ema_vwap_stochrsi']
subscribers = set()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# инициализация биржи
exchange = ccxt.binance({
    'apiKey': 'kMw0fQo3EE14MBBpkjGd2ripowlH10S4jaWs8sKF3gnRjY7uklS6QatoZ5Cp6cx',
    'secret': 'Wei80Y2PWGsuI56Pr68sqMCDKYZv0fwxWkU1zZo60QEMKoBe9aA6VcCIDQJrAjc0',
})

# ====== Хэндлер команды /start ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    await update.message.reply_text(
        '✅ Подписались на сигналы стратегий:\n' + '\n'.join(f'– {s}' for s in STRATEGIES)
    )

# ====== Топ-200 пар по объёму ======
def get_top_symbols(limit: int = 200) -> list[str]:
    tickers = exchange.fetch_tickers()
    usdt_pairs = [s for s in tickers.keys() if s.endswith('/USDT')]
    sorted_pairs = sorted(
        usdt_pairs,
        key=lambda s: tickers[s].get('quoteVolume', 0),
        reverse=True
    )
    return sorted_pairs[:limit]

# ====== Загрузка OHLCV ======
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ====== Стратегии ======
def detect_breakout(symbol: str, df: pd.DataFrame) -> str | None:
    resistance = df['high'].rolling(WINDOW).max().iloc[-2]
    support    = df['low'].rolling(WINDOW).min().iloc[-2]
    last, prev = df.iloc[-1], df.iloc[-2]
    avg_vol    = df['volume'].rolling(WINDOW).mean().iloc[-1]
    entry      = last['close']
    sl         = entry * 0.99
    tp         = entry * 1.02
    if prev['close'] < resistance and entry > resistance and last['volume'] > avg_vol * VOLUME_MULTIPLIER:
        return (
            f"🚀 [Breakout LONG] {symbol} ({TIMEFRAME})\n"
            f"Level: {resistance:.6f}, Close: {entry:.6f}\n"
            f"Entry: `{entry:.6f}`  TP: `{tp:.6f}`\n\n"
            f"SL: `{sl:.6f}`"
        )
    if prev['close'] > support and entry < support and last['volume'] > avg_vol * VOLUME_MULTIPLIER:
        sl_short = entry * 1.01
        tp_short = entry * 0.98
        return (
            f"💥 [Breakout SHORT] {symbol} ({TIMEFRAME})\n"
            f"Level: {support:.6f}, Close: {entry:.6f}\n"
            f"Entry: `{entry:.6f}`  TP: `{tp_short:.6f}`\n\n"
            f"SL: `{sl_short:.6f}`"
        )
    return None

def detect_rsi_ma_volume(symbol: str, df: pd.DataFrame) -> str | None:
    df['ema']     = ta.trend.ema_indicator(df['close'], window=EMA_WINDOW)
    df['rsi']     = ta.momentum.RSIIndicator(df['close'], window=RSI_WINDOW).rsi()
    df['avg_vol'] = df['volume'].rolling(VOLUME_WINDOW).mean()
    last, prev   = df.iloc[-1], df.iloc[-2]
    entry        = last['close']
    sl           = entry * 0.99
    tp           = entry * 1.02
    long_cond = (
        last['rsi'] < 30 and prev['close'] < prev['ema'] and
        entry > last['ema'] and last['volume'] > last['avg_vol']
    )
    if long_cond:
        return (
            f"📈 [RSI+MA+Vol LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`  TP: `{tp:.6f}`\n\n"
            f"SL: `{sl:.6f}`"
        )
    short_cond = (
        last['rsi'] > 70 and prev['close'] > prev['ema'] and
        entry < last['ema'] and last['volume'] > last['avg_vol']
    )
    if short_cond:
        sl_short = entry * 1.01
        tp_short = entry * 0.98
        return (
            f"📉 [RSI+MA+Vol SHORT] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`  TP: `{tp_short:.6f}`\n\n"
            f"SL: `{sl_short:.6f}`"
        )
    return None

def detect_ema_vwap_stochrsi(symbol: str, df: pd.DataFrame) -> str | None:
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    vp = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    vwap_val      = vp.iloc[-1]
    avg_vol       = df['volume'].rolling(WINDOW).mean().iloc[-1]
    last, prev    = df.iloc[-1], df.iloc[-2]
    entry         = last['close']
    sl            = entry * 0.99
    tp            = entry * 1.02
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(STOCHRSI_LEN).mean() / down.rolling(STOCHRSI_LEN).mean()
    rsi = 100 - (100/(1+rs))
    min_rsi = rsi.rolling(STOCHRSI_LEN).min()
    max_rsi = rsi.rolling(STOCHRSI_LEN).max()
    stoch   = (rsi - min_rsi)/(max_rsi - min_rsi)*100
    k       = stoch.rolling(STOCHRSI_K).mean().iloc[-1]
    d       = stoch.rolling(STOCHRSI_D).mean().iloc[-1]
    cross      = prev['ema_fast'] < prev['ema_slow'] and last['ema_fast'] > last['ema_slow']
    above_vwap = entry > vwap_val
    vol_ok     = last['volume'] > avg_vol
    stoch_ok   = (k > d and k < 20)
    if cross and above_vwap and vol_ok and stoch_ok:
        return (
            f"🛡 [EMA/VWAP/StochRSI LONG] {symbol} ({TIMEFRAME})\n"
            f"Entry: `{entry:.6f}`  TP: `{tp:.6f}`\n\n"
            f"SL: `{sl:.6f}`\n"
            f"VWAP: {vwap_val:.6f}  Vol: {last['volume']:.0f}/{avg_vol:.0f}\n"
            f"StochRSI: K={k:.1f} D={d:.1f}"
        )
    return None

async def check_for_signals(context: ContextTypes.DEFAULT_TYPE) -> None:
    symbols = get_top_symbols(200)
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol)
            for strat in STRATEGIES:
                if strat == 'breakout':
                    msg = detect_breakout(symbol, df)
                elif strat == 'rsi_ma_volume':
                    msg = detect_rsi_ma_volume(symbol, df)
                elif strat == 'ema_vwap_stochrsi':
                    msg = detect_ema_vwap_stochrsi(symbol, df)
                else:
                    continue
                if msg:
                    for chat_id in subscribers:
                        await context.bot.send_message(chat_id=chat_id, text=msg)
        except Exception as e:
            logger.error(f"Error on {symbol} [{strat}]: {e}")

def main() -> None:
    # Clear any previous webhooks and pending updates on startup
async def clear_webhook(app):
    # Remove webhook (if set) and clear pending updates
    await app.bot.delete_webhook(drop_pending_updates=True)

app = (
    ApplicationBuilder()
    .token(TOKEN)
    .post_init(clear_webhook)
    .build()
)
    app.add_handler(CommandHandler('start', start))
    app.job_queue.run_repeating(check_for_signals, interval=300, first=10)
    app.run_polling()

if __name__ == '__main__':
    main()
