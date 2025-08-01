import os
import time
import logging
import datetime
from typing import Optional

import pandas as pd
from joblib import dump, load
from xgboost import XGBClassifier

try:
    from iqoptionapi.stable_api import IQ_Option
except Exception:  # pragma: no cover
    IQ_Option = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(SCRIPT_DIR, 'iqoption and nautilis', 'trade_log2.csv')
HISTORY_LOG = os.path.join(SCRIPT_DIR, 'iqoption and nautilis', 'full_history_log.csv')
MARKET_DATA_LOG = os.path.join(SCRIPT_DIR, 'iqoption and nautilis', 'all_time_market_data_log.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.pkl')

MIN_SAMPLES = 50
CONFIDENCE_THRESHOLD = 0.85
PAIR = 'EURUSD'
AMOUNT = 1
DURATION = 1  # minutes

logger = logging.getLogger(__name__)

def _parse_price_history(history: str):
    try:
        prices = [float(x) for x in str(history).split(',') if x]
    except Exception:
        prices = []
    if not prices:
        return None, None, None, None, None, None
    open_p = prices[0]
    high_p = max(prices)
    low_p = min(prices)
    close_p = prices[-1]
    ma = sum(prices) / len(prices)
    std = pd.Series(prices).std()
    return open_p, high_p, low_p, close_p, ma, std

def load_labeled_data() -> pd.DataFrame:
    """Load and merge trade log with market history into labeled dataset."""
    logger.info("Loading trade/history logs...")
    trades = pd.read_csv(TRADE_LOG)
    history = pd.read_csv(HISTORY_LOG)

    trades['timestamp'] = pd.to_datetime(trades['Timestamp'])
    history['timestamp'] = pd.to_datetime(history['Timestamp'])

    trades.sort_values('timestamp', inplace=True)
    history.sort_values('timestamp', inplace=True)

    df = pd.merge_asof(trades, history, on='timestamp', direction='backward', suffixes=('', '_hist'))
    df['target'] = (df['Profit'] > 0).astype(int)

    features_cols = ['MovingAverage', 'StdDev', 'RSI', 'MACD', 'MACD_Signal', 'TraderSentiment']
    df.dropna(subset=features_cols, inplace=True)

    parsed = df['PriceHistory'].apply(_parse_price_history)
    df[['open', 'high', 'low', 'close', 'ma', 'std']] = pd.DataFrame(parsed.tolist(), index=df.index)

    return df

def train_model() -> Optional[XGBClassifier]:
    """Train model using labeled data."""
    df = load_labeled_data()
    if len(df) < MIN_SAMPLES:
        logger.warning("Not enough samples for training: %d", len(df))
        return None

    features = ['open', 'high', 'low', 'close', 'ma', 'std', 'RSI', 'MACD', 'MACD_Signal', 'TraderSentiment']
    df = df.dropna(subset=features + ['target'])

    X = df[features]
    y = df['target']

    logger.info("Training model on %d samples...", len(X))
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    dump(model, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    return model

def append_market_snapshot() -> dict:
    """Fetch a market snapshot via TradingView and append to log."""
    from tradingview_ta import TA_Handler, Interval

    handler = TA_Handler(symbol=PAIR, screener="forex", exchange="FX_IDC", interval=Interval.INTERVAL_1_MINUTE)
    analysis = handler.get_analysis()

    ts = datetime.datetime.utcnow().isoformat()
    row = {
        'Timestamp': ts,
        'Asset': PAIR,
        'LastClosePrice': analysis.indicators.get('close'),
        'MovingAverage': analysis.indicators.get('SMA'),
        'StdDev': analysis.indicators.get('STDDEV'),
        'RSI': analysis.indicators.get('RSI'),
        'MACD': analysis.indicators.get('MACD.macd'),
        'MACD_Signal': analysis.indicators.get('MACD.signal'),
        'TraderSentiment': None,
    }

    df_row = pd.DataFrame([row])
    if not os.path.exists(MARKET_DATA_LOG):
        df_row.to_csv(MARKET_DATA_LOG, index=False)
    else:
        df_row.to_csv(MARKET_DATA_LOG, mode='a', header=False, index=False)

    logger.info("Appended market snapshot to log")
    return row

def authenticate() -> Optional[object]:
    if IQ_Option is None:
        logger.warning("iqoptionapi not installed - running without live trading")
        return None
    username = os.getenv('IQOPTION_USERNAME')
    password = os.getenv('IQOPTION_PASSWORD')
    try:
        iq = IQ_Option(username, password)
        iq.connect()
        iq.change_balance("PRACTICE")
        logger.info("Authenticated with IQ Option as %s", username)
        return iq
    except Exception as e:
        logger.error("IQ Option authentication failed: %s", e)
        return None

def predict_and_trade(model, iq, features: pd.DataFrame) -> None:
    prob = model.predict_proba(features)[0][1]
    logger.info("Prediction probability=%.4f", prob)
    if prob >= CONFIDENCE_THRESHOLD:
        logger.info("Confidence above threshold: executing dry-run trade")
        if iq is not None:
            try:
                status, trade_id = iq.buy(AMOUNT, PAIR, 'call', DURATION)
                logger.info("API response: %s %s", status, trade_id)
            except Exception as e:
                logger.error("API buy error: %s", e)
        else:
            logger.info("Dry-run: would execute trade here")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    snapshot = append_market_snapshot()
    iq = authenticate()

    if not os.path.exists(MODEL_PATH):
        model = train_model()
        if model is None:
            logger.error("Model training failed; exiting")
            return
    else:
        model = load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)

    try:
        while True:
            snapshot = append_market_snapshot()
            feat_row = {
                'open': snapshot['LastClosePrice'],
                'high': snapshot['LastClosePrice'],
                'low': snapshot['LastClosePrice'],
                'close': snapshot['LastClosePrice'],
                'ma': snapshot.get('MovingAverage'),
                'std': snapshot.get('StdDev'),
                'RSI': snapshot.get('RSI'),
                'MACD': snapshot.get('MACD'),
                'MACD_Signal': snapshot.get('MACD_Signal'),
                'TraderSentiment': snapshot.get('TraderSentiment', 0.5),
            }
            features = pd.DataFrame([feat_row])
            predict_and_trade(model, iq, features)
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error("Runtime error: %s", e)
    finally:
        if iq is not None:
            iq.close()
        try:
            train_model()
        except Exception as e:
            logger.error("Retraining error: %s", e)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error("Unhandled error: %s", e)

# Checklist
# [x] Add load_labeled_data() helper to merge trade_log2.csv + full_history_log.csv
# [x] Update train_model() with sample check and joblib.save
# [x] Main routine loads or trains model and predicts
# [x] Model persisted and reloaded for predictions
# [x] Comprehensive logging for all steps
