"""
How to run
----------
# Install dependencies:
pip install tradingview_ta iqoptionapi scikit-learn xgboost pandas joblib

# Run:
python iqo18.py
"""

import os
import time
import logging
import datetime
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from tradingview_ta import TA_Handler, Interval
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

try:
    from iqoptionapi.stable_api import IQ_Option
except Exception:  # pragma: no cover - library optional for local test
    IQ_Option = None


@dataclass
class Config:
    """Central configuration for the trading bot."""

    # API Credentials
    username: str = os.getenv("IQOPTION_USERNAME", "demo@example.com")
    password: str = os.getenv("IQOPTION_PASSWORD", "password")
    practice: bool = True

    # Trading settings
    pair: str = "EURUSD"
    amount: float = 1.0
    duration: int = 1  # minutes

    # Data & model
    interval: Interval = Interval.INTERVAL_5_MINUTES
    interval_seconds: int = 300
    data_dir: str = "data"
    model_file: str = "model.pkl"
    confidence_threshold: float = 0.85
    retrain_hours: int = 6


config = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def fetch_tradingview_data() -> dict:
    """Fetch OHLCV and indicators from TradingView and store them."""
    handler = TA_Handler(
        symbol=config.pair,
        screener="forex",
        exchange="FX_IDC",
        interval=config.interval,
    )
    analysis = handler.get_analysis()

    timestamp = int(datetime.datetime.utcnow().timestamp())
    row = {
        "timestamp": timestamp,
        "open": analysis.indicators.get("open"),
        "high": analysis.indicators.get("high"),
        "low": analysis.indicators.get("low"),
        "close": analysis.indicators.get("close"),
        "volume": analysis.indicators.get("volume"),
        "RSI": analysis.indicators.get("RSI"),
        "MACD": analysis.indicators.get("MACD.macd"),
        "MACD_signal": analysis.indicators.get("MACD.signal"),
    }

    os.makedirs(config.data_dir, exist_ok=True)
    file_path = os.path.join(
        config.data_dir, datetime.datetime.utcnow().strftime("%Y-%m-%d") + ".csv"
    )
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["SMA"] = df["close"].rolling(window=9).mean()
    df["WMA"] = df["close"].rolling(window=9).apply(
        lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / sum(range(1, len(x) + 1)),
        raw=True,
    )

    df.to_csv(file_path, index=False)

    latest = df.iloc[-1]
    return {
        "timestamp": int(latest["timestamp"]),
        "ohlcv": {
            "open": latest["open"],
            "high": latest["high"],
            "low": latest["low"],
            "close": latest["close"],
            "volume": latest["volume"],
        },
        "indicators": {
            "RSI": latest["RSI"],
            "MACD": latest["MACD"],
            "EMA_9": latest["EMA_9"],
            "EMA_21": latest["EMA_21"],
            "WMA": latest["WMA"],
            "SMA": latest["SMA"],
        },
    }


def load_historical_data() -> pd.DataFrame:
    """Load all stored CSV files as a single DataFrame."""
    frames = []
    if not os.path.isdir(config.data_dir):
        return pd.DataFrame()
    for fname in os.listdir(config.data_dir):
        if fname.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(config.data_dir, fname)))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def train_model() -> None:
    """Train a binary classifier to predict next price movement."""
    df = load_historical_data()
    if df.empty or len(df) < 50:
        logger.warning("Not enough data for training")
        return

    df = df.dropna().reset_index(drop=True)
    df["rsi_diff"] = df["RSI"].diff()
    df["ema_diff"] = df["EMA_9"] - df["EMA_21"]
    df["close_diff"] = df["close"].diff()
    df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
    df["dayofweek"] = pd.to_datetime(df["timestamp"], unit="s").dt.dayofweek
    df["crossover"] = (df["EMA_9"] > df["EMA_21"]).astype(int)

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    features = [
        "RSI",
        "MACD",
        "EMA_9",
        "EMA_21",
        "WMA",
        "SMA",
        "rsi_diff",
        "ema_diff",
        "close_diff",
        "hour",
        "dayofweek",
        "crossover",
    ]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Training accuracy: %.4f", acc)

    dump(model, config.model_file)
    logger.info("Model saved to %s", config.model_file)


def predict_and_confidence(latest_record: dict) -> Tuple[str, float]:
    """Predict direction and compute combined confidence score."""
    if not os.path.exists(config.model_file):
        logger.warning("Model file not found")
        return "up", 0.0

    model = load(config.model_file)

    ind = latest_record["indicators"]
    ts = latest_record["timestamp"]
    features = pd.DataFrame([
        {
            "RSI": ind["RSI"],
            "MACD": ind["MACD"],
            "EMA_9": ind["EMA_9"],
            "EMA_21": ind["EMA_21"],
            "WMA": ind["WMA"],
            "SMA": ind["SMA"],
            "rsi_diff": 0.0,
            "ema_diff": ind["EMA_9"] - ind["EMA_21"],
            "close_diff": 0.0,
            "hour": datetime.datetime.utcfromtimestamp(ts).hour,
            "dayofweek": datetime.datetime.utcfromtimestamp(ts).weekday(),
            "crossover": int(ind["EMA_9"] > ind["EMA_21"]),
        }
    ])

    prob_up = model.predict_proba(features)[0][1]

    summary = TA_Handler(
        symbol=config.pair,
        screener="forex",
        exchange="FX_IDC",
        interval=config.interval,
    ).get_analysis().summary
    rec = summary.get("RECOMMENDATION", "NEUTRAL")
    if "BUY" in rec:
        tv_score = 1.0
    elif "SELL" in rec:
        tv_score = 0.0
    else:
        tv_score = 0.5

    if prob_up >= 0.5:
        direction = "up"
        confidence = 0.7 * prob_up + 0.3 * tv_score
    else:
        direction = "down"
        confidence = 0.7 * (1 - prob_up) + 0.3 * (1 - tv_score)

    logger.info(
        "Model prob_up=%.4f TV_score=%.4f -> %s confidence=%.4f",
        prob_up,
        tv_score,
        direction,
        confidence,
    )
    return direction, confidence


def execute_trade(direction: str, confidence: float) -> None:
    """Execute a trade on IQ Option if confidence threshold is met."""
    if confidence < config.confidence_threshold:
        logger.info("No trade: confidence %.2f below threshold", confidence)
        return

    if IQ_Option is None:
        logger.error("iqoptionapi not installed; cannot trade")
        return

    try:
        iq = IQ_Option(config.username, config.password)
        iq.connect()
        if config.practice:
            iq.change_balance("PRACTICE")
        status, trade_id = iq.buy(config.amount, config.pair, direction, config.duration)
        if status:
            logger.info("Trade placed id=%s direction=%s", trade_id, direction)
        else:
            logger.error("Trade placement failed: %s", trade_id)
        iq.close()
    except Exception as e:
        logger.error("Trade execution error: %s", e)


def model_needs_retraining() -> bool:
    if not os.path.exists(config.model_file):
        return True
    mtime = os.path.getmtime(config.model_file)
    return (time.time() - mtime) / 3600 > config.retrain_hours


def main() -> None:
    while True:
        try:
            if model_needs_retraining():
                logger.info("Retraining model...")
                train_model()

            record = fetch_tradingview_data()
            direction, confidence = predict_and_confidence(record)
            execute_trade(direction, confidence)
            time.sleep(config.interval_seconds)
        except KeyboardInterrupt:
            logger.info("Graceful shutdown requested")
            break
        except Exception as e:
            logger.error("Loop error: %s", e)
            time.sleep(config.interval_seconds)


def self_test() -> bool:
    """Run a simple startup test of all major components."""
    try:
        rec = fetch_tradingview_data()
        logger.info("TradingView fetch OK")
    except Exception as e:
        logger.error("TradingView fetch failed: %s", e)
        return False

    try:
        train_model()
        logger.info("Model training routine OK")
    except Exception as e:
        logger.error("Model training failed: %s", e)
        return False

    try:
        predict_and_confidence(rec)
        logger.info("Model load & predict OK")
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return False

    if IQ_Option is not None:
        try:
            iq = IQ_Option(config.username, config.password)
            iq.connect()
            authorized = iq.check_connect()
            iq.close()
            if authorized:
                logger.info("IQ Option authentication OK")
            else:
                logger.error("IQ Option authentication failed")
                return False
        except Exception as e:
            logger.error("IQ Option authentication error: %s", e)
            return False
    else:
        logger.warning("iqoptionapi not available; skipping auth test")

    return True


if __name__ == "__main__":
    if self_test():
        main()

    # Checklist
    # [x] TradingView OHLCV + indicators fetch
    # [x] Appending data to daily storage
    # [x] Feature engineering & model training
    # [x] Model serialization & loading
    # [x] Combined model + indicator confidence logic
    # [x] 85% confidence threshold enforcement
    # [x] IQ Option trade execution with error handling
    # [x] Configurable parameters section
    # [x] Logging of all steps & errors
    # [x] Self-test routine on startup
    # [x] Graceful shutdown on interrupt
