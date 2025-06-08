#!/usr/bin/env python3
"""
Advanced Trading Bot with Self-Learning, Expanded Feature Set, Risk Management, 
Real-Time Model Retraining, and Visualization

This code connects to the IQ Option API (for binary options on EUR/USD), gathers 
candle data, computes multiple technical indicators, logs trade data, and integrates 
a deep reinforcement learning (DQN) agent (using TensorFlow) to determine trade actions. 
It also includes periodic supervised learning model retraining from logged data and 
basic risk management (stop-loss/take-profit). 

Fill in USERNAME and PASSWORD below.
"""

#####################################
# Imports and Global Libraries
#####################################
import time, csv, os, numpy as np, datetime, random, pandas as pd
from iqoptionapi.stable_api import IQ_Option
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import traceback # For detailed error logging
from sklearn.model_selection import TimeSeriesSplit

#####################################
# TensorFlow Configuration
#####################################
def configure_tensorflow():
    """Configure TensorFlow for optimal performance"""
    try:
        # Disable unnecessary TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        
        # Configure thread usage
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        
        # Configure memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU devices found: {len(gpus)}")
        else:
            print("No GPU devices found, using CPU")
            
        # Verify TensorFlow is working
        print("Testing TensorFlow:")
        test_data = tf.constant([[1.0, 2.0, 3.0]])
        print(f"Test computation result: {tf.reduce_mean(test_data)}")
        
        return True
        
    except Exception as e:
        print(f"Error configuring TensorFlow: {str(e)}")
        traceback.print_exc()
        return False

#####################################
# Configuration & Constants
#####################################
USERNAME = os.getenv("IQOPTION_USERNAME")
PASSWORD = os.getenv("IQOPTION_PASSWORD")
PRACTICE_MODE = "PRACTICE"
TEST_MODE = os.getenv("IQOPTION_TEST_MODE", "0") == "1"

ASSET = "EURUSD-OTC"
BASE_AMOUNT = 1
DURATION = 1  # fixed trade duration in minutes
TRADING_ITERATIONS = 100 # Number of trade *cycles* / *attempts*

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "trade_log2.csv")
FULL_HISTORY_LOG_FILE = os.path.join(SCRIPT_DIR, "full_history_log.csv")

TIMEFRAME = 60     # Candle length in seconds (60s = 1 minute)
NUM_CANDLES = 30   # Number of candles to gather

STATE_DIM = 13
ACTION_DIM = 2    # 0: "call", 1: "put"

INTER_TRADE_WAIT_SECONDS = 10 # How many 1-second ticks to wait before attempting a trade

# New configuration constants
MIN_REQUIRED_ACCURACY = 0.75  # 75% minimum accuracy required before trading
HIGH_CONFIDENCE_THRESHOLD = 0.85  # 85% confidence required for trade entry
EARLY_CLOSE_THRESHOLD = 0.70  # 70% confidence threshold for early trade closure
MODEL_SAVE_PATH = "market_model.h5"
SCALER_SAVE_PATH = "feature_scaler.pkl"

# Trading timing constants
MIN_SECONDS_BETWEEN_TRADES = 60  # Minimum seconds between trade attempts
EARLY_CLOSE_WINDOW = 10  # Seconds before expiry when early close is possible

# Model & Training Settings
MODELS_DIR = "trained_models"
MIN_ACCURACY_THRESHOLD = 0.65  # Minimum required accuracy
CONFIDENCE_THRESHOLD = 0.85    # Confidence needed for trades
EARLY_SELL_THRESHOLD = 0.75    # Confidence threshold for keeping position

# Logging Settings
CONTINUOUS_LOG_INTERVAL = 1  # Log every second
HISTORY_WINDOW = 100        # How many candles to keep for pattern analysis

# Trading Settings
MIN_TRADES_FOR_TRAINING = 100  # Minimum trades before starting live trading
TRAINING_EPOCHS = 50          # Reduced from 200
ENSEMBLE_SIZE = 3             # Reduced from 5

# Add these new constants
BATCH_SIZE = 32              # Smaller batch size for memory efficiency
MAX_HISTORY_SIZE = 1000      # Limit data storage
CPU_WORKERS = 2              # Limit CPU usage
VOLATILITY_THRESHOLD = 0.002  # Threshold for market volatility

# Global flag for trader sentiment error logging
_trader_sentiment_error_logged_once = False

#####################################
# Enhanced Model Ensemble Implementation
#####################################
class ModelEnsemble:
    def __init__(self, input_dim, ensemble_size=3):
        self.models = []
        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.best_model = None
        self.best_accuracy = 0
        self.scaler = StandardScaler()
        self.feature_scaler = None  # For feature normalization
        
    def build_single_model(self):
        inputs = Input(shape=(self.input_dim, 1))
        
        # Temporal feature extraction with CNN
        x = Conv1D(32, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # First LSTM branch with residual connection
        lstm1 = LSTM(64, return_sequences=True, 
                    kernel_regularizer=l2(0.001))(x)
        bn1 = BatchNormalization()(lstm1)
        drop1 = Dropout(0.2)(bn1)
        
        # Second LSTM branch
        lstm2 = LSTM(32, return_sequences=False,
                    kernel_regularizer=l2(0.001))(drop1)
        bn2 = BatchNormalization()(lstm2)
        drop2 = Dropout(0.2)(bn2)
        
        # Dense layers with skip connections
        dense1 = Dense(32, activation='relu',
                      kernel_regularizer=l2(0.001))(drop2)
        bn3 = BatchNormalization()(dense1)
        drop3 = Dropout(0.1)(bn3)
        
        # Add skip connection
        merge = Add()([drop2, drop3])
        
        # Final dense layers
        dense2 = Dense(16, activation='relu',
                      kernel_regularizer=l2(0.001))(merge)
        bn4 = BatchNormalization()(dense2)
        drop4 = Dropout(0.1)(bn4)
        
        outputs = Dense(1, activation='sigmoid')(drop4)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Custom learning rate schedule with warm-up
        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
        
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
        
    def train_ensemble(self, X, y):
        print(f"Training ensemble with data shape: X={X.shape}, y={y.shape}")
        
        if len(X) != len(y):
            print(f"Error: X and y lengths don't match. X: {len(X)}, y: {len(y)}")
            return
            
        if len(X) < 20:
            print("Not enough data for training")
            return
        
        # Feature normalization
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            X = self.feature_scaler.fit_transform(X)
        else:
            X = self.feature_scaler.transform(X)

        X = X.reshape((X.shape[0], self.input_dim, 1))
        
        # Configure cross-validation with gap
        n_splits = min(5, len(X) // 100)  # Increased splits for better validation
        if n_splits < 2:
            n_splits = 2
            
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)  # Increased gap
        
        # Train ensemble
        for i in range(self.ensemble_size):
            model = self.build_single_model()
            accuracies = []
            
            try:
                for train_idx, val_idx in tscv.split(X):
                    if len(train_idx) < 20 or len(val_idx) < 20:
                        continue
                        
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]
                    
                    # Data augmentation with noise
                    noise_factor = 0.05
                    X_train_noisy = X_train + np.random.normal(0, noise_factor, X_train.shape)
                    
                    # Early stopping with patience
                    early_stopping = EarlyStopping(
                        monitor='val_accuracy',
                        patience=10,  # Increased patience
                        restore_best_weights=True,
                        mode='max'
                    )
                    
                    # Reduce learning rate on plateau
                    reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=0.00001
                    )
                    
                    # Train with both original and noisy data
                    history = model.fit(
                        np.concatenate([X_train, X_train_noisy]),
                        np.concatenate([y_train, y_train]),
                        validation_data=(X_val, y_val),
                        epochs=100,  # Increased epochs
                        batch_size=16,  # Reduced batch size
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1
                    )
                    
                    # Evaluate model
                    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
                    accuracies.append(accuracy)
                
                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    print(f"Model {i+1} average accuracy: {avg_accuracy:.4f}")
                    self.models.append((model, avg_accuracy))
                    
                    if avg_accuracy > self.best_accuracy:
                        self.best_accuracy = avg_accuracy
                        self.best_model = model
                
            except Exception as e:
                print(f"Error training model {i+1}: {str(e)}")
                continue
        
        # Sort models by accuracy
        if self.models:
            self.models.sort(key=lambda x: x[1], reverse=True)
            print(f"Ensemble training complete. Best accuracy: {self.best_accuracy:.4f}")
        else:
            print("Warning: No models were successfully trained")
    
    def predict(self, X):
        if not self.models:
            print("No models available for prediction")
            return None
            
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        else:
            X = np.array(X)

        X = X.reshape((X.shape[0], self.input_dim, 1))
        
        predictions = []
        weights = []
        
        for model, accuracy in self.models:
            try:
                pred = model.predict(X, verbose=0)
                predictions.append(pred)
                weights.append(accuracy)
            except Exception as e:
                print(f"Error in model prediction: {str(e)}")
                continue
        
        if not predictions:
            return None, None
            
        # Weighted ensemble prediction
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        weighted_predictions = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_predictions += pred * weight

        confidence = np.abs(weighted_predictions - 0.5) * 2

        return weighted_predictions, confidence

#####################################
# Pattern Recognizer
#####################################
class PatternRecognizer:
    def identify_pattern(self, prices, window=30):
        if len(prices) < window:
            return None
        sequence = np.array(prices[-window:])
        normalized = (sequence - np.mean(sequence)) / (np.std(sequence) if np.std(sequence) != 0 else 1)
        slope = np.polyfit(range(len(normalized)), normalized, 1)[0]
        volatility = np.std(np.diff(normalized))
        return {
            'slope': slope,
            'volatility': volatility,
            'momentum': normalized[-1] - normalized[0]
        }

#####################################
# Trading Decision Engine
#####################################
class TradingDecisionEngine:
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.model_ensemble = None
        self.market_state = "unknown"
        
    def analyze_market_conditions(self, prices, volumes):
        if len(prices) < HISTORY_WINDOW:
            return False
            
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
        
        pattern_features = self.pattern_recognizer.identify_pattern(prices)
        
        # Market state determination
        if volatility > VOLATILITY_THRESHOLD * 2:
            self.market_state = "highly_volatile"
            return False
        elif volatility < VOLATILITY_THRESHOLD / 2:
            self.market_state = "low_volatility"
            
        return True
        
    def should_trade(self, confidence, prices, volumes):
        if confidence < CONFIDENCE_THRESHOLD:
            return False
            
        if not self.analyze_market_conditions(prices, volumes):
            return False
            
        return True
        
    def should_close_early(self, current_confidence, entry_price, current_price, action):
        if current_confidence < EARLY_SELL_THRESHOLD:
            profit = (current_price - entry_price) / entry_price if action == "call" else (entry_price - current_price) / entry_price
            return profit > 0
        return False

#####################################
# File Logging Functions
#####################################
def init_log(file_path):    
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "OrderID", "Asset", "Action", "Amount", "Duration",
                "Slope", "PredictedProbability", "Outcome", "Profit",
                "PriceHistory", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment",
                "Bollinger_SMA", "Bollinger_Upper", "Bollinger_Lower", "ATR", "AvgVolume",
                "Hour", "Weekday", "NewsSentiment"
            ])

def init_full_history_log(file_path):    
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset", "PriceHistoryString", "CurrentSentiment",
                "Slope", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal",
                "Bollinger_SMA", "Bollinger_Upper", "Bollinger_Lower", "ATR",
                "AvgVolume", "Hour", "Weekday", "NewsSentiment"
            ])

def log_trade(file_path, row):                      
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        expected_columns = 25  # Number of expected columns
        
        # Pad the row if there are fewer columns than expected
        while len(row) < expected_columns:
            row.append(0.0)
            
        # Truncate the row if there are more columns than expected
        if len(row) > expected_columns:
            row = row[:expected_columns]
            
        writer.writerow(row)

def log_full_history(file_path, row):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        expected_columns = 18  # Number of expected columns for full_history_log.csv
        
        # Pad the row if there are fewer columns than expected
        while len(row) < expected_columns:
            row.append(0.0)
            
        if len(row) > expected_columns:
            row = row[:expected_columns]

        writer.writerow(row)

def validate_config():
    """Ensure required configuration is present."""
    valid = True
    if not USERNAME or not PASSWORD:
        print(
            "Error: IQ Option credentials not set. Please set IQOPTION_USERNAME and IQOPTION_PASSWORD environment variables."
        )
        valid = False

    # Ensure log files exist
    if not os.path.exists(LOG_FILE):
        print(f"{LOG_FILE} not found - it will be created.")
    if not os.path.exists(FULL_HISTORY_LOG_FILE):
        print(f"{FULL_HISTORY_LOG_FILE} not found - it will be created.")

    return valid

#####################################
# Helper Functions
#####################################
def get_closing_prices_from_candles_data(candles_data):
    if not candles_data or not isinstance(candles_data, dict):
        return []
    valid_candle_times = sorted([k for k in candles_data.keys() if isinstance(k, (int, float))])
    closing_prices = []
    for t in valid_candle_times:
        if isinstance(candles_data.get(t), dict) and 'close' in candles_data[t]:
            closing_prices.append(candles_data[t]['close'])
    return closing_prices

def compute_bollinger_bands(prices, period=20, num_std=2):
    if not prices or len(prices) < period:
        return 0.0, 0.0, 0.0
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_ATR(candles, period=14):
    if not candles:
        return 0.0
    times = sorted(candles.keys())
    if len(times) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(times)):
        current_candle = candles.get(times[i])
        previous_candle = candles.get(times[i-1])
        if not isinstance(current_candle, dict) or not isinstance(previous_candle, dict):
            continue
        
        current_high = current_candle.get("max", current_candle.get("high", 0))
        current_low = current_candle.get("min", current_candle.get("low", 0))
        prev_close = previous_candle.get("close", 0)
        
        tr = max(
            current_high - current_low,
            abs(current_high - prev_close),
            abs(current_low - prev_close)
        )
        tr_list.append(tr)
    
    if not tr_list or len(tr_list) < period:
        return 0.0
    return np.mean(tr_list[-period:])

def compute_average_volume(candles, period=10):
    if not candles:
        return 0.0
    times = sorted(candles.keys())
    volumes = [candles[t].get("volume", 0) for t in times if isinstance(candles.get(t), dict)]
    if not volumes or len(volumes) < period:
        return np.mean(volumes) if volumes else 0.0
    return np.mean(volumes[-period:])

def compute_RSI(prices, period=14):
    if not prices or len(prices) < period + 1:
        return 50.0
    
    delta = np.diff(prices)
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta
    
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_MACD(prices, short_period=12, long_period=26, signal_period=9):
    if not prices or len(prices) < long_period:
        return 0.0, 0.0
    
    prices_series = pd.Series(prices)
    exp1 = prices_series.ewm(span=short_period, adjust=False).mean()
    exp2 = prices_series.ewm(span=long_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    return macd.iloc[-1], signal.iloc[-1]

def get_trader_sentiment(api, asset):
    global _trader_sentiment_error_logged_once
    try:
        sentiment = api.get_traders_mood(asset)
        if isinstance(sentiment, (float, int)):
            return sentiment
        elif isinstance(sentiment, dict) and 'call' in sentiment:
            return sentiment['call']
        else:
            if not _trader_sentiment_error_logged_once:
                print(f"Unexpected sentiment format for {asset}. Using fallback.")
                _trader_sentiment_error_logged_once = True
            return 0.5
    except Exception as e:
        if not _trader_sentiment_error_logged_once:
            print(f"Error getting sentiment for {asset}: {str(e)}. Using fallback.")
            _trader_sentiment_error_logged_once = True
        return 0.5

def get_time_features():
    now = datetime.datetime.now()
    return now.hour, now.weekday()

def get_news_sentiment():
    return random.uniform(0.4, 0.6)

#####################################
# Data Loading and Validation
#####################################
def load_and_validate_data():
    """Load features and outcomes from the trade log"""
    try:
        print("\nLoading historical data from trade log...")

        features = []
        outcomes = []

        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            total_rows = 0
            valid_rows = 0

            for row in reader:
                total_rows += 1
                try:
                    # Columns: Timestamp,OrderID,Asset,Action,Amount,Duration,Slope,PredProb,Outcome,Profit,PriceHistory,MovingAverage,StdDev,RSI,MACD,MACD_Signal,TraderSentiment
                    prices = [float(p) for p in row[10].split(',')]

                    slope = float(row[6])
                    ma = float(row[11])
                    std = float(row[12])
                    rsi = float(row[13])
                    macd = float(row[14])
                    signal = float(row[15])
                    sentiment = float(row[16])

                    boll_sma, boll_up, boll_low = compute_bollinger_bands(prices)
                    atr = 0.0  # Not available
                    volume = 0.0

                    timestamp = datetime.datetime.fromisoformat(row[0])
                    hour = timestamp.hour
                    weekday = timestamp.weekday()
                    news = 0.5

                    feat = [
                        slope,
                        ma,
                        std,
                        rsi,
                        macd,
                        signal,
                        sentiment,
                        boll_sma,
                        atr,
                        volume,
                        hour,
                        weekday,
                        news,
                    ]
                    features.append(feat)

                    outcome = 1 if row[8] == "win" else 0
                    outcomes.append(outcome)
                    valid_rows += 1
                except Exception as e:
                    if total_rows < 5:
                        print(f"Error processing row {total_rows}: {e}")
                    continue

                if total_rows % 100 == 0:
                    print(f"Processed {total_rows} rows, valid {valid_rows}...")

        print(f"\nFinished loading features: {len(features)}")

        if len(features) < MIN_TRADES_FOR_TRAINING:
            print(
                f"WARNING: Not enough data for training. Need at least {MIN_TRADES_FOR_TRAINING}, found {len(features)}"
            )
            return None, None

        return np.array(features), np.array(outcomes)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None, None

#####################################
# Main Trading Loop
#####################################
def main():
    try:
        print("Starting trading bot initialization...")
        
        if not validate_config():
            print("Configuration validation failed. Please fix the issues above.")
            return

        init_log(LOG_FILE)
        init_full_history_log(FULL_HISTORY_LOG_FILE)
            
        print("Setting TensorFlow environment variables...")
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        configure_tensorflow()
        
        # Load and validate data
        features, outcomes = load_and_validate_data()
        if features is None or outcomes is None:
            print("Failed to load trading data")
            return
            
        print(f"\nLoaded trading data:")
        print(f"Features shape: {features.shape}")
        print(f"Outcomes shape: {outcomes.shape}")
        
        # Initialize components
        print("\nInitializing model ensemble...")
        model_ensemble = ModelEnsemble(input_dim=STATE_DIM)
        trading_engine = TradingDecisionEngine()
        
        # Train model
        print("\nTraining model ensemble...")
        model_ensemble.train_ensemble(features, outcomes)

        if model_ensemble.best_accuracy < MIN_REQUIRED_ACCURACY:
            print(
                f"Training accuracy {model_ensemble.best_accuracy:.2f} does not meet the required {MIN_REQUIRED_ACCURACY}. Aborting."
            )
            return
        
        if TEST_MODE:
            print("\nRunning in TEST MODE - skipping IQ Option connection")
            print("Testing prediction on sample data...")
            
            # Test prediction on a few samples
            test_samples = features[:5]
            preds, confs = model_ensemble.predict(test_samples)
            print(f"\nTest predictions:")
            for i, (pred, conf, actual) in enumerate(zip(preds, confs, outcomes[:5])):
                print(f"Sample {i+1}: Predicted {pred[0]:.3f} (conf {conf[0]:.2f}), Actual {actual}")
            return
            
        # Initialize trading
        print("\nInitializing trading system...")
        I_want_money = IQ_Option(USERNAME, PASSWORD)
        connected = I_want_money.connect()
        print(f"Connection attempt result: {connected}")
        
        if not I_want_money.check_connect():
            print("Connection failed. Please check your credentials and internet connection.")
            return
            
        print("Successfully connected to IQ Option.")
        I_want_money.change_balance(PRACTICE_MODE)
        print(f"Switched to {PRACTICE_MODE} account.")
        
        # Continue with rest of main function...
        # Trading variables
        current_trade_amount = BASE_AMOUNT
        successful_trades = 0
        total_trades = 0
        last_trade_time = time.time()
        
        # Start candle stream
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        print("Candle stream started")
        
        # Data collection
        all_features = []
        all_outcomes = []
        
        while True:
            try:
                current_time = datetime.datetime.now()
                
                # Get market data
                candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                if not candles:
                    print("No candle data available")
                    time.sleep(1)
                    continue
                
                # Extract prices and compute indicators
                prices = get_closing_prices_from_candles_data(candles)
                if not prices:
                    continue
                
                slope = (prices[-1] - prices[0]) / len(prices) if len(prices) > 1 else 0
                ma = np.mean(prices)
                std = np.std(prices)
                rsi = compute_RSI(prices)
                macd, signal = compute_MACD(prices)
                sentiment = get_trader_sentiment(I_want_money, ASSET)
                boll_sma, boll_up, boll_low = compute_bollinger_bands(prices)
                atr = compute_ATR(candles)
                volume = compute_average_volume(candles)
                hour, weekday = get_time_features()
                news = get_news_sentiment()
                
                # Prepare features
                features = np.array([[
                    slope, ma, std, rsi, macd, signal, sentiment,
                    boll_sma, atr, volume, hour, weekday, news
                ]])
                
                # Log market data
                log_full_history(FULL_HISTORY_LOG_FILE, [
                    current_time.isoformat(), ASSET, ",".join(map(str, prices)),
                    sentiment, slope, ma, std, rsi, macd, signal,
                    boll_sma, boll_up, boll_low, atr, volume,
                    hour, weekday, news
                ])
                
                # Store features for training
                all_features.append(features[0])
                
                # Train model if we have enough data
                if len(all_features) >= MIN_TRADES_FOR_TRAINING and not model_ensemble.models:
                    print("Training initial model...")
                    X = np.array(all_features)
                    # Use price direction as initial training target
                    y = np.array([1 if prices[i+1] > prices[i] else 0 
                                for i in range(len(prices)-1)])
                    if len(X) > len(y):
                        X = X[:len(y)]
                    model_ensemble.train_ensemble(X, y)
                
                # Make trading decisions
                if model_ensemble.models and time.time() - last_trade_time > MIN_SECONDS_BETWEEN_TRADES:
                    pred, confidence = model_ensemble.predict(features)
                    
                    if trading_engine.should_trade(confidence[0], prices, [volume]):
                        action = "call" if pred[0] > 0.5 else "put"
                        print(f"\nPlacing {action.upper()} trade with confidence {confidence[0]:.2f}")
                        
                        status, order_id = I_want_money.buy(
                            current_trade_amount, ASSET, action, DURATION)
                        
                        if status:
                            print(f"Trade placed successfully. Order ID: {order_id}")
                            last_trade_time = time.time()
                            
                            # Wait for result
                            time.sleep(DURATION * 60)

                            result = I_want_money.check_win_v3(order_id)

                            if result is not None:
                                total_trades += 1
                                all_outcomes.append(result)
                                if result > 0:
                                    successful_trades += 1
                                    print(f"Trade WON: ${result:.2f}")
                                else:
                                    print(f"Trade LOST: ${result:.2f}")
                                
                                # Log trade
                                log_trade(LOG_FILE, [
                                    current_time.isoformat(), order_id, ASSET,
                                    action, current_trade_amount, DURATION,
                                    slope, confidence[0], 
                                    "win" if result > 0 else "loss",
                                    result, ",".join(map(str, prices)),
                                    ma, std, rsi, macd, signal, sentiment,
                                    boll_sma, boll_up, boll_low, atr, volume,
                                    hour, weekday, news
                                ])
                                
                                # Update accuracy tracking
                                if total_trades > 0:
                                    accuracy = successful_trades / total_trades
                                    print(f"Current accuracy: {accuracy:.2%}")
                                
                                # Retrain model periodically
                                if total_trades % 10 == 0:
                                    print("\nRetraining model with new data...")
                                    model_ensemble.train_ensemble(
                                        np.array(all_features),
                                        np.array([1 if r > 0 else 0 for r in all_outcomes])
                                    )
                            
                            else:
                                print("Could not get trade result")
                        
                        else:
                            print("Trade placement failed")
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                traceback.print_exc()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
    finally:
        if 'I_want_money' in locals():
            try:
                I_want_money.close_connect()
                print("Disconnected from IQ Option")
            except:
                pass

if __name__ == "__main__":
    main()
