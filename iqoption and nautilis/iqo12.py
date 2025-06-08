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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import traceback # For detailed error logging
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib

#####################################
# Configuration & Constants
#####################################
USERNAME = "eaglelab23@gmail.com"      # <<< --- !!! FILL YOUR IQ OPTION EMAIL HERE !!!
PASSWORD = "Polboti1@"          # <<< --- !!! FILL YOUR IQ OPTION PASSWORD HERE !!!
PRACTICE_MODE = "PRACTICE"

ASSET = "EURUSD-OTC"
BASE_AMOUNT = 1
DURATION = 1  # fixed trade duration in minutes
TRADING_ITERATIONS = 100 # Number of trade *cycles* / *attempts*

LOG_FILE = "trade_log2.csv"             # Main trade log with advanced indicators
FULL_HISTORY_LOG_FILE = "full_history_log.csv"  # Log for full price history and ALL indicators, logged continuously

TIMEFRAME = 60     # Candle length in seconds (60s = 1 minute)
NUM_CANDLES = 30   # Number of candles to gather

STATE_DIM = 13
ACTION_DIM = 2    # 0: "call", 1: "put"

INTER_TRADE_WAIT_SECONDS = 10 # How many 1-second ticks to wait before attempting a trade

# New configuration constants
MIN_REQUIRED_ACCURACY = 0.80  # 80% minimum accuracy required before trading
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
BATCH_SIZE = 128              # Smaller batch size for memory efficiency
MAX_HISTORY_SIZE = 1000      # Limit data storage
CPU_WORKERS = 2              # Limit CPU usage

# Global flag for trader sentiment error logging
_trader_sentiment_error_logged_once = False

#####################################
# File Logging Functions
#####################################
def init_log(file_path): # For trade_log2.csv
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

def init_full_history_log(file_path): # For full_history_log.csv (now with all indicators)
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset", "PriceHistoryString", "CurrentSentiment",
                "Slope", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal",
                "Bollinger_SMA", "Bollinger_Upper", "Bollinger_Lower", "ATR",
                "AvgVolume", "Hour", "Weekday", "NewsSentiment"
            ])

def log_trade(file_path, row):          # For trade_log2.csv
    """Append exactly one well-formed line (25 columns) to trade_log2.csv."""
    EXPECTED_COLS = 25                  # <-- header length, keep in ONE place
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        # --- pad or trim so the row ALWAYS has 25 items -------------------
        if len(row) < EXPECTED_COLS:
            row.extend([0.0] * (EXPECTED_COLS - len(row)))
        elif len(row) > EXPECTED_COLS:
            row = row[:EXPECTED_COLS]
        # ------------------------------------------------------------------

        writer.writerow(row)

def log_full_history(file_path, row): # For full_history_log.csv
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        expected_columns = 18  # Number of expected columns for full_history_log.csv

        # Pad the row if there are fewer columns than expected
        while len(row) < expected_columns:
            row.append(0.0)  # Pad with 0.0 (or None if you prefer)

        if len(row) != expected_columns:
            print(f"Warning: Skipping log entry due to inconsistent number of columns: {len(row)}")
            return  # Skip writing this row if columns don't match
        writer.writerow(row)


#####################################
# Helper to get closing prices safely
#####################################
def get_closing_prices_from_candles_data(candles_data):
    """Safely extracts and sorts closing prices from candle data."""
    if not candles_data or not isinstance(candles_data, dict):
        return []
    valid_candle_times = sorted([k for k in candles_data.keys() if isinstance(k, (int, float))])
    closing_prices = []
    for t in valid_candle_times:
        if isinstance(candles_data.get(t), dict) and 'close' in candles_data[t]:
            closing_prices.append(candles_data[t]['close'])
    return closing_prices

#####################################
# Additional Technical Indicator Functions (Kept as is from user's code)
#####################################
def compute_bollinger_bands(prices, period=20, num_std=2):
    if not prices or len(prices) < period:
        sma = np.mean(prices) if prices else 0.0
        return sma, sma, sma
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_ATR(candles, period=14): # Takes raw candle dict
    if not candles: return 0.0
    times = sorted(candles.keys())
    if len(times) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(times)):
        current_candle = candles.get(times[i])
        previous_candle = candles.get(times[i-1])
        if not isinstance(current_candle, dict) or not isinstance(previous_candle, dict):
            continue
            
        current_high = current_candle.get("max", current_candle.get("high", current_candle.get("close")))
        current_low = current_candle.get("min", current_candle.get("low", current_candle.get("close")))
        prev_close = previous_candle.get("close")

        if None in [current_high, current_low, prev_close]: continue # Skip if data missing

        tr = max(current_high - current_low, abs(current_high - prev_close), abs(current_low - prev_close))
        tr_list.append(tr)
    if not tr_list or len(tr_list) < period: return 0.0
    return np.mean(tr_list[-period:])


def compute_average_volume(candles, period=10): # Takes raw candle dict
    if not candles: return 0.0
    times = sorted(candles.keys())
    volumes = [candles[t].get("volume", 0) for t in times if isinstance(candles.get(t), dict)]
    if not volumes or len(volumes) < period:
        return np.mean(volumes) if volumes else 0.0
    return np.mean(volumes[-period:])

def get_time_features():
    now = datetime.datetime.now()
    return now.hour, now.weekday()

def get_news_sentiment(): # Placeholder
    return random.uniform(0.4, 0.6)

#####################################
# Existing Technical Indicator Functions (Kept as is from user's code)
#####################################
def compute_slope(candles): # Takes raw candle dict, uses closing prices
    closing_prices = get_closing_prices_from_candles_data(candles)
    if len(closing_prices) < 2:
        return 0.0
    x = np.arange(len(closing_prices))
    try:
        slope, _ = np.polyfit(x, closing_prices, 1)
        return slope
    except (np.linalg.LinAlgError, TypeError):
        return 0.0

def compute_MA(prices): # Takes list of prices
    if not prices: return 0.0
    return np.mean(prices)

def compute_STD(prices): # Takes list of prices
    if not prices: return 0.0
    return np.std(prices)

def compute_RSI(prices, period=14): # Takes list of prices
    if not prices or len(prices) < period + 1:
        return 50.0
    changes = np.diff(np.array(prices))
    if len(changes) < period: return 50.0

    gains = changes[changes > 0]
    losses = -changes[changes < 0] # losses are negative here

    avg_gain = np.mean(gains[:period]) if len(gains) >= period else (np.sum(gains) / period if len(gains) > 0 else 0)
    avg_loss = np.mean(np.abs(losses[:period])) if len(losses) >= period else (np.sum(np.abs(losses)) / period if len(losses) > 0 else 0)
    
    if avg_loss == 0: return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_EMA(prices, period): # Takes list of prices
    if not prices or len(prices) == 0: return [] 
    try:
        return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()
    except ImportError:
        if len(prices) < period: 
            return [np.mean(prices)] * len(prices) if prices else []
        
        ema_values = []
        sma = np.mean(prices[:period])
        ema_values.append(sma)
        multiplier = 2 / (period + 1.0)
        for price in prices[period:]: 
            ema_val = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema_val)
        return ema_values


def compute_MACD(prices, short_period=12, long_period=26, signal_period=9): # Takes list of prices
    if not prices or len(prices) < long_period:
        return 0.0, 0.0
    try:
        prices_series = pd.Series(prices)
        ema_short = prices_series.ewm(span=short_period, adjust=False).mean()
        ema_long = prices_series.ewm(span=long_period, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line.iloc[-1], signal_line.iloc[-1]
    except ImportError:
        ema12 = compute_EMA_simple_for_MACD(prices, short_period)
        ema26 = compute_EMA_simple_for_MACD(prices, long_period)
        
        if not ema12 or not ema26 or len(ema12) != len(ema26): 
            return 0.0, 0.0

        macd_values = np.array(ema12) - np.array(ema26)
        
        if len(macd_values) < signal_period:
            return macd_values[-1] if len(macd_values) > 0 else 0.0, np.mean(macd_values) if len(macd_values) > 0 else 0.0
        
        signal_line_values = compute_EMA_simple_for_MACD(list(macd_values), signal_period)
        return macd_values[-1], signal_line_values[-1] if signal_line_values else 0.0

def compute_EMA_simple_for_MACD(prices, period): # Ensures full length for MACD if pandas not used
    if not prices: return []
    ema = [0.0] * len(prices)
    if not ema: return []
    
    if len(prices) < period : 
        if prices:
            initial_avg = np.mean(prices)
            for i in range(len(prices)):
                if i == 0:
                    ema[i] = initial_avg
                else:
                    k = 2 / (min(i + 1, period) + 1)
                    ema[i] = prices[i] * k + ema[i-1] * (1-k)
            return ema
        else:
            return []

    ema[0] = prices[0]
    k = 2 / (period + 1.0)
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i-1] * (1-k)
    return ema

#####################################
# Trader Sentiment Function - FIX: Log error only once
#####################################
def get_trader_sentiment(api, asset):
    global _trader_sentiment_error_logged_once
    try:
        # NOTE: get_traders_mood is a hypothetical name and might be different in your library version.
        sentiment_data = api.get_traders_mood(asset) 
        if isinstance(sentiment_data, (float, int)):
             return sentiment_data
        elif isinstance(sentiment_data, dict) and 'call' in sentiment_data:
             return sentiment_data['call'] 
        else:
            if not _trader_sentiment_error_logged_once:
                print(f"Trader sentiment for {asset} in unexpected format: {sentiment_data}. Using fallback.")
                _trader_sentiment_error_logged_once = True
            return random.uniform(0.4, 0.6)
    except AttributeError: 
        if not _trader_sentiment_error_logged_once:
            print(f"API method for trader sentiment (e.g., get_traders_mood) not found for {asset}. Using fallback.")
            _trader_sentiment_error_logged_once = True
        return random.uniform(0.4, 0.6)
    except Exception as e:
        if not _trader_sentiment_error_logged_once:
            print(f"Real trader sentiment not available for {asset}, using fallback. Error: {e}")
            _trader_sentiment_error_logged_once = True
        return random.uniform(0.4, 0.6)

#####################################
# Advanced Prediction Function (Kept as is from user's code)
#####################################
def advanced_predict_probability(slope, ma, std, rsi, macd, macd_signal, trader_sentiment, action, boll_sma, news_sentiment):
    prob = 0.5
    if action == "call":
        prob += 0.15 if slope > 0 else -0.15
        prob += 0.10 if rsi < 70 else -0.10 
        prob += 0.10 if macd > macd_signal else -0.10
    elif action == "put":
        prob += 0.15 if slope < 0 else -0.15
        prob += 0.10 if rsi > 30 else -0.10 
        prob += 0.10 if macd < macd_signal else -0.10

    if trader_sentiment > 0.6: prob += 0.1
    elif trader_sentiment < 0.4: prob -= 0.1

    if news_sentiment > 0.6: prob += 0.05
    elif news_sentiment < 0.4: prob -= 0.05
    
    return max(0.01, min(0.99, prob)) 

#####################################
# Risk Management Functions (Kept as is from user's code)
#####################################
def is_stop_loss_triggered(current_price, entry_price, stop_loss_pct=0.02, action_type="call"):
    if entry_price == 0: return False 
    if action_type == "call":
        return (entry_price - current_price) / entry_price >= stop_loss_pct
    else: 
        return (current_price - entry_price) / entry_price >= stop_loss_pct


def is_take_profit_triggered(current_price, entry_price, take_profit_pct=0.05, action_type="call"):
    if entry_price == 0: return False
    if action_type == "call":
        return (current_price - entry_price) / entry_price >= take_profit_pct
    else:
        return (entry_price - current_price) / entry_price >= take_profit_pct

#####################################
# Adaptive Trade Sizing Function (Kept as is from user's code)
#####################################
def update_trade_amount(current_amount, outcome, predicted_prob, threshold=0.9):
    if outcome == "win" and predicted_prob >= threshold:
        return current_amount * 1.5
    else:
        return BASE_AMOUNT

#####################################
# RL Agent (Deep Q-Network using TensorFlow/Keras)
#####################################
class RLAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_dim,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q = self.model.predict(np.array([next_state]), verbose=0)[0]
            target += self.gamma * np.amax(next_q)
        target_f = self.model.predict(np.array([state]), verbose=0)
        target_f[0][action] = target 
        self.model.fit(np.array([state]), target_f, epochs=100, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#####################################
# Enhanced RL Agent with LSTM and Advanced Features
#####################################
class EnhancedRLAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.batch_size = 32
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_enhanced_model()
        self.target_model = self._build_enhanced_model()
        self.update_target_model()

    def _build_enhanced_model(self):
        model = Sequential([
            # First LSTM layer with regularization
            LSTM(128, input_shape=(self.state_dim,1), return_sequences=True,
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            
            Dense(self.action_dim, activation='linear')
        ])
        
        # Custom loss function with Huber loss
        def huber_loss(y_true, y_pred, clip_delta=1.0):
            error = y_true - y_pred
            cond = K.abs(error) <= clip_delta
            squared_loss = 0.5 * K.square(error)
            quadratic_loss = clip_delta * K.abs(error) - 0.5 * K.square(clip_delta)
            return K.mean(K.switch(cond, squared_loss, quadratic_loss))
        
        model.compile(
            optimizer=Adam(learning_rate=self.lr, clipnorm=1.0),
            loss=huber_loss,
            metrics=['accuracy']
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state = np.reshape(state, [1, self.state_dim, 1])
        q_values = self.model.predict(state, verbose=0)[0]
        
        # Add noise to q_values for exploration
        noise = np.random.normal(0, 0.1, self.action_dim)
        q_values += noise
        
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_dim, 1))
        next_states = np.zeros((self.batch_size, self.state_dim, 1))
        
        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = np.reshape(state, (self.state_dim, 1))
            next_states[i] = np.reshape(next_state, (self.state_dim, 1))
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, _, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Periodically update target network
        if len(self.memory) % 100 == 0:
            self.update_target_model()

#####################################
# Supervised Learning: Retrain Model from Logs - FIX: Added robust CSV reading
#####################################
def retrain_supervised_model(log_file, state_dim=STATE_DIM, action_dim=ACTION_DIM, epochs=123):
    if not os.path.exists(log_file) or os.path.getsize(log_file) < 100: 
        print(f"Log file {log_file} not found or too small for supervised retraining.")
        return None
    try:
        feature_cols_for_training = ["Slope", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment",
                                     "Bollinger_SMA", "ATR", "AvgVolume", "Hour", "Weekday", "NewsSentiment"]
        cols_to_read = feature_cols_for_training + ["Profit", "Outcome"]
        
        try:
            # FIX: Attempt to read CSV, skipping bad lines to prevent ParserError
            data = pd.read_csv(log_file, usecols=lambda c: c in cols_to_read, on_bad_lines='skip')
        except pd.errors.ParserError as pe:
            print(f"Pandas ParserError reading {log_file}: {pe}. The file might be corrupted.")
            return None
        except Exception as e_read:
            print(f"An unexpected error occurred while reading {log_file}: {e_read}")
            return None

        if data.empty:
            print(f"No valid data could be read from {log_file}.")
            return None

        if 'Outcome' not in data.columns or 'Profit' not in data.columns:
            print(f"Critical: 'Outcome' or 'Profit' column not found in {log_file} after reading. Cannot retrain.")
            return None
        
        data_filtered = data[data['Outcome'].isin(['win', 'loss'])].copy()

        if data_filtered.empty:
            print("No 'win' or 'loss' outcomes in log for retraining after filtering.")
            return None

        for col in feature_cols_for_training:
            if col not in data_filtered.columns:
                data_filtered[col] = 0.0 
        
        X = data_filtered[feature_cols_for_training].values.astype(np.float32)
        y = data_filtered["Profit"].values.astype(np.float32)

        if X.shape[0] < 10: 
            print(f"Not enough samples ({X.shape[0]}) for retraining.")
            return None

        model = Sequential()
        model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear')) 
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        print(f"Retraining supervised model with {X.shape[0]} samples...")
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=1, shuffle=True)
        
        model_save_path = "supervised_model.keras" 
        model.save(model_save_path)
        print(f"Supervised model retrained and saved to {model_save_path}.")
        return model
    except Exception as e:
        print(f"Error during supervised model retraining: {e}")
        traceback.print_exc()
        return None

#####################################
# Visualization and Monitoring Dashboard (Kept as is from user's code)
#####################################
def create_dashboard(log_file):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed, skipping dashboard creation.")
        return
    if not os.path.exists(log_file) or os.path.getsize(log_file) < 100:
        print(f"Log file {log_file} not found or empty, skipping dashboard creation.")
        return
    
    try:
        data = pd.read_csv(log_file, on_bad_lines="skip")
        if 'Timestamp' not in data.columns or 'Profit' not in data.columns:
            print("Dashboard creation: Required columns (Timestamp, Profit) not in log.")
            return

        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.sort_values('Timestamp', inplace=True)
        data['CumulativeProfit'] = data['Profit'].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Timestamp'], y=data['CumulativeProfit'], mode='lines+markers', name='Cumulative Profit'))
        fig.update_layout(title="Cumulative Profit Over Time", xaxis_title="Time", yaxis_title="Profit")
        
        dashboard_path = "dashboard.html"
        fig.write_html(dashboard_path)
        print(f"Dashboard created and saved to {dashboard_path}.")
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        traceback.print_exc()

#####################################
# Real-Time Model Retraining Function (Kept as is from user's code)
#####################################
def update_model_from_logs(): 
    print("Attempting to update supervised model from logs...")
    model = retrain_supervised_model(LOG_FILE) 
    if model:
        print("Supervised model update from logs complete.")
    else:
        print("Supervised model update from logs failed or no data.")
    return model 

#####################################
# Advanced Feature Engineering
#####################################
def compute_price_momentum(prices, period=14):
    """Compute price momentum indicator"""
    if len(prices) < period:
        return 0
    return (prices[-1] / prices[-period]) - 1

def compute_volatility(prices, period=20):
    """Compute price volatility"""
    if len(prices) < period:
        return 0
    return np.std(np.diff(prices[-period:]))

def detect_price_pattern(prices, window=20):
    """Detect common price patterns"""
    if len(prices) < window:
        return 0
    
    # Simple pattern detection (can be expanded)
    recent_prices = prices[-window:]
    price_changes = np.diff(recent_prices)
    
    # Detect trend reversals
    if all(price_changes[:window//2] < 0) and all(price_changes[window//2:] > 0):
        return 1  # Potential reversal from downtrend
    elif all(price_changes[:window//2] > 0) and all(price_changes[window//2:] < 0):
        return -1  # Potential reversal from uptrend
    return 0

#####################################
# Enhanced Market State Detection
#####################################
class MarketStateAnalyzer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.state_history = []
        
    def analyze_market_state(self, prices, volumes, volatility):
        """Determine current market state"""
        if len(prices) < self.window_size:
            return "unknown"
            
        vol = np.std(prices[-self.window_size:])
        avg_volume = np.mean(volumes[-self.window_size:])
        
        # Detect trending vs ranging market
        if vol > np.mean(self.state_history[-10:]) if self.state_history else 0:
            if np.all(np.diff(prices[-5:]) > 0):
                return "strong_uptrend"
            elif np.all(np.diff(prices[-5:]) < 0):
                return "strong_downtrend"
            return "volatile"
        else:
            return "ranging"
            
    def update(self, state):
        """Update market state history"""
        self.state_history.append(state)
        if len(self.state_history) > 100:  # Keep limited history
            self.state_history.pop(0)

#####################################
# Enhanced Model Architecture
#####################################
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
        
    def build_single_model(self):
        # Create model with functional API for better flexibility
        inputs = Input(shape=(self.input_dim, 1))
        
        # First LSTM layer
        x = LSTM(64, return_sequences=True, 
                kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Second LSTM layer
        x = LSTM(32, return_sequences=False, 
                kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(32, activation='relu',
                 kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use optimizer with learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
          optimizer = Adam(learning_rate=lr_schedule,
                       clipnorm=1.0)
        
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
            
        # Normalize features
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Configure cross-validation
        n_splits = min(3, len(X) // 20)
        if n_splits < 2:
            n_splits = 2
            
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=2)
        
        # Train ensemble
        for i in range(self.ensemble_size):
            model = self.build_single_model()
            accuracies = []
            
            try:
                for train_idx, val_idx in tscv.split(X):
                    if len(train_idx) < 10 or len(val_idx) < 10:
                        continue
                        
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_val = X[val_idx]
                    y_val = y[val_idx]
                    
                    # Early stopping callback
                    early_stopping = EarlyStopping(
                        monitor='val_accuracy',
                        patience=5,
                        restore_best_weights=True
                    )
                    
                    # Train the model
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        callbacks=[early_stopping],
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
        inputs = Input(shape=(self.input_dim, 1))
        
        # First LSTM branch
        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
        bn1 = BatchNormalization()(lstm1)
        drop1 = Dropout(0.2)(bn1)
        
        # Second LSTM branch with residual connection
        lstm2 = LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))(drop1)
        bn2 = BatchNormalization()(lstm2)
        drop2 = Dropout(0.2)(bn2)
        
        # Dense layers with residual connections
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(drop2)
        bn3 = BatchNormalization()(dense1)
        res1 = Add()([bn3, drop2])  # Residual connection
        
        dense2 = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(res1)
        bn4 = BatchNormalization()(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(bn4)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Custom optimizer with learning rate schedule
        initial_learning_rate = 0.001
        decay_steps = 1000
        decay_rate = 0.9
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )
        
        optimizer = Adam(
            learning_rate=learning_rate_fn,
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
          def build_single_model(self):
        inputs = Input(shape=(self.input_dim, 1))
        
        # First LSTM branch
        lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
        bn1 = BatchNormalization()(lstm1)
        drop1 = Dropout(0.2)(bn1)
        
        # Second LSTM branch with residual connection
        lstm2 = LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))(drop1)
        bn2 = BatchNormalization()(lstm2)
        drop2 = Dropout(0.2)(bn2)
        
        # Dense layers with residual connections
        dense1 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(drop2)
        bn3 = BatchNormalization()(dense1)
        res1 = Add()([bn3, drop2])  # Residual connection
        
        dense2 = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(res1)
        bn4 = BatchNormalization()(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(bn4)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Custom optimizer with learning rate schedule
        initial_learning_rate = 0.001
        decay_steps = 1000
        decay_rate = 0.9
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps, decay_rate
        )
        
        optimizer = Adam(
            learning_rate=learning_rate_fn,
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
      def train_ensemble(self, X, y):
        try:
            print(f"Training ensemble with data shape: X={X.shape}, y={y.shape}")
            
            if len(X) != len(y):
                print(f"Error: X and y lengths don't match. X: {len(X)}, y: {len(y)}")
                return
                
            if len(X) < 20:  # Need minimum data for training
                print("Not enough data for training")
                return
                
            # Normalize features
            X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Create more balanced splits
            n_splits = min(3, len(X) // 20)  # Ensure we have enough data for splits
            if n_splits < 2:
                n_splits = 2
                
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)  # Add gap between train and test
        
        # Ensure X and y are the same length before processing
        max_samples = min(len(X), MAX_HISTORY_SIZE)
        X = X[-max_samples:]
        y = y[-max_samples:]
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
        
        for i in range(self.ensemble_size):
            model = self.build_single_model()
            accuracies = []
            
            try:                # Time series cross-validation with proper indexing
                try:
                    for train_idx, val_idx in tscv.split(X):
                        if len(train_idx) < 10 or len(val_idx) < 10:
                            continue
                            
                        X_train = X[train_idx]
                        X_val = X[val_idx]
                        y_train = y[train_idx]
                        y_val = y[val_idx]
                        
                        # Add early stopping to prevent overfitting
                        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        )
                        
                        # Fit model with early stopping
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=TRAINING_EPOCHS,
                            batch_size=min(BATCH_SIZE, len(X_train) // 4),  # Ensure batch size isn't too large
                            callbacks=[early_stopping],
                            verbose=1  # Show progress
                        )
                    
                    accuracy = model.evaluate(X_val, y_val, 
                                           batch_size=BATCH_SIZE,
                                           verbose=0)[1]
                    accuracies.append(accuracy)
                
                if accuracies:  # Only if we have valid accuracies
                    avg_accuracy = np.mean(accuracies)
                    self.models.append((model, avg_accuracy))
                    
                    if avg_accuracy > self.best_accuracy:
                        self.best_accuracy = avg_accuracy
                        self.best_model = model
                        
            except Exception as e:
                print(f"Error in training model {i}: {str(e)}")
                continue
        
        # Sort models by accuracy
        if self.models:
            self.models.sort(key=lambda x: x[1], reverse=True)
            print(f"Ensemble training complete. Best accuracy: {self.best_accuracy:.4f}")
        else:
            print("Warning: No models were successfully trained")
              def predict(self, X):
        if not self.models:
            print("No trained models available")
            return None

        try:
            # Normalize input data
            X_norm = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            predictions = []
            weights = []
            confidences = []
            
            # Get predictions from top 3 models
            for i, (model, accuracy) in enumerate(self.models[:3]):
                pred = model.predict(X_norm, verbose=0)
                predictions.append(pred)
                weights.append(accuracy)
                
                # Calculate prediction confidence
                confidence = np.mean([abs(p - 0.5) * 2 for p in pred])
                confidences.append(confidence)
            
            # Combine predictions with weighted average
            combined_weights = [w * c for w, c in zip(weights, confidences)]
            weighted_pred = np.average(predictions, weights=combined_weights, axis=0)
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidences)
            if avg_confidence < 0.6:
                print(f"Warning: Low prediction confidence ({avg_confidence:.2f})")
            
            return weighted_pred, avg_confidence
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, 0.0

#####################################
# Pattern Recognizer (missing class)
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

# Update main() function to use new components...

#####################################
# Main Trading Loop - Enhanced
#####################################
def main():
    try:
        # Disable OneDNN optimizations to prevent warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        print("Initializing advanced trading bot...")

        # Initialize components
        I_want_money = None
        trading_decision_engine = TradingDecisionEngine()
        model_ensemble = ModelEnsemble(input_dim=STATE_DIM)
        
        # Initialize logs
        init_log(LOG_FILE)
        init_full_history_log(FULL_HISTORY_LOG_FILE)
        
        # Trading state variables
        current_trade_amount = BASE_AMOUNT
        main_loop_active = True
        trade_attempt_count = 0
        last_trade_time = time.time()
        successful_trades = 0
        total_trades = 0
        
        # Performance tracking
        trade_results = []
        accuracy_window = []  # Track recent accuracy
    
    # Data storage for training
    all_prices = []
    all_volumes = []
    all_features = []
    all_outcomes = []
    
    # Load existing model if available
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading existing model...")
        model_ensemble.load()

    try:
        print(f"Attempting to connect to IQ Option as {USERNAME}...")
        I_want_money = IQ_Option(USERNAME, PASSWORD)
        I_want_money.connect()

        if not I_want_money.check_connect():
            print("Connection failed. Check credentials/network.")
            return
        print("Successfully connected to IQ Option.")

        I_want_money.change_balance(PRACTICE_MODE)
        print(f"Switched to {PRACTICE_MODE} account.")
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        print("Candle stream started")

        # --- Main Loop for Continuous Logging and Trading ---
        while main_loop_active:
            current_iso_timestamp = datetime.datetime.now().isoformat()

            try:
                # Fetch real-time candles
                continuous_candles_data = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                continuous_closing_prices_list = get_closing_prices_from_candles_data(continuous_candles_data)
                
                if not continuous_closing_prices_list:
                    print("No fresh candle data. Skipping...")
                    time.sleep(CONTINUOUS_LOG_INTERVAL)
                    continue
                
                # Compute indicators
                cont_slope = compute_slope(continuous_candles_data)
                cont_ma = compute_MA(continuous_closing_prices_list)
                cont_std = compute_STD(continuous_closing_prices_list)
                cont_rsi = compute_RSI(continuous_closing_prices_list)
                cont_macd, cont_macd_signal = compute_MACD(continuous_closing_prices_list)
                cont_trader_sentiment = get_trader_sentiment(I_want_money, ASSET)
                cont_boll_sma, cont_boll_up, cont_boll_low = compute_bollinger_bands(continuous_closing_prices_list)
                cont_atr = compute_ATR(continuous_candles_data)
                cont_avg_volume = compute_average_volume(continuous_candles_data)
                cont_hour, cont_weekday = get_time_features()
                cont_news_sentiment = get_news_sentiment()
                
                # Log full history continuously
                log_full_history(
                    FULL_HISTORY_LOG_FILE,
                    [
                        current_iso_timestamp, ASSET, ",".join(map(str, continuous_closing_prices_list)),
                        cont_trader_sentiment,
                        cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal,
                        cont_boll_sma, cont_boll_up, cont_boll_low, cont_atr, cont_avg_volume,
                        cont_hour, cont_weekday, cont_news_sentiment
                    ]
                )
                
                # Store data for training
                all_prices.append(continuous_closing_prices_list[-1])
                all_volumes.append(cont_avg_volume)
                all_features.append([
                    cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, cont_trader_sentiment,
                    cont_boll_sma, cont_atr, cont_avg_volume, cont_hour, cont_weekday, cont_news_sentiment
                ])
                
                # Check if enough data for training
                if len(all_prices) > MIN_TRADES_FOR_TRAINING and not model_ensemble.models:
                    print("Training initial model...")
                    X = np.array(all_features)
                    y = np.array([1 if i > 0 else 0 for i in np.diff(all_prices)])  # Simple direction prediction
                    model_ensemble.train_ensemble(X, y)
                    print(f"Initial model trained. Best accuracy: {model_ensemble.best_accuracy:.4f}")
                
                # Trading Logic
                if model_ensemble.models and time.time() - last_trade_time > MIN_SECONDS_BETWEEN_TRADES:
                    # Prepare features for prediction
                    current_features = np.array([all_features[-1]])
                    
                    # Make prediction
                    confidence = model_ensemble.predict(current_features)[0][0]
                    
                    # Trading decision
                    if trading_decision_engine.should_trade(confidence, all_prices, all_volumes):
                        action = "call" if confidence > 0.5 else "put"
                        print(f"Placing trade: {action.upper()} with confidence {confidence:.2f}")
                        
                        trade_status, order_id = I_want_money.buy(current_trade_amount, ASSET, action, DURATION)
                        
                        if trade_status:
                            print(f"Trade placed. Order ID: {order_id}. Waiting for result...")
                            last_trade_time = time.time()
                            trade_start_time = time.time()
                            
                            # Monitor trade and potentially close early
                            while time.time() - trade_start_time < DURATION * 60:
                                # Check for early close
                                current_price = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                                if current_price:
                                    current_price_list = get_closing_prices_from_candles_data(current_price)
                                    current_features = np.array([all_features[-1]])
                                    current_confidence = model_ensemble.predict(current_features)[0][0]
                                    
                                    if trading_decision_engine.should_close_early(
                                        current_confidence, continuous_closing_prices_list[-1], current_price_list[-1], action
                                    ):
                                        print("Closing trade early...")
                                        # Implement early close logic here (if IQ Option API supports it)
                                        # For now, just log the intention
                                        print("Early close triggered but not implemented.")
                                        break
                                
                                time.sleep(5)  # Check every 5 seconds
                            
                            # Get trade result
                            api_trade_result = I_want_money.check_win_v3(order_id)
                            if api_trade_result is not None:
                                profit_val = api_trade_result
                                outcome_str = "win" if profit_val > 0 else ("loss" if profit_val < 0 else "tie")
                                print(f"Trade Result: {outcome_str.upper()}, Profit: ${profit_val:.2f}")
                                
                                # Log trade
                                log_row_for_trade_log = [
                                    current_iso_timestamp, order_id, ASSET, action, current_trade_amount, DURATION,
                                    cont_slope, confidence, outcome_str, profit_val,
                                    ",".join(map(str, continuous_closing_prices_list)), cont_ma, cont_std, cont_rsi,
                                    cont_macd, cont_macd_signal, cont_trader_sentiment,
                                    cont_boll_sma, cont_boll_up, cont_boll_low, cont_atr, cont_avg_volume,
                                    cont_hour, cont_weekday, cont_news_sentiment
                                ]
                                log_trade(LOG_FILE, log_row_for_trade_log)
                                
                                # Update training data with outcome
                                all_outcomes.append(1 if outcome_str == "win" else 0)
                                
                                # Retrain model periodically
                                if len(all_outcomes) > MIN_TRADES_FOR_TRAINING and len(all_outcomes) % 50 == 0:
                                    print("Retraining model...")
                                    X = np.array(all_features)
                                    y = np.array(all_outcomes)
                                    model_ensemble.train_ensemble(X, y)
                                    print(f"Model retrained. Best accuracy: {model_ensemble.best_accuracy:.4f}")
                            else:
                                print("Trade result not found.")
                        else:
                            print("Trade placement failed.")
                
            except Exception as e_cont_log:
                print(f"Error during main loop: {e_cont_log}")
                traceback.print_exc()
                
                if I_want_money and not I_want_money.check_connect():
                    print("Connection lost. Attempting reconnect...")
                    try:
                        I_want_money.connect()
                        if I_want_money.check_connect():
                            print("Reconnected.")
                            I_want_money.change_balance(PRACTICE_MODE)
                        else:
                            print("Reconnect failed. Stopping bot.")
                            main_loop_active = False
                    except Exception as e_reconnect:
                        print(f"Exception during reconnect: {e_reconnect}. Stopping bot.")
                        main_loop_active = False
            
            time.sleep(CONTINUOUS_LOG_INTERVAL)

    except KeyboardInterrupt:
        print("\nTrading bot stopped by user (Ctrl+C).")
    except Exception as e_main:
        print(f"A critical error occurred in the main bot function: {e_main}")
        traceback.print_exc()
    finally:
        if I_want_money:
            try:
                if I_want_money.check_connect():
                    print("Disconnecting from IQ Option in finally block...")
                    I_want_money.close_connect()
                    print("Disconnected.")
                else:
                    print("IQ Option connection was not active in finally block or already closed.")
            except Exception as e_disconnect:
                print(f"Error during disconnection: {e_disconnect}")
        print("Trading bot shutdown sequence complete.")

if __name__ == "__main__":
    main()