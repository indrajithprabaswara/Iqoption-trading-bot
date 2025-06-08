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

# TensorFlow and Keras for RL agent
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import traceback # For detailed error logging

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
# Main Trading Loop - FIX: Restructured for truly continuous logging
#####################################
def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print("Initializing trading bot...")
    rl_agent = RLAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    current_trade_amount = BASE_AMOUNT

    init_log(LOG_FILE) 
    init_full_history_log(FULL_HISTORY_LOG_FILE) 

    I_want_money = None  
    main_loop_active = True
    trade_attempt_count = 0 
    seconds_since_last_trade_attempt = 0

    try:
        print(f"Attempting to connect to IQ Option as {USERNAME}...")
        I_want_money = IQ_Option(USERNAME, PASSWORD)
        I_want_money.connect() 

        if not I_want_money.check_connect():
            print("Connection failed after connect() call. Please check credentials/network.")
            return
        print("Successfully connected to IQ Option.")

        I_want_money.change_balance(PRACTICE_MODE)
        print(f"Switched to {PRACTICE_MODE} account.")
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        print("Candle stream started")

        # --- Train model with existing log data before the main loop starts ---
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 100:
            update_model_from_logs()  # Train with initial log data before starting the loop

        # --- Main Loop for Continuous Logging and Periodic Trading ---
        while main_loop_active:
            current_iso_timestamp = datetime.datetime.now().isoformat()

            # Continuous Data Fetching and Logging (Every Second)
            try:
                # NOTE: start_candles_stream now runs once after you connect ­– NOT here
                continuous_candles_data = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                continuous_closing_prices_list = get_closing_prices_from_candles_data(continuous_candles_data)

                if continuous_closing_prices_list:
                    cont_slope          = compute_slope(continuous_candles_data)
                    cont_ma             = compute_MA(continuous_closing_prices_list)
                    cont_std            = compute_STD(continuous_closing_prices_list)
                    cont_rsi            = compute_RSI(continuous_closing_prices_list)
                    cont_macd, cont_macd_signal = compute_MACD(continuous_closing_prices_list)
                    cont_trader_sentiment       = get_trader_sentiment(I_want_money, ASSET)
                    cont_boll_sma, cont_boll_up, cont_boll_low = compute_bollinger_bands(continuous_closing_prices_list)
                    cont_atr            = compute_ATR(continuous_candles_data)
                    cont_avg_volume     = compute_average_volume(continuous_candles_data)
                    cont_hour, cont_weekday = get_time_features()
                    cont_news_sentiment = get_news_sentiment()
                    price_history_str_cont = ",".join(map(str, continuous_closing_prices_list))


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

            except Exception as e_cont_log:
                print(f"Error during continuous data logging: {e_cont_log}")
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

            if not main_loop_active: break

            # Trade Execution Logic
            seconds_since_last_trade_attempt += 1
            if seconds_since_last_trade_attempt >= INTER_TRADE_WAIT_SECONDS:
                if trade_attempt_count < TRADING_ITERATIONS:
                    seconds_since_last_trade_attempt = 0 
                    trade_attempt_count += 1
                    print(f"\n--- Trade Attempt Cycle {trade_attempt_count}/{TRADING_ITERATIONS} ---")

                    if not (continuous_candles_data and continuous_closing_prices_list):
                        print("Skipping trade attempt due to lack of fresh data.")
                    else:
                        state_features = np.array([
                            cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, cont_trader_sentiment,
                            cont_boll_sma, cont_atr, cont_avg_volume, cont_hour, cont_weekday, cont_news_sentiment
                        ], dtype=np.float32)

                        entry_price_for_trade = continuous_closing_prices_list[-1]
                        print(f"Indicators for Trade: Price={entry_price_for_trade:.5f}, Slope={cont_slope:.4f}, MA={cont_ma:.4f}, ..., Sentiment={cont_trader_sentiment:.2f}")

                        action_idx = rl_agent.choose_action(state_features)
                        action_str = "call" if action_idx == 0 else "put"
                        print(f"RL Agent chose action: {action_str.upper()}")

                        heuristic_prob = advanced_predict_probability(
                            cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, 
                            cont_trader_sentiment, action_str, cont_boll_sma, cont_news_sentiment
                        )
                        print(f"Heuristic Probability: {heuristic_prob:.2%}")

                        print(f"Placing trade with amount: ${current_trade_amount:.2f}")
                        trade_status, order_id = I_want_money.buy(current_trade_amount, ASSET, action_str, DURATION)

                        outcome_str = "N/A"; profit_val = 0.0

                        if trade_status:
                            print(f"Trade placed. Order ID: {order_id}. Waiting for result...")
                            trade_wait_start = time.time()
                            max_wait_trade = DURATION * 60 + 20
                            api_trade_result = None
                            stop_loss_triggered_flag = False
                            take_profit_triggered_flag = False

                            # --- START REPLACEMENT: keep logging during trade wait ---
                            while time.time() - trade_wait_start < max_wait_trade:
                                # Pull live candles
                                current_candles_trade_wait = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                                current_prices_trade_wait  = get_closing_prices_from_candles_data(current_candles_trade_wait)

                                # >>> NEW: write a row to full_history_log.csv every second <<<
                                if current_candles_trade_wait and current_prices_trade_wait:
                                    now_iso             = datetime.datetime.now().isoformat()
                                    slope_now           = compute_slope(current_candles_trade_wait)
                                    ma_now              = compute_MA(current_prices_trade_wait)
                                    std_now             = compute_STD(current_prices_trade_wait)
                                    rsi_now             = compute_RSI(current_prices_trade_wait)
                                    macd_now, macd_sig  = compute_MACD(current_prices_trade_wait)
                                    sentiment_now       = get_trader_sentiment(I_want_money, ASSET)
                                    boll_sma_now, _, _  = compute_bollinger_bands(current_prices_trade_wait)
                                    atr_now             = compute_ATR(current_candles_trade_wait)
                                    avg_vol_now         = compute_average_volume(current_candles_trade_wait)
                                    hr_now, wkday_now   = get_time_features()
                                    news_sent_now       = get_news_sentiment()

                                    price_hist_str_now  = ",".join(map(str, current_prices_trade_wait))
                                    log_full_history(
                                        FULL_HISTORY_LOG_FILE,
                                        [
                                            now_iso, ASSET, price_hist_str_now, sentiment_now,
                                            slope_now, ma_now, std_now, rsi_now, macd_now, macd_sig,
                                            boll_sma_now, atr_now, avg_vol_now, hr_now, wkday_now,
                                            news_sent_now
                                        ]
                                    )

                                    # --- stop-loss / take-profit check (unchanged) ---
                                    live_price = current_prices_trade_wait[-1]
                                    if not stop_loss_triggered_flag and is_stop_loss_triggered(
                                            live_price, entry_price_for_trade, action_type=action_str):
                                        print(f"STOP LOSS for order {order_id} at {live_price}")
                                        profit_val = -current_trade_amount
                                        outcome_str = "loss_sl"
                                        stop_loss_triggered_flag = True
                                        break

                                    if not take_profit_triggered_flag and is_take_profit_triggered(
                                            live_price, entry_price_for_trade, action_type=action_str):
                                        print(f"TAKE PROFIT for order {order_id} at {live_price}")
                                        take_profit_triggered_flag = True
                                # -------------------------------------------------------

                                api_trade_result = I_want_money.check_win_v3(order_id)
                                if api_trade_result is not None:
                                    break

                                time.sleep(1)   # keep the 1-second cadence

                            if api_trade_result is not None:
                                profit_val = api_trade_result
                                outcome_str = "win" if profit_val > 0 else ("loss" if profit_val < 0 else "tie")
                            elif stop_loss_triggered_flag: pass
                            else: outcome_str = "unknown_timeout"; profit_val = 0.0
                            print(f"Trade Result: {outcome_str.upper()}, Profit: ${profit_val:.2f}")
                        else:
                            order_id = "PLACEMENT_FAILED"; outcome_str = "failed_execution"; profit_val = 0.0
                            print("Trade execution failed.")

                        log_row_for_trade_log = [
                            current_iso_timestamp, order_id, ASSET, action_str, current_trade_amount, DURATION,
                            cont_slope, heuristic_prob, outcome_str, profit_val,
                            price_history_str_cont, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, cont_trader_sentiment,
                            cont_boll_sma, cont_boll_up, cont_boll_low, cont_atr, cont_avg_volume, 
                            cont_hour, cont_weekday, cont_news_sentiment
                        ]
                        log_trade(LOG_FILE, log_row_for_trade_log)
                        print(f"Trade logged to {LOG_FILE}")

                        rl_reward = 1 if profit_val > 0 else (-1 if profit_val < 0 else 0)
                        next_state_features = state_features 
                        done_rl = (trade_attempt_count >= TRADING_ITERATIONS)
                        rl_agent.train(state_features, action_idx, rl_reward, next_state_features, done_rl)
                        print("RL Agent trained.")

                        current_trade_amount = update_trade_amount(current_trade_amount, outcome_str, heuristic_prob)
                        print(f"Next trade amount: ${current_trade_amount:.2f}")

                        if trade_attempt_count > 0 and trade_attempt_count % 20 == 0: 
                            update_model_from_logs()
                            create_dashboard(LOG_FILE)

            if trade_attempt_count >= TRADING_ITERATIONS:
                print(f"All {TRADING_ITERATIONS} trade attempts are complete. Entering continuous logging only mode.")

            time.sleep(1) # Ensure the main loop ticks every second

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