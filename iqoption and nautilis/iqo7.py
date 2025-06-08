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
TRADING_ITERATIONS = 100

LOG_FILE = "trade_log2.csv"             # Main trade log with advanced indicators
FULL_HISTORY_LOG_FILE = "full_history_log.csv"  # Log for full price history and ALL indicators, logged continuously

TIMEFRAME = 60     # Candle length in seconds (60s = 1 minute)
NUM_CANDLES = 30   # Number of candles to gather

STATE_DIM = 13
ACTION_DIM = 2    # 0: "call", 1: "put"

INTER_TRADE_WAIT_SECONDS = 10 # Seconds to wait and log data between trade attempts

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
                "Bollinger_SMA", "ATR", "AvgVolume", "Hour", "Weekday", "NewsSentiment"
            ])

def log_trade(file_path, row): # For trade_log2.csv
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def log_full_history(file_path, row): # For full_history_log.csv
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
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
    losses = -changes[changes < 0]

    avg_gain = np.mean(gains[:period]) if len(gains) >= period else (np.sum(gains) / period if len(gains) > 0 else 0)
    avg_loss = np.mean(losses[:period]) if len(losses) >= period else (np.sum(losses) / period if len(losses) > 0 else 0)
    
    if avg_loss == 0: return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_EMA(prices, period): # Takes list of prices
    if not prices or len(prices) == 0: return [] # Return empty list if no prices
    # Using pandas for robust EMA calculation if available
    try:
        return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()
    except ImportError:
        # Fallback to manual calculation if pandas is not available
        if len(prices) < period: # Not enough data for a full EMA based on initial SMA
            return [np.mean(prices)] * len(prices) # Or some other handling like returning prices itself
        
        ema_values = []
        # Calculate initial SMA for the first EMA value
        sma = np.mean(prices[:period])
        ema_values.append(sma)
        
        # Multiplier for EMA calculation
        multiplier = 2 / (period + 1.0)
        
        # Calculate subsequent EMA values
        for price in prices[period:]: # Start from the (period)-th index
            ema_val = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema_val)
        # The EMA list will be shorter than prices by period-1. For consistency, one might pad.
        # For MACD, it's common to use pandas or ensure full length EMAs.
        return ema_values # Note: This list is shorter than original if period > 1


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
        if not ema12 or not ema26 or len(ema12) != len(ema26): return 0.0,0.0 # Should not happen if EMA is full length

        macd_values = np.array(ema12) - np.array(ema26)
        if len(macd_values) < signal_period:
            return macd_values[-1] if len(macd_values) > 0 else 0.0, np.mean(macd_values) if len(macd_values) > 0 else 0.0
        
        signal_line_values = compute_EMA_simple_for_MACD(list(macd_values), signal_period)
        return macd_values[-1], signal_line_values[-1] if signal_line_values else 0.0

def compute_EMA_simple_for_MACD(prices, period): # Ensures full length for MACD if pandas not used
    if not prices: return []
    ema = [0.0] * len(prices)
    if not ema: return []
    ema[0] = prices[0]
    k = 2 / (period + 1)
    for i in range(1, len(prices)):
        ema[i] = prices[i] * k + ema[i-1] * (1-k)
    return ema

#####################################
# Trader Sentiment Function (Kept as is from user's code)
#####################################
def get_trader_sentiment(api, asset):
    try:
        # This method might not exist or work as expected in all iqoptionapi forks.
        # It often requires an active_id, not just the asset string.
        sentiment_data = api.get_traders_mood(asset) # Assuming this is the intended method
        # The structure of sentiment_data needs to be known to extract a single value.
        # Example: if it returns {'call': 0.6, 'put': 0.4}, you might use sentiment_data['call']
        if isinstance(sentiment_data, (float, int)): # If it directly returns a sentiment value
             return sentiment_data
        elif isinstance(sentiment_data, dict) and 'call' in sentiment_data:
             return sentiment_data['call'] # Assuming 'call' proportion represents bullish sentiment
        else:
            print(f"Trader sentiment for {asset} in unexpected format: {sentiment_data}. Using fallback.")
            return random.uniform(0.4, 0.6)
    except AttributeError: # If get_traders_mood doesn't exist
        print(f"API method for trader sentiment (e.g., get_traders_mood) not found for {asset}. Using fallback.")
        return random.uniform(0.4, 0.6)
    except Exception as e:
        print(f"Real trader sentiment not available for {asset}, using fallback. Error: {e}")
        return random.uniform(0.4, 0.6)

#####################################
# Advanced Prediction Function (Kept as is from user's code)
#####################################
def advanced_predict_probability(slope, ma, std, rsi, macd, macd_signal, trader_sentiment, action, boll_sma, news_sentiment):
    prob = 0.5
    if action == "call":
        prob += 0.15 if slope > 0 else -0.15
        prob += 0.10 if rsi < 70 else -0.10 # Original logic
        prob += 0.10 if macd > macd_signal else -0.10
    elif action == "put":
        prob += 0.15 if slope < 0 else -0.15
        prob += 0.10 if rsi > 30 else -0.10 # Original logic
        prob += 0.10 if macd < macd_signal else -0.10

    if trader_sentiment > 0.6: prob += 0.1
    elif trader_sentiment < 0.4: prob -= 0.1

    if news_sentiment > 0.6: prob += 0.05
    elif news_sentiment < 0.4: prob -= 0.05
    
    # Considering Bollinger Bands (BollSMA is the middle band)
    # If action is call, price above SMA is good. If put, price below SMA is good.
    # This is somewhat redundant with MA, but kept for consistency with state_dim
    # current_price would be needed here. Assuming MA vs BollSMA for trend confirmation.
    # if action == "call" and ma > boll_sma : prob += 0.02 # Slight boost
    # elif action == "put" and ma < boll_sma : prob += 0.02

    return max(0.01, min(0.99, prob)) # Clamped probability

#####################################
# Risk Management Functions (Kept as is from user's code)
#####################################
def is_stop_loss_triggered(current_price, entry_price, stop_loss_pct=0.02, action_type="call"):
    if entry_price == 0: return False # Avoid division by zero
    if action_type == "call":
        return (entry_price - current_price) / entry_price >= stop_loss_pct
    else: # put
        return (current_price - entry_price) / entry_price >= stop_loss_pct


def is_take_profit_triggered(current_price, entry_price, take_profit_pct=0.05, action_type="call"):
    if entry_price == 0: return False
    if action_type == "call":
        return (current_price - entry_price) / entry_price >= take_profit_pct
    else: #put
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
# RL Agent (Deep Q-Network using TensorFlow/Keras) - Corrected input_shape
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
        # Corrected: Use input_shape for the first layer in Sequential models
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
        target_f[0][action] = target # action is already an index
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#####################################
# Supervised Learning: Retrain Model from Logs (Kept as is from user's code)
#####################################
def retrain_supervised_model(log_file, state_dim=STATE_DIM, action_dim=ACTION_DIM, epochs=5):
    if not os.path.exists(log_file) or os.path.getsize(log_file) < 100: # Basic check
        print(f"Log file {log_file} not found or too small for supervised retraining.")
        return None
    try:
        data = pd.read_csv(log_file)
        feature_cols = ["Slope", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment",
                        "Bollinger_SMA", "ATR", "AvgVolume", "Hour", "Weekday", "NewsSentiment"]
        
        # Filter for conclusive outcomes
        data_filtered = data[data['Outcome'].isin(['win', 'loss'])].copy()
        if data_filtered.empty:
            print("No 'win' or 'loss' outcomes in log for retraining.")
            return None

        for col in feature_cols:
            if col not in data_filtered.columns:
                data_filtered[col] = 0.0 # Fill missing features, ideally logs are complete
        
        X = data_filtered[feature_cols].values.astype(np.float32)
        # For regression, predict profit. For classification, predict win/loss.
        # User's original code predicted profit (linear activation).
        y = data_filtered["Profit"].values.astype(np.float32)

        if X.shape[0] < 10:
            print(f"Not enough samples ({X.shape[0]}) for retraining.")
            return None

        model = Sequential()
        model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear')) # Linear for profit regression
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        print(f"Retraining supervised model with {X.shape[0]} samples...")
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=1, shuffle=True)
        
        model_save_path = "supervised_model.keras" # Using .keras extension
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
        data = pd.read_csv(log_file)
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
def update_model_from_logs(): # This refers to the supervised model
    print("Attempting to update supervised model from logs...")
    model = retrain_supervised_model(LOG_FILE) # Using LOG_FILE for trades
    if model:
        print("Supervised model update from logs complete.")
    else:
        print("Supervised model update from logs failed or no data.")
    return model # This model is currently not re-assigned to any active trading logic

#####################################
# Main Trading Loop
#####################################
def main():
    # Suppress TensorFlow oneDNN custom operations warning
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Optional: Control TensorFlow logging level more globally
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all messages, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
    # tf.get_logger().setLevel('ERROR')


    print("Initializing trading bot...")
    rl_agent = RLAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    current_trade_amount = BASE_AMOUNT

    # Initialize log files with appropriate headers
    init_log(LOG_FILE) # For individual trade details
    init_full_history_log(FULL_HISTORY_LOG_FILE) # For continuous market data with all indicators

    I_want_money = None  # Initialize connection object to None

    try:
        print(f"Attempting to connect to IQ Option as {USERNAME}...")
        I_want_money = IQ_Option(USERNAME, PASSWORD)
        I_want_money.connect() # Library should handle connection feedback or errors

        if not I_want_money.check_connect():
            print("Connection failed after connect() call. Please check credentials/network.")
            return
        print("Successfully connected to IQ Option.")

        I_want_money.change_balance(PRACTICE_MODE)
        print(f"Switched to {PRACTICE_MODE} account.")
        
        try:
            profile = I_want_money.get_profile_ansyc() # Can be problematic if not handled well
            print(f"Profile data (async): {profile}")
        except Exception as e_profile:
            print(f"Could not fetch profile data (async call): {e_profile}")

        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 100: # Only create if log has data
            create_dashboard(LOG_FILE)

        # Main loop for trading iterations and continuous logging
        for i in range(TRADING_ITERATIONS + INTER_TRADE_WAIT_SECONDS * TRADING_ITERATIONS): # Adjusted loop for continuous nature
            
            current_iso_timestamp = datetime.datetime.now().isoformat()
            
            # --- Continuous Logging Section (every second) ---
            try:
                I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
                time.sleep(0.2) # Shorter sleep, get_realtime_candles will wait if needed or return latest
                continuous_candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                # It's often better to have one persistent stream if API allows, 
                # rather than start/stop or relying on get_realtime_candles to manage.
                # For now, assuming get_realtime_candles fetches latest from an implicitly managed stream.

                if continuous_candles and isinstance(continuous_candles, dict) and any(continuous_candles):
                    continuous_closing_prices = get_closing_prices_from_candles_data(continuous_candles)
                    
                    if continuous_closing_prices:
                        cont_slope = compute_slope(continuous_candles) # Pass raw candles
                        cont_ma = compute_MA(continuous_closing_prices)
                        cont_std = compute_STD(continuous_closing_prices)
                        cont_rsi = compute_RSI(continuous_closing_prices)
                        cont_macd, cont_macd_signal = compute_MACD(continuous_closing_prices)
                        cont_trader_sentiment = get_trader_sentiment(I_want_money, ASSET) # Real sentiment attempt
                        cont_boll_sma, _, _ = compute_bollinger_bands(continuous_closing_prices)
                        cont_atr = compute_ATR(continuous_candles) # Pass raw candles
                        cont_avg_volume = compute_average_volume(continuous_candles) # Pass raw candles
                        cont_hour, cont_weekday = get_time_features()
                        cont_news_sentiment = get_news_sentiment()
                        
                        price_history_str_cont = ",".join(map(str, continuous_closing_prices))

                        full_history_continuous_row = [
                            current_iso_timestamp, ASSET, price_history_str_cont, cont_trader_sentiment,
                            cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal,
                            cont_boll_sma, cont_atr, cont_avg_volume, cont_hour, cont_weekday, cont_news_sentiment
                        ]
                        log_full_history(FULL_HISTORY_LOG_FILE, full_history_continuous_row)
                else: # Log minimal if no candle data
                    log_full_history(FULL_HISTORY_LOG_FILE, [current_iso_timestamp, ASSET, "NO_CANDLE_DATA", 0.5] + [0.0]*(16-4) )
            
            except Exception as e_cont_log:
                print(f"Error during continuous logging: {e_cont_log}")
                if I_want_money and not I_want_money.check_connect():
                    print("Connection lost during continuous log. Attempting reconnect...")
                    I_want_money.connect() # Attempt to reconnect
                    if I_want_money.check_connect(): I_want_money.change_balance(PRACTICE_MODE)
                    else: print("Reconnect failed in continuous log."); break 


            # --- Trade Execution Logic (every 'INTER_TRADE_WAIT_SECONDS' seconds approx) ---
            if i % INTER_TRADE_WAIT_SECONDS == 0:
                trade_cycle_num = i // INTER_TRADE_WAIT_SECONDS
                print(f"\n--- Trade Cycle {trade_cycle_num + 1}/{TRADING_ITERATIONS} ---")
                
                # Use the most recently fetched continuous_candles and indicators for trade decision
                # This ensures trade decisions are based on up-to-date data from the continuous log cycle
                if not (continuous_candles and isinstance(continuous_candles, dict) and any(continuous_candles) and continuous_closing_prices):
                    print("Skipping trade cycle due to lack of fresh data from continuous logging.")
                    time.sleep(1) # Wait for the next 1-second log cycle
                    continue
                
                # State for RL agent (using indicators from continuous log)
                state_features = np.array([
                    cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, cont_trader_sentiment,
                    cont_boll_sma, cont_atr, cont_avg_volume, cont_hour, cont_weekday, cont_news_sentiment
                ], dtype=np.float32)
                
                entry_price_for_trade = continuous_closing_prices[-1]

                print(f"Indicators for Trade: Price={entry_price_for_trade:.5f}, Slope={cont_slope:.4f}, MA={cont_ma:.4f}, STD={cont_std:.4f}, RSI={cont_rsi:.2f}, MACD={cont_macd:.4f}, Signal={cont_macd_signal:.4f}, Sentiment={cont_trader_sentiment:.2f}, ATR={cont_atr:.4f}, News={cont_news_sentiment:.2f}")
                
                action_idx = rl_agent.choose_action(state_features)
                action_str = "call" if action_idx == 0 else "put"
                print(f"RL Agent chose action: {action_str.upper()}")
                
                # Heuristic probability for logging (not for RL agent's decision directly)
                heuristic_prob = advanced_predict_probability(
                    cont_slope, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, 
                    cont_trader_sentiment, action_str, cont_boll_sma, cont_news_sentiment
                )
                print(f"Heuristic Probability: {heuristic_prob:.2%}")

                print(f"Placing trade with amount: ${current_trade_amount:.2f}")
                trade_status, order_id = I_want_money.buy(current_trade_amount, ASSET, action_str, DURATION)
                
                outcome_str = "N/A"
                profit_val = 0.0

                if trade_status:
                    print(f"Trade placed. Order ID: {order_id}. Waiting for result...")
                    # Wait for trade result logic (simplified from user's original logic for this part)
                    # The inner while loop for risk management and frequent logging during trade wait:
                    trade_wait_start = time.time()
                    max_wait_trade = DURATION * 60 + 15 # seconds
                    api_trade_result = None
                    
                    stop_loss_triggered_flag = False
                    take_profit_triggered_flag = False

                    while time.time() - trade_wait_start < max_wait_trade:
                        current_candles_trade_wait = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                        current_prices_trade_wait = get_closing_prices_from_candles_data(current_candles_trade_wait)
                        
                        if current_prices_trade_wait: # Log to full_history during trade wait
                            live_price = current_prices_trade_wait[-1]
                            sentiment_trade_wait = get_trader_sentiment(I_want_money, ASSET)
                            history_str_trade_wait = ",".join(map(str, current_prices_trade_wait))
                            # Note: No need to re-calculate all indicators here for full_history_log,
                            # as the main 1-sec loop is already doing that. This is just for price & sentiment.
                            # If more detail needed here, calculate them. For now, keep it lean.
                            # log_full_history(FULL_HISTORY_LOG_FILE, [datetime.datetime.now().isoformat(), ASSET, history_str_trade_wait, sentiment_trade_wait] + [0.0]*(16-4) ) # Simpler log
                            
                            # Risk management check
                            if not stop_loss_triggered_flag and is_stop_loss_triggered(live_price, entry_price_for_trade, action_type=action_str):
                                print(f"STOP LOSS for order {order_id} at {live_price}")
                                profit_val = -current_trade_amount
                                outcome_str = "loss_sl"
                                stop_loss_triggered_flag = True
                                break 
                            if not take_profit_triggered_flag and is_take_profit_triggered(live_price, entry_price_for_trade, action_type=action_str):
                                print(f"TAKE PROFIT for order {order_id} at {live_price}")
                                # For binary, profit is fixed. This indicates win condition met early.
                                # We still rely on check_win for actual profit.
                                take_profit_triggered_flag = True 
                                # break # Usually wait for binary expiry

                        api_trade_result = I_want_money.check_win_v3(order_id)
                        if api_trade_result is not None:
                            break
                        time.sleep(1) # Check result every second

                    if api_trade_result is not None:
                        profit_val = api_trade_result
                        outcome_str = "win" if profit_val > 0 else ("loss" if profit_val < 0 else "tie")
                    elif stop_loss_triggered_flag: # SL already determined profit and outcome
                        pass
                    else:
                        print(f"Trade {order_id} outcome timed out.")
                        outcome_str = "unknown_timeout"
                        profit_val = 0.0
                    
                    print(f"Trade Result: {outcome_str.upper()}, Profit: ${profit_val:.2f}")
                
                else: # Trade placement failed
                    print("Trade execution failed.")
                    order_id = "PLACEMENT_FAILED"
                    outcome_str = "failed_execution"
                    profit_val = 0.0

                # Log the trade to trade_log2.csv
                log_row_for_trade_log = [
                    current_iso_timestamp, order_id, ASSET, action_str, current_trade_amount, DURATION,
                    cont_slope, heuristic_prob, outcome_str, profit_val,
                    price_history_str_cont, cont_ma, cont_std, cont_rsi, cont_macd, cont_macd_signal, cont_trader_sentiment,
                    cont_boll_sma, 0, 0, cont_atr, cont_avg_volume, # Placeholder for boll_upper, boll_lower if not used in state
                    cont_hour, cont_weekday, cont_news_sentiment
                ]
                log_trade(LOG_FILE, log_row_for_trade_log)
                print(f"Trade logged to {LOG_FILE}")

                # Train RL Agent
                rl_reward = 1 if profit_val > 0 else (-1 if profit_val < 0 else 0)
                # For next_state, ideally fetch fresh state after trade. Using current as approximation.
                next_state_features = state_features 
                done_rl = (trade_cycle_num == TRADING_ITERATIONS - 1)
                rl_agent.train(state_features, action_idx, rl_reward, next_state_features, done_rl)
                print("RL Agent trained.")

                current_trade_amount = update_trade_amount(current_trade_amount, outcome_str, heuristic_prob)
                print(f"Next trade amount: ${current_trade_amount:.2f}")
                
                if (trade_cycle_num + 1) % 20 == 0: # Every 20 trade cycles
                    update_model_from_logs() # Retrain supervised model
                    create_dashboard(LOG_FILE)


            time.sleep(1) # Main loop ticks every second for continuous logging

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
                    I_want_money.close_connect() # Or I_want_money.disconnect() if that's the method
                    print("Disconnected.")
                else:
                    print("IQ Option connection was not active in finally block.")
            except Exception as e_disconnect:
                print(f"Error during disconnection: {e_disconnect}")
        print("Trading bot shutdown sequence complete.")

if __name__ == "__main__":
    main()
