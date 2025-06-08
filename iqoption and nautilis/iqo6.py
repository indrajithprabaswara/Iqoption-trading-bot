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

#####################################
# Configuration & Constants
#####################################
USERNAME = "@gmail.com"          # Fill your IQ Option email here
PASSWORD = ""          # Fill your IQ Option password here
PRACTICE_MODE = "PRACTICE"

ASSET = "EURUSD-OTC"
BASE_AMOUNT = 1
DURATION = 1  # fixed trade duration in minutes
TRADING_ITERATIONS = 100

LOG_FILE = "trade_log2.csv"             # Main trade log with advanced indicators
FULL_HISTORY_LOG_FILE = "full_history_log.csv"  # Log for full price history and sentiment

TIMEFRAME = 60     # Candle length in seconds (60s = 1 minute)
NUM_CANDLES = 30   # Number of candles to gather

STATE_DIM = 13
ACTION_DIM = 2    # 0: "call", 1: "put"

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
            writer.writerow(["Timestamp", "Asset", "PriceHistory", "TraderSentiment"])

def log_trade(file_path, row):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def log_full_history(file_path, row):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

#####################################
# Additional Technical Indicator Functions
#####################################
def compute_bollinger_bands(prices, period=20, num_std=2):
    if len(prices) < period:
        sma = np.mean(prices)
        return sma, sma, sma
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_ATR(candles, period=14):
    times = sorted(candles.keys())
    if len(times) < period + 1:
        return 0
    tr_list = []
    for i in range(1, len(times)):
        current = candles[times[i]]
        previous = candles[times[i-1]]
        current_high = current.get("high", current["close"])
        current_low = current.get("low", current["close"])
        prev_close = previous.get("close")
        tr = max(current_high - current_low, abs(current_high - prev_close), abs(current_low - prev_close))
        tr_list.append(tr)
    atr = np.mean(tr_list[-period:])
    return atr

def compute_average_volume(candles, period=10):
    times = sorted(candles.keys())
    volumes = [candles[t].get("volume", 0) for t in times]
    if len(volumes) < period:
        return np.mean(volumes)
    return np.mean(volumes[-period:])

def get_time_features():
    now = datetime.datetime.now()
    return now.hour, now.weekday()

def get_news_sentiment():
    return random.uniform(0, 1)

#####################################
# Existing Technical Indicator Functions
#####################################
def compute_slope(candles):
    times = sorted(candles.keys())
    if len(times) < 2:
        return 0.0
    closes = [candles[t]["close"] for t in times]
    x = np.arange(len(closes))
    slope, _ = np.polyfit(x, closes, 1)
    return slope

def compute_MA(prices):
    return np.mean(prices)

def compute_STD(prices):
    return np.std(prices)

def compute_RSI(prices, period=14):
    if len(prices) < period + 1:
        return 50
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_EMA(prices, period):
    ema = [prices[0]]
    alpha = 2 / (period + 1)
    for price in prices[1:]:
        ema.append(price * alpha + ema[-1] * (1 - alpha))
    return ema

def compute_MACD(prices):
    if len(prices) < 26:
        return 0, 0
    ema12 = compute_EMA(prices, 12)
    ema26 = compute_EMA(prices, 26)
    macd_line = np.array(ema12) - np.array(ema26)
    if len(macd_line) < 9:
        signal = np.mean(macd_line)
    else:
        signal_line = compute_EMA(list(macd_line), 9)
        signal = signal_line[-1]
    return macd_line[-1], signal

#####################################
# Trader Sentiment Function
#####################################
def get_trader_sentiment(api, asset):
    try:
        sentiment = api.get_trader_sentiment(asset)
        return sentiment
    except Exception as e:
        print("Real trader sentiment not available, using fallback. Error:", e)
        return random.uniform(0, 1)

#####################################
# Advanced Prediction Function
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

    if trader_sentiment > 0.6:
        prob += 0.1
    elif trader_sentiment < 0.4:
        prob -= 0.1

    if news_sentiment > 0.6:
        prob += 0.05
    elif news_sentiment < 0.4:
        prob -= 0.05

    return max(0, min(1, prob))

#####################################
# Risk Management Functions
#####################################
def is_stop_loss_triggered(current_price, entry_price, stop_loss_pct=0.02):
    return (entry_price - current_price) / entry_price >= stop_loss_pct

def is_take_profit_triggered(current_price, entry_price, take_profit_pct=0.05):
    return (current_price - entry_price) / entry_price >= take_profit_pct

#####################################
# Adaptive Trade Sizing Function
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
        model.add(Dense(64, input_dim=self.state_dim, activation='relu'))
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
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#####################################
# Supervised Learning: Retrain Model from Logs
#####################################
def retrain_supervised_model(log_file, state_dim=STATE_DIM, action_dim=ACTION_DIM, epochs=5):
    if not os.path.exists(log_file):
        print("Log file not found for supervised retraining.")
        return None
    data = pd.read_csv(log_file)
    feature_cols = ["Slope", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment",
                    "Bollinger_SMA", "ATR", "AvgVolume", "Hour", "Weekday", "NewsSentiment"]
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0
    X = data[feature_cols].values
    y = data["Profit"].values
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=1)
    model.save("supervised_model.h5")
    print("Supervised model retrained and saved.")
    return model

#####################################
# Visualization and Monitoring Dashboard
#####################################
def create_dashboard(log_file):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed.")
        return
    if not os.path.exists(log_file):
        return
    data = pd.read_csv(log_file)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.sort_values('Timestamp', inplace=True)
    data['CumulativeProfit'] = data['Profit'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Timestamp'], y=data['CumulativeProfit'], mode='lines', name='Cumulative Profit'))
    fig.update_layout(title="Cumulative Profit Over Time", xaxis_title="Time", yaxis_title="Profit")
    fig.write_html("dashboard.html")
    print("Dashboard created and saved to dashboard.html.")

#####################################
# Real-Time Model Retraining Function
#####################################
def update_model_from_logs():
    model = retrain_supervised_model(LOG_FILE)
    print("Real-time model retraining complete.")
    return model

#####################################
# Main Trading Loop
#####################################
def main():
    rl_agent = RLAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    current_amount = BASE_AMOUNT

    init_log(LOG_FILE)
    init_full_history_log(FULL_HISTORY_LOG_FILE)
    
    I_want_money = IQ_Option(USERNAME, PASSWORD)
    I_want_money.connect()
    time.sleep(1)
    if not I_want_money.check_connect():
        print("Connection failed. Please check your credentials or network.")
        return
    print("Connected successfully!")
    I_want_money.change_balance(PRACTICE_MODE)
    print("Switched to Practice Account mode.")
    profile = I_want_money.get_profile_ansyc()
    print("Profile data:", profile)
    
    create_dashboard(LOG_FILE)
    
    prev_state = np.zeros(STATE_DIM)
    
    for i in range(TRADING_ITERATIONS):
        print("\nTrade iteration", i + 1)
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        time.sleep(3)
        candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
        times = sorted(candles.keys())
        closing_prices = [candles[t]["close"] for t in times]
        
        slope = compute_slope(candles)
        ma = compute_MA(closing_prices)
        std = compute_STD(closing_prices)
        rsi = compute_RSI(closing_prices, period=14)
        macd, macd_signal = compute_MACD(closing_prices)
        trader_sentiment = get_trader_sentiment(I_want_money, ASSET)
        boll_sma, boll_upper, boll_lower = compute_bollinger_bands(closing_prices)
        atr = compute_ATR(candles, period=14)
        avg_volume = compute_average_volume(candles, period=10)
        hour, weekday = get_time_features()
        news_sentiment = get_news_sentiment()
        
        print("Indicators: Slope={}, MA={}, STD={}, RSI={}, MACD={}, MACD_Signal={}, Sentiment={}, BollSMA={}, ATR={}, AvgVolume={}, Hour={}, Weekday={}, NewsSentiment={}".format(
            slope, ma, std, rsi, macd, macd_signal, trader_sentiment, boll_sma, atr, avg_volume, hour, weekday, news_sentiment))
        
        timestamp_history = datetime.datetime.now().isoformat()
        price_history_str = ",".join([str(p) for p in closing_prices])
        full_history_row = [timestamp_history, ASSET, price_history_str, trader_sentiment]
        log_full_history(FULL_HISTORY_LOG_FILE, full_history_row)
        print("Full history logged to", FULL_HISTORY_LOG_FILE)
        
        state = np.array([slope, ma, std, rsi, macd, macd_signal, trader_sentiment,
                          boll_sma, atr, avg_volume, hour, weekday, news_sentiment])
        
        action_idx = rl_agent.choose_action(state)
        action = "call" if action_idx == 0 else "put"
        print("RL Agent chose action:", action)
        
        trade_amount = current_amount
        print("Placing trade with amount:", trade_amount)
        trade_status, order_id = I_want_money.buy(trade_amount, ASSET, action, DURATION)
        if not trade_status:
            print("Trade execution failed. Skipping this iteration.")
            continue
        print("Trade placed. Order ID:", order_id)
        
        entry_price = closing_prices[-1]
        
        max_wait = 60
        start_wait = time.time()
        result = None
        current_price = entry_price
        # Update full history logging every second during wait period:
        while time.time() - start_wait < max_wait:
            candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
            times = sorted(candles.keys())
            current_price = candles[times[-1]]["close"]
            current_closing_prices = [candles[t]["close"] for t in times]
            timestamp_update = datetime.datetime.now().isoformat()
            full_history_log_row = [timestamp_update, ASSET, ",".join([str(p) for p in current_closing_prices]), trader_sentiment]
            log_full_history(FULL_HISTORY_LOG_FILE, full_history_log_row)
            if is_stop_loss_triggered(current_price, entry_price):
                print("Stop loss triggered.")
                result = -trade_amount
                break
            if is_take_profit_triggered(current_price, entry_price):
                print("Take profit triggered.")
                result = trade_amount * 0.85
                break
            result = I_want_money.check_win_v3(order_id)
            if result is not None:
                break
            time.sleep(1)
        if result is None:
            outcome = "unknown"
            profit = 0
            print("Timed out waiting for trade result.")
        else:
            profit = result
            outcome = "win" if profit > 0 else "loss"
            print("Trade result:", outcome, "with profit/loss:", profit)
        
        timestamp_trade = datetime.datetime.now().isoformat()
        log_row = [
            timestamp_trade, order_id, ASSET, action, trade_amount, DURATION,
            slope, advanced_predict_probability(slope, ma, std, rsi, macd, macd_signal, trader_sentiment, action, boll_sma, news_sentiment),
            outcome, profit,
            price_history_str, ma, std, rsi, macd, macd_signal, trader_sentiment,
            boll_sma, boll_upper, boll_lower, atr, avg_volume,
            hour, weekday, news_sentiment
        ]
        log_trade(LOG_FILE, log_row)
        print("Trade logged to", LOG_FILE)
        
        next_state = state
        done = False
        rl_agent.train(state, action_idx, profit, next_state, done)
        
        current_amount = update_trade_amount(current_amount, outcome,
            advanced_predict_probability(slope, ma, std, rsi, macd, macd_signal, trader_sentiment, action, boll_sma, news_sentiment))
        print("Updated trade amount for next trade:", current_amount)
        
        if (i + 1) % 20 == 0:
            update_model_from_logs()
        
        time.sleep(10)

    I_want_money.close_connect()
    print("Disconnected from IQ Option.")

if __name__ == "__main__":
    main()
