from iqoptionapi.stable_api import IQ_Option
import time
import csv
import os
import numpy as np
import datetime
import random

# ---------------------------
# Configuration & Constants
# ---------------------------
USERNAME = "eaglelab23@gmail.com"          # Your IQ Option email
PASSWORD = "Polboti1@"                     # Your IQ Option password
PRACTICE_MODE = "PRACTICE"                 # Switch to demo mode

ASSET = "EURUSD-OTC"                       # The asset to trade (ensure it is open for trading)
ACTION = "call"                            # "call" (buy) or "put" (sell)
BASE_AMOUNT = 1                            # Base trade amount (in USD, for example)
DURATION = 1                               # Trade duration in minutes
TRADING_ITERATIONS = 100                   # How many trades before ending
LOG_FILE = "trade_log2.csv"                # Log file with advanced indicators
FULL_HISTORY_LOG_FILE = "full_history_log.csv"  # Log file for full price history and sentiment

# Candle parameters (we use more candles so we have enough data for our indicators)
TIMEFRAME = 60                             # Candle length in seconds (e.g., 60s = 1 minute)
NUM_CANDLES = 30                           # Number of candles to gather

# ---------------------------
# Helper Functions
# ---------------------------
def init_log(file_path):
    """Initialize the CSV log file with headers if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "OrderID", "Asset", "Action", "Amount", "Duration",
                "Slope", "PredictedProbability", "Outcome", "Profit",
                "PriceHistory", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment"
            ])

def init_full_history_log(file_path):
    """Initialize the full history log file with headers if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset", "PriceHistory", "TraderSentiment"
            ])

def log_trade(file_path, row):
    """Append a row of trade data to the CSV file."""
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def log_full_history(file_path, row):
    """Append a row of full history data to the CSV file."""
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def compute_slope(candles):
    """
    Compute a simple linear regression slope from candle data.
    `candles` is a dict with timestamps as keys; each value is assumed to have a 'close' price.
    """
    times = sorted(candles.keys())
    if len(times) < 2:
        return 0.0
    closes = [candles[t]["close"] for t in times]
    x = np.arange(len(closes))
    slope, intercept = np.polyfit(x, closes, 1)
    return slope

def compute_MA(prices):
    """Compute the simple moving average of the closing prices."""
    return np.mean(prices)

def compute_STD(prices):
    """Compute the standard deviation of the closing prices (a volatility indicator)."""
    return np.std(prices)

def compute_RSI(prices, period=14):
    """Compute the Relative Strength Index (RSI) for the given list of prices."""
    if len(prices) < period + 1:
        return 50  # Neutral value if insufficient data
    changes = np.diff(prices)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_EMA(prices, period):
    """Compute the Exponential Moving Average (EMA) for the list of prices."""
    ema = [prices[0]]
    alpha = 2 / (period + 1)
    for price in prices[1:]:
        ema.append(price * alpha + ema[-1] * (1 - alpha))
    return ema

def compute_MACD(prices):
    """
    Compute MACD and its signal line.
    MACD = EMA(12) - EMA(26)
    Signal line = EMA of MACD using period 9
    Returns the latest MACD and signal line values.
    """
    if len(prices) < 26:
        return 0, 0  # Insufficient data
    ema12 = compute_EMA(prices, 12)
    ema26 = compute_EMA(prices, 26)
    macd_line = np.array(ema12) - np.array(ema26)
    if len(macd_line) < 9:
        signal = np.mean(macd_line)
    else:
        signal_line = compute_EMA(list(macd_line), 9)
        signal = signal_line[-1]
    return macd_line[-1], signal

def get_trader_sentiment(api, asset):
    """
    Attempt to fetch real trader sentiment for the given asset from IQ Option.
    If the method is not supported, it falls back to a random value.
    """
    try:
        # If your API supports a method like this, it will return a real sentiment value.
        sentiment = api.get_trader_sentiment(asset)
        return sentiment
    except Exception as e:
        print("Real trader sentiment not available, using fallback. Error:", e)
        return random.uniform(0, 1)

def advanced_predict_probability(slope, ma, std, rsi, macd, macd_signal, trader_sentiment, action):
    """
    An advanced dummy prediction function that combines multiple indicators.
    This heuristic is for demonstration only; later, you could train a model with these features.
    
    For a "call" trade:
      - Prefer a positive slope, moderately low RSI (not overbought), a bullish MACD (MACD > signal),
        and higher trader sentiment.
    For a "put" trade, reverse the conditions.
    """
    prob = 0.5  # Base probability
    
    if action == "call":
        if slope > 0:
            prob += 0.15
        else:
            prob -= 0.15
        if rsi < 70:
            prob += 0.10
        else:
            prob -= 0.10
        if macd > macd_signal:
            prob += 0.10
        else:
            prob -= 0.10
    elif action == "put":
        if slope < 0:
            prob += 0.15
        else:
            prob -= 0.15
        if rsi > 30:
            prob += 0.10
        else:
            prob -= 0.10
        if macd < macd_signal:
            prob += 0.10
        else:
            prob -= 0.10

    # Adjust based on trader sentiment:
    if trader_sentiment > 0.6:
        prob += 0.1
    elif trader_sentiment < 0.4:
        prob -= 0.1

    prob = max(0, min(1, prob))
    return prob

def wait_for_result(api, order_id, timeout=60):
    """
    Poll the API for the result of a trade.
    (Uses your API's method; here we assume 'check_win_v3' is available.)
    """
    start_time = time.time()
    result = None
    while time.time() - start_time < timeout:
        result = api.check_win_v3(order_id)
        if result is not None:
            return result
        time.sleep(1)
    return None

def update_trade_amount(current_amount, outcome, predicted_prob, threshold=0.9):
    """
    Adaptive position sizing: increase the trade amount if a win is achieved with high prediction confidence,
    otherwise reset to the base amount.
    """
    if outcome == "win" and predicted_prob >= threshold:
        return current_amount * 1.5
    else:
        return BASE_AMOUNT

# ---------------------------
# Main Trading Loop
# ---------------------------
def main():
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
    
    for i in range(TRADING_ITERATIONS):
        print("\nTrade iteration", i + 1)
        # --- STEP 1: Gather Market Data & Compute Indicators ---
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        time.sleep(3)  # Allow time for candles to accumulate.
        candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
        
        # Sort candles by time and extract closing prices.
        times = sorted(candles.keys())
        closing_prices = [candles[t]["close"] for t in times]
        
        slope = compute_slope(candles)
        ma = compute_MA(closing_prices)
        std = compute_STD(closing_prices)
        rsi = compute_RSI(closing_prices, period=14)
        macd, macd_signal = compute_MACD(closing_prices)
        trader_sentiment = get_trader_sentiment(I_want_money, ASSET)
        
        print("Indicators: Slope={}, MA={}, STD={}, RSI={}, MACD={}, MACD_Signal={}, Sentiment={}".format(
            slope, ma, std, rsi, macd, macd_signal, trader_sentiment))
        
        # --- STEP 1B: Log Full Price History and Sentiment Data ---
        timestamp_history = datetime.datetime.now().isoformat()
        price_history_str = ",".join([str(p) for p in closing_prices])
        full_history_row = [timestamp_history, ASSET, price_history_str, trader_sentiment]
        log_full_history(FULL_HISTORY_LOG_FILE, full_history_row)
        print("Full history logged to", FULL_HISTORY_LOG_FILE)
        
        # --- STEP 2: Predict Trade Success Probability using Advanced Model ---
        predicted_prob = advanced_predict_probability(
            slope, ma, std, rsi, macd, macd_signal, trader_sentiment, ACTION)
        print("Predicted probability for action {}: {}".format(ACTION, predicted_prob))
        
        trade_amount = current_amount
        print("Placing trade with amount: {}".format(trade_amount))
        
        # --- STEP 3: Place the Trade ---
        trade_status, order_id = I_want_money.buy(trade_amount, ASSET, ACTION, DURATION)
        if not trade_status:
            print("Trade execution failed. Check parameters, balance, or asset status.")
            continue
        print("Trade placed. Order ID: {}".format(order_id))
        
        # --- STEP 4: Wait for Trade Result ---
        print("Waiting for trade outcome...")
        result = wait_for_result(I_want_money, order_id, timeout=60)
        if result is None:
            outcome = "unknown"
            profit = 0
            print("Timed out waiting for trade result.")
        else:
            profit = result
            outcome = "win" if profit > 0 else "loss"
            print("Trade result: {} with profit/loss: {}".format(outcome, profit))
        
        # --- STEP 5: Log All Trade Data Including Advanced Features ---
        timestamp_trade = datetime.datetime.now().isoformat()
        log_row = [
            timestamp_trade, order_id, ASSET, ACTION, trade_amount, DURATION,
            slope, predicted_prob, outcome, profit,
            price_history_str, ma, std, rsi, macd, macd_signal, trader_sentiment
        ]
        log_trade(LOG_FILE, log_row)
        print("Trade logged to", LOG_FILE)
        
        # --- STEP 6: Update Trade Amount for Next Trade ---
        current_amount = update_trade_amount(current_amount, outcome, predicted_prob)
        print("Updated trade amount for next trade: {}".format(current_amount))
        
        time.sleep(10)  # Pause between trades
    
    I_want_money.close_connect()
    print("Disconnected from IQ Option.")

if __name__ == "__main__":
    main()
