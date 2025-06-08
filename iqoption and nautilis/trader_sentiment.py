from iqoptionapi.stable_api import IQ_Option
import random
import time
not done
# User credentials
USERNAME = "eaglelab23@gmail.com"
PASSWORD = "Polboti1@"

# Asset to check sentiment
ASSET = "EURUSD-OTC"

# Connect to IQ Option
I_want_money = IQ_Option(USERNAME, PASSWORD)
I_want_money.connect()
import time
import csv
import os
import numpy as np
import datetime
import random
from iqoptionapi.stable_api import IQ_Option

# Try to import sklearn for online incremental model training
try:
    from sklearn.linear_model import SGDClassifier
    ONLINE_MODEL = SGDClassifier(loss='log', penalty='l2', max_iter=1000, tol=1e-3)
    ONLINE_MODEL_INITIALIZED = False
except ImportError:
    print("sklearn not installed, online model training disabled.")
    ONLINE_MODEL = None
    ONLINE_MODEL_INITIALIZED = False

# ---------------------------
# Configuration & Constants
# ---------------------------
USERNAME = "1@gmail.com"           # Your IQ Option email
PASSWORD = "1"                     # Your IQ Option password
PRACTICE_MODE = "PRACTICE"         # Switch to demo/practice mode
ASSET = "EURUSD-OTC"               # The asset to trade
ACTION = "call"                    # "call" (buy) or "put" (sell)
BASE_AMOUNT = 1                    # Base trade amount
DURATION = 1                       # Trade duration in minutes
TRADING_ITERATIONS = 100           # Number of trades before ending
LOG_FILE = "trade_log2.csv"        # Trade log file
ALL_TIME_LOG_FILE = "all_time_market_data_log.csv"  # Continuous market log file
TIMEFRAME = 60                     # Candle timeframe in seconds (60s = 1 minute)
NUM_CANDLES = 30                   # Number of candles to gather for indicator calculations

# --------------
# Helper Functions
# --------------

def init_log(file_path):
    """Initialize the trade log file with headers if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "OrderID", "Asset", "Action", "Amount", "Duration", 
                "Slope", "PredictedProbability", "Outcome", "Profit", 
                "PriceHistory", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment"
            ])

def log_trade(file_path, row):
    """Append a row of trade data to the trade log file."""
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def init_all_time_log(file_path):
    """Initialize the all‑time market data log file with headers."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset", "LastClosePrice", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment"
            ])

def log_all_time_data(file_path, row):
    """Append a row of continuous market data to the all‑time log."""
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def compute_slope(candles):
    """Compute a simple linear regression slope from candle closing prices."""
    times = sorted(candles.keys())
    if len(times) < 2:
        return 0.0
    closes = [candles[t]['close'] for t in times]
    x = np.arange(len(closes))
    slope, intercept = np.polyfit(x, closes, 1)
    return slope

def compute_MA(prices):
    """Compute the simple moving average (MA) for closing prices."""
    return np.mean(prices)

def compute_STD(prices):
    """Compute the standard deviation (STD) of closing prices (a measure of volatility)."""
    return np.std(prices)

def compute_RSI(prices, period=14):
    """Compute the Relative Strength Index (RSI) for a list of prices."""
    if len(prices) < period + 1:
        return 50  # Return neutral if insufficient data
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
    """Compute the Exponential Moving Average (EMA) for a given period."""
    ema = [prices[0]]
    alpha = 2 / (period + 1)
    for price in prices[1:]:
        ema.append(price * alpha + ema[-1] * (1 - alpha))
    return ema

def compute_MACD(prices):
    """
    Compute MACD (Moving Average Convergence Divergence) and its signal line.
    MACD = EMA(12) - EMA(26); Signal line = EMA(MACD, 9)
    """
    if len(prices) < 26:
        return 0, 0  # Return zero if not enough data
    ema12 = compute_EMA(prices, 12)
    ema26 = compute_EMA(prices, 26)
    macd_line = np.array(ema12[-len(ema26):]) - np.array(ema26)
    if len(macd_line) < 9:
        signal = np.mean(macd_line)
    else:
        signal_line = compute_EMA(list(macd_line), 9)
        signal = signal_line[-1]
    return macd_line[-1], signal

def get_active_id(api, asset_name):
    """
    Retrieve the unique active ID for the given asset name.
    Attempts multiple methods (e.g. get_all_ACTIVES or get_all_digital) as available.
    """
    # Try using get_all_ACTIVES if available
    try:
        actives = api.get_all_ACTIVES()  # Expected to return a dict grouped by asset type
        for category in actives.values():
            for asset in category:
                if asset['name'].upper() == asset_name.upper():
                    return asset['active_id']
    except Exception as ex:
        print("get_all_ACTIVES not available, trying alternative. Error:", ex)

    # Try using digital assets list if available
    try:
        digital_actives = api.get_all_digital()
        for asset in digital_actives:
            if asset_name.upper() in asset['name'].upper():
                return asset['active_id']
    except Exception as ex:
        print("get_all_digital not available:", ex)

    return None

def get_trader_sentiment(api, asset_name):
    """
    Retrieve real trader sentiment using the IQ Option API.
    It attempts to use the asset's unique ID to call get_traders_mood().
    If this isn’t possible (or the API doesn’t provide it), falls back to a simulated random value.
    """
    active_id = get_active_id(api, asset_name)
    if active_id is None:
        print("Could not find active ID for asset:", asset_name, "- using simulated sentiment.")
        return random.uniform(0.4, 0.6)
    try:
        # Expected format: {'call': <value>, 'put': <value>}
        sentiment_data = api.get_traders_mood(active_id)
        if sentiment_data and 'call' in sentiment_data:
            return sentiment_data['call']  # Use call sentiment for this example
        else:
            print("Trader sentiment data not found; using simulated value.")
            return random.uniform(0.4, 0.6)
    except Exception as e:
        print("Error fetching trader sentiment:", e, "- using simulated value.")
        return random.uniform(0.4, 0.6)

def advanced_predict_probability(slope, ma_val, std_val, rsi_val, macd_val, macd_signal_val, trader_sentiment, action, current_price):
    """
    Predict the probability of a successful trade using a heuristic that combines multiple indicators.
    If an online learning model (SGDClassifier) is available and trained, it combines the model’s prediction
    with the heuristic value.
    """
    # Base heuristic model:
    prob = 0.5
    if action == "call":
        prob += 0.15 if slope > 0 else -0.15
        prob += 0.10 if rsi_val < 70 else -0.10
        prob += 0.10 if macd_val > macd_signal_val else -0.10
    elif action == "put":
        prob += 0.15 if slope < 0 else -0.15
        prob += 0.10 if rsi_val > 30 else -0.10
        prob += 0.10 if macd_val < macd_signal_val else -0.10
    # Adjust based on trader sentiment:
    if trader_sentiment > 0.6:
        prob += 0.1
    elif trader_sentiment < 0.4:
        prob -= 0.1
    heuristic_prob = max(0, min(1, prob))
    
    global ONLINE_MODEL, ONLINE_MODEL_INITIALIZED
    if ONLINE_MODEL is not None and ONLINE_MODEL_INITIALIZED:
        features = np.array([[slope, ma_val, std_val, rsi_val, macd_val, macd_signal_val, trader_sentiment]])
        # Get decision value and convert to a probability via the logistic function
        decision_value = ONLINE_MODEL.decision_function(features)[0]
        model_prob = 1 / (1 + np.exp(-decision_value))
        combined_prob = (heuristic_prob + model_prob) / 2.0
        return combined_prob
    else:
        return heuristic_prob

def wait_for_result(api, order_id, timeout=60):
    """
    Poll the API for the result of a trade (using check_win_v3) until the timeout expires.
    Returns the numerical profit/loss when available.
    """
    start_time = time.time()
    result = None
    while time.time() - start_time < timeout:
        try:
            result = api.check_win_v3(order_id)
        except Exception as e:
            print("Error checking trade result:", e)
            result = None
        if result is not None:
            return result
        time.sleep(1)
    return None

def update_trade_amount(current_amount, outcome, predicted_prob, threshold=0.9):
    """
    Adaptive position sizing: Increase the trade amount when winning with high probability,
    otherwise reset to the base amount.
    """
    if outcome == "win" and predicted_prob >= threshold:
        return current_amount * 1.5
    else:
        return BASE_AMOUNT

def update_online_model(features, label):
    """
    Update the online machine learning model with the latest trade outcome.
    'features' is an array of indicator values and 'label' is 1 for win, 0 for loss.
    Uses partial_fit for incremental learning.
    """
    global ONLINE_MODEL, ONLINE_MODEL_INITIALIZED
    if ONLINE_MODEL is None:
        return
    if not ONLINE_MODEL_INITIALIZED:
        ONLINE_MODEL.partial_fit(features, [label], classes=[0, 1])
        ONLINE_MODEL_INITIALIZED = True
    else:
        ONLINE_MODEL.partial_fit(features, [label])

# ---------------------------
# Main Trading Loop
# ---------------------------

def main():
    current_amount = BASE_AMOUNT
    init_log(LOG_FILE)
    init_all_time_log(ALL_TIME_LOG_FILE)
    
    # Connect to IQ Option
    I_want_money = IQ_Option(USERNAME, PASSWORD)
    I_want_money.connect()
    time.sleep(2)
    if not I_want_money.check_connect():
        print("Connection failed. Please check credentials or network.")
        return
    print("Connected successfully!")
    
    I_want_money.change_balance(PRACTICE_MODE)
    print("Switched to", PRACTICE_MODE, "account mode.")
    profile = I_want_money.get_profile_ansyc()
    print("Profile data:", profile)
    
    for i in range(TRADING_ITERATIONS):
        print("\nTrade iteration", i + 1)
        # STEP 1: Gather Market Data & Compute Indicators
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        time.sleep(3)  # Allow candles to accumulate
        candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
        times = sorted(candles.keys())
        closing_prices = [candles[t]['close'] for t in times]
        
        if len(closing_prices) < NUM_CANDLES:
            print("Insufficient candle data; skipping iteration.")
            continue
        
        slope = compute_slope(candles)
        ma_val = compute_MA(closing_prices)
        std_val = compute_STD(closing_prices)
        rsi_val = compute_RSI(closing_prices, period=14)
        macd_val, macd_signal_val = compute_MACD(closing_prices)
        current_price = closing_prices[-1]
        trader_sentiment = get_trader_sentiment(I_want_money, ASSET)
        
        print(f"Indicators: Slope={slope:.4f}, MA={ma_val:.4f}, STD={std_val:.4f}, RSI={rsi_val:.2f}, "
              f"MACD={macd_val:.4f}, MACD_Signal={macd_signal_val:.4f}, Sentiment={trader_sentiment:.4f}")
        
        # Log all‑time market data for training purposes
        timestamp = datetime.datetime.now().isoformat()
        all_time_row = [timestamp, ASSET, current_price, ma_val, std_val, rsi_val, macd_val, macd_signal_val, trader_sentiment]
        log_all_time_data(ALL_TIME_LOG_FILE, all_time_row)
        
        # STEP 2: Predict Trade Success Probability using Advanced Model
        predicted_prob = advanced_predict_probability(slope, ma_val, std_val, rsi_val, macd_val, macd_signal_val, trader_sentiment, ACTION, current_price)
        print(f"Predicted probability for action {ACTION}: {predicted_prob:.4f}")
        
        trade_amount = current_amount
        print("Placing trade with amount:", trade_amount)
        
        # STEP 3: Place the Trade
        trade_status, order_id = I_want_money.buy(trade_amount, ASSET, ACTION, DURATION)
        if not trade_status:
            print("Trade execution failed. Check parameters, balance, or asset status.")
            continue
        print("Trade placed. Order ID:", order_id)
        
        # STEP 4: Wait for Trade Result
        print("Waiting for trade outcome...")
        result = wait_for_result(I_want_money, order_id, timeout=60)
        if result is None:
            outcome = "unknown"
            profit = 0
            print("Timed out waiting for trade result.")
        else:
            profit = result
            outcome = "win" if profit > 0 else "loss"
            print(f"Trade result: {outcome} with profit/loss: {profit}")
        
        # STEP 5: Log All Trade Data
        price_history_str = ",".join(map(str, closing_prices))
        trade_log_row = [
            timestamp, order_id, ASSET, ACTION, trade_amount, DURATION, 
            slope, predicted_prob, outcome, profit, 
            price_history_str, ma_val, std_val, rsi_val, macd_val, macd_signal_val, trader_sentiment
        ]
        log_trade(LOG_FILE, trade_log_row)
        print("Trade logged to", LOG_FILE)
        
        # STEP 6: Update Online Model (if available) and Adjust Trade Amount
        features = np.array([slope, ma_val, std_val, rsi_val, macd_val, macd_signal_val, trader_sentiment]).reshape(1, -1)
        label = 1 if outcome == "win" else 0
        update_online_model(features, label)
        current_amount = update_trade_amount(current_amount, outcome, predicted_prob)
        print("Updated trade amount for next trade:", current_amount)
        
        time.sleep(10)  # Pause between iterations
    
    I_want_money.close_connect()
    print("Disconnected from IQ Option.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
    except Exception as ex:
        print("An error occurred:", ex)

if not I_want_money.check_connect():
    print("Connection failed! Check credentials or network.")
    exit()

print("Connected successfully!")

# Attempt multiple methods to retrieve trader sentiment
def get_trader_sentiment(api, asset):
    asset_id = api.get_active_by_name(asset)  # Try to get asset ID
    
    if asset_id is None:
        print("Could not find asset ID.")
        return None
    
    methods = {
        "get_traders_mood": lambda: api.get_traders_mood(asset_id),
        "get_financial_information": lambda: api.get_financial_information(asset),
        "get_technical_indicators": lambda: api.get_technical_indicators(asset_id)
    }

    results = {}

    for method_name, func in methods.items():
        try:
            result = func()
            results[method_name] = result if result else "No data returned"
        except Exception as e:
            results[method_name] = f"Error: {str(e)}"
    
    return results

# Run the function
sentiment_data = get_trader_sentiment(I_want_money, ASSET)

# Print the results to see which method returns valid data
print("\n### Trader Sentiment Check Results ###")
for method, value in sentiment_data.items():
    print(f"{method}: {value}")

I_want_money.close_connect()
