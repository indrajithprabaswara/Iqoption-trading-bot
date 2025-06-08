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
USERNAME = "@gmail.com"             # Your IQ Option email
PASSWORD = ""                      # Your IQ Option password - PLEASE USE YOUR ACTUAL PASSWORD
PRACTICE_MODE = "PRACTICE"          # Switch to demo mode ("PRACTICE") or real ("REAL")

ASSET = "EURUSD-OTC"                # The asset to trade (ensure it is open for trading)
ACTION = "call"                     # Default action: "call" (buy) or "put" (sell) - can be dynamic later
BASE_AMOUNT = 1                     # Base trade amount (in USD, for example)
DURATION = 1                        # Trade duration in minutes
TRADING_ITERATIONS = 100            # How many trading cycles before ending
LOG_FILE = "trade_log2.csv"         # Log file for individual trade details
ALL_TIME_LOG_FILE = "all_time_market_data_log.csv" # Log file for continuous market data

# Candle parameters
TIMEFRAME = 60                      # Candle length in seconds (e.g., 60s = 1 minute)
NUM_CANDLES = 30                    # Number of candles to gather for indicators (e.g., 30 for 30 minutes of data)

# ---------------------------
# Helper Functions - Logging
# ---------------------------
def init_log(file_path, headers):
    """Initialize a CSV log file with headers if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def log_data(file_path, row):
    """Append a row of data to a CSV file."""
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---------------------------
# Helper Functions - Technical Indicators
# ---------------------------
def get_closing_prices(candles):
    """Extracts and sorts closing prices from candle data."""
    if not candles or not isinstance(candles, dict):
        return []
    
    # Filter out potential non-integer keys and ensure values are dicts with 'close'
    valid_candle_times = sorted([k for k in candles.keys() if isinstance(k, (int, float))])
    
    closing_prices = []
    for t in valid_candle_times:
        if isinstance(candles[t], dict) and 'close' in candles[t]:
            closing_prices.append(candles[t]['close'])
        # else:
            # print(f"Warning: Candle data for timestamp {t} is not in expected format: {candles[t]}")
    return closing_prices

def compute_slope(closing_prices):
    """Compute a simple linear regression slope from closing prices."""
    if len(closing_prices) < 2:
        return 0.0
    x = np.arange(len(closing_prices))
    try:
        slope, _ = np.polyfit(x, closing_prices, 1)
        return slope
    except Exception as e:
        # print(f"Error computing slope: {e}. Prices: {closing_prices}")
        return 0.0


def compute_MA(closing_prices, period=14):
    """Compute the simple moving average of the closing prices."""
    if len(closing_prices) < period:
        return np.mean(closing_prices) if closing_prices else 0.0 # Return mean of available if less than period
    return np.mean(closing_prices[-period:])

def compute_STD(closing_prices, period=14):
    """Compute the standard deviation of the closing prices (a volatility indicator)."""
    if len(closing_prices) < period:
        return np.std(closing_prices) if closing_prices else 0.0
    return np.std(closing_prices[-period:])

def compute_RSI(closing_prices, period=14):
    """Compute the Relative Strength Index (RSI) for the given list of prices."""
    if len(closing_prices) < period + 1:
        return 50.0  # Neutral value if insufficient data

    prices_array = np.array(closing_prices)
    changes = np.diff(prices_array)
    
    if len(changes) < period: # Not enough changes to calculate RSI for the period
        return 50.0

    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0) # Losses are positive values

    # Calculate average gains and losses using Wilder's smoothing method (exponential moving average)
    # For the first calculation, simple average is fine
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0: # Avoid division by zero
        return 100.0 if avg_gain > 0 else 50.0 # RSI is 100 if all losses are zero and gains exist
        
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_EMA(closing_prices, period):
    """Compute the Exponential Moving Average (EMA) for the list of prices."""
    if not closing_prices:
        return []
    if len(closing_prices) < period: # Not enough data for a full EMA, can return simple MA or partial EMA
        # For simplicity, returning what can be computed or an empty list if too short
        if len(closing_prices) == 0: return []
        # Fallback to SMA if not enough data for proper EMA start
        # return [np.mean(closing_prices)] * len(closing_prices) # Or handle differently

    ema = []
    # First EMA is SMA
    sma = np.mean(closing_prices[:period])
    ema.append(sma)
    
    alpha = 2 / (period + 1.0)
    for price in closing_prices[period:]:
        ema_val = price * alpha + ema[-1] * (1 - alpha)
        ema.append(ema_val)
    return ema # This EMA list will be shorter than input if period > 1

def compute_MACD(closing_prices, short_period=12, long_period=26, signal_period=9):
    """
    Compute MACD and its signal line.
    Returns the latest MACD value and latest signal line value.
    """
    if len(closing_prices) < long_period:
        return 0.0, 0.0  # Insufficient data

    ema_short = compute_EMA(closing_prices, short_period)
    ema_long = compute_EMA(closing_prices, long_period)

    # Align lengths of EMAs for MACD calculation
    # MACD line calculation needs aligned series. EMA calculation above might return shorter series.
    # A more robust EMA function would return series of same length as input, padded at start.
    # For now, let's re-calculate EMAs ensuring full length for simplicity of alignment.
    
    prices_arr = np.array(closing_prices)
    if len(prices_arr) < long_period: return 0.0, 0.0

    # Using pandas for robust EMA calculation if available, otherwise simplified numpy
    try:
        import pandas as pd
        df = pd.DataFrame(prices_arr, columns=['price'])
        ema_short_pd = df['price'].ewm(span=short_period, adjust=False).mean()
        ema_long_pd = df['price'].ewm(span=long_period, adjust=False).mean()
        macd_line_pd = ema_short_pd - ema_long_pd
        signal_line_pd = macd_line_pd.ewm(span=signal_period, adjust=False).mean()
        return macd_line_pd.iloc[-1], signal_line_pd.iloc[-1]

    except ImportError: # Fallback to numpy based if pandas not available
        # This simplified EMA needs adjustment for MACD calculation
        # For a proper MACD, EMA series should be of same length as price series.
        # The compute_EMA above returns a shorter series. We need to adjust.
        # A common way is to pad with NaNs or use a library.
        # Given the constraints, this part might be less accurate than pandas version.
        # print("Warning: pandas not installed. MACD calculation might be less precise.")
        
        # Recalculate EMAs for full length (simplified)
        alpha_short = 2 / (short_period + 1.0)
        alpha_long = 2 / (long_period + 1.0)
        
        ema_s = [0.0] * len(prices_arr)
        ema_l = [0.0] * len(prices_arr)
        
        ema_s[0] = prices_arr[0]
        ema_l[0] = prices_arr[0]
        
        for i in range(1, len(prices_arr)):
            ema_s[i] = prices_arr[i] * alpha_short + ema_s[i-1] * (1-alpha_short)
            ema_l[i] = prices_arr[i] * alpha_long + ema_l[i-1] * (1-alpha_long)

        macd_line = np.array(ema_s) - np.array(ema_l)
        
        if len(macd_line) < signal_period:
            return macd_line[-1] if macd_line.size > 0 else 0.0, np.mean(macd_line) if macd_line.size > 0 else 0.0

        alpha_signal = 2 / (signal_period + 1.0)
        signal_l = [0.0] * len(macd_line)
        signal_l[0] = macd_line[0]
        for i in range(1, len(macd_line)):
            signal_l[i] = macd_line[i] * alpha_signal + signal_l[i-1] * (1-alpha_signal)
            
        return macd_line[-1], signal_l[-1]


def get_trader_sentiment(api, asset_name):
    """
    Simulates fetching trader sentiment.
    In a real scenario, this function would make an API call to IQ Option
    to get the current trader sentiment for the given asset.
    Example: api.get_traders_mood(asset_id) or similar.
    For now, it returns a random value and prints a message.
    """
    # print(f"INFO: Attempting to fetch real trader sentiment for {asset_name} (currently simulated).")
    # TODO: Replace with actual API call to fetch trader sentiment.
    # Example (conceptual - requires knowing the actual API method and asset_id):
    # try:
    #     asset_id = get_asset_id_from_name(api, asset_name) # You'd need to implement this
    #     if asset_id:
    #         mood = api.get_traders_mood(asset_id) # This is a hypothetical method name
    #         # Mood might be {'call': 0.6, 'put': 0.4}
    #         # You could return mood['call'], or (mood['call'] - mood['put'])
    #         # For a single sentiment value (e.g., % bullish):
    #         # return mood.get('call', 0.5) # Default to neutral if not found
    # except Exception as e:
    #     print(f"Warning: Could not fetch real trader sentiment due to: {e}. Using simulated value.")
    
    # Returning a random sentiment value (0.0 to 1.0, where >0.5 is bullish)
    simulated_sentiment = random.uniform(0.3, 0.7) # Simulate a sentiment range
    return simulated_sentiment

# ---------------------------
# Prediction Logic
# ---------------------------
def advanced_predict_probability(slope, ma_val, std_val, rsi_val, macd_val, macd_signal_val, sentiment_val, current_price, action):
    """
    Advanced heuristic prediction function combining multiple indicators.
    This is a placeholder for a more sophisticated model (e.g., trained ML model).
    """
    prob = 0.5  # Base probability (neutral)
    confidence_score = 0 # Accumulate confidence points

    # Factor 1: Trend (Slope & MA)
    if action == "call":
        if slope > 0.00001: confidence_score += 15 # Slight positive slope
        if current_price > ma_val: confidence_score += 10 # Price above MA
    elif action == "put":
        if slope < -0.00001: confidence_score += 15 # Slight negative slope
        if current_price < ma_val: confidence_score += 10 # Price below MA

    # Factor 2: Momentum (RSI)
    if action == "call":
        if rsi_val < 30: confidence_score += 10 # Oversold, potential bounce up
        elif rsi_val > 70: confidence_score -= 5 # Overbought, risky for call
    elif action == "put":
        if rsi_val > 70: confidence_score += 10 # Overbought, potential drop
        elif rsi_val < 30: confidence_score -= 5 # Oversold, risky for put
    
    # Factor 3: MACD
    if macd_val > macd_signal_val: # Bullish crossover/momentum
        if action == "call": confidence_score += 15
        else: confidence_score -= 5 # Contradicts put
    elif macd_val < macd_signal_val: # Bearish crossover/momentum
        if action == "put": confidence_score += 15
        else: confidence_score -= 5 # Contradicts call

    # Factor 4: Volatility (STD) - Higher volatility can mean more risk or opportunity
    # This is a simplified interpretation: lower std might mean more predictable trend
    if std_val < (current_price * 0.001): # If std is less than 0.1% of price (low vol)
        confidence_score += 5 # Slightly more confident in current trend signals
    elif std_val > (current_price * 0.005): # If std is more than 0.5% of price (high vol)
        confidence_score -= 5 # Slightly less confident due to high volatility

    # Factor 5: Trader Sentiment
    # Assuming sentiment_val is % bullish (0 to 1)
    if sentiment_val > 0.6: # More traders are bullish
        if action == "call": confidence_score += 10
        else: confidence_score -= 5
    elif sentiment_val < 0.4: # More traders are bearish
        if action == "put": confidence_score += 10
        else: confidence_score -= 5

    # Convert confidence score to probability (simple scaling)
    # Max possible positive score could be around 15+10+10+15+5+10 = 65
    # Min possible negative score could be around -5-5-5-5 = -20 (less symmetric)
    # Let's scale score from a range (e.g., -30 to +70) to probability (0 to 1)
    # For simplicity, let's use a direct mapping:
    prob = 0.5 + (confidence_score / 100.0) * 0.5 # Scale score to affect +/- 0.25 of probability from base 0.5
                                                # Max change is 70/100 * 0.35 = 0.35. So prob can go up to 0.85 or down to 0.15

    return max(0.01, min(0.99, prob)) # Clamp probability between 1% and 99%

# ---------------------------
# Trade Execution & Management
# ---------------------------
def wait_for_result(api, order_id, timeout_seconds=DURATION*60 + 15): 
    """Poll the API for the result of a trade reliably."""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            print(f"Timeout waiting for result of order {order_id}.")
            return None

        try:
            result = api.check_win_v3(order_id)
            if result is not None:
                return result
        except Exception as e:
            print(f"Exception while checking result: {e}. Reconnecting now...")
            api.connect()
            time.sleep(5)  # Allow time for reconnection
        time.sleep(2)  # Regular check interval

def update_trade_amount(current_amount, outcome, predicted_prob, confidence_threshold=0.75, increase_factor=1.5, max_amount_factor=5):
    """Adaptive position sizing."""
    max_amount = BASE_AMOUNT * max_amount_factor
    if outcome == "win" and predicted_prob >= confidence_threshold:
        new_amount = current_amount * increase_factor
        return min(new_amount, max_amount) # Cap the amount
    else: # Loss or low confidence win
        return BASE_AMOUNT # Reset to base amount

# ---------------------------
# Main Trading Loop
# ---------------------------
def main():
    # Initialize log files with their respective headers
    trade_log_headers = [
        "Timestamp", "OrderID", "Asset", "Action", "Amount", "Duration",
        "Slope", "PredictedProbability", "Outcome", "Profit",
        "PriceHistory", "CurrentPrice", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment"
    ]
    init_log(LOG_FILE, trade_log_headers)

    all_time_log_headers = [
        "Timestamp", "Asset", "LastClosePrice", "MovingAverage", "StdDev", "RSI", "MACD", "MACD_Signal", "TraderSentiment"
    ]
    init_log(ALL_TIME_LOG_FILE, all_time_log_headers)

    current_trade_amount = BASE_AMOUNT
    
    # Connect to IQ Option
    iq_api = IQ_Option(USERNAME, PASSWORD)
    # print("Connecting to IQ Option...")
    iq_api.connect() # This will print connection status internally or raise error

    if not iq_api.check_connect():
        # print("Connection failed. Please check credentials or network and try again.")
        return
    # print("Connected successfully!")
    
    iq_api.change_balance(PRACTICE_MODE)
    # print(f"Switched to {PRACTICE_MODE} account.")
    
    # Optional: Get and print profile info once
    # profile = iq_api.get_profile_ansyc() # Async, careful with use
    # print("Profile data:", profile)
    
    for i in range(TRADING_ITERATIONS):
        # print(f"\n----- Trade Iteration {i+1}/{TRADING_ITERATIONS} -----")
        
        # STEP 1: Gather Market Data & Compute Indicators
        # print(f"Fetching {NUM_CANDLES} candles for {ASSET}, timeframe {TIMEFRAME}s...")
        iq_api.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        time.sleep(2) # Allow some time for candles to be fetched
        raw_candles = iq_api.get_realtime_candles(ASSET, TIMEFRAME)
        
        closing_prices = get_closing_prices(raw_candles)

        if not closing_prices or len(closing_prices) < 2: # Need at least 2 points for slope, more for others
            # print(f"Warning: Not enough closing price data to proceed ({len(closing_prices)} points). Skipping iteration.")
            time.sleep(TIMEFRAME // 2) # Wait before retrying
            continue
            
        current_price = closing_prices[-1]

        # Compute technical indicators
        slope = compute_slope(closing_prices)
        ma = compute_MA(closing_prices)
        std = compute_STD(closing_prices)
        rsi = compute_RSI(closing_prices)
        macd, macd_signal = compute_MACD(closing_prices)
        trader_sentiment = get_trader_sentiment(iq_api, ASSET) # Pass API and asset

        # print(f"Indicators: Price={current_price:.5f}, Slope={slope:.5f}, MA={ma:.5f}, STD={std:.5f}, RSI={rsi:.2f}, MACD={macd:.5f}, Signal={macd_signal:.5f}, Sentiment={trader_sentiment:.2f}")

        # Log data to all_time_market_data_log.csv
        all_time_log_row = [
            datetime.datetime.now().isoformat(), ASSET, current_price,
            ma, std, rsi, macd, macd_signal, trader_sentiment
        ]
        log_data(ALL_TIME_LOG_FILE, all_time_log_row)
        # print(f"Market data logged to {ALL_TIME_LOG_FILE}")

        # STEP 2: Determine Action (Dynamic Action Strategy - Example)
        # For this example, we'll keep ACTION fixed as per global config,
        # but you could implement logic here to decide 'call' or 'put'
        current_action = ACTION 
        # Example: if macd > macd_signal and rsi < 70: current_action = "call" else: current_action = "put"


        # STEP 3: Predict Trade Success Probability
        predicted_prob = advanced_predict_probability(slope, ma, std, rsi, macd, macd_signal, trader_sentiment, current_price, current_action)
        # print(f"Predicted probability for action '{current_action}': {predicted_prob:.2%}")
        
        # STEP 4: Decide whether to trade (e.g., based on probability threshold)
        MIN_PROB_THRESHOLD = 0.60 # Example: Only trade if probability is > 60%
        if predicted_prob < MIN_PROB_THRESHOLD:
            # print(f"Skipping trade: Predicted probability {predicted_prob:.2%} is below threshold {MIN_PROB_THRESHOLD:.0%}.")
            time.sleep(10) # Wait before next iteration
            continue

        # print(f"Placing trade with amount: {current_trade_amount:.2f}")
        
        # STEP 5: Place the Trade
        # Note: The buy method might be buy_digital_spot or buy_binary depending on API version / asset type
        # Using generic 'buy' which often defaults or handles it.
        trade_successful, order_id = iq_api.buy(current_trade_amount, ASSET, current_action, DURATION)
        
        outcome = "N/A"
        profit = 0.0

        if trade_successful:
            # print(f"Trade placed successfully! Order ID: {order_id}")
            
            # STEP 6: Wait for Trade Result
            # print("Waiting for trade outcome...")
            profit_or_loss = wait_for_result(iq_api, order_id)
            
            if profit_or_loss is not None:
                profit = profit_or_loss
                outcome = "win" if profit > 0 else ("loss" if profit < 0 else "tie")
                # print(f"Trade result: {outcome.upper()}! Profit/Loss: {profit:.2f}")
            else:
                outcome = "unknown_timeout"
                profit = 0.0
                # print("Trade result unknown (timed out).")
        else:
            # print("Trade execution failed. Check parameters, balance, or asset status.")
            order_id = "FAILED_TO_PLACE" # Placeholder for logging
            outcome = "failed_execution"
            profit = 0.0

        # STEP 7: Log Trade Data
        price_history_str = ",".join(map(str, closing_prices))
        trade_log_row = [
            datetime.datetime.now().isoformat(), order_id, ASSET, current_action, current_trade_amount, DURATION,
            slope, predicted_prob, outcome, profit,
            price_history_str, current_price, ma, std, rsi, macd, macd_signal, trader_sentiment
        ]
        log_data(LOG_FILE, trade_log_row)
        # print(f"Trade details logged to {LOG_FILE}")
        
        # STEP 8: Update Trade Amount for Next Trade
        if trade_successful and outcome != "unknown_timeout" and outcome != "failed_execution": # Only update if trade was conclusive
            current_trade_amount = update_trade_amount(current_trade_amount, outcome, predicted_prob)
            # print(f"Next trade amount will be: {current_trade_amount:.2f}")
        elif not trade_successful: # If trade failed to place, reset amount
             current_trade_amount = BASE_AMOUNT
             # print(f"Trade placement failed. Resetting trade amount to base: {current_trade_amount:.2f}")


        # Pause between iterations
        # print("Pausing before next iteration...")
        time.sleep(10) 
    
    # print("----- Trading session finished -----")
    iq_api.disconnect() # Changed from close_connect as disconnect is more common
    # print("Disconnected from IQ Option.")

if __name__ == "__main__":
    # print("Starting trading bot...") # Uncomment for debug
    try:
        main()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")
    except Exception as e:
        print(f"An critical error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()  # This will print the full traceback
    finally:
        print("Bot shutdown sequence complete.")