from iqoptionapi.stable_api import IQ_Option
import time
import csv
import os
import numpy as np
import datetime

# ---------------------------
# Configuration & Constants
# ---------------------------

USERNAME = "eaglelab23@gmail.com"          # Your IQ Option email
PASSWORD = "Polboti1@"                   # Your IQ Option password
PRACTICE_MODE = "PRACTICE"         # Switch to demo mode

ASSET = "EURUSD-OTC"              # The asset to trade (ensure it is open for trading)
ACTION = "call"                   # "call" (buy) or "put" (sell) – this is our chosen direction
BASE_AMOUNT = 1                   # Base trade amount (in USD, for example)
DURATION = 1                      # Trade duration in minutes

# How many trades before ending (for this example)
TRADING_ITERATIONS = 100

# File for logging trade data (for later analysis or model re-training)
LOG_FILE = "trade_log.csv"

# ---------------------------
# Helper Functions
# ---------------------------

def init_log(file_path):
    """Initialize the CSV log file with headers if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "OrderID", "Asset", "Action", "Amount", "Duration",
                "Slope", "PredictedProbability", "Outcome", "Profit"
            ])
            
def log_trade(file_path, row):
    """Append a row of trade data to the CSV file."""
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def compute_slope(candles):
    """
    Compute a simple linear regression slope from candle data.
    `candles` is assumed to be a dict with timestamps as keys and each value a dict with a 'close' price.
    """
    times = sorted(candles.keys())
    if len(times) < 2:
        return 0.0
    # Extract closing prices in time order.
    closes = [candles[t]['close'] for t in times]
    x = np.arange(len(closes))
    slope, intercept = np.polyfit(x, closes, 1)
    return slope

def predict_probability(slope, action):
    """
    Dummy prediction function.
    For example, if trading a 'call' and the slope is positive, we return a high probability (e.g. 0.95).
    Otherwise, we return a lower probability.
    (Replace this with your real model in a production system.)
    """
    if action == "call":
        return 0.95 if slope > 0 else 0.45
    elif action == "put":
        return 0.95 if slope < 0 else 0.45
    return 0.5

def wait_for_result(api, order_id, timeout=60):
    """
    Poll the API for the result of a trade.
    This example uses a (commonly available in forks) 'check_win_v3' method.
    (Your API version or fork may differ; adjust accordingly.)
    """
    start_time = time.time()
    result = None
    while time.time() - start_time < timeout:
        result = api.check_win_v3(order_id)
        if result is not None:
            return result  # Expected to return a numeric profit/loss value.
        time.sleep(1)
    return None

def update_trade_amount(current_amount, outcome, predicted_prob, threshold=0.9):
    """
    Adaptive position sizing.
    If the trade wins and our prediction was very confident (>= threshold), increase the amount.
    Otherwise, if the trade loses, reset to the base amount.
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

    # Initialize the log file.
    init_log(LOG_FILE)

    # -------- Connect to IQ Option --------
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

    # -------------- Trading Loop --------------
    for i in range(TRADING_ITERATIONS):
        print("\nTrade iteration", i+1)

        # ---------------------------
        # STEP 1: Gather Market Data & Compute Indicator
        # ---------------------------
        # Start a candle stream (we ask for 10 recent candles with a 60-second timeframe)
        timeframe = 60  
        I_want_money.start_candles_stream(ASSET, timeframe, 10)
        time.sleep(2)  # Allow time for candles to accumulate.
        candles = I_want_money.get_realtime_candles(ASSET, timeframe)
        slope = compute_slope(candles)
        print("Computed slope from recent candles:", slope)

        # ---------------------------
        # STEP 2: Predict Trade Success Probability
        # ---------------------------
        predicted_prob = predict_probability(slope, ACTION)
        print("Predicted probability for action", ACTION, ":", predicted_prob)

        # Determine the trade amount (adaptive sizing)
        trade_amount = current_amount
        print("Placing a trade with amount:", trade_amount)

        # ---------------------------
        # STEP 3: Place the Trade
        # ---------------------------
        trade_status, order_id = I_want_money.buy(trade_amount, ASSET, ACTION, DURATION)
        if not trade_status:
            print("Trade execution failed. Check parameters, balance, or asset status.")
            continue
        print(f"Trade placed. Order ID: {order_id}")

        # ---------------------------
        # STEP 4: Wait for Trade Result
        # ---------------------------
        print("Waiting for trade outcome...")
        result = wait_for_result(I_want_money, order_id, timeout=60)
        if result is None:
            outcome = "unknown"
            profit = 0
            print("Timed out waiting for trade result.")
        else:
            # Here we assume a positive result means win; a negative result means loss.
            profit = result
            outcome = "win" if profit > 0 else "loss"
            print(f"Trade result: {outcome} with profit/loss: {profit}")

        # ---------------------------
        # STEP 5: Log the Trade Data
        # ---------------------------
        timestamp = datetime.datetime.now().isoformat()
        log_row = [
            timestamp, order_id, ASSET, ACTION, trade_amount, DURATION,
            slope, predicted_prob, outcome, profit
        ]
        log_trade(LOG_FILE, log_row)
        print("Trade logged to", LOG_FILE)

        # ---------------------------
        # STEP 6: Update Trade Amount for Next Trade
        # ---------------------------
        current_amount = update_trade_amount(current_amount, outcome, predicted_prob)
        print("Updated trade amount for next trade:", current_amount)

        # Optional pause between trades.
        time.sleep(10)

    # -------------- Shutdown --------------
    I_want_money.close_connect()
    print("Disconnected from IQ Option.")

if __name__ == "__main__":
    main()
