from iqoptionapi.stable_api import IQ_Option
import time

# Replace these with your IQ Option credentials
USERNAME = "eaglelab23@gmail.com"
PASSWORD = "Polboti1@"

PRACTICE_MODE = "PRACTICE"             # The mode needed to switch to demo account

ASSET = "EURUSD-OTC"  # Replace with the asset you want to trade; ensure it’s open for trading
ACTION = "call"       # "call" (buy) or "put" (sell) – choose based on your strategy
AMOUNT = 1            # The amount in USD (or your account currency) for the trade
DURATION = 1          # Duration (in minutes) of the binary trade

# --- Connect to IQ Option ---
I_want_money = IQ_Option(USERNAME, PASSWORD)
I_want_money.connect()
time.sleep(1)  # Allow a moment for connection setup

if not I_want_money.check_connect():
    print("Connection failed. Please check your credentials or network.")
    exit()

print("Connected successfully!")

# --- Switch to the Practice (Demo) Account ---
I_want_money.change_balance(PRACTICE_MODE)
print("Switched to Practice Account mode.")

# --- Retrieve and Display Profile Data ---
profile = I_want_money.get_profile_ansyc()
print("Profile data:", profile)

# --- Optional: Subscribe to Market Data (if needed) ---
# Uncomment the lines below if you want to monitor real-time candles or other data.
# I_want_money.start_candles_stream(ASSET, 60, 300)
# time.sleep(1)
# candles = I_want_money.get_realtime_candles(ASSET, 60)
# print("Realtime Candle Data:", candles)

# --- Place a Trade ---
print("Placing trade on the demo account...")
trade_status, order_id = I_want_money.buy(AMOUNT, ASSET, ACTION, DURATION)

if trade_status:
    print(f"Trade placed successfully on practice account! Order ID: {order_id}")
else:
    print("Trade execution failed. Please check your parameters, balance, or asset status.")

# --- Wait a Bit to View the Result ---
time.sleep(5)  # Wait 5 seconds (adjust as needed)

# --- Clean Up by Disconnecting ---
I_want_money.close_connect()
print("Disconnected from IQ Option.")