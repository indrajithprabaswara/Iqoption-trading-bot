import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from iqoptionapi.stable_api import IQ_Option
import time
import numpy as np
import datetime
from model_ensemble import ModelEnsemble

# Configuration
USERNAME = "eaglelab23@gmail.com"
PASSWORD = "Polboti1@"
PRACTICE_MODE = "PRACTICE"
ASSET = "EURUSD-OTC"
BASE_AMOUNT = 1
DURATION = 1
TIMEFRAME = 60
NUM_CANDLES = 30
STATE_DIM = 13
MIN_TRADES_FOR_TRAINING = 20

def main():
    try:
        print("Initializing trading bot...")
        
        # Connect to IQ Option
        I_want_money = IQ_Option(USERNAME, PASSWORD)
        I_want_money.connect()
        
        if not I_want_money.check_connect():
            print("Connection failed")
            return
            
        print("Successfully connected to IQ Option")
        I_want_money.change_balance(PRACTICE_MODE)
        print(f"Switched to {PRACTICE_MODE} account")
        
        # Initialize model
        model = ModelEnsemble(input_dim=STATE_DIM)
        
        # Start candle stream
        I_want_money.start_candles_stream(ASSET, TIMEFRAME, NUM_CANDLES)
        print("Candle stream started")
        
        # Data collection
        features = []
        labels = []
        
        while True:
            try:
                # Get market data
                candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                if not candles:
                    time.sleep(1)
                    continue
                    
                # Process candle data
                closing_prices = []
                for timestamp in sorted(candles.keys()):
                    candle = candles[timestamp]
                    if isinstance(candle, dict) and 'close' in candle:
                        closing_prices.append(candle['close'])
                
                if len(closing_prices) < 2:
                    continue
                    
                # Calculate features
                feature_vector = np.array([
                    closing_prices[-1],  # Current price
                    np.mean(closing_prices),  # Mean
                    np.std(closing_prices),  # Standard deviation
                    (closing_prices[-1] - closing_prices[0]) / closing_prices[0],  # Price change
                    np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0,  # Short-term MA
                    np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0,  # Long-term MA
                    float(datetime.datetime.now().hour),  # Hour
                    float(datetime.datetime.now().weekday()),  # Day of week
                    0.5,  # Placeholder for sentiment
                    0.0, 0.0, 0.0, 0.0  # Placeholders for additional features
                ]).reshape(1, STATE_DIM)
                
                features.append(feature_vector[0])
                
                # Generate simple labels (1 if price went up, 0 if down)
                if len(closing_prices) >= 2:
                    label = 1 if closing_prices[-1] > closing_prices[-2] else 0
                    labels.append(label)
                
                # Train model when we have enough data
                if len(features) >= MIN_TRADES_FOR_TRAINING and len(features) == len(labels):
                    print("\nTraining model...")
                    X = np.array(features)
                    y = np.array(labels)
                    model.train_ensemble(X, y)
                    
                    # Make prediction
                    pred, confidence = model.predict(feature_vector)
                    
                    if confidence > 0.7:  # Only trade with high confidence
                        action = "call" if pred > 0.5 else "put"
                        print(f"\nPlacing {action} trade with confidence {confidence:.2f}")
                        
                        # Place trade
                        status, order_id = I_want_money.buy(
                            BASE_AMOUNT, ASSET, action, DURATION)
                            
                        if status:
                            print(f"Trade placed. Order ID: {order_id}")
                            time.sleep(DURATION * 60)  # Wait for trade to complete
                            
                            # Check result
                            result = I_want_money.check_win_v3(order_id)
                            
                            if result is not None:
                                print(f"Trade result: {'WIN' if result > 0 else 'LOSS'}")
                            else:
                                print("Could not get trade result")
                        else:
                            print("Trade placement failed")
                            
                        # Wait before next trade
                        time.sleep(30)
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Critical error: {str(e)}")
    finally:
        if 'I_want_money' in locals():
            I_want_money.close_connect()
            print("Disconnected from IQ Option")

if __name__ == "__main__":
    main()
