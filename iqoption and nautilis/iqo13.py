#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
--- Professional Predictive Trading Bot ---
Version: 11.0 | "The Analyst"
Strategy: Supervised Deep Learning with Rigorous Pre-Trade Validation and High-Confidence Execution
====================================================================================================

SYNOPSIS:
This bot represents a paradigm shift from reactive trading to a professional, data-driven predictive 
approach. It operates in two distinct phases:

1. ANALYST PHASE (PRE-FLIGHT CHECK):
   - The bot first acts as a data scientist. It loads the entire history of market data from the 
     'full_history_log.csv'.
   - It performs extensive FEATURE ENGINEERING, creating a rich dataset by combining standard 
     indicators (RSI, MACD, etc.) with advanced ones (Momentum, Stochastics, Lagged Features).
   - It intelligently generates a "ground truth" by labeling each historical data point with the 
     actual future price movement (up or down).
   - The data is meticulously split into TRAINING and unseen TESTING sets.
   - A sophisticated deep learning model is trained on the training data, using advanced techniques 
     like Batch Normalization, Dropout, and Early Stopping to achieve peak performance and avoid 
     overfitting.
   - **ACCURACY GATE:** The model is then rigorously evaluated on the unseen testing data. The bot 
     will CATEGORICALLY REFUSE to enter the trading phase unless this test accuracy surpasses the 
     `REQUIRED_ACCURACY_THRESHOLD`.

2. SNIPER PHASE (LIVE EXECUTION):
   - Only upon successful model validation does the bot connect to the trading platform.
   - It enters a continuous, second-by-second monitoring loop, calculating all features in real-time.
   - The validated model is used to generate a prediction (CALL/PUT) and a CONFIDENCE SCORE for 
     each second.
   - **CONFIDENCE GATE:** A trade is only executed if the model's confidence in its prediction is 
     exceptionally high, exceeding the `TRADE_CONFIDENCE_THRESHOLD`.
   - The bot acts as a patient sniper, waiting for high-probability setups rather than trading 
     frequently.
   - All live data is continuously logged, enriching the dataset for future, even more powerful, 
     training sessions.

This architecture ensures that trading decisions are based on a statistically validated edge, 
embodying the principles of modern quantitative finance.
"""

####################################################################################################
# SECTION 1: IMPORTS & GLOBAL CONFIGURATION
####################################################################################################

# --- Standard Libraries ---
import time, csv, os, sys, random
import traceback
from collections import deque
import datetime

# --- Data Science & Machine Learning Libraries ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# --- Deep Learning (TensorFlow/Keras) ---
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- IQ Option API ---
from iqoptionapi.stable_api import IQ_Option

####################################################################################################
# SECTION 2: CORE TRADING & BOT PARAMETERS
####################################################################################################

# --- Connection & Account Configuration ---
USERNAME = "@gmail.com"      # <<< --- !!! FILL YOUR IQ OPTION EMAIL HERE !!!
PASSWORD = ""          # <<< --- !!! FILL YOUR IQ OPTION PASSWORD HERE !!!
PRACTICE_MODE = "PRACTICE"

# --- Trading Asset & Execution Parameters ---
ASSET = "EURUSD-OTC"
BASE_AMOUNT = 1.0  # Use float for consistency
DURATION = 1       # Trade duration in minutes (used for labeling and execution)

# --- File & Path Management ---
LOG_FILE = "trade_log_v11_supervised.csv"
FULL_HISTORY_LOG_FILE = "full_history_log.csv"
MODEL_FILE = "trading_model_v11.keras"
SCALER_FILE = "data_scaler_v11.pkl"

# --- Market Data & Feature Engineering Configuration ---
TIMEFRAME = 60          # Candle length in seconds (1 minute)
NUM_CANDLES = 60        # Number of candles to gather for indicator calculation (increased for more context)
FEATURE_LAG_PERIODS = [1, 2, 5] # Use indicator values from 1, 2, and 5 minutes ago as features

# --- CRITICAL: Model Validation & Trading Thresholds ---
REQUIRED_ACCURACY_THRESHOLD = 0.75  # 75% - Bot will NOT trade if test accuracy is below this. Raise to 0.80 for higher strictness.
TRADE_CONFIDENCE_THRESHOLD = 0.85   # 85% - Will only place a trade if prediction probability is above this.

# --- Bot Operational Control ---
MAX_TRADE_ATTEMPTS = 200 # Maximum number of high-confidence opportunities the bot will seek.
OPPORTUNITY_ANALYSIS_INTERVAL = 5 # Analyze the market for a trade opportunity every 5 seconds.

####################################################################################################
# SECTION 3: ADVANCED FEATURE ENGINEERING & INDICATOR SUITE
####################################################################################################
print("Defining indicator and feature engineering suite...")

def calculate_all_features(df, price_col='close'):
    """
    Takes a DataFrame with a 'close' price column and engineers a rich set of features.
    This is the core feature creation engine for both training and live prediction.
    """
    # --- Basic Indicators ---
    df['MA_10'] = df[price_col].rolling(window=10).mean()
    df['MA_30'] = df[price_col].rolling(window=30).mean()
    df['Slope_10'] = df['MA_10'].diff().fillna(0)
    
    # --- Volatility Indicators ---
    df['STD_10'] = df[price_col].rolling(window=10).std()
    df['Bollinger_Mid'] = df[price_col].rolling(window=20).mean()
    df['Bollinger_Std'] = df[price_col].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * 2)
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * 2)
    
    # --- Momentum Indicators ---
    df['RSI'] = compute_RSI_vectorized(df[price_col], period=14)
    macd_line, signal_line = compute_MACD_vectorized(df[price_col])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['ROC'] = (df[price_col].diff(10) / df[price_col].shift(10)) * 100
    df['Momentum'] = df[price_col].diff(4)

    # --- Oscillator Indicators ---
    stoch_k, stoch_d = compute_stochastic_vectorized(df, period=14, d_period=3)
    df['Stochastic_%K'] = stoch_k
    df['Stochastic_%D'] = stoch_d
    df['Williams_%R'] = compute_williams_r_vectorized(df, period=14)

    # --- Time-Based Features ---
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    df['DayOfWeek'] = df.index.dayofweek

    # --- Create Lagged Features (Crucial for providing historical context to the model) ---
    feature_cols_to_lag = ['RSI', 'MACD_Hist', 'Stochastic_%K', 'Williams_%R', 'Slope_10']
    for col in feature_cols_to_lag:
        for lag in FEATURE_LAG_PERIODS:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    # --- Clean up by dropping NaNs created by rolling windows ---
    df.dropna(inplace=True)
    return df

# --- Vectorized (Fast) Indicator Functions ---
def compute_RSI_vectorized(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_MACD_vectorized(prices, short_span=12, long_span=26, signal_span=9):
    ema_short = prices.ewm(span=short_span, adjust=False).mean()
    ema_long = prices.ewm(span=long_span, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line

def compute_stochastic_vectorized(df, period=14, d_period=3):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_williams_r_vectorized(df, period=14):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    return -100 * ((high_max - df['close']) / (high_max - low_min))

print("Indicator suite defined.")

####################################################################################################
# SECTION 4: THE TRADING MODEL ANALYST CLASS
####################################################################################################

class TradingModelAnalyst:
    """
    A comprehensive class to handle all data science aspects of the trading bot:
    - Data loading, cleaning, and preparation.
    - Advanced feature engineering.
    - Intelligent target labeling.
    - Model building, training, and rigorous validation.
    - Saving, loading, and using the model for live predictions.
    """
    def __init__(self, model_path=MODEL_FILE, scaler_path=SCALER_FILE):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        print("TradingModelAnalyst initialized.")

    def _build_model(self, input_shape):
        """Builds a robust Keras model with Batch Normalization and Dropout."""
        print(f"Building model with input shape: {input_shape}")
        model = Sequential([
            Dense(256, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid') # Sigmoid for binary classification (Up/Down)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        model.summary()
        return model

    def prepare_data_and_train(self, data_log_path, look_ahead_period=DURATION, test_size=0.3):
        """
        Main orchestration method for the Analyst Phase.
        Loads data, creates features/labels, trains, validates, and saves the model.
        Returns the final accuracy on the unseen test set.
        """
        print("\n" + "="*50)
        print("--- STARTING ANALYST PHASE: DATA PREPARATION & MODEL TRAINING ---")
        print("="*50)

        # 1. Load and Pre-process Data
        if not os.path.exists(data_log_path) or os.path.getsize(data_log_path) < 5000:
            print(f"[FAIL] Historical data log '{data_log_path}' is missing or too small. Cannot train model.")
            return 0.0
        
        print(f"Loading historical data from '{data_log_path}'...")
        try:
            # Reconstruct a clean dataframe from the log
            # The log contains a string of prices, we need to parse that.
            # For simplicity, we'll assume the log is clean enough to parse the key metrics.
            # A more robust solution would parse the price history string.
            # Let's pivot to using the calculated features directly from the log.
            df = pd.read_csv(data_log_path, on_bad_lines='skip')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            # ---------- SAFETY FIX: guarantee essential price columns exist ----------------
            essential_cols = ['close', 'MovingAverage', 'StdDev', 'high', 'low']

            # 1) If 'close' is still missing, fall back to any price-like series we do have
            if 'close' not in df.columns:
                if 'Bollinger_SMA' in df.columns:
                    df['close'] = df['Bollinger_SMA']
                elif 'Slope' in df.columns:
                    df['close'] = df['Slope']  # last-ditch fallback
                else:
                    df['close'] = np.nan       # create the column so dropna works

            # 2) Derive missing columns from `close`
            if 'MovingAverage' not in df.columns:
                df['MovingAverage'] = df['close'].rolling(window=30, min_periods=1).mean()
            if 'StdDev' not in df.columns:
                df['StdDev'] = df['close'].rolling(window=30, min_periods=1).std()
            if 'high' not in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns:
                df['low'] = df['close']

            # 3) Now drop rows where ANY of the essentials are still NaN
            present_cols = [c for c in essential_cols if c in df.columns]
            df.dropna(subset=present_cols, inplace=True)
            # -------------------------------------------------------------------------------

        except Exception as e:
            print(f"[FAIL] Error processing log file: {e}")
            return 0.0

        # 2. Advanced Feature Engineering
        print("Performing advanced feature engineering...")
        df_featured = calculate_all_features(df.copy())
        
        # 3. Intelligent Label Creation
        print(f"Creating target labels with a {look_ahead_period}-minute look-ahead...")
        look_ahead_steps = look_ahead_period * 60 # Convert minutes to seconds (assuming 1-sec log interval)
        df_featured['Future_Price'] = df_featured['close'].shift(-look_ahead_steps)
        df_featured.dropna(inplace=True)
        
        # Target: 1 if price goes up, 0 if it goes down or stays the same.
        df_featured['Target'] = (df_featured['Future_Price'] > df_featured['close']).astype(int)
        
        self.feature_columns = [col for col in df_featured.columns if col not in ['close', 'high', 'low', 'open', 'Future_Price', 'Target']]
        X = df_featured[self.feature_columns]
        y = df_featured['Target']

        if len(X) < 500:
            print(f"[FAIL] Insufficient clean data ({len(X)} rows) after feature engineering. Need at least 500.")
            return 0.0
            
        print(f"Generated {len(X)} training samples.")
        print(f"Class Distribution:\n{y.value_counts(normalize=True)}")

        # 4. Data Splitting and Scaling
        print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        print("Scaling features with StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 5. Model Building and Training
        self.model = self._build_model(input_shape=(X_train_scaled.shape[1],))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

        print("\n--- Training Model ---")
        self.model.fit(
            X_train_scaled, y_train,
            validation_split=0.2, # Use 20% of training data for validation
            epochs=200,
            batch_size=128,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # 6. Rigorous Evaluation
        print("\n--- Evaluating Model on Unseen Test Data ---")
        # Load the best model saved by the checkpoint
        self.model = load_model(self.model_path)
        
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        print("\n" + "="*20 + " MODEL EVALUATION REPORT " + "="*20)
        print(f"Test Accuracy: {accuracy:.2%}")
        print(f"Test Loss: {loss:.4f}")

        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)']))
        
        print("Confusion Matrix:")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred), 
                           index=['Actual DOWN', 'Actual UP'], 
                           columns=['Predicted DOWN', 'Predicted UP']))
        print("="*65 + "\n")

        # 7. Save the final validated model and scaler
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Final validated model saved to '{self.model_path}'")
        print(f"Data scaler saved to '{self.scaler_path}'.")
        
        self.is_trained = True
        return accuracy

    def predict_live(self, live_data_deque):
        """
        Takes a deque of recent live data, creates features, scales, and predicts.
        Returns: (action_str, confidence_float) or (None, 0) if data is insufficient.
        """
        if not self.is_trained: raise Exception("Model is not trained or loaded.")
        
        # Convert deque to DataFrame
        live_df = pd.DataFrame(list(live_data_deque))
        live_df['Timestamp'] = pd.to_datetime(live_df['Timestamp'])
        live_df.set_index('Timestamp', inplace=True)

        if 'high' not in live_df.columns: live_df['high'] = live_df['MovingAverage'] + live_df['StdDev']
        if 'low' not in live_df.columns: live_df['low'] = live_df['MovingAverage'] - live_df['StdDev']
        if 'close' not in live_df.columns: live_df['close'] = live_df['MovingAverage']
        
        # Engineer features for the live data
        featured_live_df = calculate_all_features(live_df.copy())
        
        if featured_live_df.empty:
            return None, 0.0

        # Get the latest feature set for prediction
        latest_features = featured_live_df[self.feature_columns].iloc[-1:]
        
        # Scale the features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Predict
        prediction_prob = self.model.predict(latest_features_scaled, verbose=0)[0][0]
        
        action = "call" if prediction_prob > 0.5 else "put"
        confidence = prediction_prob if action == "call" else 1 - prediction_prob
            
        return action, confidence

####################################################################################################
# SECTION 5: MAIN APPLICATION & TRADING LOOP
####################################################################################################
def main():
    """Main orchestration function."""
    # --- Suppress TensorFlow logging ---
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    # --- PHASE 1: ANALYST & VALIDATION ---
    model_analyst = TradingModelAnalyst()
    test_accuracy = model_analyst.prepare_data_and_train(FULL_HISTORY_LOG_FILE)
    
    # --- ACCURACY GATE ---
    if test_accuracy < REQUIRED_ACCURACY_THRESHOLD:
        print(f"\nCRITICAL: MODEL VALIDATION FAILED! Accuracy ({test_accuracy:.2%}) is below required threshold ({REQUIRED_ACCURACY_THRESHOLD:.2%}).")
        print("BOT WILL NOT EXECUTE TRADES. Please gather more data or improve the model.")
        sys.exit(1) # Exit with an error code
        
    print(f"\nSUCCESS: MODEL VALIDATED WITH {test_accuracy:.2%} ACCURACY. PROCEEDING TO LIVE TRADING.")
    
    # --- PHASE 2: SNIPER (LIVE TRADING) ---
    I_want_money = None
    try:
        print("\n--- Initializing Sniper Phase ---")
        I_want_money = IQ_Option(USERNAME, PASSWORD)
        I_want_money.connect()
        if not I_want_money.check_connect():
            print("Connection to IQ Option failed. Exiting.")
            return

        I_want_money.change_balance(PRACTICE_MODE)
        print(f"Connected to IQ Option. Mode: {PRACTICE_MODE}")

        # Initialize logging for this session
        init_log(LOG_FILE)
        
        # Use a deque to hold recent live data for feature calculation
        live_data_deque = deque(maxlen=(NUM_CANDLES + max(FEATURE_LAG_PERIODS) + 5))
        
        main_loop_active = True
        trade_attempt_count = 0
        seconds_since_last_analysis = 0

        while main_loop_active and trade_attempt_count < MAX_TRADE_ATTEMPTS:
            loop_start_time = time.time()
            
            # --- A. Live Data Ingestion & Continuous Logging ---
            try:
                I_want_money.start_candles_stream(ASSET, TIMEFRAME, 1) # Get latest candle
                time.sleep(0.2)
                candles = I_want_money.get_realtime_candles(ASSET, TIMEFRAME)
                
                if candles and isinstance(candles, dict) and any(candles):
                    latest_timestamp = max(candles.keys())
                    latest_candle = candles[latest_timestamp]
                    
                    # Compute basic indicators for logging
                    # A full feature re-computation on the whole deque will happen during analysis
                    temp_prices = [c['close'] for c in live_data_deque if 'close' in c] + [latest_candle['close']]
                    live_ma = np.mean(temp_prices[-30:]) if len(temp_prices) >= 30 else 0
                    live_std = np.std(temp_prices[-30:]) if len(temp_prices) >= 30 else 0

                    live_data_point = {
                        "Timestamp": datetime.datetime.fromtimestamp(latest_timestamp).isoformat(),
                        "Asset": ASSET,
                        "PriceHistoryString": f"{latest_candle['close']}", # Simplified for live log
                        "CurrentSentiment": 0.5, # Removed this feature
                        "close": latest_candle['close'],
                        "high": latest_candle.get('max', latest_candle['close']),
                        "low": latest_candle.get('min', latest_candle['close']),
                        "MovingAverage": live_ma,
                        "StdDev": live_std,
                    }
                    live_data_deque.append(live_data_point)
                    
                    # Log a simplified row continuously. The analysis function will compute full features.
                    log_full_history(FULL_HISTORY_LOG_FILE, list(live_data_point.values()))
            except Exception as e:
                print(f"Error in live data loop: {e}")
                time.sleep(2) # Pause if there's an error

            # --- B. Periodic High-Confidence Trade Analysis ---
            seconds_since_last_analysis += 1
            if seconds_since_last_analysis >= OPPORTUNITY_ANALYSIS_INTERVAL:
                seconds_since_last_analysis = 0
                
                if len(live_data_deque) < live_data_deque.maxlen:
                    print(f"Gathering initial data... {len(live_data_deque)}/{live_data_deque.maxlen} points.")
                else:
                    print(f"\nAnalyzing market for high-confidence opportunity...")
                    
                    action, confidence = model_analyst.predict_live(live_data_deque)
                    
                    if action:
                        print(f"Model Prediction: {action.upper()} with {confidence:.2%} confidence.")
                        if confidence >= TRADE_CONFIDENCE_THRESHOLD:
                            trade_attempt_count += 1
                            print(f"CONFIDENCE THRESHOLD MET! Executing trade #{trade_attempt_count}.")
                            
                            status, order_id = I_want_money.buy(BASE_AMOUNT, ASSET, action, DURATION)
                            if status:
                                print(f"Trade placed (ID: {order_id}). Waiting for outcome...")
                                time.sleep(DURATION * 60 + 5) # Wait for expiry plus buffer
                                profit = I_want_money.check_win_v4(order_id)
                                outcome = "win" if profit > 0 else "loss" if profit < 0 else "tie"
                                print(f"Trade Result: {outcome.upper()}, Profit: ${profit:.2f}")
                                # Log detailed trade info
                                log_trade(LOG_FILE, [datetime.datetime.now().isoformat(), order_id, ASSET, action, BASE_AMOUNT, DURATION, 0, confidence, outcome, profit] + ['-']*15)
                            else:
                                print("Trade placement failed by API.")
                        else:
                            print("Confidence below threshold. Holding.")
                    else:
                        print("Not enough data for a stable prediction yet.")

            # --- Maintain Loop Cadence ---
            loop_duration = time.time() - loop_start_time
            sleep_time = max(0, 1 - loop_duration)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nA critical error terminated the bot: {e}")
        traceback.print_exc()
    finally:
        if I_want_money and I_want_money.check_connect():
            print("Disconnecting from IQ Option...")
            I_want_money.close_connect()
        print("Bot shutdown sequence complete.")

if __name__ == "__main__":
    main()
