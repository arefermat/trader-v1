import yfinance as yf
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import schedule
import time
import config
from trained-models import *
import keyboard
import os


# Alpaca API Credentials
API_KEY = config.alpaca_api_key
API_SECRET = config.alpaca_secret_api_key
BASE_URL = 'https://paper-api.alpaca.markets'



# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def clear():
    os.System("cls")

# Fetch historical stock data
def fetch_data(stock_symbol):
    data = yf.download(stock_symbol, start='2010-01-01', end='2023-01-01')
    return data[['Close']]

# Prepare the data for the LSTM model
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    return np.array(X), np.array(y), scaler

# Build and train the LSTM model
def build_and_train_model(X_train, y_train):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=10)
    return model

# Trading functions
def get_current_price(symbol):
    barset = api.get_barset(symbol, 'minute', 1)
    stock_bars = barset[symbol]
    return stock_bars[-1].c

def buy_stock(symbol, qty):
    api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')

def sell_stock(symbol, qty):
    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')

# Trading logic
def run_trading(model, scaler, stock_symbol):
    global paper_stocks
    current_price = get_current_price(stock_symbol)
    recent_data = scaled_data[-60:].reshape(1, 60, 1)
    predicted_price = model.predict(recent_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    if predicted_price > current_price:
        print(f"Predicted price is higher ({predicted_price}), buying stock.")
        buy_stock(stock_symbol, 1) 
    else:
        print(f"Predicted price is lower ({predicted_price}), selling stock.")
        sell_stock(stock_symbol, 1)

# Save model to a file
def save_model(model, filename):
    model.save(filename)

# Load the model from a file
def load_model(model, filename):
    model.load(filename)
    
# Main execution
if __name__ == "__main__":
    model_decision = input("Do you want to train a new model or load an existing one")
    if model_decision == "train":
        clear()
        stock_symbol = input("Stock Symbol : ")
        clear()
        start = time.perf_counter()
        data = fetch_data(stock_symbol)
        X_train, y_train, scaler = prepare_data(data)
        model = build_and_train_model(X_train, y_train)
        end = time.perf_counter()
        time_taken = round(end-start, 2)
        print("Done!")
        if time_taken > 60:
            print(f"It took {time_taken} seconds.")
        else:
            print(f"It took {round(time_taken/60, 2)} minutes.")
        time.sleep(3)
        clear()
        save_decision = input("Would you like to save this model? (Y/N) : ")
        if save_decision == "Y":
            clear()
            file_name = open(input("What's the file directory? "), "x")
            save_model(model, file_name)
        elif save_decision == "N":
            print("Ok")
            time.sleep(2)
            clear()
    elif model_decision == "load":
        clear()
        file_name = input("What's the file directory? ")
        model = load_model(model, file_name)
        print("Done!")
        time.sleep(2)
        clear()
    
    # Schedule trading every 5 minutes
    schedule.every(interval).minutes.do(run_trading, model, scaler, stock_symbol)

    # Keep the script running
    while True:
        schedule.run_pending()
        if keyboard.is_pressed("ctrl+s"):
            break
        time.sleep(1)

