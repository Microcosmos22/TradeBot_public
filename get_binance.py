import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
from calc_tools import *
from plotting import *

def candle2interval(candle_length):
    if candle_length == Client.KLINE_INTERVAL_1MINUTE:
        interval_duration = timedelta(minutes=1)
    elif candle_length == Client.KLINE_INTERVAL_5MINUTE:
        interval_duration = timedelta(minutes=5)
    elif candle_length == Client.KLINE_INTERVAL_15MINUTE:
        interval_duration = timedelta(minutes=15)
    elif candle_length == Client.KLINE_INTERVAL_30MINUTE:
        interval_duration = timedelta(minutes=30)
    elif candle_length == Client.KLINE_INTERVAL_1HOUR:
        interval_duration = timedelta(hours=1)
    elif candle_length == Client.KLINE_INTERVAL_2HOUR:
        interval_duration = timedelta(hours=2)
    elif candle_length == Client.KLINE_INTERVAL_4HOUR:
        interval_duration = timedelta(hours=4)
    elif candle_length == Client.KLINE_INTERVAL_6HOUR:
        interval_duration = timedelta(hours=6)
    elif candle_length == Client.KLINE_INTERVAL_8HOUR:
        interval_duration = timedelta(hours=8)
        
    return interval_duration



def get_historical_data_trim(start_date_or_lastNcandles, nfeatures, candle_length = Client.KLINE_INTERVAL_1HOUR, normalize=True, plot=False):
    # Gets all candles and features of a crypto coin for a fixed candle length. There is three options:
    # 1: Give it an int to retrieve the last N candles
    # 2: Give it a start datetime to retrieve candles until now
    # 3: Give it [end_date, int] to retrieve last N candles before "end_date"
    
    # Binance API key and secret
    api_key = "pxRONzQcbpDoImQXzHqkO6XJWd7WMIKSTyBPtTlkvaCbIGJ0Whcnz8LDw7SavMIx"
    api_secret = "hFXzByh1Fg90Vcxvx8uakDq9n6reH32KswXuYOTzFxxjmAVvMbHRh1lOvMSgHlex"
    # Initialize Binance client
    client = Client(api_key, api_secret)
    #print("Interval: {}".format(str(candle_length)))
    
    if type(start_date_or_lastNcandles) == int:
        now = datetime.now()
        end_date = now.strftime("%Y-%m-%d %H:%M:%S")
        n_candles = start_date_or_lastNcandles
        
        interval = candle2interval(candle_length)
        time_to_subtract = interval * n_candles
        # Calculate the start_date by subtracting the total time from the end_date
        start_date = now - time_to_subtract
        
        #start_date = end_date - timedelta(hours=n_candles)
        data = np.asarray(client.get_historical_klines(
        "BTCUSDT", candle_length, 
        start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"), 
        end_str=now.strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        print("Calling last {} candles".format(data[:, 0].shape[0]))
        
    elif isinstance(start_date_or_lastNcandles, datetime):
        
        symbol = "BTCUSDT"
        start_date_or_lastNcandles = start_date_or_lastNcandles.strftime("%Y-%m-%d %H:%M:%S")
        data = np.asarray(client.get_historical_klines(symbol, candle_length, start_date_or_lastNcandles)).astype(float)
        print("Calling candles since {}".format(start_date_or_lastNcandles))
        
    elif (isinstance(start_date_or_lastNcandles[0], datetime) and type(start_date_or_lastNcandles[1]) == int):
        
        end_date = start_date_or_lastNcandles[0]
        n_candles = start_date_or_lastNcandles[1]
        
        
        interval = candle2interval(candle_length)
        time_to_subtract = interval * n_candles
        # Calculate the start_date by subtracting the total time from the end_date
        start_date = end_date - time_to_subtract
        #start_date = end_date - timedelta(hours=n_candles)
        data = np.asarray(client.get_historical_klines(
        "BTCUSDT", candle_length, 
        start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"), 
        end_str=end_date.strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        print("Calling last {} candles before {} (starting {})".format(n_candles, end_date, start_date))
        
        
        
    elif (isinstance(start_date_or_lastNcandles[0], datetime) and isinstance(start_date_or_lastNcandles[1], datetime)):
        start_date = start_date_or_lastNcandles[0]
        end_date = start_date_or_lastNcandles[1]
        
        
        data = np.asarray(client.get_historical_klines(
        "BTCUSDT", candle_length, 
        start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
        end_str=end_date.strftime("%Y-%m-%d %H:%M:%S")))
        print("calling {} candles between {} and {}".format(data.shape, start_date_or_lastNcandles[0].strftime("%d %b, %Y %H:%M:%S"), start_date_or_lastNcandles[1].strftime("%Y-%m-%d %H:%M:%S")))
            
    print(data.shape)
    """ (time, open, high, low, close, volume, close_time etc.) """
    # Use closing prices
    
    timestamps = data[:, 0]  # First column contains the timestamp (open time)
    #dates = [datetime.utcfromtimestamp(float(timestamp) / 1000).strftime('%Y-%m-%d %H h') for timestamp in timestamps]

    target, features = compute_features_trim(data, timestamps, nfeatures)
    
    if plot:
        plot_historical_data(target, features)
    
    return np.asarray(target), np.asarray(features), np.asarray(timestamps).astype(np.int64), data


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    
    
    start_str = "1 January 2024 00:00:00"
    end_str = "1 December 2024 00:00:00"
    
    start_of_training = datetime.strptime(start_str, "%d %B %Y %H:%M:%S")
    end_of_training = datetime.strptime(end_str, "%d %B %Y %H:%M:%S")
    
    target, features, dates = get_historical_data_trim(Client.KLINE_INTERVAL_1HOUR, [start_of_training, end_of_training], normalize=False, plot=True)
    
    