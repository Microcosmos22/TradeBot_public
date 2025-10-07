import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
from calc_tools import *
from get_binance import *
from simulation import *
from plotting import *
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.layers import LSTM
import logging
import copy
import os, time

api_key = "pxRONzQcbpDoImQXzHqkO6XJWd7WMIKSTyBPtTlkvaCbIGJ0Whcnz8LDw7SavMIx"
api_secret = "hFXzByh1Fg90Vcxvx8uakDq9n6reH32KswXuYOTzFxxjmAVvMbHRh1lOvMSgHlex"
# Configure logging
logging.basicConfig(
    filename="trading_bot.log",  # Log file name
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(message)s",  # Log format: timestamp and message
)


    
def write_trackingtable(modelstr, date, values, verbose = True):
    if os.path.isfile('tracking.pkl'):
        with open('tracking.pkl', 'rb') as f:
            df = pickle.load(f)
    else:
        df = pd.DataFrame()
    
    if modelstr not in df.columns:
        df[modelstr] = [None] * len(df)  # Column initialized to None (for dictionary)

    # If the date does not exist as an index, add it with None values for each column
    if date not in df.index:
        df.loc[date] = [None] * len(df.columns)  # Row initialized to None (for dictionary)

    
    values_to_store = {
        "portfolio": values[0],
        "btc_price": values[1],
        "bet": values[2],
        "out_of_eq": values[3]
    }
    df.at[date, modelstr] = values_to_store
    with open('tracking.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    
    if verbose:
        print(" New entry for {}, {} = {}".format(modelstr, date, values[0]))
        
def is_within_5_minutes_of_full_hour(time: datetime = None, tolerance = 5):
    # Use current UTC time if no time is provided
    if time is None:
        time = datetime.utcnow()

    # Round the time to the nearest hour
    full_hour = time.replace(minute=0, second=0, microsecond=0)

    # Check if the time is within ±5 minutes of the full hour
    if abs((time - full_hour).total_seconds()) <= tolerance * 60:
        return True
    else:
        return False
        
        
    
def live_tracking(verbose = True, candletime_tolerance = 5):
    """ Should be running constantly. Performs trades with all machines and fix budget.
    Writes the portfolio increases of a single in a pandas table, reset kasse after each trade """
    
    modeln, machinestrn, scalern, lookfn = load_machines_scalers()
    target, features, dates, _ = get_historical_data_trim(75, nfeatures=5)
    start_price = client.get_symbol_ticker(symbol="BTCUSDT")['price']
    
    #print("First candle is {} UTC".format(datetime.utcfromtimestamp(int(dates[-1])/1000).strftime('%Y-%m-%d %H:%M:%S')))
    
    
    # Reset portfolio, want the unbiased bets
    kasse = [10000 for k in range(len(modeln))] # Euros
    btc_kasse = [10000/float(start_price) for k in range(len(modeln))]
    portfolio = [kasse[k]+btc_kasse[k]*float(start_price) for k in range(len(modeln))] # Euros
    
    
    i = True
    last_candle_date = dates[-2]/1000
    bought = None
    
    print("Tracking {} different machines ".format(len(modeln)))
    while i:
        current_time = datetime.utcnow()
        
        if is_within_5_minutes_of_full_hour(current_time, candletime_tolerance):
            for k in range(len(modeln)):
                target, features, dates, _ = get_historical_data_trim(75, nfeatures=scalern[k].n_features_in_)
                candle_date = dates[-1]/1000
                lasttime = datetime.utcfromtimestamp(candle_date).strftime('%Y-%m-%d %H:%M:%S')
                now_price = float(client.get_symbol_ticker(symbol="BTCUSDT")['price'])
            
                price_obs = target[-1]
                print("############################################")
                
                
                if verbose:
                    """ Only for printing"""
                    stacked_pred = raws_predict(modeln[k], scalern[k], target, features, lookfn[k])
                    price_pred = price_obs*(1+stacked_pred[-1,0])
                    forw_diff = price_pred-price_obs
                    _, _, bought, _ = equil_trader(forw_diff, kasse[k], btc_kasse[k], now_price, equil_const = 1, forward_const = 10)
                    printstep(target[-1], forw_diff, bought, lasttime, kasse[k], btc_kasse[k])
                
                   
                """ New candle got released, perform a trade. First trade will be done
                when script is run, for the sake of debugging """
                if candle_date != last_candle_date:
                    print("____ Diff candle dates, trade! _____")
                    # Bilanz ziehen
                    print("Now price {} Last price obs (candle): {}".format(now_price,target[-2]))
                    portfolio[k] = kasse[k]+btc_kasse[k]*now_price
                    
                    
                    stacked_pred = raws_predict(modeln[k], scalern[k], target, features, lookfn[k])
                    price_pred = price_obs*(1+stacked_pred[-1,0])
                    forw_diff = price_pred-price_obs
                    
                    # Setze kasse, BTC kasse für nächste Bilanz aus standard kasse
                    kasse[k], btc_kasse[k], bought, out_of_equil = equil_trader(forw_diff, kasse[k], btc_kasse[k], now_price, equil_const = 1, forward_const = 10)
                    print("After the trade: {} $ and {} BTC".format(kasse[k], btc_kasse[k]))
                    
                    
                    write_trackingtable(machinestrn[k], lasttime, [portfolio[k], now_price, bought, out_of_equil])
                
            if candle_date != last_candle_date:
                print("diff candle dates, trade!")
                last_candle_date = candle_date
        time.sleep(60)
    
    
if __name__ == "__main__":
    
    api_key = "pxRONzQcbpDoImQXzHqkO6XJWd7WMIKSTyBPtTlkvaCbIGJ0Whcnz8LDw7SavMIx"
    api_secret = "hFXzByh1Fg90Vcxvx8uakDq9n6reH32KswXuYOTzFxxjmAVvMbHRh1lOvMSgHlex"
    # Initialize Binance client
    client = Client(api_key, api_secret)
    
    
    #init_trackingtable()
    live_tracking()