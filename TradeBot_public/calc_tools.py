import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def calc_returns(target):
    returns = []
    for i in range(len(target)-1):
        returns.append((target[i+1]-target[i])/target[i])
    return np.asarray(returns)

def calculate_rsi1(target, period=14):
    # Calculate RSI (14-period by default)
    delta = target.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bb_width1(target, window=20):
    # Calculate Bollinger Bands (20-period by default)
    sma_20_bollinger = target.rolling(window=20).mean()
    std_20 = target.rolling(window=20).std()
    bb_upper = sma_20_bollinger + (2 * std_20)
    bb_lower = sma_20_bollinger - (2 * std_20)
    bb_width = (bb_upper - bb_lower) / sma_20_bollinger
    return bb_width

def calculate_momentum1(target, period=10):
    # Calculate momentum as the difference between the current and the value 'period' steps back
    momentum = target - target.shift(period)
    return momentum
    
def calculate_macd_histogram(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the MACD Histogram (MACD - Signal) and return it as a numpy array.
    
    Parameters:
    - df: pandas DataFrame with 'close' column representing closing prices.
    - fast_period: The period for the fast EMA (default 12).
    - slow_period: The period for the slow EMA (default 26).
    - signal_period: The period for the signal line (default 9).
    
    Returns:
    - macd_histogram: numpy array with the MACD Histogram values (MACD - Signal).
    """
    # Calculate fast and slow EMAs
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean().values
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean().values

    # Calculate the MACD Line (Fast EMA - Slow EMA)
    macd = ema_fast - ema_slow

    # Calculate the Signal Line (9-period EMA of MACD)
    signal = pd.Series(macd).ewm(span=signal_period, adjust=False).mean().values

    # Calculate the MACD Histogram (MACD - Signal)
    macd_histogram = macd - signal
    
    return macd_histogram
    
def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    """
    Calculate the Stochastic Oscillator %K and %D and return as numpy arrays.
    
    Parameters:
    - df: pandas DataFrame with 'high', 'low', 'close' columns representing price data.
    - k_period: The period for %K (default 14).
    - d_period: The period for %D (default 3).
    
    Returns:
    - k_array: numpy array with %K values.
    - d_array: numpy array with %D values.
    """
    df.loc[:, 'close'] = pd.to_numeric(df['close'], errors='coerce')
    df.loc[:, 'high'] = pd.to_numeric(df['high'], errors='coerce')
    df.loc[:, 'low'] = pd.to_numeric(df['low'], errors='coerce')

    # Calculate the rolling highest high and lowest low over the k_period
    lowest_low = df['low'].rolling(window=k_period, min_periods=1).min().values
    highest_high = df['high'].rolling(window=k_period, min_periods=1).max().values
    
    # Calculate the %K (Stochastic Oscillator)
    k_array = ((df['close'].values - lowest_low) / (highest_high - lowest_low)) * 100

    # Calculate the %D (3-period SMA of %K)
    d_array = pd.Series(k_array).rolling(window=d_period, min_periods=1).mean().values
    
    return k_array, d_array
    
def extract_times(timestamp_ms):
    dmonth, dweek, hour = [], [], []
    
    for ts in timestamp_ms:
        timestamp_s = int(ts) / 1000

        # Convert to datetime object (in UTC)
        dt_utc = datetime.utcfromtimestamp(timestamp_s)

        # Extract the day of the month, day of the week, and UTC hour
        dmonth.append(dt_utc.day)
        dweek.append(dt_utc.weekday())  # Full weekday name (e.g., 'Monday')
        hour.append(dt_utc.hour)
        
        
    return np.asarray(dmonth), np.asarray(dweek), np.asarray(hour)
    
def compute_features_trim(data, timestamp_ms, nfeatures):
    # Compute features of the target (which is usually closing_prices)
    #sma_20 = np.asarray([np.sum([close_prices[i] for i in range(p-20, p)]) for p in range(20, len(close_prices))])/20  # Simple Moving Average
    close_prices = data[:, 4].astype(float)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
           'close_time', 'quote_asset_volume', 'number_of_trades', 
           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    # Convert the NumPy array to a Pandas DataFrame
    df = pd.DataFrame(data, columns=columns)
    df_for_so = df[['high', 'low', 'close']]
    df_for_macd = df[['close']]
    
    close_prices = pd.Series(close_prices)
    
    
    
    # Calculate SMA_20 and SMA_50
    sma_20 = close_prices.rolling(window=20).mean()
    sma_50 = close_prices.rolling(window=50).mean()
    k, d = calculate_stochastic_oscillator(df_for_so)
    macddiff = calculate_macd_histogram(df_for_macd)
    dmonth, dweek, hour = extract_times(timestamp_ms)
    
    
    # sma's start at 50
    rsi = calculate_rsi1(close_prices, period=14)
    bb_width = calculate_bb_width1(close_prices, window=20)
    momentum = calculate_momentum1(close_prices, period=10)

    # Combine features into a single array
    # Align lengths of features with the target
    #min_length = min(len(sma_20), len(sma_50), len(rsi), len(bb_width), len(momentum),len(volume), len(k), len(d), len(macddiff), len(dmonth), len(dweek), len(hour))print("Lengths {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} get cut to {}".format(len(sma_20), len(sma_50),len(rsi), len(bb_width), len(momentum), len(volume), len(k), len(d),len(macddiff), len(dmonth), len(dweek), len(hour)))
    
    sma_20 = pd.to_numeric(sma_20, errors='coerce').values
    sma_50 = pd.to_numeric(sma_50, errors='coerce').values
    rsi = pd.to_numeric(rsi, errors='coerce').values
    bb_width = pd.to_numeric(bb_width, errors='coerce').values
    momentum = pd.to_numeric(momentum, errors='coerce').values
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    volume = df[['volume']].values
    
    min_length = close_prices.shape[0]-50 # len(sma_50)#max(min_length - 50, 0)
    print("min length: {}".format(min_length))
    if nfeatures == 13:
        features = np.column_stack([
            sma_20[-min_length:],  # SMA 20
            sma_50[-min_length:],  # SMA 50
            rsi[-min_length:],     # RSI
            bb_width[-min_length:],# Bollinger Bands Width
            momentum[-min_length:],# Momentum
            volume[-min_length:],  # Volume
            k[-min_length:],       # Stochastic %K
            d[-min_length:],       # Stochastic %D
            macddiff[-min_length:],# MACD difference
            dmonth[-min_length:],  # Day of the month
            dweek[-min_length:],   # Day of the week
            hour[-min_length:]     # Hour of the day
        ]).astype(float)
    elif nfeatures == 5:
        features = np.column_stack([sma_20[-min_length:],rsi[-min_length:],
        bb_width[-min_length:],
        momentum[-min_length:]])
    
    
    target = close_prices[-min_length:]
    if np.any(np.isnan(features)):
        raise ValueError("There is NaNs in features")
    else:
        return target, features

 
def prepare_extended_data(features, target, dates, train_ratio = 0.01, verbose = False):
    if verbose:
        print("prep extended data from: {}, {}, {}".format(features.shape, target.shape, dates.shape))
    
    dates = dates[10:-5]
    
    input, output = [], []
    input_train = []
    output_train = []
    input_test = []
    output_test = []
    
    for i in range(10, len(target) - 25):
        # Create input: last 10 closing prices + 5 technical indicators
        input.append(np.concatenate((features[i],target[i-10:i])))
        
        # Create output: next 5 closing prices
        output.append(target[i:i+5]) #close_prices[i + lookback_prices:i + lookback_prices + future_steps]
        #outputs.append(output_prices)
        
    if verbose:
        print("Example of training data preprocessing (last extended datapoint)")
        print("input datapoint 5 TI and 10 last prices")
        print(input[i-10])
        print("output data point next 5 prices")
        print(output[i-10])
        
    split_index = int(len(input) * train_ratio)
    
    input_train = input[:-split_index]
    input_test = input[-split_index:]
    
    output_train = output[:-split_index]
    output_test = output[-split_index:]
    
    date_train = dates[:-split_index]
    date_test = dates[-split_index:]
    
    return np.asarray(input_train), np.asarray(input_test), np.asarray(output_train), np.asarray(output_test), np.asarray(date_train), np.asarray(date_test)

def slice_tapes(target, features, lookf, lookb, steps, stab_slope, scaler = None, onlyfirstpoints = None, indices = None, nowstr = " ", trainorval = None):
    """ Prepares the input for the ML model from tha API fetched data. It cuts and slices the
        data in samples (tapes), and filters out tapes that are not in a stationary distribution"""
    # Indices is either a numpy.loadtxt containing indices or the running index for machines
    # that are searching good subsets of data
        
    if stab_slope != None:
        stab_slope = float(stab_slope)
        print("Filter out any tapes containing return > {}% of the max return".format(stab_slope*100))
    
    returns = calc_returns(target)
    stacked = np.hstack((returns.reshape(-1,1), features.reshape(-1,12)[1:,:]))
    
    print("Returns: {} +- {}".format(np.mean(returns), np.std(returns)))
    print("Min: {}, Max: {}".format(np.min(returns), np.max(returns)))
    
    
    if scaler == None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        stacked_n = scaler.fit_transform(stacked.reshape(-1,13))
    else:
        stacked_n = scaler.transform(stacked.reshape(-1,13))
        
    stacked_n = stacked_n.reshape(-1,13,1)
    x, y = [], []
    
    if isinstance(indices, np.ndarray): # Indices file was passed
        #random_indices = np.random.choice(len(stacked), size=onlyfirstpoints, replace=False)
        print(indices.shape)
        select_indices = indices
        stacked_n = stacked_n[select_indices]
    elif isinstance(indices, int):    # Generate subsets from a Running index
        #select_indices = np.random.choice(len(x), size=onlyfirstpoints, replace=False)
        select_indices = range(indices*onlyfirstpoints, (indices+1)*onlyfirstpoints)
        print("Subset from {} to {}".format(select_indices[0], select_indices[-1]))
        stacked_n = stacked_n[select_indices]
        
        np.savetxt("../indices/"+str(trainorval)+"_indices"+nowstr+".txt", select_indices, fmt='%d')
    
    """ Slicing to get the single tapes (batch_size, time_steps, features)
    If tape contains any return > stab_slope, throw away.     """
    for i in range(0,len(stacked_n)-lookb-lookf,steps):
        if stab_slope == None: # Dont filter out static distribution
            x.append(stacked_n[i:i+lookb,:])
            y.append(stacked_n[i+lookb:i+lookf+lookb,0])
        elif np.all(stacked_n[i:i+lookb,0] < stab_slope):
            x.append(stacked_n[i:i+lookb,:])
            y.append(stacked_n[i+lookb:i+lookf+lookb,0])
            
            
                
    return np.asarray(x),  np.asarray(y), scaler


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    print("runnin")