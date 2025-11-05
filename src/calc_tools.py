import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calc_returns(target):
    returns = []
    for i in range(len(target)-1):
        returns.append((target[i+1]-target[i])/target[i]*100)
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

        """ encode_time_cyclically(dmonth, dweek, hour) """
        # Day of month (1–31)
        dmonth_sin = np.sin(2 * np.pi * dmonth / 31)
        dmonth_cos = np.cos(2 * np.pi * dmonth / 31)

        # Day of week (0–6)
        dweek_sin = np.sin(2 * np.pi * dweek / 7)
        dweek_cos = np.cos(2 * np.pi * dweek / 7)

        # Hour of day (0–23)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        return np.column_stack([dmonth_sin, dmonth_cos,
                                dweek_sin, dweek_cos,
                                hour_sin, hour_cos])

def compute_features_trim(data, timestamp_ms, nfeatures = 13):
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
    #dmonth, dweek, hour = extract_times(timestamp_ms)


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
        np.zeros(np.asarray(k[-min_length:]).shape),    # day of month
        np.zeros(np.asarray(k[-min_length:]).shape),    # day of week
        np.zeros(np.asarray(k[-min_length:]).shape)     # Hour of the day
    ]).astype(float)

    """ DATES FEATURES ARE PLACEHOLDERS.
     NEED 6 FEATURES FOR CYCLICAL DATES  """

    target = close_prices[-min_length:].to_numpy()
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

if __name__ == "__main__":
    np.set_printoptions(precision=2)
    print("runnin")
