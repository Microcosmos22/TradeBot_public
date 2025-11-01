import numpy as np
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
from calc_tools import *
from plotting import *
import copy

def candle2interval(candle_length):
    """ From a candle length, it returns a timedelta of the candle time window """
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


def k_driver(target0, features, traderpower = 50):
    k = features[:,6]
    target = copy.copy(target0)

    for i in range(len(target)):
        pricedrive = (50-k[i])*traderpower
        target[i] = target[i] + pricedrive

    return target

class CryptoDataGetter:
    def __init__(self):
        self.end_of_training = None
        self.ncandles = None
        self.coin = None
        self.lookf, self.lookb, self.stability_slope, self.val_train_proportion = 0,0,0,0.2
        self.x_train, self.y_train, self.x_val, self.y_val, self.scaler = None, None, None, None, None
        self.target_train, self.target_val, self.features_train, self.features_val = 0,0,0,0

    def load_simdata(self, sim_N):
        self.target_total = np.load("../target_sim.npy")[:sim_N]
        self.features_total = np.load("../features_sim.npy")[:sim_N]
        print("Loading target: {} and features: {}".format(self.target_total.shape, self.features_total.shape))
        #self.dates_total = np.load("../../dates_sim.npy")
        return self.target_total, self.features_total

    def get_historical_data_trim(self, timedef, coin = "BTCUSDT", candle_length = Client.KLINE_INTERVAL_1HOUR):
        """ Given a end_of_training datetime and N of candles, returns the past N candles and computes the features/technical indicators.
        Gets all candles and features of a crypto coin for a fixed candle length. There is three options:
        # 1: timedef is an int -> retrieve the last N candles
        # 2: timedef is a start_datetime -> retrieve candles until now
        # 3: COMMONLY: timedef is [end_date, int] to retrieve last N candles before "end_date"
        """
        # Binance API key and secret
        api_key = "pxRONzQcbpDoImQXzHqkO6XJWd7WMIKSTyBPtTlkvaCbIGJ0Whcnz8LDw7SavMIx"
        api_secret = "hFXzByh1Fg90Vcxvx8uakDq9n6reH32KswXuYOTzFxxjmAVvMbHRh1lOvMSgHlex"
        client = Client(api_key, api_secret)

        if type(timedef) == int:
            now = datetime.now()
            end_date = now.strftime("%Y-%m-%d %H:%M:%S")
            n_candles = timedef

            interval = candle2interval(candle_length)
            time_to_subtract = interval * n_candles
            # Calculate the start_date by subtracting the total time from the end_date
            start_date = now - time_to_subtract

            data = np.asarray(client.get_historical_klines(
            coin, candle_length,
            start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=now.strftime("%Y-%m-%d %H:%M:%S")
            ))

            print("Calling last {} candles".format(data[:, 0].shape[0]))

        elif isinstance(timedef, datetime):

            timedef = timedef.strftime("%Y-%m-%d %H:%M:%S")
            data = np.asarray(client.get_historical_klines(coin, candle_length, timedef)).astype(float)
            print("Calling candles since {}".format(timedef))

        elif (isinstance(timedef[0], datetime) and type(timedef[1]) == int):

            end_date = timedef[0]
            n_candles = timedef[1]

            interval = candle2interval(candle_length)
            time_to_subtract = interval * n_candles
            # Calculate the start_date by subtracting the total time from the end_date
            start_date = end_date - time_to_subtract

            data = np.asarray(client.get_historical_klines(
            coin, candle_length,
            start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end_date.strftime("%Y-%m-%d %H:%M:%S")
            ))

            print("Calling last {} candles before {} (starting {})".format(n_candles, end_date, start_date))


        elif (isinstance(timedef[0], datetime) and isinstance(timedef[1], datetime)):
            start_date = timedef[0]
            end_date = timedef[1]

            data = np.asarray(client.get_historical_klines(
            coin, candle_length,
            start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end_date.strftime("%Y-%m-%d %H:%M:%S")))
            print("calling {} candles between {} and {}".format(data.shape, timedef[0].strftime("%d %b, %Y %H:%M:%S"), timedef[1].strftime("%Y-%m-%d %H:%M:%S")))

        print(data.shape)
        """ (time, open, high, low, close, volume, close_time etc.) """
        # Use closing prices

        self.timestamps = data[:, 0]  # First column contains the timestamp (open time)
        #print(self.timestamps)
        self.target_total, self.features_total = compute_features_trim(data, self.timestamps)
        #self.dates = np.asarray([datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S") for ts in self.timestamps])

        #target_total = k_driver(target_total, features_total)

        """ Split target and features into training and validation """
        self.target_train = self.target_total[:int(self.target_total.shape[0]*(1-self.val_train_proportion))]
        self.target_val = self.target_total[int(self.target_total.shape[0]*(1-self.val_train_proportion)):]
        self.features_train = self.features_total[:int(self.features_total.shape[0]*(1-self.val_train_proportion))]
        self.features_val = self.features_total[int(self.features_total.shape[0]*(1-self.val_train_proportion)):]
        return np.asarray(self.target_total), np.asarray(self.features_total)#, np.asarray(timestamps).astype(np.int64), data

    def slice_train_and_val(self, lookb, lookf):
        self.lookb, self.lookf = lookb, lookf

        self.x_train, self.y_train = self.slice_tapes(self.target_total, self.features_total, self.lookf, self.lookb, self.stability_slope, None, trainorval = "train")
        print("LSTM In- & Out shapes (training): {}, {}".format(self.x_train.shape, self.y_train.shape))
        print()
        self.x_val, self.y_val= self.slice_tapes(self.target_total, self.features_total, self.lookf, self.lookb, self.stability_slope, self.scaler, trainorval="val")
        print("LSTM In- & Out shapes (validation): {}, {}".format(self.x_val.shape, self.y_val.shape))
        return self.x_train, self.y_train, self.x_val, self.y_val, self.scaler

    def slice_tapes(self, target, features, lookf, lookb, stab_slope, scaler = None, nowstr = " ", trainorval = None):
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
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            stacked_n = self.scaler.fit_transform(stacked.reshape(-1,13))
        else:
            stacked_n = self.scaler.transform(stacked.reshape(-1,13))

        stacked_n = stacked_n.reshape(-1,13,1)
        x, y = [], []

        """ Slicing to get the single tapes (batch_size, time_steps, features)
        If tape contains any return > stab_slope, throw away.     """
        for i in range(0,len(stacked_n)-lookb-lookf):
            if stab_slope == None: # Dont filter out static distribution
                x.append(stacked_n[i:i+lookb,:])
                y.append(stacked_n[i+lookb:i+lookf+lookb,0])
            elif np.all(stacked_n[i:i+lookb,0] < stab_slope):
                x.append(stacked_n[i:i+lookb,:])
                y.append(stacked_n[i+lookb:i+lookf+lookb,0])

        return np.asarray(x),  np.asarray(y)

def load_example_train_val(ncandles=3200, coin = "BTCUSDT", candle=Client.KLINE_INTERVAL_1HOUR):
    #start_of_training = datetime.strptime("1 September 2020 00:00:00", "%d %B %Y %H:%M:%S")
    end_of_training = datetime.strptime("1 June 2023 00:00:00", "%d %B %Y %H:%M:%S")
    target_train, features_train, _, _ = get_historical_data_trim([end_of_training, ncandles], coin = "BTCUSDT")
    target_train = k_driver(target_train, features_train)
    np.save("../../target_train_1h.npy", target_train)
    np.save("../../features_train_1h.npy", features_train)


    #target_train = np.load("../../target_train_1h.npy")
    #features_train = np.load("../../features_train_1h.npy")

    #start_of_val = datetime.strptime("2 January 2024 00:00:00", "%d %B %Y %H:%M:%S")
    end_of_val = datetime.strptime("1 August 2024 00:00:00", "%d %B %Y %H:%M:%S")

    target_val, features_val, _, _ = get_historical_data_trim([end_of_val, int(ncandles/4)], coin = "BTCUSDT")
    target_val = k_driver(target_val, features_val)
    np.save("../../target_val_1h.npy", target_val)
    np.save("../../features_val_1h.npy", features_val)

    #target_val = np.load("../../target_val_1h.npy")
    #features_val = np.load("../../features_val_1h.npy")


    return target_train, target_val, features_train, features_val


if __name__ == "__main__":


    print("features tr length: {}".format(features_train.shape))
    print("features val length: {}".format(features_val.shape))

    x_train, y_train, scaler_fitted = slice_tapes(target_train, features_train, lookf, lookb, stability_slope, None, onlyfirstpoints, indices, nowstr, trainorval = "train")
    print("LSTM In- & Out shapes (training): {}, {}".format(x_train.shape, y_train.shape))
    print()
    x_val, y_val, scaler = slice_tapes(target_val, features_val, lookf, lookb, stability_slope, scaler_fitted, int(onlyfirstpoints/4), int(indices), nowstr, trainorval="val")
    print("LSTM In- & Out shapes (validation): {}, {}".format(x_val.shape, y_val.shape))


    x_train = x_train.reshape((-1, lookb, 13))
    x_val = x_val.reshape((-1, lookb, 13))


    np.set_printoptions(precision=2)


    start_str = "1 January 2024 00:00:00"
    end_str = "1 December 2024 00:00:00"

    start_of_training = datetime.strptime(start_str, "%d %B %Y %H:%M:%S")
    end_of_training = datetime.strptime(end_str, "%d %B %Y %H:%M:%S")
