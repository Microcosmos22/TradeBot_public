import numpy as np
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
from calc_tools import *
from plotting import *
from scaler import FeatureAwareScaler
import copy
import plotly
import plotly.graph_objects as go

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
        self.data = None
        self.lookf, self.lookb, self.stability_slope, self.val_train_proportion = 0,0,0,0.2
        self.x_train, self.y_train, self.x_val, self.y_val, self.scaler = None, None, None, None, None
        self.target_train, self.target_val, self.features_train, self.features_val = 0,0,0,0
        self.synth_target, self.synth_features = None, None
        self.target_total, self.features_total = None, None
        self.stacked, self.stacked_n = None, None

    def load_simdata(self, sim_N):
        self.target_total = np.load("../target_sim.npy")[:sim_N]
        self.features_total = np.load("../features_sim.npy")[:sim_N]
        print("Loading target: {} and features: {}".format(self.target_total.shape, self.features_total.shape))
        #self.dates_total = np.load("../../dates_sim.npy")
        return self.target_total, self.features_total

    def get_historical_data_trim(self, timedef, coin = "BTCUSDT", candle_length = Client.KLINE_INTERVAL_1HOUR, transform_func=None, transform_strength = 3, plot = False):
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

        elif isinstance(timedef, str):

            data = np.asarray(client.get_historical_klines(coin, candle_length, timedef)).astype(float)
            print("Calling candles since {}".format(timedef))

        elif (isinstance(timedef[0], str) and type(timedef[1]) == int):
            end_date = datetime.strptime(timedef[0], "%d %B %Y %H:%M:%S")
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

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        self.data = pd.DataFrame(data, columns=columns)
        print(data.shape)

        """ (time, open, high, low, close, volume, close_time etc.) """
        # Use closing prices

        self.timestamps = data[:, 0]  # First column contains the timestamp (open time)
        #print(self.timestamps)
        self.target_total, self.features_total = compute_features_trim(data, self.timestamps)

        if transform_func is not None:
            return_shift = transform_func(self.target_total, self.features_total, transform_strength)

            returns = calc_returns(data[:,1].astype(float))
            print(" Original Returns [%]: {:.6f} +- {:.6f}".format(np.mean(returns), np.std(returns)))

            for k in range(1,5):
                data[50:,k] = np.multiply(data[50:,k].astype(float), return_shift)

            #print(" Average trader return impact [%] {:.6f} +- {:.6f} ".format(100-(np.mean(return_shift)*100), np.std(return_shift)*100))
            returns = calc_returns(data[:,1].astype(float))
            print(" Synth. returns after trader [%]: {:.6f} +- {:.6f}".format(np.mean(returns), np.std(returns)))


            # This for without recomputing synth features
            self.synth_features = self.features_total
            self.synth_target = np.multiply(self.target_total, return_shift)
            # This is for re-computing the synth features after trader
            #self.synth_target, self.synth_features = compute_features_trim(data, self.timestamps)

        if plot:
            self.plot_candlechart(200)
            plt.plot(self.target_total, label = " Orig. BTCUSD")
            plt.plot(self.synth_target, label = " Synth BTCUSDT")
            plt.xlabel(" Time (5min candles)")
            plt.ylabel(" BTCUSDT in $")
            plt.legend()
            plt.show()

        return np.asarray(self.target_total), np.asarray(self.features_total), np.asarray(self.synth_target), np.asarray(self.synth_features)

    def split_train_val(self, target_total = None, features_total = None):
        if target_total is None:
            target_total = self.target_total
        if features_total is None:
            features_total = self.features_total

        split_idx = int(self.target_total.shape[0]*(1-self.val_train_proportion))

        """ Split target and features into training and validation """
        self.target_train = target_total[:split_idx]
        self.target_val = target_total[split_idx:]
        self.features_train = features_total[:split_idx]
        self.features_val = features_total[split_idx:]

        return self.target_train, self.target_val, self.features_train, self.features_val

    def slice_alltapes_normalize(self, lookb, lookf):
        self.lookb, self.lookf = lookb, lookf

        self.x_train, self.y_train, _ = self.slice_tapes_normalize(self.target_train, self.features_train, self.lookf, self.lookb)
        print("LSTM In- & Out shapes (training): {}, {}".format(self.x_train.shape, self.y_train.shape))
        print()
        self.x_val, self.y_val, _ = self.slice_tapes_normalize(self.target_val, self.features_val, self.lookf, self.lookb)
        print("LSTM In- & Out shapes (validation): {}, {}".format(self.x_val.shape, self.y_val.shape))
        return self.x_train, self.y_train, self.x_val, self.y_val, self.scaler

    def slice_tapes_normalize(self, target, features, lookf, lookb, scaler = None):
        """ Prepares the input for the ML model from tha API fetched data. It cuts and slices the
            data in samples (tapes), and filters out tapes that are not in a stationary distribution """
        # Indices is either a numpy.loadtxt containing indices or the running index for machines
        # that are searching good subsets of data

        returns = calc_returns(target)
        self.stacked = np.hstack((returns.reshape(-1,1), features.reshape(-1,12)[1:,:]))

        if scaler == None:
            self.scaler = FeatureAwareScaler()
            self.stacked_n = self.scaler.fit_transform(self.stacked.reshape(-1,13))
        else:
            self.stacked_n = self.scaler.transform(self.stacked.reshape(-1,13))

        x, y = [], []

        """ Slicing to get the single tapes (batch_size, time_steps, features)
        If tape contains any return > stab_slope, throw away.     """
        for i in range(0,len(self.stacked_n)-lookb-lookf):
            x.append(self.stacked_n[i:i+lookb,:])
            y.append(np.mean(self.stacked_n[i+lookb:i+lookf+lookb,0]))

        return np.asarray(x),  np.asarray(y), self.scaler

    def split_slice_normalize(self, lookb, lookf, target_total = None, features_total = None):
        if target_total is None:
            target_total = self.target_total
        if features_total is None:
            features_total = self.features_total
        self.lookb, self.lookf = lookb, lookf

        self.split_train_val(target_total, features_total)
        self.slice_alltapes_normalize(self.lookb, self.lookf)
        return self.x_train, self.y_train, self.x_val, self.y_val, self.scaler

    def plot_candlechart(self, N):

        fig = go.Figure(data=[
        go.Candlestick(
            x=self.data[['timestamp']].to_numpy().flatten()[:N],
            open=self.data[['open']].to_numpy().flatten()[:N],
            high=self.data[['high']].to_numpy().flatten()[:N],
            low=self.data[['low']].to_numpy().flatten()[:N],
            close=self.data['close'].to_numpy().flatten()[:N]
            )
        ])

        fig.update_layout(
            title={
                'text': "BTC/USDT Candlestick Chart",
                'x': 0.5,  # centers the title
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Time",
            yaxis_title="Price $",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"  # optional: gives a nice look
        )

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()
        return

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

    cryptodata = CryptoDataGetter()
    target, features, _, _ = cryptodata.get_historical_data_trim(
    ["1 August 2024 00:00:00", 15000],
    "BTCUSDT",
    Client.KLINE_INTERVAL_5MINUTE)

    target_train, target_val, features_train, features_val = cryptodata.split_train_val(target, features)

    x_train, y_train, x_val, y_val, scaler = cryptodata.slice_alltapes(lookb = 10, lookf = 5)

    print(len(cryptodata.stacked[0]))
    col = plot_scaling_stacked(cryptodata.stacked, cryptodata.stacked_n)

    col = ["ret", "sma20", "sma50", "RSI", "BBwidth", "mom", "vol", "K", "D", "MACD", "d_month", "d_week", "h_day"]


    for i in range(len(cryptodata.stacked[0])):
        print("col: {} min: {} max: {} ".format(col[i], np.min(cryptodata.stacked[:,i]), np.max(cryptodata.stacked[:,i])))
    for i in range(len(cryptodata.stacked_n[0])):
        print("col: {} min: {} max: {} ".format(col[i], np.min(cryptodata.stacked_n[:,i]), np.max(cryptodata.stacked_n[:,i])))
