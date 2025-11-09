import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
from .calc_tools import *
from .get_binance import *
from .plotting import *
from .synthetic_driver import *
from .simulation import *
import pandas as pd
import scipy.signal as signal
import pickle
import copy


if __name__ == "__main__":

    cryptodata = CryptoDataGetter()
    synth = SyntheticTrader(cryptodata)
    synth_machine = LSTMachine().init(candle = "5min", layer1 = 40, layer2 = 15, lookb = 10, learn_rate = 0.03 , dropout = 0.1, reg = 1e-4)

    """ ## Call historical data, simulate and apply an artificial trader ## """

    _, _, synth_target, synth_features = cryptodata.get_historical_data_trim(
    ["1 August 2024 00:00:00", 15000], "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE,
    transform_func=synth.linear_RSI, transform_strength = 0.02, plot = False)

    """ ############# Prepare Inputs, train the Neural Network ########### """
    x_train, y_train, x_val, y_val, scaler = cryptodata.split_slice_normalize(lookb = 10, lookf = 5, target_total = synth_target, features_total = synth_features)

    trainmean, train_std, valmean, val_std = synth_machine.fit(x_train, y_train, x_val, y_val, epochs = 50, batch = 16)

    """ ############## Plot training and some examples #################### """
    plot = MachinePlotter(synth_machine)
    plot.plotmachine(trainmean, train_std, valmean, val_std)
    plot.plot_tape_eval(x_val, y_val)
