import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
from calc_tools import *
from get_binance import *
from plotting import *
from synthetic_driver import *
from simulation import *
import pandas as pd
import scipy.signal as signal
import pickle
import copy


if __name__ == "__main__":

    cryptodata = CryptoDataGetter()
    synth = SyntheticTrader(cryptodata)

    target, features, synth_target, synth_features = cryptodata.get_historical_data_trim(
    ["1 August 2024 00:00:00", 15000], "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE,
    transform_func=synth.linear_RSI, transform_strength = 0.02)

    """ Plotting some info on the data """
    cryptodata.plot_candlechart(200)
    epochs = 50

    """ ###################### --- ###################### """
    cryptodata.split_train_val(synth_target, synth_features)
    x_trains, y_trains, x_vals, y_vals, scaler = cryptodata.slice_alltapes_normalize(lookb = 10, lookf = 5)

    synth_machine = CryptoMachine()
    synth_machine.init(candle = "1h", layer1 = 40, layer2 = 15, lookb = 10, learn_rate = 0.03 , dropout = 0.1, reg = 1e-4)
    trainmean2, train_std2, valmean2, val_stdfinal2 = synth_machine.fit(x_trains, y_trains, x_vals, y_vals, epochs = epochs, batch = 16)

    """ ###################### --- ###################### """
    plot = MachinePlotter(synth_machine, synth_machine, x_val, y_val, x_vals, y_vals)
    plot.plotmachines([trainmean1, trainmean2], [train_std1, train_std2], [valmean1, valmean2], [val_stdfinal1, val_stdfinal2])
    plot.plot_tape_eval(x_val, y_val)

    print(" Final prediction errors on the validation of each series. ")
    print(" Original: {:.4f} +- {:.4f}".format(np.mean(valmean1), val_stdfinal1))
    print(" Synth:   {:.4f} +- {:.4f}".format(np.mean(valmean2), val_stdfinal2))

    """ ############# PROFIT SIMULATION ############## """
