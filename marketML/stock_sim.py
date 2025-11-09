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
from .calc_tools import *
from .get_binance import *
from .plotting import *
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.layers import LSTM
import logging
import copy
import os

def k_driver(target0, features, traderpower = 10):
    k = features[:,6]
    target = copy.copy(target0)

    for i in range(len(target)):
        pricedrive = (50-k[i])*traderpower
        target[i] = target[i] + pricedrive

    return target


if __name__ == "__main__":
    end_of_training = datetime.strptime("1 June 2023 00:00:00", "%d %B %Y %H:%M:%S")
    target0, features, timestamps, data = get_historical_data_trim([end_of_training, 3200], candle_length = Client.KLINE_INTERVAL_1HOUR, plot=False)




    plt.plot(target0, label="orig")
    plt.plot(target, label="driven")
    plt.plot(k, label="k")
    plt.legend()
    plt.show()

    # Once the prices have been modified by the simulated daytraders, have to recompute the features
    target, features = compute_features_trim(data, timestamps)
