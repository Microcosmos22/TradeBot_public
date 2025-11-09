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
from .simulation import *
from .plotting import *
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.layers import LSTM
import logging
import copy
import os, time


def calculate_mean_std_last_n_returns(df, n=6):
    """Calculate the mean and std of the last 'n' returns for each model."""
    mean_returns = df.iloc[:, -n:].mean(axis=0)  # Mean of the last 'n' returns
    std_returns = df.iloc[:, -n:].std(axis=0)    # Standard deviation of the last 'n' returns
    return mean_returns, std_returns

def find_best_models(df, n=5):
    """Find the top 'n' models with the highest mean returns."""
    # Calculate mean returns and standard deviations for the last 6 returns
    mean_returns, _ = calculate_mean_std_last_n_returns(df, n=6)

    # Sort models by the mean return in descending order and select top n
    top_models = mean_returns.nlargest(n)
    return top_models


# Run the program
if __name__ == "__main__":
    with open("tracking.pkl", 'rb') as f:
        df = pickle.load(f)

    print("Rendite in %/jahr")
    print(df.values)
    for idx in df.index:
        print(idx)


    df = df
    mean_returns, std_returns = calculate_mean_std_last_n_returns(df, n=6)
    print(mean_returns.shape, std_returns.shape)

    for i in range(len(mean_returns)):
        print("{} +- {}".format(mean_returns[i], std_returns[i]))





    best_models = find_best_models(df, n=5)


    print("Top 5 models with the highest mean returns (anteil / year):")
    for model, mean_return in best_models.items():
        print(f"{model}: {mean_return:.2f}")

    # Optionally, print the mean and std of the last 6 returns for each model
    print("\nMean returns (last 6 periods):")
    print(mean_returns)
    print("\nStandard Deviation of returns (last 6 periods):")
    print(std_returns)
