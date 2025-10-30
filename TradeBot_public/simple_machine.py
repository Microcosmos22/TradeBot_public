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
from plotting import *
from keras.models import Sequential, load_model
import pandas as pd
import scipy.signal as signal
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed
import pickle
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
import copy


def deleteplots(file1, file2, file3):
    """ Gets three filepaths, deletes all three plots """
    files_to_delete = [file1, file2, file3]

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File {file_path[45:]} removed successfully.")
            except Exception as e:
                print(f"Error removing file {file_path[45:]}: {e}")

class SimpleMachine:
    def __init__(self, layer1, layer2, lookb, lookf, learn_rate, dropout):
        self.layer1 = layer1
        self.layer2 = layer2
        self.lookb = lookb
        self.lookf = lookf
        self.learn_rate = learn_rate
        self.dropout = dropout

        # Define the model
        self.model = Sequential()
        self.model.add(LSTM(layer1, return_sequences=True, input_shape=(lookb, 13)))
        self.model.add(Dropout(dropout))

        if lookf == 1:
            self.model.add(LSTM(layer2, activation="sigmoid", return_sequences=False))
            self.model.add(Dense(1, activation="sigmoid"))
        else:
            self.model.add(LSTM(layer2, activation="sigmoid", return_sequences=True))
            self.model.add(TimeDistributed(Dense(1, activation="sigmoid")))

        optimizer = Adam(learning_rate=float(learn_rate))
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model.summary()

        # Attributes to store training curves
        self.errorplots = None
        self.valplots = None
        self.mae = None
        self.modelpath = None
        self.scalerpath = None

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch,
            scaler=None, steps=1, stability_slope=0, onlyfirstpoints=0, candle=1, nowstr="NOW", plot_curves=True):

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch,
            validation_data=(x_val, y_val)
        )

        # Save training metrics
        self.errorplots = np.log(history.history['loss'])
        self.valplots = np.log(history.history['val_loss'])
        self.mae = np.log(history.history['mae'])

        # Define file paths
        self.modelpath = f"../machines/1H1KTAPES_machine_{nowstr}_f{self.lookf}_b{self.lookb}_s{steps}_sbs{stability_slope}_l{self.learn_rate}_b{batch}_one{self.layer1}_two{self.layer2}_only{onlyfirstpoints}_dr{self.dropout}_candle{candle}.h5"
        self.scalerpath = f"../scalers/1K1KTAPES_scaler_{nowstr}_f{self.lookf}_b{self.lookb}_s{steps}_sbs{stability_slope}_l{self.learn_rate}_b{batch}_one{self.layer1}_two{self.layer2}_only{onlyfirstpoints}_dr{self.dropout}_candle{candle}.pkl"

        # Save model and scaler
        print(f"saving model {self.modelpath} \n and scaler {self.scalerpath}")
        self.model.save(self.modelpath)
        if scaler is not None:
            with open(self.scalerpath, 'wb') as f:
                pickle.dump(scaler, f)

        # Plot curves if requested
        if plot_curves:
            self.plot(nowstr)

    def plot(self, nowstr="NOW"):
        if self.errorplots is None or self.valplots is None:
            print("No training history to plot.")
            return

        # File paths
        file1 = f"..\\traincurves\\trainerror_{self.modelpath[12:]}.png"
        file2 = f"..\\traincurves\\mae_{self.modelpath[12:]}.png"
        file3 = f"..\\traincurves\\valerror_{self.modelpath[12:]}.png"
        deleteplots(file1, file2, file3)
        time.sleep(1)

        # Plot training loss
        plt.plot(self.errorplots, label='Train Loss')
        plt.legend()
        plt.title("Training Error")
        plt.savefig(f"../traincurves/trainerror_{nowstr}{self.modelpath[12:]}.png")

        # Plot validation loss
        plt.plot(self.valplots, label='Validation Loss')
        plt.legend()
        plt.title("Validation Error")
        plt.savefig(f"../traincurves/valerror_{nowstr}{self.modelpath[12:]}.png")
        plt.clf()

        print("Saved training curves")



if __name__ == "__main__":

    #target_train, target_val, features_train, features_val = load_example_train_val(32200, coin = "BTCUSDT", candle = Client.KLINE_INTERVAL_1HOUR)
    #x_train, y_train, scaler_fitted = slice_tapes(target_train, features_train, lookf, lookb, steps, stability_slope, None, onlyfirstpoints, indices, nowstr, trainorval = "train")

    epochs = 100
    #stability_cut = None#0.5 # 0.05 # where stab_cut * lookb (=100) is the maximal return observed in the train/val/test set

    cryptodata = CryptoDataGetter()
    cryptodata.get_historical_data_trim("1 August 2024 00:00:00", 32000, "BTCUSDT", lookf = 10, lookb = 5, steps = 1)
    cryptodata.slice_train_and_val()

    x_train = cryptodata.x_train
    y_train = cryptodata.y_train
    x_val = cryptodata.x_val
    y_val = cryptodata.y_val

    simple_machine = SimpleMachine(layer1 = 40, layer2 = 15, lookb = 10, lookf = 5, learn_rate = 0.09 , dropout = 0.0)
    simple_machine.fit(x_train, y_train, x_val, y_val, epochs = epochs, batch = 16)
