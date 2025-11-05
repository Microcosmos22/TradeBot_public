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
import pandas as pd
import scipy.signal as signal
import pickle
import copy

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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

class CryptoMachine:
    def __init__(self):
        # Attributes to store training curves
        self.candle, self.layer1, self.layer2, self.lookb, self.lookf, self.learn_rate, self.dropout = None, None, None, None, None, None, None

        self.model = None
        self.scaler = None

        self.trainmean = None
        self.valmean = None
        self.mae = None
        self.modelpath = None
        self.scalerpath = None
        self.epochs = None
        self.batch = None
        self.error_batch_epoch = None
        self.logger = None

    def init(self, candle, layer1, layer2, lookb, lookf, learn_rate, dropout):
        self.candle, self.layer1, self.layer2, self.lookb, self.lookf, self.learn_rate, self.dropout = candle, layer1, layer2, lookb, lookf, learn_rate, dropout

        # Define the model
        self.model = Sequential()
        self.model.add(LSTM(layer1, return_sequences=True, input_shape=(lookb, 13), kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)))
        self.model.add(Dropout(dropout))

        self.model.add(LSTM(layer2, activation="sigmoid", return_sequences=False, kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=float(learn_rate))
        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model.summary()

    def load_machine(self, machinestrn, scalerstrn):

        self.model = load_model(os.path.join("../../machines", machinestrn))
        with open(os.path.join("../../scalers", scalerstrn), 'rb') as f:
            scalern = pickle.load(f)
            self.scaler = scalern
        self.lookf = 1

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch,
            save=False, candle=1, nowstr="NOW", plot_curves=True):
        self.epochs = epochs

        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        self.logger = BatchLossLogger(validation_data=(x_val, y_val), batch = batch)

        self.batch = batch
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch,
            validation_data=(x_val, y_val),
            callbacks=[self.logger],
            verbose=0
        )

        # Save training metrics
        self.trainmean = np.asarray(history.history['loss'])
        self.valmean = np.asarray(history.history['val_loss'])

        self.train_std = np.std(self.logger.train_batch_losses, axis=1)
        self.val_stdfinal = self.calc_val_std(x_val, y_val)

        return self.trainmean, self.train_std, self.valmean, self.val_stdfinal


    def save_model_scaler(self, scaler):
        # Define file paths
        nowstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.modelpath = f"../../machines/1H1KTAPES_machine_{nowstr}_f{self.lookf}_b{self.lookb}_l{self.learn_rate}_b{self.batch}_one{self.layer1}_two{self.layer2}_dr{self.dropout}_candle{self.candle}.h5"
        self.scalerpath = f"../../scalers/1K1KTAPES_scaler_{nowstr}_f{self.lookf}_b{self.lookb}_l{self.learn_rate}_b{self.batch}_one{self.layer1}_two{self.layer2}_dr{self.dropout}_candle{self.candle}.pkl"

        print(f"saving model {self.modelpath} \n and scaler {self.scalerpath}")
        self.model.save(self.modelpath)
        if scaler is not None:
            with open(self.scalerpath, 'wb') as f:
                pickle.dump(scaler, f)

    def raws_predict(self, target, features):
        """ Makes the LAST tape of the given target, features (uses internal lookb/f)
        and makes a prediction for the next candle. Accepts both lookf=10 or 1 models,
        as it grabs only the first following value    """

        returns_test = calc_returns(target) # true returns
        stacked_test = np.hstack((returns_test.reshape(-1,1), features[1:]))[-10:]

        stacked_test_n = self.scaler.transform(stacked_test.reshape(-1, self.scaler.n_features_in_))

        stacked_test_n = stacked_test_n.reshape(-1,self.scaler.n_features_in_,1)
        return_pred_n = self.model.predict(stacked_test_n[-self.lookb:,:].reshape(1, self.lookb, self.scaler.n_features_in_), verbose=0)

        stacked_n = np.full((self.lookf, self.scaler.n_features_in_), np.nan)

        stacked_n[:,0] = return_pred_n.reshape(-1)[:self.lookf]
        stacked_pred = self.scaler.inverse_transform(stacked_n) # prediction is returns

        return stacked_pred

    def tape_predict(self, x):
        """ Takes a (normalized) tape and makes a prediction for the next candle """
        """ RAISE WARNING IF TAPES lookb DOES NOT MATCH THE MACHINE """
        return_pred_n = self.model.predict(x.reshape(1, self.lookb, 13), verbose=0)[0,0,0]

        stacked_n = np.full((1, 13), np.nan)
        stacked_n[0,0] = return_pred_n.reshape(-1)
        stacked_pred = self.scaler.inverse_transform(stacked_n) # prediction is returns

        return stacked_pred

    def calc_val_std(self, x_val, y_val):
        # x_val, y_val are your validation sets
        val_batch_losses = []

        for i in range(0, len(x_val), self.batch):
            X_batch = x_val[i:i+self.batch]
            y_batch = y_val[i:i+self.batch]
            batch_loss = simple_machine.model.evaluate(X_batch, y_batch, verbose=0)
            val_batch_losses.append(batch_loss)

        val_batch_losses = np.array(val_batch_losses)
        self.val_mean = np.mean(val_batch_losses)
        self.val_std = np.std(val_batch_losses)
        return self.val_std

class BatchLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, batch):
        super().__init__()
        self.validation_data = validation_data
        self.batch = batch

    def on_train_begin(self, logs=None):
        self.train_batch_losses = []  # list of lists
        self.val_batch_losses = []

    def on_epoch_begin(self, epoch, logs=None):
        # start a new list for this epoch
        self.current_epoch_losses = []

    def on_batch_end(self, batch, logs=None):
        # append the batch loss for this epoch
        self.current_epoch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        # store the full list of batch losses for this epoch
        self.train_batch_losses.append(self.current_epoch_losses)


if __name__ == "__main__":

    cryptodata = CryptoDataGetter()
    synth = SyntheticDriver(cryptodata)

    target, features, synth_target, synth_features = cryptodata.get_historical_data_trim(
    ["1 August 2024 00:00:00", 15000],
    "BTCUSDT",
    Client.KLINE_INTERVAL_5MINUTE,
    transform_func=synth.linear_RSI,
    transform_strength = 0.003)
    #cryptodata.plot_candlechart(200)

    epochs = 10

    """ ############################################ """
    target_train, target_val, features_train, features_val = cryptodata.split_train_val(target, features)

    x_train, y_train, x_val, y_val, scaler = cryptodata.slice_alltapes(lookb = 10, lookf = 5)

    simple_machine = CryptoMachine()
    simple_machine.init(candle = "1h", layer1 = 40, layer2 = 15, lookb = 10, lookf = 1, learn_rate = 0.001 , dropout = 0.1)
    trainmean1, train_std1, valmean1, val_stdfinal1 = simple_machine.fit(x_train, y_train, x_val, y_val, epochs = epochs, batch = 16)

    """ ############################################ """
    target_train, target_val, features_train, features_val = cryptodata.split_train_val(synth_target, synth_features)

    x_trains, y_trains, x_vals, y_vals, scaler = cryptodata.slice_alltapes(lookb = 10, lookf = 5)

    synth_machine = CryptoMachine()
    synth_machine.init(candle = "1h", layer1 = 40, layer2 = 15, lookb = 10, lookf = 1, learn_rate = 0.001 , dropout = 0.1)
    trainmean2, train_std2, valmean2, val_stdfinal2 = synth_machine.fit(x_trains, y_trains, x_vals, y_vals, epochs = epochs, batch = 16)

    plot = MachinePlotter(simple_machine, synth_machine, x_val, y_val, x_vals, y_vals)
    plot.plotmachines([trainmean1, trainmean2], [train_std1, train_std2], [valmean1, valmean2], [val_stdfinal1, val_stdfinal2])

    plot.plot_tape_eval(x_train, y_train)
    plot.plot_tape_eval(x_val, y_val)

    print(" Final errors of ")
    print(" Natural: {} +- {}".format(np.mean(valmean1), val_stdfinal1))
    print(" Synth:   {} +- {}".format(np.mean(valmean2), val_stdfinal2))
