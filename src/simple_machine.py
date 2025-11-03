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
from keras.models import Sequential, load_model
import pandas as pd
import scipy.signal as signal
from tensorflow.keras.layers import LSTM, Dropout, Dense, TimeDistributed
import pickle
from tensorflow.keras.optimizers import RMSprop, Adam
import copy
from tensorflow.keras.callbacks import Callback


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

        self.errorplots = None
        self.valplots = None
        self.mae = None
        self.modelpath = None
        self.scalerpath = None
        self.epochs = None
        self.batch = None
        self.error_batch_epoch = None

    def init(self, candle, layer1, layer2, lookb, lookf, learn_rate, dropout):
        self.candle, self.layer1, self.layer2, self.lookb, self.lookf, self.learn_rate, self.dropout = candle, layer1, layer2, lookb, lookf, learn_rate, dropout

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

    def load_machine(self, machinestrn, scalerstrn):

        self.model = load_model(os.path.join("../../machines", machinestrn))
        with open(os.path.join("../../scalers", scalerstrn), 'rb') as f:
            scalern = pickle.load(f)
            self.scaler = scalern
        self.lookf = 1

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch,
            save=False, candle=1, nowstr="NOW", plot_curves=True):
        self.epochs = epochs

        logger = BatchLossLogger(validation_data=(x_val, y_val), batch = batch)

        self.batch = batch
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch,
            validation_data=(x_val, y_val),
            callbacks=[logger]
        )

        self.error_batch_epoch = logger.epoch_batch_losses

        # Save training metrics
        self.errorplots = np.log(history.history['loss'])
        self.valplots = np.log(history.history['val_loss'])


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

    def plot(self, nowstr="NOW"):
        if self.errorplots is None or self.valplots is None:
            print("No training history to plot.")
            return

        plt.style.use('ggplot') #Change/Remove This If you Want

        print(np.std(self.error_batch_epoch, axis=1))
        print(len(self.error_batch_epoch))

        epochs = self.epochs-1
        train_err = self.errorplots[1:]
        val_err = self.valplots[1:]
        train_std = np.std(self.error_batch_epoch, axis=1)[1:] / 2  # your shaded region

        # Create 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

        # --- Training error plot ---
        axes[0].plot(np.arange(epochs), train_err, color='blue', label='Train Error', linewidth=1.0)
        axes[0].fill_between(np.arange(epochs),
                             train_err - train_std,
                             train_err + train_std,
                             color='blue', alpha=0.4)
        axes[0].set_title("Training Error")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Error")
        axes[0].legend(loc='best')

        # --- Validation error plot ---
        axes[1].plot(np.arange(epochs), val_err, color='red', label='Validation Error', linewidth=1.0)
        axes[1].set_title("Validation Error")
        axes[1].set_xlabel("Epochs")
        axes[1].legend(loc='best')

        plt.tight_layout()
        plt.show()

        print("Saved training curves")

class BatchLossLogger(Callback):
    def __init__(self, validation_data, batch):
        super().__init__()
        self.validation_data = validation_data
        self.batch = batch

    def on_train_begin(self, logs=None):
        self.epoch_batch_losses = []  # list of lists
        self.val_batch_losses = []

    def on_epoch_begin(self, epoch, logs=None):
        # start a new list for this epoch
        self.current_epoch_losses = []

    def on_batch_end(self, batch, logs=None):
        # append the batch loss for this epoch
        self.current_epoch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        # store the full list of batch losses for this epoch
        self.epoch_batch_losses.append(self.current_epoch_losses)

        # Run validation in batches
        X_val, y_val = self.validation_data
        val_losses = []

        for i in range(0, len(X_val), self.batch):
            X_batch = X_val[i:i+self.batch]
            y_batch = y_val[i:i+self.batch]
            val_loss = self.model.evaluate(X_batch, y_batch, verbose=0)
            val_losses.append(val_loss)

        self.val_batch_losses.append(val_losses)


if __name__ == "__main__":

    #target_train, target_val, features_train, features_val = load_example_train_val(32200, coin = "BTCUSDT", candle = Client.KLINE_INTERVAL_1HOUR)
    #x_train, y_train, scaler_fitted = slice_tapes(target_train, features_train, lookf, lookb, steps, stability_slope, None, onlyfirstpoints, indices, nowstr, trainorval = "train")

    #stability_cut = None#0.5 # 0.05 # where stab_cut * lookb (=100) is the maximal return observed in the train/val/test set

    cryptodata = CryptoDataGetter()
    cryptodata.get_historical_data_trim(["1 August 2024 00:00:00", 3200], "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE)
    #cryptodata.plot_candlechart(200)

    synth = SyntheticDriver(cryptodata.target_total, cryptodata.features_total)
    synth_target = synth.discrete_MA(1)

    x_train, y_train, x_val, y_val, scaler = cryptodata.slice_train_and_val(lookb = 10, lookf = 5, target = synth_target, features=cryptodata.features_total)

    simple_machine = CryptoMachine()
    simple_machine.init(candle = "1h", layer1 = 40, layer2 = 15, lookb = 10, lookf = 1, learn_rate = 0.09 , dropout = 0.0)
    simple_machine.fit(x_train, y_train, x_val, y_val, epochs = 30, batch = 16)
    simple_machine.plot()
