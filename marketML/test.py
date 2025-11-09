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
from .simple_machine import *
from .get_binance import * 

def print_machine(model):
    # Print model summary
    model.summary()

    weights = model.get_weights()

    # Loop through all the layers' weights
    for i, weight in enumerate(weights):
        print(f"Weight matrix {i} shape: {weight.shape}")
        print(weight)  # Print the weight matrix itself
        print("="*50)  # Separator between layers

    # Access and print weights for the first LSTM layer
    print("LSTM Layer 1 Weights:")
    lstm_layer = model.layers[0]
    print("Kernel weights shape:", lstm_layer.get_weights()[0].shape)
    print("Recurrent weights shape:", lstm_layer.get_weights()[1].shape)
    print("Biases shape:", lstm_layer.get_weights()[2].shape)

    # Iterate over all layers and print their trainable parameters
    for layer in model.layers:
        print(f"{layer.name} has {layer.count_params()} parameters")
        if hasattr(layer, 'trainable_weights'):
            print(f"  Trainable weights: {len(layer.trainable_weights)}")
            for i, weight in enumerate(layer.trainable_weights):
                print(f"    Weight {i} shape: {weight.shape}")


def variable_outcome(pred):
    sq_slope_diff = []

    for i in range(len(pred)-1):

        sq_slope_diff.append(np.sum(np.power((np.diff(pred[i])-np.diff(pred[i+1])), 2)))

    mean_squaresum_slope_diff = np.mean(sq_slope_diff)

    return mean_squaresum_slope_diff

def concatenate_recompute_features(x, value_to_concatenate, scaler):
    """ out of an x and the predicted norm return, computes the x extended by 1 """
    x_old = x.copy()

    # 1. Concatenate the returns
    extended_returns_n = pd.Series(np.concatenate((x.reshape(-1,5,1)[:,0],value_to_concatenate.reshape(1,1)))[:,0])

    # 2. Un-normalize the returns
    stacked_n = np.full((extended_returns_n.shape[0], 5), np.nan)
    stacked_n[:,0] = extended_returns_n.to_numpy()
    stacked = scaler.inverse_transform(stacked_n)
    extended_returns = pd.Series(stacked[:,0])

    # 3. Compute the features
    stacked[:,1] = extended_returns.rolling(window=20).mean()
    stacked[:,2] = calculate_rsi1(extended_returns, period=14)
    stacked[:,3] = calculate_bb_width1(extended_returns, window=20)
    stacked[:,4] = calculate_momentum1(extended_returns, period=10)

    # 4. Normalize again! Then paste to patch the NaN values (first values of x unchanged)
    stacked_n = scaler.transform(stacked)
    stacked[:23,1] = x_old[:23,1].reshape(-1)
    stacked[:15,2] = x_old[:15,2].reshape(-1)
    stacked[:21,3] = x_old[:21,3].reshape(-1)
    stacked[:11,4] = x_old[:11,4].reshape(-1)

    return stacked

def projected_prediction(x_testi, scaler, model, projsteps = 5, verbose = False):
    """ Given a x, predicts the upcoming values and returns an extended x """
    x_testi = np.asarray(x_testi, dtype=np.float32)
    x_pred = []

    for i in range(projsteps):

        #print("predict from: {}".format(x_testi[-lookb:].reshape(1, lookb, 5)))
        pred_n_return = model.predict(x_testi[-lookb:].reshape(1, lookb, 5), verbose=0)

        #print("single prediction {}".format(pred_n_return))

        x_pred.append(pred_n_return)
        x_testi = concatenate_recompute_features(x_testi,  pred_n_return.reshape(1,1,1), scaler)

    return np.asarray(x_testi).reshape(-1,5)

def sample_predictions(target, features, scaler, model, onlyfirstpoints, sequential = True):
    target_test = target
    features_test = features

    returns_test = calc_returns(target_test) # true returns
    stacked_test = np.hstack((returns_test.reshape(-1,1), features_test[1:]))[:onlyfirstpoints]
    stacked_test_n = scaler.transform(stacked_test.reshape(-1,5))
    stacked_test_n = stacked_test_n.reshape(-1,5,1)
    x_true_list, predictions_list = [], []

    """ Loop through test samples """
    for i in range(0, onlyfirstpoints-lookb-lookf, steps):
        if (len(stacked_test_n) < (i+lookf+lookb)):
            break
        """ x_test is a single tape """
        x_test = stacked_test_n[i:i+lookb,:] # input (norm returns)
        y_test_n = stacked_test_n[i+lookb:i+lookf+lookb,0] # y : true returns
        stacked_n = np.full((lookf, 5), np.nan)

        if(x_test.shape[0] == lookb):
            if sequential == False:
                x_ext = projected_prediction(x_test, scaler, model, verbose = False)
                return_pred = scaler.inverse_transform(x_ext) # prediction is returns
            else:
                return_pred_n = model.predict(x_test.reshape(1, lookb, 5), verbose=0)
                stacked_n[:,0] = return_pred_n.reshape(-1)
                return_pred = scaler.inverse_transform(stacked_n) # prediction is returns

            # Un-scale the true values
            stacked_n[:,0] = y_test_n.reshape(-1)
            y_test = scaler.inverse_transform(stacked_n) # prediction is returns
            # Save true and predicted values, whether seq. generated or projected

            pred_list.append(return_pred.reshape(-lookf,5)[:,0])
            x_true_list.append(y_test.reshape(-lookf,5)[:,0])


    print("Sample predictions: true {} and pred {}".format(np.asarray(x_true_list).shape, np.asarray(pred_list).shape))
    return x_true_list, pred_list


if __name__ == "__main__":
    if len(sys.argv) == 1: # if there is no argument
        modelpath = "../machines/1H1KTAPES_machine_2024-12-30_20-38-11_f1_b10_s5_sbsNone_l0.009_b16_one20_two15_only5001_dr0.0_candle1h.h5"
        scalerpath = "../scalers/1K1KTAPES_scaler_2024-12-30_20-38-11_f1_b10_s5_sbsNone_l0.009_b16_one20_two15_only5001_dr0.0_candle1h.pkl"

    else:
        modelpath = sys.argv[1]
        scalerpath = sys.argv[2]

    lookb = 10
    lookf = 10
    steps = 5
    sequential = True

    model = load_model(modelpath)
    with open(scalerpath, 'rb') as f:
        scaler = pickle.load(f)


    """ ###### """

    start_of_training = datetime.strptime("2 December 2024 00:00:00", "%d %B %Y %H:%M:%S")
    end_of_training = datetime.strptime("28 December 2024 00:00:00", "%d %B %Y %H:%M:%S")

    target_test, features_test, _ = get_historical_data_trim(1500, Client.KLINE_INTERVAL_1HOUR)
    pred_list = []
    # FOR PURPOSE OF OVERFITTING
    target_train, target_val, features_train, features_val = load_train_val(Client.KLINE_INTERVAL_1HOUR)
    print(target_train.shape, features_train.shape)

    print("Train")
    #x_true_list_train, pred_list_train = sample_predictions(target_train, features_train, scaler, model, onlyfirstpoints, sequential)
    print("Test")
    x_true_list_test, pred_list_test = sample_predictions(target_test, features_test, scaler, model, len(target_test), sequential)


    plot_test_train_predictions_grid(x_true_list_test, pred_list_test, x_true_list_test, pred_list_test, modelpath[12:])


    mean_squaresum_slope_diff = variable_outcome(pred_list)
    print("\n \n Mean squaresum of slope differences (Variability): {}".format(mean_squaresum_slope_diff))
