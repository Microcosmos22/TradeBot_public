import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from .calc_tools import *
from datetime import datetime, timedelta
import os, sys

class MachinePlotter:
    def __init__(self, model, synthmodel = None):
        self.blue_colors = [
            "lightblue", "skyblue", "deepskyblue", "dodgerblue",
            "cornflowerblue", "royalblue", "blue", "mediumblue", "navy"
        ]
        self.model = model
        self.synthmodel = synthmodel
        self.red_colors = [
            "lightcoral", "salmon", "darksalmon", "tomato",
            "red", "firebrick", "darkred", "indianred", "crimson"
        ]
        return

    def plot_tape_eval(self, x, y):
        idx = np.random.choice(np.arange(len(x)))

        y_pred = self.model.model.predict(x[idx].reshape(1, 10, 13), verbose=0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        x_target = x[idx,:,0]

        # Top-left
        axes[0].plot(np.arange(10), x_target, color='blue')
        axes[0].set_title(" Past 10 Returns [%] ")

        # Top-right
        axes[1].scatter(1, y_pred, label="y_pred", color='cyan')
        axes[1].scatter(1, y[idx], label="y_true", color='blue')
        axes[1].legend()
        axes[1].set_title(" Mean of Next 5 Returns [%]")

        #axes[1, 1].plot(np.arange(len(self.x_val[idx,:,:])), self.x_val[idx,:,:], color='blue')
        #axes[1, 1].set_title("Features. X")

        # Adjust layout and show
        plt.tight_layout()
        plt.show()

    def plotmachines(self, train_mean, train_std, val_mean, val_std):

        plt.style.use('ggplot') #Change/Remove This If you Want

        epochs = len(train_mean[0])

        # Create 1 row, 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(14, 5), sharex=True, sharey=False)

        """ --- Training error plot --- """

        axes[0,0].plot(np.arange(epochs), train_mean[0], color='blue', label='Train Error', linewidth=1.0)
        axes[0,0].fill_between(np.arange(epochs),
                             train_mean[0] - train_std[0],
                             train_mean[0] + train_std[0],
                             color='blue', alpha=0.4)
        axes[0,0].errorbar(x=[epochs - 1], y=[train_mean[0][-1]], yerr=[train_std[0][-1]],
            fmt='o', color='blue', ecolor='blue',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ')

        axes[1,0].plot(np.arange(epochs), train_mean[1], color='cyan', label='Synth Train Error', linewidth=1.0)
        axes[1,0].fill_between(np.arange(epochs),
                             train_mean[1] - train_std[1],
                             train_mean[1] + train_std[1],
                             color='cyan', alpha=0.4)
        axes[1,0].errorbar(x=[epochs - 1], y=[train_mean[1][-1]], yerr=[train_std[1][-1]],
            fmt='o', color='cyan', ecolor='cyan',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ')

        axes[0,0].set_title("Training Error")
        axes[1,0].set_title("Training Error")
        axes[1,0].set_xlabel("Epochs")
        axes[0,0].set_ylabel("Error")
        axes[0,0].legend(loc='best')
        axes[1,0].legend(loc='best')


        """ --- Validation error plot --- """
        axes[0,1].plot(np.arange(epochs), val_mean[0], color='red', label='Validation Error', linewidth=1.0)

        axes[1,1].plot(np.arange(epochs), val_mean[1], color='orange', label='Synth Validation Error', linewidth=1.0)

        axes[0,1].set_title("Validation Error")
        axes[1,1].set_title("Validation Error")
        axes[1,1].set_xlabel("Epochs")
        axes[0,1].legend(loc='best')
        axes[1,1].legend(loc='best')

        plt.tight_layout()
        plt.show()

        print("Saved training curves")

    def plotmachine(self, train_mean, train_std, val_mean, val_std):

        plt.style.use('ggplot') #Change/Remove This If you Want

        epochs = len(train_mean)

        # Create 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

        """ --- Training error plot --- """

        axes[0].plot(np.arange(epochs), train_mean, color='blue', label='Train Error', linewidth=1.0)
        axes[0].fill_between(np.arange(epochs),
                             np.subtract(train_mean,train_std),
                             np.add(train_mean,train_std),
                             color='blue', alpha=0.4)
        axes[0].errorbar(x=[epochs - 1], y=[train_mean[-1]], yerr=[train_std[-1]],
            fmt='o', color='blue', ecolor='blue',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ')


        axes[0].set_title("Training Error")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Error")
        axes[0].legend(loc='best')


        """ --- Validation error plot --- """
        axes[1].plot(np.arange(epochs), val_mean, color='orange', label='Synth Validation Error', linewidth=1.0)

        axes[1].set_title("Validation Error")
        axes[1].set_xlabel("Epochs")
        axes[1].legend(loc='best')

        plt.tight_layout()
        plt.show()

        print("Saved training curves")

    def plot_tape(x):

        plt.plot(x[:,0], label="Returns")
        plt.plot(x[:,1], label="SMA_20")
        plt.plot(x[:,2], label="SMA_50")
        plt.plot(x[:,3], label="RSI")
        plt.plot(x[:,4], label="BB_w")
        plt.plot(x[:,5], label="momentum")
        plt.plot(x[:,6], label="vol")
        plt.plot(x[:,7], label="k")
        plt.plot(x[:,8], label="d")
        plt.plot(x[:,9], label="macddiff")
        plt.plot(x[:,10], label="dmonth")
        plt.plot(x[:,11], label="dweek")
        plt.plot(x[:,12], label="hday")
        plt.legend()
        plt.title(" Input X (normalized)")
        plt.show()

def plot_scaling_stacked(stacked, stacked_n):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=False)

    """ --- Training error plot --- """

    col = ["ret", "sma20", "sma50", "RSI", "BBwidth", "mom", "vol", "K", "D", "MACD", "d_month", "d_week", "h_day"]

    for i in range(len(stacked[0])): # along features
        axes[0].plot(np.arange(len(stacked[:,0])), stacked[:,i], label=col[i])
        axes[1].plot(np.arange(len(stacked_n[:,0])), stacked_n[:,i], label=col[i])

    plt.tight_layout()
    plt.legend()
    plt.show()
    return col


def plot_test_train_predictions_grid(trues_test, preds_test, tres_train, preds_train, identifier):
    # Create a 4x5 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Adjust figsize as needed

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Loop over your data and plot it on each subplot
    for i in range(2):  # 4x5 = 20 subplots
        axes[i].plot(tres_train[i], label="true")  # Example plot, replace with your data
        axes[i].plot(preds_train[i], label="pred")
        axes[i].set_title(f"Train")  # Set title for each subplot
        axes[i].legend()

    for i in range(2,4):  # 4x5 = 20 subplots
        axes[i].plot(trues_test[i], label="true")  # Example plot, replace with your data
        axes[i].plot(preds_test[i], label="pred")
        axes[i].set_title(f"Test")  # Set title for each subplot
        axes[i].legend()

    # Adjust layout for better spacing between plots
    plt.tight_layout()
    plt.savefig("../traincurves/predictions"+str(identifier)+".png")

    # Show the plots
    plt.show()

def plot_test_predictions_grid(trues, preds):
    # Create a 4x5 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Adjust figsize as needed

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Loop over your data and plot it on each subplot
    for i in range(4):  # 4x5 = 20 subplots
        axes[i].plot(trues[i], label="true")  # Example plot, replace with your data
        axes[i].plot(preds[i], label="pred")
        axes[i].set_title(f"Plot {i + 1}")  # Set title for each subplot
        axes[i].legend()

    # Adjust layout for better spacing between plots
    plt.tight_layout()
    plt.savefig("constant_predictions.png")

    # Show the plots
    plt.show()

def plot_after_split(y_test, par_test):

    plt.title("After split")
    plt.plot(y_test, label="true price")
    plt.plot(par_test[:,0], label="SMA_20")
    plt.plot(par_test[:,1], label="SMA_50")
    plt.plot(par_test[:,2], label="RSI")
    plt.plot(par_test[:,3], label="BB_w")
    plt.plot(par_test[:,4], label="momentum")
    plt.legend()
    plt.show()
    return None

def plot_correlation(a, b, title):

    lags = range(1, 41)
    acf = [np.corrcoef(a[:-lag], b[lag:])[0, 1] for lag in lags]

    # --- Original series ---
    plt.stem(lags, acf)
    plt.title(title)
    plt.ylabel("Correlation")

    plt.tight_layout()
    plt.show()

def plot_returns_histo(target, synth_target):
    plt.hist(target, 50, alpha=0.5, label='Original returns')
    plt.hist(synth_target, 50, alpha=0.5, label='Synth returns')
    plt.legend(loc='upper right')
    plt.title(" Prices histogram ")
    plt.xlabel(" Prices [BTCUSDT]")
    plt.show()

    returns = calc_returns(target)
    synth_returns = calc_returns(synth_target)

    plt.hist(returns, 50, alpha=0.5, label='Original returns')
    plt.hist(synth_returns, 50, alpha=0.5, label='Synth returns')
    plt.xlabel("Returns [%]")
    plt.title(" Returns histogram")
    plt.legend(loc='upper right')
    plt.show()

def plot_future_results(y_pred, y_test, date_test):
    # Plot results

    plt.figure(figsize=(12, 6))
    plt.plot(date_test, y_test, label='True Prices', color='blue')
    plt.plot(date_test, y_pred, label='Predicted Prices', color='red', linestyle='dashed')
    plt.title("True vs Predicted Closing Prices")
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show()

    # Plot buy/sell signals
    """plt.figure(figsize=(12, 6))
    plt.plot([i for i in range(len(y_test))], y_test, label='True Prices', color='blue')

    plt.scatter(np.arange(len(buy_signal))[buy_signal == 1], y_test[buy_signal == 1], color='green', label='Buy Signal')

    plt.scatter(np.arange(len(sell_signal))[sell_signal == -1], y_test[sell_signal == -1], color='red', label='Sell Signal')

    plt.title("Buy/Sell Signals on True Prices")
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show()
    """


if __name__ == "__main__":
    print("runnin")
