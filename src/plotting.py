import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys

class MachinePlotter:
    def __init__(self, model, synthmodel, x_val, y_val, x_vals, y_vals):
        self.blue_colors = [
            "lightblue", "skyblue", "deepskyblue", "dodgerblue",
            "cornflowerblue", "royalblue", "blue", "mediumblue", "navy"
        ]
        self.model = model
        self.synthmodel = synthmodel
        self.x_val, self.y_val, self.x_vals, self.y_vals = x_val, y_val, x_vals, y_vals

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
        axes[0].set_title("Target. X")

        # Top-right
        axes[1].scatter(1, y_pred, label="y_pred", color='cyan')
        axes[1].scatter(1, y[idx], label="y_true", color='blue')
        axes[1].legend()
        axes[1].set_title("Target. Y")

        #axes[1, 1].plot(np.arange(len(self.x_val[idx,:,:])), self.x_val[idx,:,:], color='blue')
        #axes[1, 1].set_title("Features. X")


        # Adjust layout and show
        plt.tight_layout()
        plt.show()

    def plotmachines(self, train_mean, train_std, val_mean, val_std):

        plt.style.use('ggplot') #Change/Remove This If you Want

        epochs = len(train_mean[0])

        # Create 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

        """ --- Training error plot --- """

        axes[0].plot(np.arange(epochs), train_mean[0], color='blue', label='Train Error', linewidth=1.0)
        axes[0].fill_between(np.arange(epochs),
                             train_mean[0] - train_std[0],
                             train_mean[0] + train_std[0],
                             color='blue', alpha=0.4)
        axes[0].errorbar(x=[epochs - 1], y=[train_mean[0][-1]], yerr=[train_std[0][-1]],
            fmt='o', color='blue', ecolor='blue',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ')

        axes[0].plot(np.arange(epochs), train_mean[1], color='cyan', label='Synth Train Error', linewidth=1.0)
        axes[0].fill_between(np.arange(epochs),
                             train_mean[1] - train_std[1],
                             train_mean[1] + train_std[1],
                             color='cyan', alpha=0.4)
        axes[0].errorbar(x=[epochs - 1], y=[train_mean[1][-1]], yerr=[train_std[1][-1]],
            fmt='o', color='cyan', ecolor='cyan',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ')

        axes[0].set_title("Training Error")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Error")
        axes[0].legend(loc='best')


        """ --- Validation error plot --- """
        axes[1].plot(np.arange(epochs), val_mean[0], color='red', label='Validation Error', linewidth=1.0)
        axes[1].errorbar(x=[epochs - 1], y=[val_mean[0][-1]], yerr=[val_std[0]],
            fmt='o', color='red', ecolor='red',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ'
        )

        axes[1].plot(np.arange(epochs), val_mean[1], color='orange', label='Synth Validation Error', linewidth=1.0)
        axes[1].errorbar(x=[epochs - 1], y=[val_mean[1][-1]], yerr=[val_std[1]],
            fmt='o', color='orange', ecolor='orange',           # color of error bar
            elinewidth=1.5, capsize=4, label='Final ±1σ'
        )
        axes[1].set_title("Validation Error")
        axes[1].set_xlabel("Epochs")
        axes[1].legend(loc='best')

        plt.tight_layout()
        plt.show()

        print("Saved training curves")

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

def plot_historical_data(target, features):

    plt.plot(target)
    plt.plot(features[:,0], label="SMA_20")
    plt.plot(features[:,1], label="SMA_50")
    plt.plot(features[:,2], label="RSI")
    plt.plot(features[:,3], label="BB_w")
    plt.plot(features[:,4], label="momentum")
    plt.plot(features[:,5], label="vol")
    plt.plot(features[:,6], label="k")
    plt.plot(features[:,7], label="d")
    plt.plot(features[:,8], label="macddiff")
    plt.plot(features[:,9], label="dmonth")
    plt.plot(features[:,10], label="dweek")
    plt.plot(features[:,11], label="hour")
    plt.legend()
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

def plot_returns_histo(target, synth_target):
    pyplot.hist(target, 50, alpha=0.5, label='Original returns')
    pyplot.hist(synth_target, 50, alpha=0.5, label='Synth returns')
    pyplot.legend(loc='upper right')
    pyplot.show()


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
