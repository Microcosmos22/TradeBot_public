import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from binance.client import Client
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys

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