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
from simple_machine import *
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.layers import LSTM
import logging
import copy
import os

api_key = "pxRONzQcbpDoImQXzHqkO6XJWd7WMIKSTyBPtTlkvaCbIGJ0Whcnz8LDw7SavMIx"
api_secret = "hFXzByh1Fg90Vcxvx8uakDq9n6reH32KswXuYOTzFxxjmAVvMbHRh1lOvMSgHlex"
# Configure logging
logging.basicConfig(
    filename="trading_bot.log",  # Log file name
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(message)s",  # Log format: timestamp and message
)

def get_last_n_full_hours(N):
    # The target date: 1st January 2025, 00:00:00
    target_date = datetime(2025, 1, 1, 0, 0, 0)

    # List to store the last N full hours
    last_n_hours = []

    # Loop to get the last N full hours before the target date
    for i in range(N):
        # Subtract full hours (i+1) from the target date
        full_hour = target_date - timedelta(hours=i+1)

        # Add the formatted hour to the list
        last_n_hours.append(full_hour)

    return last_n_hours

def live_trading1(model, scaler):
    # This method will be activated once the best performing machine has been determined

    target, features, dates = get_historical_data_trim(61)

    print("First candle is {} UTC".format(datetime.utcfromtimestamp(int(dates[-1])/1000).strftime('%Y-%m-%d %H:%M:%S')))
    firstdate = dates[-1]/1000
    price_obs = target[-1]

    kasse = 10000 # Euros
    btc_kasse = 10000/target[lookb+1]

    j = 0
    i = True
    last_candle_date = dates[-1]/1000
    bought = None


    while i:
        # Every 15 minutes since first candle or inmediatly if first
        print("############################################")
        # Get the current Bitcoin price (BTC/USDT pair)
        candle_date = dates[-1]/1000
        lasttime = datetime.utcfromtimestamp(candle_date).strftime('%Y-%m-%d %H:%M:%S')
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")


        # Grabs the last tape and makes a prediction for the next candle
        target, features, timestamps = get_historical_data_trim(61)
        stacked_pred = raws_predict(model, scaler, target, features)
        price_pred = price_obs*(1+stacked_pred[-1,0])
        forw_diff = price_pred-price_obs

        #print(price_pred, price_obs, stacked_pred)
        _, _, bought, out_of_equil = equil_trader(forw_diff, kasse, btc_kasse, price_obs, equil_const = 1, forward_const = 10)

        """ New candle got released, perform a trade """
        if candle_date != last_candle_date:
            j += 1
            last_candle_date = candle_date
            kasse, btc_kasse, bought, out_of_equil = equil_trader(forw_diff, kasse, btc_kasse, price_obs, equil_const = 1, forward_const = 10)
            logging.info(f"Kasse: {kasse}, BTC Kasse: {btc_kasse}, Bet: {bought}, Price Pred: {price_pred}, Price Obs: {price_obs}")
        printstep(target[-1], forw_diff, bought, lasttime, kasse, btc_kasse)


    return None


def load_machines_scalers(onlyfirst = None):
    """ Loads all machines that have a corresponding scaler.
    It is searched by substituting machine->'scaler'. We also read the lookf from the filename"""
    machinepaths = os.listdir("../machines")
    scalerpaths = os.listdir("../scalers")

    machines = [mpath for mpath in machinepaths if os.path.isfile(os.path.join("../machines", mpath))]
    scalers = [spath for spath in scalerpaths if os.path.isfile(os.path.join("../scalers", spath))]

    print("Found {} machines and {} scalers".format(len(machines), len(scalers)))

    modeln = []
    scalern = []
    machinestr  = []
    lookfn = []

    matches = 0

    # Load all machines and scalers
    for i in range(len(machines)):
        if os.path.isfile(os.path.join("../machines", machines[i])):
            if "machine" in machines[i]:
                scalername = machines[i].replace("machine", "scaler")

            if os.path.isfile(os.path.normpath(os.path.join("..\scalers", scalername[:-2]+"pkl"))): # scaler exists
                matches += 1
                print("Loading {}".format(scalername[:-2]+"pkl"))

                model = load_model(os.path.join("../machines", machines[i]))
                with open(os.path.join("../scalers", scalername[:-2]+"pkl"), 'rb') as f:
                    scaleri = pickle.load(f)

                if machines[i][machines[i].find("_f")+3] == "_": # only one character between f and _
                    lookf = 1 # int(machines[i][machines[i].find("_f")+2])
                else:
                    lookf = 10 #machines[i][machines[i].find("_f")+2]

                lookfn.append(lookf)
                modeln.append(model)
                scalern.append(scaleri)
                machinestr.append(machines[i])

                if matches == onlyfirst:
                    break

    print("Found {} pairs of machine & scaler".format(matches))
    return modeln, machinestr, scalern, lookfn

def printstep(btc_price, forw_diff, bought, kasse, btc_kasse, out_of_equil, forw_dependency):
    #print()
    #print(f"Current Bitcoin price: {btc_price} USDT")
    print()
    print("Forward price difference: {:.2f}".format(forw_diff))
    print()
    print("Bet0: {:.4f} $ equil {:.2f}  -> {:.2f}".format(forw_dependency, out_of_equil, bought))
    #if lasttime is not None:
    #    print("Last candle time {}".format(lasttime))
    print(" Kontostand: {:.2f} $, {:.4f} BTC".format(kasse, btc_kasse))
    print(" Portfolio (bef trade): {:.2f}".format(kasse+btc_kasse*btc_price))
    #print()

def equil_trader(forw_diff, kasse, btc_kasse, btc_price, equil_const = 100, forward_const = 50):
    portfolio = kasse+btc_kasse*btc_price
    # (a-b) / tot is [+1, -1]
    equil = (kasse-btc_kasse*btc_price)/portfolio # 0 when more $, +2 when more BTC
    sign = np.sign(equil)


    if equil > 0: # More Kasse
        out_of_equil = (equil)*equil_const+1 # gives [1, infty]
    else: # More BTC
        out_of_equil = np.absolute(equil*equil_const)+1   # gives [1,infty]
    # Is now always in interval [1, infty]


    # Introduce assymetry to ensure the kasse doesnt run out
    if forw_diff > 0:
        if sign > 0:# More Kasse
            bought0 = forw_diff*out_of_equil
        else:
            bought0 = forw_diff/out_of_equil
    elif forw_diff < 0:
        if sign > 0:# More Kasse
            bought0 = forw_diff/out_of_equil
        else:
            bought0 = forw_diff*out_of_equil

    #bought = bought0 # np.sign(bought0)*np.sqrt(np.absolute(bought0))*forward_const

    bought = np.sign(bought0)*np.sqrt(np.absolute(bought0))


    if ((kasse - bought > 0) and (btc_kasse + bought/float(btc_price)) > 0):
        kasse = kasse - bought
        btc_kasse = btc_kasse + bought/float(btc_price)

    #print("____")
    #print(out_of_equil)
    #print()
    #print(forw_diff)
    #print(bought0)
    #print(bought)

    """
    if ((bought > 0) and kasse-bought > 0 ):
        kasse -= bought
        btc_kasse += bought/float(btc_price)
    elif ((bought < 0) and btc_kasse-bought/btc_price > 0 ):
        kasse -= bought
        btc_kasse += bought/float(btc_price)
    """
    return kasse, btc_kasse, bought, equil, bought0

def simulate_rendite(simple_machine, cryptodata = None, verboseeach = 250, onlyfirstmachines=None, equil_const = 2, subsets = 200):
    """ Runs through hourly candles, trades using each machine and computes obtained returns """
    # Careful, amount of tapes is less than target bec. they span an interval (lookb+lookf).
    # Also the return computation reduces amount by -1.
    # First tape (x) spans [0:lookb+1] and the predicted return applies on target[lookb]

    if cryptodata is None:
        cryptodata = CryptoDataGetter()
        _, _, target_full, features_full = cryptodata.get_historical_data_trim(
        ["1 August 2025 00:00:00", 3000], "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE,
        transform_func=synth.linear_RSI, transform_strength = 0.008)
        x_train, y_train, scaler = self.slice_tapes_normalize(target_full, features_full, 10, 5)
        simple_machine.set_scaler(scaler)


    #print("Last candle is")
    #print(datetime.utcfromtimestamp(dates[-1]/1000).strftime('%Y-%m-%d %H:%M:%S'))
    """ Slicing to get the tapes """
    targetn, featuresn = [],[]
    for i in range(len(target_full)-20):
        targetn.append(target_full[i:i+11])
        featuresn.append(features_full[i:i+11])
    sim_hours = len(target_full)-20

    start_price = target_full[10]
    btc_price_now = target_full[10]

    print(" Starting price: {}".format(target_full[10]))

    kasse = 10000 # Euros
    btc_kasse = 10000/float(start_price)
    portfolio = kasse+btc_kasse*float(start_price) # Euros
    port_track = []

    old_price_pred = target_full[10]
    bought, forw_diff = 0, 0

    bets = []
    price_error = []
    forw_diffs = []

    for i in range(len(targetn)): # Loop through the tapes and simulate trading
        old_btc_price = copy.copy(btc_price_now)

        # grab the ith-slice [0:11] which has indexes 0, 1, ..., 10
        target = targetn[i]
        features = featuresn[i]

        #candle_date = dates[i+10]/1000
        #time = datetime.utcfromtimestamp(candle_date).strftime('%Y-%m-%d %H:%M:%S')

        btc_price_now = target[-1]

        # Bilanz ziehen
        portfolio = kasse+btc_kasse*btc_price_now

        stacked_pred = simple_machine.raws_predict(target, features)
        price_pred = btc_price_now*(1+stacked_pred[-1,0])
        forw_diff = price_pred-btc_price_now

        # Trade (Bet)
        kasse, btc_kasse, bought, out_of_equil, bought0 = equil_trader(forw_diff, kasse, btc_kasse, btc_price_now, equil_const, forward_const = 10)

        if i%verboseeach==0:
            print("############################################")
            print("step {}, machine".format(i))
            printstep(btc_price_now, forw_diff, bought, kasse, btc_kasse, out_of_equil, bought0)
            #print("Now price {} Last price obs (candle): {}".format(btc_price_now,target[i+lookb]))

        bets.append(bought)
        price_error.append(old_price_pred-btc_price_now)
        forw_diffs.append(forw_diff)

        btc_gewinn = (10000*btc_price_now/start_price-10000)
        port_track.append((portfolio-20000)-btc_gewinn)

        old_price_pred = copy.copy(price_pred)


    start_price = target_full[10]
    end_price = target_full[-1]

    # Bilanz ziehen, wertänderung nach trade und nach Kursänderung
    portfolio = kasse+btc_kasse*end_price

    returns = 0.0
    btc_gewinn = (10000*end_price/start_price-10000)
    trade_returns = 0.0

    print(" @@@@@  Anfangszustand   @@@@  ")
    print(" Kasse: 1000 $, BTC-Kasse: {}".format(10000/float(start_price)))

    print()
    print(machinestrn)
    print(" Kassen: {} $ {} BTC ".format(kasse, btc_kasse))
    #print(" Portfolio: {}".format(portfolio))
    print()
    print(" Haben einen Gewinn von {:.2f} $ oder {:.2f} $/h ".format((portfolio-20000),(portfolio-20000)/sim_hours))
    print(" Purer BTC Gewinn {:.2f} $ oder {:.2f} $/h  (Ursprüngliche BTCs)".format(btc_gewinn, btc_gewinn/sim_hours))
    #print(" durch den {:.2f} % Anstieg von {:.2f} auf {:.2f} ".format((end_price-start_price)/start_price , start_price, end_price))
    print(" ___________________________________________________" )
    print(" Korrigierter Trade-Gewinn {:.6f} $/h ".format(((portfolio-20000)/sim_hours)-(btc_gewinn/sim_hours)))
    print()

    #print(" Wetten: {:.2f} +- {:.2f}".format(np.mean(bets), np.std(bets)))
    #print(" Price prediction error: {:.2f} +- {:.2f}".format(np.mean(price_error), np.std(price_error)))
    print(" Delta_price (forward): {:.2f} +- {:.2f}".format(np.mean(forw_diffs), np.std(forw_diffs)))

    #print(" Directions accuracy: {:.4f} +- {:.4f}".format(np.mean(np.asarray(correct_direction)), np.std(np.asarray(correct_direction))))
    print(" BTC Preisanstieg von {} auf {} ".format(start_price, end_price))

    print(" Haben einen Gewinn von {:.2f} $, korrigiert {:.2f} $, Treffer {:.2f} %, Avg bet {:.2f} $  ".format((portfolio-20000),(portfolio-20000)-btc_gewinn, np.mean(np.asarray(correct_direction))*100, np.mean(bets)))

    return returns


if __name__ == "__main__":

    #client = Client(api_key, api_secret)
    #lookb = 10

    #live_trading1(model, scaler)
    """ Subsets must subdivide sim_N"""
    simulate_rendite("1H1KTAPES_machine_2025-10-09_10-57-01_f1_b10_s2_sbs0.1_l0.09_b160_one40_two20_only8000_dr0.1_candle1h.h5", "1K1KTAPES_scaler_2025-10-09_10-57-01_f1_b10_s2_sbs0.1_l0.09_b160_one40_two20_only8000_dr0.1_candle1h.pkl", timedef = ["sim", 3600], verboseeach=10, onlyfirstmachines=None, equil_const=50000, subsets=200)
