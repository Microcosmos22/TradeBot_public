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


def k_driver(target0, features, traderpower = 50):
    k = features[:,6]
    target = copy.copy(target0)
    
    for i in range(len(target)):
        pricedrive = (50-k[i])*traderpower
        target[i] = target[i] + pricedrive
        
    return target
    
def load_train_val(ncandles=32000, candle=Client.KLINE_INTERVAL_1HOUR):
    #start_of_training = datetime.strptime("1 September 2020 00:00:00", "%d %B %Y %H:%M:%S")
    end_of_training = datetime.strptime("1 June 2023 00:00:00", "%d %B %Y %H:%M:%S")
    target_train, features_train, _, _ = get_historical_data_trim([end_of_training, ncandles], candle_length = candle, normalize=False, plot=False)
    target_train = k_driver(target_train, features_train)
    np.save("../target_train_1h.npy", target_train)
    np.save("../features_train_1h.npy", features_train)
    
    
    #target_train = np.load("../target_train_1h.npy")
    #features_train = np.load("../features_train_1h.npy")
    
    #start_of_val = datetime.strptime("2 January 2024 00:00:00", "%d %B %Y %H:%M:%S")
    end_of_val = datetime.strptime("1 August 2024 00:00:00", "%d %B %Y %H:%M:%S")
    
    target_val, features_val, _, _ = get_historical_data_trim([end_of_val, int(ncandles/4)], candle_length = candle, normalize=False, plot=False)
    target_val = k_driver(target_val, features_val)
    np.save("../target_val_1h.npy", target_val)
    np.save("../features_val_1h.npy", features_val)
    
    #target_val = np.load("../target_val_1h.npy")
    #features_val = np.load("../features_val_1h.npy")
    
    
    return target_train, target_val, features_train, features_val
    

def deleteplots(file1, file2, file3):
    # Construct file paths
    modelpath = "some/path/to/model"
   
    # List of files to delete
    files_to_delete = [file1, file2, file3]

    # Check if each file exists and delete it
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File {file_path[45:]} removed successfully.")
            except Exception as e:
                print(f"Error removing file {file_path[45:]}: {e}")
                
                


if __name__ == "__main__":
    
    
    candl = [Client.KLINE_INTERVAL_1HOUR]
    
    for candle in candl:
        
        target_train, target_val, features_train, features_val = load_train_val(32200, candle)
            
        indices = 0
        # Will divide data into subsets and train a machine on each subset
        while ((indices == 0) or (indices+1)*onlyfirstpoints < target_train.shape[0]):
            print()
            print(indices)
            print()
            
            errorplots, valplots, mae, val_mae, mse, val_mse = [], [], [], [], [], []
            machinename = []
            traintime = 100
            #stability_cut = None#0.5 # 0.05 # where stab_cut * lookb (=100) is the maximal return observed in the train/val/test set
            
            
            
            identifier = "LLSTM"
            # learn rate, batch, layer1, layer2, dropout, lookf, lookb, steps, SBS, onlyfirstpoints, candle
            # params =  np.asarray([[0.01, 16, 75, 30, 0.0, 5, 100, 5], [0.01, 16, 20, 15, 0.0, 5, 100, 5]])
                #j = p[10]
            #    Client.KLINE_INTERVAL_1HOUR

            
            
            p = np.asarray([0.09, 160, 40, 20, 0.1, 1, 10, 2, 0.10, 8000, candle])
            
            trainvalproportion = 4
            nowstr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
                
            learn_rate = p[0]
            batch = int(p[1])
            layer1 = int(p[2])
            layer2 = int(p[3])
            dropout = float(p[4])
            lookf = int(p[5])
            lookb = int(p[6])
            steps = int(p[7])
            stability_slope = p[8]
            onlyfirstpoints = int(p[9])
            #loaded_indices = np.loadtxt('random_indices'+indices+'.txt', dtype=int)
            #j = p[10]
            
            
            print("features tr length: {}".format(features_train.shape))
            print("features val length: {}".format(features_val.shape))
            
            x_train, y_train, scaler_fitted = slice_tapes(target_train, features_train, lookf, lookb, steps, stability_slope, None, onlyfirstpoints, indices, nowstr, trainorval = "train")
            print("LSTM In- & Out shapes (training): {}, {}".format(x_train.shape, y_train.shape))
            print()
            x_val, y_val, scaler = slice_tapes(          target_val, features_val, lookf, lookb, steps,      stability_slope, scaler_fitted, int(onlyfirstpoints/trainvalproportion), int(indices), nowstr, trainorval="val")
            print("LSTM In- & Out shapes (validation): {}, {}".format(x_val.shape, y_val.shape))
            
            
            x_train = x_train.reshape((-1, lookb, 13))
            x_val = x_val.reshape((-1, lookb, 13))
            
            # Define the model
            model = Sequential()
            model.add(LSTM(layer1, return_sequences=True, input_shape=(lookb, 13)))
            model.add(Dropout(dropout))
            if lookf == 1:
                model.add(LSTM(layer2, activation="sigmoid", return_sequences=False))  # Keep return_sequences=True
                model.add(Dense(1, activation="sigmoid"))
            else:
                model.add(LSTM(layer2, activation="sigmoid", return_sequences=True))
                model.add(TimeDistributed(Dense(1, activation="sigmoid")))
                
            optimizer = Adam(learning_rate=learn_rate)  # You can change 0.001 to your desired learning rate
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', 'mse'])
            model.summary()
            
            #history = model.fit(x_train, y_train, epochs=traintime, batch_size=batch, verbose = 1)
            history  = model.fit(x_train, y_train, epochs=traintime, batch_size=batch, validation_data=(x_val, y_val))
            
            
            modelpath = "../machines/1H1KTAPES_machine_"+nowstr+"_f"+str(lookf)+"_b"+str(lookb)+"_s"+str(steps)+"_sbs"+str(stability_slope)+"_l"+str(learn_rate)+"_b"+str(batch)+"_one"+str(layer1)+"_two"+str(layer2)+"_only"+str(onlyfirstpoints)+"_dr"+str(dropout)+"_candle"+str(candle)+".h5"
            scalerpath = "../scalers/1K1KTAPES_scaler_"+ nowstr+"_f"+str(lookf)+"_b"+str(lookb)+"_s"+str(steps)+"_sbs"+str(stability_slope)+"_l"+str(learn_rate)+"_b"+str(batch)+"_one"+str(layer1)+"_two"+str(layer2)+"_only"+str(onlyfirstpoints)+"_dr"+str(dropout)+"_candle"+str(candle)+".pkl"
            print("saving model {} \n and scaler {}".format(modelpath, scalerpath))
            
            model.save(modelpath)
            with open(scalerpath, 'wb') as f:
                pickle.dump(scaler, f)
                
            errorplots.append(np.log(history.history['loss']))
            valplots.append(np.log(history.history['val_loss']))
            mae.append(np.log(history.history['mae']))
            
            machinename.append(modelpath[12:])
                
                
            import os, time
            file1 = r"C:\Users\usuario\Documents\Trade\traincurves\trainerror_" + modelpath[12:] + ".png"
            file2 = r"C:\Users\usuario\Documents\Trade\traincurves\mae_" + modelpath[12:] + ".png"
            file3 = r"C:\Users\usuario\Documents\Trade\traincurves\valerror_" + modelpath[12:] + ".png"
            deleteplots(file1, file2, file3)
            time.sleep(1)
            
            
            
            plt.clf()
            plt.figure(figsize=(16, 9))
            for i in range(len(errorplots)):
                plt.plot(errorplots[i], label='Train Loss '+machinename[i])
            plt.legend()
            plt.title("Training Error")
            plt.savefig("../traincurves/trainerror_"+nowstr+".png")
            
            
            plt.clf()
            plt.figure(figsize=(16, 9))
            for i in range(len(valplots)):
                #for j in range(len(valplots[i])): ## Avg out to get smooth curve
                #    valplots[i] = np.mean(valplots[i,j-2:j+2])
                plt.plot(valplots[i], label='Train Loss '+machinename[i])
            plt.legend()
            plt.title("Val Error")
            plt.savefig("../traincurves/valerror_"+nowstr+modelpath[12:]+".png")
            plt.clf()
            
            print("Saved training curves")
            indices += 1
                
            import os
            #os.system(f"python test.py {modelpath} {scalerpath}")