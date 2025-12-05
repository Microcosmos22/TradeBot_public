# TradeBot — Learning Market Dynamics from Synthetic Correlations
Simulate common trading bots on real crypto data and prove that a neural networks is able to capture the pattern!

This project proves that a neural network can learn meaningful market patterns
that stem from commonly used pre-programmed bots on the real market.
It also permits its implementation in a real-time trading bot,
and includes different tools for getting, analyzing and pre-processing data.
It is meant to make this whole process more easy!


<img src="images/example/candlestick.png" width="41%">

## First install with: 

pip install marketML

## Example code:

```python
    cryptodata = CryptoDataGetter()
    synth = SyntheticTrader(cryptodata)
    synth_machine = LSTMachine().init(candle = "5min", layer1 = 40, layer2 = 15, lookb = 10, learn_rate = 0.03 , dropout = 0.1, reg = 1e-4)

    """ ## Call historical data, simulate and apply an artificial trader ## """

    _, _, synth_target, synth_features = cryptodata.get_historical_data_trim(
    ["1 August 2024 00:00:00", 15000], "BTCUSDT", Client.KLINE_INTERVAL_5MINUTE,
    transform_func=synth.linear_RSI, transform_strength = 0.02, plot = False)

    """ ############# Prepare Inputs, train the Neural Network ########### """
    x_train, y_train, x_val, y_val, scaler = cryptodata.split_slice_normalize(lookb = 10, lookf = 5, target_total = synth_target, features_total = synth_features)

    trainmean, train_std, valmean, val_std = synth_machine.fit(x_train, y_train, x_val, y_val, epochs = 50, batch = 16)

    """ ############## Plot training and some examples #################### """
    plot = MachinePlotter(synth_machine)
    plot.plotmachine(trainmean, train_std, valmean, val_std)
    plot.plot_tape_eval(x_val, y_val)
```

From the original candlestick crypto data above,

We modify it by introducing a linear RSI trader that causes a price shift:
<img src="images/example/orig_synth_price.png" width="50%">

The model learns much more from this synthetic data than from the original pattern.
![](images/train_val.png)

From the correlations between target and features_i: C_i(tau):
<img src="images/example/crosscorr.png" width="50%">

We can see theoretically the "predictiveness" of future prices based on their linear (there are more correlatios above) correlation with past features/technical indicators:

This is not surprising, but this framework provides the opportunity to test different models and traders,
with the goal to find the model that can capture the most trade bots. 




For a more detailed tutorial that goes through all functions of this library,
i uploaded a Kaggle notebook, showing the capture of a real-time trading bot:



## 🧠 Project Overview

1. Load historical BTCUSDT data  
2. Compute technical indicators and prepare model inputs  
3. Diagnose data distributions and correlations  
4. Generate synthetic market data with controlled structure  
5. Train neural network on real vs synthetic data  
6. Simulate a trading bot using trained model  







