# TradeBot — Learning Market Dynamics from Synthetic Correlations
<img src="images/example/candlestick.png" width="41%">

Simulate common trading bots on real crypto data and prove that a neural networks is able to capture the pattern!

## Motivation
This project proves that a neural network can learn meaningful market patterns
that stem from commonly used pre-programmed bots on the real market.
It also permits its implementation in a real-time trading bot,
and includes different tools for getting, analyzing and pre-processing data.
It is meant to make this whole process more easy!

## Approach & Architecture



First install with: 

`pip install marketML`

Example code:

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

The input of the model, X, is always composed of whichever price over the last N timesteps and M features (we use 13). that can be computed directly.

<img src="images/Tapes.png" width="50%">

In the histogram, we can wee the broadening of the price fluctuations due to the synthetic trader.

<img src="images/prices_hist.png" width="50%">

## Results

The model learns much more from this synthetic data than from the original pattern. This is due to the explicitly introduced correlation between the target price and a certain features: RSI.
![](images/train_val.png)

After averaging of the price returns by lookf, we can compute the correlations C_i(tau) between returns and the i-th feature (RSI):

<img src="images/crosscorr.png" width="50%">


We can see theoretically the "predictiveness" of future prices based on their linear (there are higher order correlatios) correlation with past features/technical indicators. It remains the question of why our model could not capture this pattern, a safe assumption is that the signal is too weak compared to the noise.

## Future work

- A question arises, to whether is correlation pattern is stable over time, this needs to be tested an quantified; that will give a notion of how often the model has to be retrained.
- Implementation of the trading bot strategy: Although this is working, there is no assurance of profit until the above pattern can be captured.
- A relation between the training and the trading must be established. Even if the synthetic trader could be modelled, would this give profits on future synthetic prices?

###################################################################








