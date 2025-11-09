import numpy as np
import matplotlib.pyplot as plt

class SyntheticTrader:
    def __init__(self, cryptodata):
        self.target, self.features = cryptodata.target_total, cryptodata.features_total

        self.traded = None
        """ Accumulated contains the shift in % """
        self.accumulated = None
        self.synth_price = None

    def discrete_MA(self, target, features, perc):
        """ When short MA crosses above long MA, the bot buys self.traded[i] = perc
        where perc is given in percentage of the price, thus the price shift
        that it will cause. This is then accumulated and stored into a synthetic price chart"""

        self.target = target
        self.features = features
        self.traded = [0 for k in range(len(target))]
        """ Accumulated contains the shift in % """
        self.accumulated = [0 for k in range(len(target))]
        self.synth_price = [0 for k in range(len(target))]

        shortMA, longMA = features[:, 0], features[:, 1]
        short_prev, long_prev = 0.4, 0.5 # First trade is golden cross

        for i in range(len(self.traded)):
            short_now, long_now = shortMA[i], longMA[i]

            if short_prev <= long_prev and short_now > long_now: # Golden cross = BUY

                self.traded[i] = perc
                short_prev = short_now
                long_prev = long_now
            elif short_prev >= long_prev and short_now < long_now: # Death cross = SELL
                self.traded[i] = -perc
                short_prev = short_now
                long_prev = long_now

        self.shift_prices(perc)
        return self.accumulated

    def discrete_RSI(self, target, features, perc):
        """ When RSI < 30 (or > 70 for sell), the bot buys self.traded[i] = perc
        where perc is given in percentage of the price, thus the price shift
        that it will cause. This is then accumulated and stored into a synthetic price chart"""

        self.target = target
        self.features = features
        self.traded = [0 for k in range(len(target))]
        """ Accumulated contains the shift in % """
        self.accumulated = [0 for k in range(len(target))]
        self.synth_price = [0 for k in range(len(target))]

        RSI = features[:,2]
        bought = False

        for i in range(len(self.traded)):
            if RSI[i] > 70 and not bought:
                self.traded[i] = perc
                bought = True
            if RSI[i] < 30 and bought:
                self.traded[i] = -perc
                bought = False

        self.shift_prices(perc)
        return self.accumulated

    def continuous_RSI(self, target, features, perc):
        """ When RSI < 30 (or > 70 for sell), the bot buys every candle,
        proportionately to the distance above 70 or sells below 30.
        This is then accumulated and stored into a synthetic price chart"""

        self.target = target
        self.features = features
        self.traded = [0 for k in range(len(target))]
        """ Accumulated contains the shift in % """
        self.accumulated = [0 for k in range(len(target))]
        self.synth_price = [0 for k in range(len(target))]

        RSI = features[:,2]
        bought = False

        for i in range(len(self.traded)):
            if RSI[i] > 70:
                self.traded[i] = (RSI[i]-70)/30*perc
                bought = True
            if RSI[i] < 30:
                self.traded[i] = (RSI[i]-30)/30*perc
                bought = False

        self.shift_prices(perc)
        return self.accumulated

    def linear_RSI(self, target, features, perc):
        """ When RSI != 50, the bot buys or sells just linearly depending on RSI-50
        This is then accumulated and stored into a synthetic price chart"""

        self.target = target
        self.features = features
        self.traded = [0 for k in range(len(target))] # also "volume" or "pressure"
        """ Accumulated contains the shift in % """
        self.accumulated = [0 for k in range(len(target))]
        self.synth_price = [0 for k in range(len(target))]

        RSI = features[:,2]
        bought = False

        for i in range(len(self.traded)):
            """ Square root impact (literature) in interval [-1,+1] """
            dP = np.sqrt(np.abs((RSI[i]-50))/50) * np.sign( RSI[i]-50)
            self.traded[i] = dP*perc + 1 # 0.01 + 1
            """ Linear impact """
            #self.traded[i] = (RSI[i]-50)/50*perc + 1 # 0.01 + 1

            self.synth_price[i] = self.target[i]*self.traded[i]

        return self.traded

    def shift_prices(self, perc):
        for i in range(len(self.traded)):
            self.accumulated[i] = -perc/2+float(np.sum(self.traded[:i])) # Accumulated price shift in %
            self.synth_price[i] = float(self.target[i] + self.accumulated[i]/100*self.target[i])

        plt.plot(self.target, label = " Orig. BTCUSD")
        plt.plot(self.synth_price, label = " Synth BTCUSDT")
        plt.show()
        return

    def continuous_MA(self):
        diff = short_MA - long_MA
        signal_strength = np.tanh(diff / price)  # normalize, smooth
        target_position = max_position * signal_strength
        return

if __name__ == "__main__":
    print("test")
