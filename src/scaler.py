import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureAwareScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that applies feature-specific scaling strategies.
    """

    def __init__(self):
        self.feature_names = ["ret", "sma20", "sma50", "RSI", "BBwidth", "mom", "vol", "K", "D", "MACD", "d_month", "d_week", "h_day"]
        self.params_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.feature_names)
        self.params_ = {}

        for col in self.feature_names:
            x = X[col].values

            if col in ['RSI', 'K', 'F']:
                # 0–100 oscillator → no fitting needed
                self.params_[col] = {'type': 'oscillator'}

            elif col in ['sma20', 'sma50']: # 0 to inf
                self.params_[col] = {'mean': x.mean(), 'std': x.std(), 'type': 'zscore'}

            elif col in ['mom', 'MACD', 'BBwidth']: # 0 to inf
                self.params_[col] = {'mean': x.mean(), 'std': x.std(), 'type': 'zscore'}

            elif col == 'vol':
                xlog = np.log1p(x)
                self.params_[col] = {'mean': xlog.mean(), 'std': xlog.std(), 'type': 'log_zscore'}

            elif col in ['d_month', 'd_week', 'h_day']:
                self.params_[col] = {'type': 'cyclical'}

            else:
                self.params_[col] = {'mean': x.mean(), 'std': x.std(), 'type': 'zscore'}

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names)
        X_scaled = pd.DataFrame(index=X.index)

        for col in self.feature_names:
            p = self.params_[col]
            x = X[col].values

            if p['type'] == 'oscillator':
                X_scaled[col] = (x - 50) / 50

            elif p['type'] == 'zscore':
                X_scaled[col] = (x - p['mean']) / (p['std'] + 1e-8)

            elif p['type'] == 'log_zscore':
                xlog = np.log1p(x)
                X_scaled[col] = (xlog - p['mean']) / (p['std'] + 1e-8)

            elif p['type'] == 'cyclical':
                if col == 'd_month':
                    maxv = 31
                elif col == 'd_week':
                    maxv = 7
                else:
                    maxv = 24
                X_scaled[f'{col}_sin'] = np.sin(2 * np.pi * x / maxv)
                X_scaled[f'{col}_cos'] = np.cos(2 * np.pi * x / maxv)
                continue

        return X_scaled.values

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it — returns transformed array, not self"""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_scaled):
        # Optional: reconstruct original values if needed
        pass
