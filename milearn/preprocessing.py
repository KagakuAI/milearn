import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

class BagScaler(BaseEstimator, TransformerMixin):
    """A wrapper to apply sklearn scalers to bags of instances."""
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else MinMaxScaler()

    def fit(self, x, y=None):
        all_instances = np.vstack(x)  # stack all bags
        self.scaler.fit(all_instances, y)
        return self

    def transform(self, x):
        x_scaled = []
        for bag in x:
            x_scaled.append(self.scaler.transform(bag))
        return x_scaled

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

class BagMinMaxScaler(BagScaler):
    def __init__(self, **kwargs):
        super().__init__(scaler=MinMaxScaler(**kwargs))

class BagStandardScaler(BagScaler):
    def __init__(self, **kwargs):
        super().__init__(scaler=StandardScaler(**kwargs))

class BagMaxAbsScaler(BagScaler):
    def __init__(self, **kwargs):
        super().__init__(scaler=MaxAbsScaler(**kwargs))

class BagRobustScaler(BagScaler):
    def __init__(self, **kwargs):
        super().__init__(scaler=RobustScaler(**kwargs))
