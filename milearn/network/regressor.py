import numpy as np

from sklearn.preprocessing import MinMaxScaler

from .module.attention import AdditiveAttentionNetwork, SelfAttentionNetwork, HopfieldAttentionNetwork
from .module.base import BaseRegressor
from .module.dynamic import DynamicPoolingNetwork
from .module.classic import InstanceNetwork, BagNetwork
from .module.mlp import BagWrapperMLPNetwork, InstanceWrapperMLPNetwork


class BagNetworkRegressor(BagNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class InstanceNetworkRegressor(InstanceNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AdditiveAttentionNetworkRegressor(AdditiveAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SelfAttentionNetworkRegressor(SelfAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HopfieldAttentionNetworkRegressor(HopfieldAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BagWrapperMLPNetworkRegressor(BagWrapperMLPNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class InstanceWrapperMLPNetworkRegressor(InstanceWrapperMLPNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DynamicPoolingNetworkRegressor(DynamicPoolingNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        y = np.array(y).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        y = self.scaler.fit_transform(y).flatten()

        return super().fit(x, y)

    def predict(self, x):
        y_pred = super().predict(x)
        y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred
