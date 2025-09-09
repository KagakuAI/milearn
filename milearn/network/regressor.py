import numpy as np
from sklearn.model_selection import train_test_split
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

    def _train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x, dtype="object"), np.asarray(y, dtype="object")
        # x, m = add_padding(x)
        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        if isinstance(self, BaseRegressor):
            self.scaler = MinMaxScaler()
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = self.scaler.transform(y_val.reshape(-1, 1)).flatten()

        x_train, y_train, m_train = self._array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self._array_to_tensor(x_val, y_val, m_val)
        return x_train, x_val, y_train, y_val, m_train, m_val
