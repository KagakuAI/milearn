import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from milearn.network.module.attention import (AttentionNetwork,
                                                  SelfAttentionNetwork,
                                                  GatedAttentionNetwork,
                                                  MultiHeadAttentionNetwork,
                                                  HopfieldAttentionNetwork)

from milearn.network.module.attention import TempAttentionNetwork

from milearn.network.module.base import BaseRegressor
from milearn.network.module.utils import add_padding
from milearn.network.module.dynamic import DynamicPoolingNetwork
from milearn.network.module.classic import InstanceNetwork, BagNetwork


class AttentionNetworkRegressor(AttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MultiHeadAttentionNetworkRegressor(MultiHeadAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SelfAttentionNetworkRegressor(SelfAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GatedAttentionNetworkRegressor(GatedAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HopfieldAttentionNetworkRegressor(HopfieldAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TempAttentionNetworkRegressor(TempAttentionNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InstanceNetworkRegressor(InstanceNetwork, BaseRegressor):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)
        self.pool = pool


class BagNetworkRegressor(BagNetwork, BaseRegressor):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)


class DynamicPoolingNetworkRegressor(DynamicPoolingNetwork, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_val_split(self, x, y, val_size=0.2, random_state=42):
        x, y = np.asarray(x, dtype="object"), np.asarray(y, dtype="object")
        x, m = add_padding(x)
        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(x, y, m, test_size=val_size,
                                                                          random_state=random_state)
        if isinstance(self, BaseRegressor):
            self.scaler = MinMaxScaler()
            y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = self.scaler.transform(y_val.reshape(-1, 1)).flatten()

        x_train, y_train, m_train = self._array_to_tensor(x_train, y_train, m_train)
        x_val, y_val, m_val = self._array_to_tensor(x_val, y_val, m_val)
        return x_train, x_val, y_train, y_val, m_train, m_val
