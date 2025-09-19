from .module.attention import AdditiveAttentionNetwork, SelfAttentionNetwork, HopfieldAttentionNetwork
from .module.base import BaseClassifier
from .module.dynamic import DynamicPoolingNetwork, MarginLoss
from .module.classic import BagNetwork, InstanceNetwork
from .module.mlp import BagWrapperMLPNetwork, InstanceWrapperMLPNetwork

class BagNetworkClassifier(BagNetwork, BaseClassifier):

    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)

class InstanceNetworkClassifier(InstanceNetwork, BaseClassifier):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(pool=pool, **kwargs)

class AdditiveAttentionNetworkClassifier(AdditiveAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SelfAttentionNetworkClassifier(SelfAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HopfieldAttentionNetworkClassifier(HopfieldAttentionNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BagWrapperMLPNetworkClassifier(BagWrapperMLPNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class InstanceWrapperMLPNetworkClassifier(InstanceWrapperMLPNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DynamicPoolingNetworkClassifier(DynamicPoolingNetwork, BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, y_pred, y_true):
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true)
        return loss
