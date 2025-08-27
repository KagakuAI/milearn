import torch
from torch.nn import Linear
from .base import BaseNetwork, FeatureExtractor
from .hopt import StepwiseHopt

def apply_instance_dropout(m, p=0.0, training=True):
    if training and p > 0.0:
        drop_mask = (torch.rand_like(m.float()) > p).float()
        m = m * drop_mask
    return m

class BagNetwork(BaseNetwork, StepwiseHopt):
    def __init__(self, pool='mean', instance_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pool = pool
        self.instance_dropout = instance_dropout

    def _pooling(self, X, M):
        if self.pool == 'mean':
            bag_embed = X.sum(axis=1) / M.sum(axis=1)
        elif self.pool == 'max':
            bag_embed = X.max(dim=1)[0]
        elif self.pool == 'lse':
            bag_embed = X.exp().sum(dim=1).log()
        else:
            TypeError(f"Pooling type {self.pool} is not supported.")
            return None
        bag_embed = bag_embed.unsqueeze(1)
        return bag_embed

    def forward(self, X, M):

        # 1. Compute instance embeddings
        H = self.extractor(X)

        # 2. Apply instance dropout and mask
        M = apply_instance_dropout(M, self.instance_dropout, self.training)
        H = M * H

        # 3. Apply pooling and compute bag embedding
        bag_embed = self._pooling(H, M)

        # 4. Compute final bag prediction
        bag_score = self.estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, None, bag_pred

    def hopt(self, x, y, param_grid, verbose=True):
        valid_pools = ['mean', 'max', 'lse']
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)


class InstanceNetwork(BaseNetwork):

    def __init__(self, pool='mean', instance_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pool = pool
        self.instance_dropout = instance_dropout

    def _pooling(self, Y, M):
        if self.pool == 'mean':
            y_agg = Y.sum(axis=1) / M.sum(axis=1)
        elif self.pool == 'max':
            idx = Y.abs().argmax(dim=1, keepdim=True)  # [B, 1, D]
            y_agg = Y.gather(1, idx).squeeze(1)  # [B, D]
        else:
            TypeError(f"Pooling type {self.pool} is not supported.")
            return None
        y_agg = y_agg.unsqueeze(1)
        return y_agg

    def forward(self, X, M):

        # 1. Compute instance embeddings
        H = self.extractor(X)

        # 2. Apply instance dropout and mask
        M = apply_instance_dropout(M, self.instance_dropout, self.training)
        H = M * H

        # 3. Compute instance predictions
        inst_score = self.estimator(H)
        inst_pred = self.prediction(inst_score)

        # 4. Apply pooling and compute final bag prediction
        bag_pred = self._pooling(inst_pred, M)

        return None, inst_pred, bag_pred

    def hopt(self, x, y, param_grid, verbose=True):
        valid_pools = ['mean', 'max']
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)




