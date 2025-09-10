from .base import BaseNetwork, apply_instance_dropout
from .hopt import StepwiseHopt

class BagNetwork(BaseNetwork, StepwiseHopt):
    def __init__(self, pool='mean', instance_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, X, M):
        if self.pool == 'mean':
            bag_embed = X.sum(axis=1) / M.sum(axis=1)
        elif self.pool == 'sum':
            bag_embed = X.sum(axis=1)
        elif self.pool == 'max':
            bag_embed = X.max(dim=1)[0]
        elif self.pool == 'lse':
            bag_embed = X.exp().sum(dim=1).log()
        else:
            raise TypeError(f"Pooling type {self.pool} is not supported.")

        bag_embed = bag_embed.unsqueeze(1)
        return bag_embed

    def forward(self, X, M):

        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(X)

        # 2. Apply instance dropout and mask
        M = apply_instance_dropout(M, self.hparams.instance_dropout, self.training)
        inst_embed = M * inst_embed

        # 3. Apply pooling and compute bag embedding
        bag_embed = self._pooling(inst_embed, M)

        # 4. Compute final bag prediction
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, None, bag_pred

    def hopt(self, x, y, param_grid,  verbose=True):
        valid_pools = ['mean', 'sum', 'max', 'lse']
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)


class InstanceNetwork(BaseNetwork):

    def __init__(self, pool='mean', instance_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, Y, M):
        if self.pool == 'mean':
            y_agg = Y.sum(axis=1) / M.sum(axis=1)
        elif self.pool == 'sum':
            y_agg = Y.sum(axis=1)
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
        inst_embed = self.instance_transformer(X)

        # 2. Apply instance dropout and mask
        M = apply_instance_dropout(M, self.hparams.instance_dropout, self.training)
        inst_embed = M * inst_embed

        # 3. Compute instance predictions
        inst_score = self.bag_estimator(inst_embed)
        inst_pred = self.prediction(inst_score)

        # 4. Apply pooling and compute final bag prediction
        bag_pred = self._pooling(inst_pred, M)

        return None, inst_pred, bag_pred

    def hopt(self, x, y, param_grid, verbose=True):
        valid_pools = ['mean', "sum", 'max']
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)




