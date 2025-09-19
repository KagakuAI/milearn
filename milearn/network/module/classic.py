from .base import BaseNetwork, instance_dropout
from .hopt import StepwiseHopt

class BagNetwork(BaseNetwork, StepwiseHopt):
    def __init__(self, pool='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, bags, inst_mask):
        if self.pool == 'mean':
            bag_embed = bags.sum(axis=1) / inst_mask.sum(axis=1)
        elif self.pool == 'sum':
            bag_embed = bags.sum(axis=1)
        elif self.pool == 'max':
            bag_embed = bags.max(dim=1)[0]
        elif self.pool == 'lse':
            bag_embed = bags.exp().sum(dim=1).log()
        else:
            raise TypeError(f"Pooling type {self.pool} is not supported.")

        bag_embed = bag_embed.unsqueeze(1)
        return bag_embed

    def forward(self, bags, inst_mask):

        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(bags)

        # 2. Apply instance dropout and mask
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed

        # 3. Apply pooling and compute bag embedding
        bag_embed = self._pooling(inst_embed, inst_mask)

        # 4. Compute final bag prediction
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, None, bag_pred

    def hopt(self, x, y, param_grid,  verbose=False):
        valid_pools = ['mean', 'sum', 'max', 'lse']
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)


class InstanceNetwork(BaseNetwork):

    def __init__(self, pool='mean', **kwargs):
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, inst_pred, inst_mask):
        if self.pool == 'mean':
            bag_pred = inst_pred.sum(axis=1) / inst_mask.sum(axis=1)
        elif self.pool == 'sum':
            bag_pred = inst_pred.sum(axis=1)
        elif self.pool == 'max':
            idx = inst_pred.abs().argmax(dim=1, keepdim=True)  # [B, 1, D]
            bag_pred = inst_pred.gather(1, idx).squeeze(1)  # [B, D]
        else:
            TypeError(f"Pooling type {self.pool} is not supported.")
            return None
        bag_pred = bag_pred.unsqueeze(1)
        return bag_pred

    def forward(self, bags, inst_mask):

        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(bags)

        # 2. Apply instance dropout and mask
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed

        # 3. Compute instance predictions
        inst_score = self.bag_estimator(inst_embed)
        bag_score = self._pooling(inst_score, inst_mask)

        # 4. Apply pooling and compute final bag prediction
        bag_pred = self.prediction(bag_score)

        return None, None, bag_pred

    def hopt(self, x, y, param_grid, verbose=True):
        valid_pools = ['mean', "sum", 'max']
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)




