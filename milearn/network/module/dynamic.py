import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Softmax
from .base import InstanceTransformer, BaseNetwork, apply_instance_dropout

class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, alpha=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.alpha = alpha

    def forward(self, lengths, labels):
        left = F.relu(self.m_pos - lengths, inplace=True) ** 2
        right = F.relu(lengths - self.m_neg, inplace=True) ** 2
        margin_loss = labels * left + self.alpha * (1. - labels) * right
        return margin_loss.mean()

class Squash(nn.Module):
    def forward(self, bag_embed):
        norm = torch.norm(bag_embed, p=2, dim=2, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
        return scale * bag_embed

class Norm(nn.Module):
    def forward(self, bag_squash):
        return torch.norm(bag_squash, p=2, dim=2, keepdim=True)

class DynamicPooling(nn.Module):
    def __init__(self, n_iter=3):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, inst_embed, inst_mask):

        # 1. Apply mask to instance embeddings
        inst_embed = inst_mask * inst_embed

        # 2. Initialize temporary instance logits
        inst_logits = torch.zeros(*inst_embed.shape[:2], 1)

        # 3. Do dynamic routing
        for t in range(self.n_iter):

            # 4. Compute instance weights
            inst_weights = torch.softmax(inst_mask * inst_logits, dim=1)
            # inst_weights = torch.transpose(inst_weights, 2, 1)

            # 5. Compute and squash bag embedding
            bag_embed = torch.sum(inst_weights * inst_embed, dim=1, keepdim=True)
            bag_squash = Squash()(bag_embed)

            # 6. Compute similarity between each instance embedding and current bag embedding
            new_logits = torch.sum(bag_squash * inst_embed, dim=2, keepdim=True)

            # 7. Update instance logits
            inst_logits = inst_logits + new_logits

        # 8. Compute final instance weights
        inst_weights = torch.softmax(inst_logits, dim=1)

        return bag_squash, inst_weights

class DynamicPoolingNetwork(BaseNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_basic_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        super()._create_basic_layers(input_layer_size, hidden_layer_sizes)
        self.bag_estimator = Norm()

    def _create_special_layers(self, input_layer_size, hidden_layer_sizes):
        self.dynamic_pooling = DynamicPooling()

    def forward(self, x, inst_mask):

        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(x)

        # 2. Apply instance dropout and mask
        inst_mask = apply_instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed

        # 3. Compute bag embedding and instance weights
        bag_embed, inst_weights = self.dynamic_pooling(inst_embed, inst_mask)

        # 4. Compute final bag prediction
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, inst_weights, bag_pred



