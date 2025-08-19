import logging
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities import rank_zero


def add_padding(x):
    bag_size = max(len(i) for i in x)
    mask = np.ones((len(x), bag_size, 1))

    out = []
    for i, bag in enumerate(x):
        bag = np.asarray(bag)
        if len(bag) < bag_size:
            mask[i][len(bag):] = 0
            padding = np.zeros((bag_size - bag.shape[0], bag.shape[1]))
            bag = np.vstack((bag, padding))
        out.append(bag)
    out_bags = np.asarray(out)
    return out_bags, mask


def get_mini_batches(x, y, m, batch_size=16):
    data = MBSplitter(x, y, m)
    mb = DataLoader(data, batch_size=batch_size, shuffle=True)
    return mb


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MBSplitter(Dataset):
    def __init__(self, x, y, m):
        super(MBSplitter, self).__init__()
        self.x = x
        self.y = y
        self.m = m

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]

    def __len__(self):
        return len(self.y)


def silence_lightning(level=logging.ERROR):
    """
    Suppress PyTorch Lightning info logs like:
    'GPU available: ...', 'TPU available: ...', and model summaries.

    Args:
        level: logging level to set (default=logging.ERROR).
               Use logging.WARNING if you still want warnings.
    """
    modules = [
        "lightning.pytorch",                       # global lightning logs
        "lightning.pytorch.accelerators.cuda",     # GPU availability
        "lightning.pytorch.accelerators.tpu",      # TPU availability
        "lightning.pytorch.accelerators.hpu",      # HPU availability
    ]
    for m in modules:
        logging.getLogger(m).setLevel(level)
    rank_zero.rank_zero_info = lambda *args, **kwargs: None
    rank_zero.rank_zero_warn = lambda *args, **kwargs: None


class TrainLogging(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", 0.0)
        val_loss = metrics.get("val_loss", 0.0)

        # Align epoch numbers and losses
        print(
            f"Epoch {trainer.current_epoch+1:3d}/{trainer.max_epochs:<3d} | "
            f"train_loss={train_loss:7.4f} | "
            f"val_loss={val_loss:7.4f}"
        )
