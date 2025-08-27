import logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from lightning.pytorch.utilities import rank_zero

def silence_and_seed_lightning(seed=42, level=logging.ERROR):

    # 1. Silence standard loggers
    modules = [
        "lightning.pytorch",  # global lightning logs
        "lightning.pytorch.accelerators.cuda",  # GPU availability
        "lightning.pytorch.accelerators.tpu",  # TPU availability
        "lightning.pytorch.accelerators.hpu",  # HPU availability
        "lightning.pytorch.utilities.seed",  # <- the seed message
    ]
    for m in modules:
        logging.getLogger(m).setLevel(level)

    # 2. Silence rank_zero log helpers
    rank_zero.rank_zero_info = lambda *a, **k: None
    rank_zero.rank_zero_warn = lambda *a, **k: None
    rank_zero.rank_zero_debug = lambda *a, **k: None
    rank_zero.rank_zero_only = lambda f, *a, **k: None

    # 3. Seed everything
    seed_everything(seed, workers=True, verbose=False)


class TrainLogging(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", 0.0)
        val_loss = metrics.get("val_loss", 0.0)

        # Align epoch numbers and losses
        print(
            f"Epoch {trainer.current_epoch+1:3d}/{trainer.max_epochs:<3d} | "
            f"train_loss={train_loss:7.3f} | "
            f"val_loss={val_loss:7.3f}"
        )
