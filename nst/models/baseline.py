import pytorch_lightning as pl

from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss
from nst.modules import FeatureExtractor


class Baseline(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
