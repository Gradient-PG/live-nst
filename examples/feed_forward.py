import torchvision.io as io
import pytorch_lightning as pl

from nst.models import FeedForward
from nst.datamodules import COCO128DataModule


if __name__ == "__main__":
    style_image = "images/style_image.jpg"

    model = FeedForward(style_image)
    datamodule = COCO128DataModule()

    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10, gpus=1, gradient_clip_val=1e-6, gradient_clip_algorithm="value")
    trainer.fit(model, datamodule)
