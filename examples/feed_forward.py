import torchvision.io as io
import pytorch_lightning as pl

from nst.models import FeedForward
from nst.datamodules import COCO128DataModule


if __name__ == "__main__":
    style_image = "examples/images/style_image.jpg"

    model = FeedForward(style_image)
    datamodule = COCO128DataModule()

    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=10, gpus=1)
    trainer.fit(model, datamodule)
