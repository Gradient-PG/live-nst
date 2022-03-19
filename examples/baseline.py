import torchvision.io as io
import pytorch_lightning as pl

from nst.models import Baseline


if __name__ == "__main__":
    content_image = "examples/images/content_image.jpg"
    style_image = "examples/images/style_image.jpg"

    model = Baseline(content_image, style_image)

    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=10)
    trainer.fit(model)
