import torch
from nst.losses.l1_batch_loss import L1BatchLoss


class TotalVariationLoss:
    def __init__(self):
        """
        Class computes total variation loss for the input image.
        It measures how smooth input image is.
        """
        self._loss = L1BatchLoss()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss of the input image.

        :param image: tensor representing input image with shape [batch, channels, height, width]
        :returns: vector of len = batch representing total variation loss
        """
        tv_loss = self._loss(image[:, :, :, :-1], image[:, :, :, 1:]) + self._loss(
            image[:, :, :-1, :], image[:, :, 1:, :]
        )
        return tv_loss
