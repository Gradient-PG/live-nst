import torch
from nst.losses.l1_batch_loss import L1BatchLoss
import torch.nn as nn


class TotalVariationLoss:
    def __init__(self, reduce_batch=False):
        """
        Class computes total variation loss for the input image.
        It measures how smooth input image is.

        :param reduce_batch: if false loss is computed for each image in batch
        """
        if reduce_batch:
            self._loss = nn.L1Loss()
        else:
            self._loss = L1BatchLoss()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss of the input image.

        :param image: tensor representing input image with shape [batch, channels, height, width]
        :returns: vector of len = batch representing total variation loss or scalar if reduce_batch is true
        """
        tv_loss = self._loss(image[:, :, :, :-1], image[:, :, :, 1:]) + self._loss(
            image[:, :, :-1, :], image[:, :, 1:, :]
        )
        return tv_loss
