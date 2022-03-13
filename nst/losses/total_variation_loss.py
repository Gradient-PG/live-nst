import torch
import torch.nn.functional as F
from typing import List


class TotalVariationLoss:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss of the input image.

        :param image: tensor representing input image with shape [batch, channels, height, width]
        :returns: vector of len = batch representing total variation loss
        """
        tv_loss = F.l1_loss(image[:, :, :, :-1], image[:, :, :, 1:]) + F.l1_loss(
            image[:, :, :-1, :], image[:, :, 1:, :]
        )
        return tv_loss
