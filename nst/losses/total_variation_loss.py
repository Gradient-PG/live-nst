import torch
from typing import List


class TotalVariationLoss:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss of the input image.

        :param image: tensor representing input image with shape [batch, channels, height, width]
        :returns: vector of len = batch representing total variation loss
        """
        batch, _, _, _ = image.size()

        tv_loss = torch.sum(
            torch.abs(image[:, :, :, :-1].view(batch, -1).sum(1) - image[:, :, :, 1:].view(batch, -1).sum(1))
        ) + torch.sum(
            torch.abs(image[:, :, :-1, :].view(batch, -1).sum(1) - image[:, :, 1:, :].view(batch, -1).sum(1))
        )
        return tv_loss
