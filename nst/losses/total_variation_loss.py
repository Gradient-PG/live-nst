import torch
from typing import List
from nst.models import Baseline


class TotalVariationLoss:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + torch.sum(
            torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        )
        return image
        """
        Compute total variation loss of the input image.

        :param image: tensor representing input image with shape [batch, channels, height, width]
        :returns: vector of len = batch representing total variation loss
        """
