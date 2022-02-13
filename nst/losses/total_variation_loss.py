import torch
from typing import List


class TotalVariationLoss:
    def __call__(self, image: torch.tensor) -> torch.tensor:
        """
        Compute total variation loss of the input image.

        :param image: tensor representing input image with shape [batch, channels, height, width]
        :returns: vector of len = batch representing total variation loss 
        """
        # TODO compute total variation loss of the input image
        pass
