import torch
from typing import List


class ContentLoss:
    def __init__(self, content_target_features: List[torch.tensor]):
        """
        Loss between content target feature maps and input image feature maps.

        :param content_target_features: list of conten target feature maps
        """
        # TODO initialize content target features field
        pass

    def __call__(self, image_features: List[torch.tensor]) -> torch.tensor:
        """
        Compute and sum content loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing content loss 
        """
        # TODO compute loss based on input and target feature maps
        pass
