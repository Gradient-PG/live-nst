import torch
from typing import List


class StyleLoss:
    def __init__(self, style_target_features: List[torch.tensor]):
        """
        Loss between style target feature maps and input image feature maps.
        Computed using gram matrix of the feature maps.

        :param style_target_features: list of style target feature maps
        """
        # TODO compute gram matrix for each style target feature map
        pass

    def __call__(self, image_features: List[torch.tensor]) -> torch.tensor:
        """
        Compute and sum style loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing style loss
        """
        # TODO compute gram matrix for each input image feature map
        # TODO compute loss based on input and target gram matrices
        pass

    @staticmethod
    def _gram_matrix(x: torch.tensor) -> torch.tensor:
        """
        Compute gram matrix of given input matrix.

        :param x: input matrix with shape [batch, channels, height, width]
        :returns: tensor representing gram matrix for each example in batch
        """
        # TODO compute input gram matrix
        pass
