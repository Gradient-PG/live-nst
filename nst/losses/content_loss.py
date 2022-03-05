import torch
from typing import List
from torch.nn.functional import mse_loss


class ContentLoss:
    def __init__(
        self,
        content_target_features: List[torch.Tensor],
    ):
        """
        Loss between content target feature maps and input image feature maps.

        :param content_target_features: list of conten target feature maps
        """
        self.content_target_features = content_target_features

    def __call__(self, image_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute and sum content loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image
            (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing content loss
        """
        sum_of_losses = torch.zeros(image_features[0].shape[0])
        for index, feature_map in enumerate(image_features):
            sum_of_losses += mse_loss(self.content_target_features[index], feature_map)
        return sum_of_losses
