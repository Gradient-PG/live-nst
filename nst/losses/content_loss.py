import torch
from typing import List
from nst.models import Baseline


class ContentLoss:
    def __init__(
        self,
        content_target_features: List[torch.Tensor],
    ):
        self.content_target_features = content_target_features
        """
        Loss between content target feature maps and input image feature maps.

        :param content_target_features: list of conten target feature maps
        """

    def __call__(self, image_features: List[torch.Tensor]) -> torch.Tensor:
        current_content_loss = torch.nn.MSELoss(reduction="mean")(self.content_target_features, image_features)
        return current_content_loss
        """
        Compute and sum content loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image
        (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing content loss
        """
