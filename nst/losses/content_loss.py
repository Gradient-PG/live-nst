import torch
from typing import List, Optional
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
        self._content_target_features = content_target_features

    def __call__(self, image_features: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute and sum content loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image
            (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing content loss or None if image_features is empty
        """
        loss = None

        for input_features, target_features in zip(image_features, self._content_target_features):
            if loss is None:
                loss = mse_loss(input_features, target_features)
            else:
                loss += mse_loss(input_features, target_features)

        return loss
