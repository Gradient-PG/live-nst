import torch
from typing import List
from nst.models import Baseline


# style_target_features, image_features are lists of tensor containing respective feature maps
class StyleLoss:
    def __init__(self, style_target_features: List[torch.Tensor]):  # Gram matrix operation on target style image
        self.style_target_features = StyleLoss._gram_matrix(style_target_features)
        """
        Loss between style target feature maps and input image feature maps.
        Computed using gram matrix of the feature maps.

        :param style_target_features: list of style target feature maps
        """

    def __call__(self, image_features: List[torch.Tensor]) -> torch.Tensor:  # Style loss for specific iteration
        current_style_loss = torch.nn.MSELoss(reduction="mean")(
            StyleLoss._gram_matrix(image_features), self.style_target_features
        )
        return current_style_loss
        """
        Compute and sum style loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image
         (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing style loss
        """

    @staticmethod  # Gram matrix operation on the input
    def _gram_matrix(x: List[torch.Tensor]) -> torch.Tensor:
        for feature_map in x:
            b, ch, height, width = feature_map.size()
            feature_map.view(b, ch, height * width)
            Gram = torch.bmm(feature_map, feature_map.transpose(1, 2))  # transpose in the first and second dimension
            Gram.div_(height * width)

        return Gram
        """
        Compute gram matrix of given input matrix.

        :param x: input matrix with shape [batch, channels, height, width]
        :returns: tensor representing gram matrix for each example in batch
        """
