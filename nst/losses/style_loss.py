import torch
from typing import List
from torch.nn.functional import mse_loss


class StyleLoss:
    def __init__(self, style_target_features: List[torch.Tensor]):
        """
        Loss between style target feature maps and input image feature maps.
        Computed using gram matrix of the feature maps.

        :param style_target_features: list of style target feature maps
        """
        self.style_gram_matrix_list = []
        for feature_map in style_target_features:
            self.style_gram_matrix_list.append(StyleLoss._gram_matrix(feature_map))

    def __call__(self, image_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute and sum style loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image
            (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing style loss
        """
        sum_of_losses = torch.zeros(image_features[0].shape[0])
        for index, feature_map in enumerate(image_features):
            feature_map_gram_matrix = StyleLoss._gram_matrix(feature_map)
            sum_of_losses += mse_loss(self.style_gram_matrix_list[index], feature_map_gram_matrix)
        return sum_of_losses

    @staticmethod
    def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
        """
        Compute gram matrix of given input matrix.

        :param x: input matrix with shape [batch, channels, height, width]
        :returns: tensor representing gram matrix for each example in batch
        """
        batch, channel, height, width = x.size()
        x_view = x.view(batch, channel, height * width)
        x_gram = torch.bmm(x_view, x_view.transpose(1, 2))  # transpose in the first and second dimension
        return x_gram.div(height * width)
