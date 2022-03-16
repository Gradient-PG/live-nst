import torch
from typing import List, Optional
from nst.losses.mse_batch_loss import MSEBatchLoss


class StyleLoss:
    def __init__(self, style_target_features: List[torch.Tensor]):
        """
        Loss between style target feature maps and input image feature maps.
        Computed using gram matrix of the feature maps.

        :param style_target_features: list of style target feature maps
        """
        self._style_target_gram_matrices = []
        for feature_map in style_target_features:
            self._style_target_gram_matrices.append(self._gram_matrix(feature_map))

        self._loss = MSEBatchLoss()

    def __call__(self, image_features: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute and sum style loss for each feature map of the input image.

        :param image_features: list of feature maps of the input image
            (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing style loss or None if image_features is empty
        """
        loss = None

        for input_features, target_gram_matrix in zip(image_features, self._style_target_gram_matrices):
            input_gram_matrix = self._gram_matrix(input_features)

            if loss is None:
                loss = self._loss(input_gram_matrix, target_gram_matrix)
            else:
                loss += self._loss(input_gram_matrix, target_gram_matrix)

        if loss is not None:
            loss /= len(image_features)
        return loss

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
        return x_gram.div(channel * height * width)
