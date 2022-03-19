import torch
from typing import List, Optional
from nst.losses.mse_batch_loss import MSEBatchLoss


class StyleLoss:
    def __init__(self):
        """
        Loss between style target feature maps and input image feature maps.
        Computed using gram matrix of the feature maps.
        """
        self._loss = MSEBatchLoss()

    def __call__(
        self,
        input_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
        compute_target_gram: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Compute and sum style loss for each feature map of the input image.

        :param input_features: list of feature maps of the input image
            (feature map shape [batch, channels, height, width])
        :param target_features: list of feature maps of the target image
            (feature map shape [batch, channels, height, width])
        :param compute_target_gram: if false assume that list of target features contains gram matrices
        :returns: vector of len = batch representing style loss or None if image_features is empty
        """
        loss = None

        for input_feature_map, target_feature_map in zip(input_features, target_features):
            input_gram_matrix = self.gram_matrix(input_feature_map)

            if compute_target_gram:
                target_gram_matrix = self.gram_matrix(target_feature_map)
            else:
                target_gram_matrix = target_feature_map

            if loss is None:
                loss = self._loss(input_gram_matrix, target_gram_matrix)
            else:
                loss += self._loss(input_gram_matrix, target_gram_matrix)

        if loss is not None:
            loss /= len(input_features)
        return loss

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
        """
        Compute gram matrix of given input matrix.

        :param x: input matrix with shape [batch, channels, height, width]
        :returns: tensor representing gram matrix for each example in batch
        """
        batch, channel, height, width = x.size()
        x_view = x.view(batch, channel, height * width)
        x_gram = torch.bmm(x_view, x_view.transpose(1, 2))  # transpose in the first and second dimension
        return x_gram.div(channel * height * width)
