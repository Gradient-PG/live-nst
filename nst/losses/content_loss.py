import torch
from typing import List, Optional
from nst.losses.mse_batch_loss import MSEBatchLoss
import torch.nn as nn


class ContentLoss:
    def __init__(self, reduce_batch=False):
        """
        Loss between content target feature maps and input image feature maps.

        :param reduce_batch: if false loss is computed for each image in batch
        """
        if reduce_batch:
            self._loss = nn.MSELoss()
        else:
            self._loss = MSEBatchLoss()

    def __call__(
        self, input_features: List[torch.Tensor], target_features: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute and sum content loss for each feature map of the input image.

        :param input_features: list of feature maps of the input image
            (feature map shape [batch, channels, height, width])
        :param target_features: list of feature maps of the target image
            (feature map shape [batch, channels, height, width])
        :returns: vector of len = batch representing style loss, scalar if reduce_batch is true
            or None if image_features is empty
        """
        loss = None

        for input_feature_map, target_feature_map in zip(input_features, target_features):
            if loss is None:
                loss = self._loss(input_feature_map, target_feature_map)
            else:
                loss += self._loss(input_feature_map, target_feature_map)

        return loss
