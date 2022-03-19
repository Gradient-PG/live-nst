import torch
import torch.nn.functional as F
from typing import List


class MSEBatchLoss:
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute mean squared error for each tensor pair in batch.

        :param input: batch of input tensors
        :param target: batch of target tensors
        :returns: vector of len = batch representing mean squared error loss
        """
        loss_unreduced = F.mse_loss(input, target, reduction="none")
        batch_size = loss_unreduced.size()[0]
        loss = loss_unreduced.view(batch_size, -1).mean(1)
        return loss
