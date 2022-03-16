import torch
import torch.nn.functional as F


class L1BatchLoss:
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute l1 loss for each tensor pair in batch.

        :param input: batch of input tensors
        :param target: batch of target tensors
        :returns: vector of len = batch representing l1 loss
        """
        loss_unreduced = F.l1_loss(input, target, reduction="none")
        batch_size = loss_unreduced.size()[0]
        loss = loss_unreduced.view(batch_size, -1).mean(1)
        return loss
