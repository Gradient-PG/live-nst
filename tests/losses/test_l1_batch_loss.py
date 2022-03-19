import pytest
import torch

from nst.losses import MSEBatchLoss


class TestL1BatchLoss:
    @pytest.fixture
    def ones_vector_batch(self):
        return torch.ones((5, 4))

    @pytest.fixture
    def zeros_vector_batch(self):
        return torch.zeros((5, 4))

    @pytest.fixture
    def l1_batch_loss(self):
        return MSEBatchLoss()

    def test_l1_batch_loss_should_keep_batch_dim(self, l1_batch_loss, ones_vector_batch, zeros_vector_batch):
        loss = l1_batch_loss(ones_vector_batch, zeros_vector_batch)
        assert loss.size() == (5,)
