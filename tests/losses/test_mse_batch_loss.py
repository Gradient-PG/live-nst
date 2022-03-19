import pytest
import torch

from nst.losses import MSEBatchLoss


class TestMSEBatchLoss:
    @pytest.fixture
    def ones_vector_batch(self):
        return torch.ones((5, 4))

    @pytest.fixture
    def zeros_vector_batch(self):
        return torch.zeros((5, 4))

    @pytest.fixture
    def mse_batch_loss(self):
        return MSEBatchLoss()

    def test_mse_batch_loss_should_keep_batch_dim(self, mse_batch_loss, ones_vector_batch, zeros_vector_batch):
        loss = mse_batch_loss(ones_vector_batch, zeros_vector_batch)
        assert loss.size() == (5,)
