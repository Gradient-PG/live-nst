import pytest
import torch

from nst.losses import TotalVariationLoss


class TestTotalVariationLoss:
    @pytest.fixture
    def ones_image_batch(self):
        return torch.ones((1, 3, 250, 250))

    @pytest.fixture
    def tv_loss(self):
        return TotalVariationLoss()

    def test_tv_loss_should_be_zero_for_same_pixel_image(self, tv_loss, ones_image_batch):
        loss = tv_loss(ones_image_batch)
        assert loss.item() == 0
