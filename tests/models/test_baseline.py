import pytest
import torch

from nst.models.baseline import Baseline


class TestBaseline:
    @pytest.fixture
    def random_image(self):
        rand_tensor = 255 * torch.rand((3, 250, 250))
        return torch.round(rand_tensor)

    def test_baseline_should_initialize_without_error(self, random_image):
        model = Baseline(random_image, random_image)
