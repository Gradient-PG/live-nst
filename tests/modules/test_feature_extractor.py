import pytest
import torch

from nst.modules import FeatureExtractor


class TestFeatureExtractor:
    @pytest.fixture
    def random_image_batch(self):
        return torch.rand((1, 3, 250, 250))

    @pytest.fixture
    def feature_extractor(self):
        return FeatureExtractor()

    def test_feature_extractor_should_extract_features_without_error(self, feature_extractor, random_image_batch):
        content_features, style_features = feature_extractor(random_image_batch)
