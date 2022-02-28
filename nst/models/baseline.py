import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from typing import List, Tuple
import torchvision.transforms as T

from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss
from nst.modules import FeatureExtractor


class Baseline(pl.LightningModule):
    def __init__(
        self,
        content_image: torch.tensor,
        style_image: torch.tensor,
        image_size: Tuple[int, int] = (225, 225),
        learing_rate: float = 1e-1,
        content_weight: float = 0.4,
        style_weight: float = 0.4,
        total_variation_weight: float = 0.2,
        conten_layers: List[str] = None,
        style_layers: List[str] = None,
    ):
        """
        Classic iterative style transfer algorithm.

        :param content_image: tensor with shape [3, image_height, image_width] and uint8 values
        :param content_image: tensor with shape [3, image_height, image_width] and uint8 values
        :param image_size: tuple representing output image size (at least 224x224)
        :param learning_rate: learning rate passed to optimizer
        :param content_weight: content loss weight
        :param style_weight: style loss weight
        :param total_variation_weight: total variation loss weight
        :param content_layers: list of string names of vgg19 layers to use to compoute content feature maps
        :param style_layers: list of string names of vgg19 layers to use to compoute style feature maps
        """
        super().__init__()

        if conten_layers is None:
            conten_layers = ["conv4_2"]

        if style_layers is None:
            style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Resize(image_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Lambda(lambda x: x.unsqueeze(0)),
            ]
        )

        content_image = preprocess(content_image)
        style_image = preprocess(style_image)

        self._feature_extractor = FeatureExtractor(conten_layers, style_layers)

        # content feature maps of content_image without batch dimension
        target_content_features_maps = self._feature_extractor(content_image)[0]
        for x in range(len(target_content_features_maps)):
            target_content_features_maps[x] = target_content_features_maps[x].squeeze(0)

        self._content_loss = ContentLoss(target_content_features_maps)

        # style feature maps of style_image without batch dimension
        target_style_features_maps = self._feature_extractor(style_image)[1]
        for x in range(len(target_style_features_maps)):
            target_style_features_maps[x] = target_style_features_maps[x].squeeze(0)

        self._style_loss = StyleLoss(target_style_features_maps)

        self._total_variation_loss = TotalVariationLoss
        
        self._optimized_image = content_image
        pass

    def training_step(self, batch, batch_idx):
        optimized_image_feature_maps = self._feature_extractor(self._optimized_image)

        content_loss = self._content_loss(optimized_image_feature_maps[0])
        style_loss = self._style_loss(optimized_image_feature_maps[1])
        total_variation_loss = self._total_variation_loss(self._optimized_image)

        sum_of_losses = self.content_weight * content_loss + self.style_weight * style_loss

        # TODO log losses and _optimized_image to tensorboard
        return sum_of_losses

    def configure_optimizers(self):
        # TODO configure Adam optimizer for self._optimized_image
        pass

    def train_dataloader(self):
        """Configure dummy dataset with one empty tensor."""
        dummy_tensor = torch.empty((1, 1))
        dummy_tensor_dataset = TensorDataset(dummy_tensor)
        return DataLoader(dummy_tensor_dataset)

    @property
    def optimized_image(self) -> torch.tensor:
        """Return a detached copy of optimized image tensor."""
        # TODO detach optimized image from computational graph, copy to cpu
        # TODO remove dummy batch dimmension and denormalize
        return self._optimized_image
