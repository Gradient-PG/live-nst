from asyncore import write
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import write_jpeg
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss
from nst.modules import FeatureExtractor


class Baseline(pl.LightningModule):
    def __init__(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        image_size: Tuple[int, int] = (225, 225),
        learning_rate: float = 1e-1,
        content_weight: float = 0.4,
        style_weight: float = 0.4,
        total_variation_weight: float = 0.2,
        content_layers: List[str] = None,
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

        if content_layers is None:
            content_layers = ["conv4_2"]

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

        self._feature_extractor = FeatureExtractor(content_layers, style_layers)

        target_content_feature_maps = self._feature_extractor(content_image)[0]
        self._content_loss = ContentLoss(target_content_feature_maps)

        target_style_feature_maps = self._feature_extractor(style_image)[1]
        self._style_loss = StyleLoss(target_style_feature_maps)

        self._total_variation_loss = TotalVariationLoss()

        self._optimized_image = content_image

        self._content_weight = content_weight
        self._style_weight = style_weight
        self._total_variation_weight = total_variation_weight

        self._learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        content_feature_maps, style_feature_maps = self._feature_extractor(self._optimized_image)

        content_loss = self._content_loss(content_feature_maps)
        style_loss = self._style_loss(style_feature_maps)

        total_variation_loss = self._total_variation_loss(self._optimized_image)
        sum_of_losses = self._content_weight * content_loss + self._style_weight * style_loss

        writer = SummaryWriter()
        writer.add_image("Working Image", self._optimized_image.squeeze(), global_step=self.global_step)
        writer.add_scalar("Loss/content/", content_loss, global_step=self.global_step)
        writer.add_scalar("Loss/style/", style_loss, global_step=self.global_step)
        writer.add_scalar("Loss/total_variation/", total_variation_loss, global_step=self.global_step)
        writer.close()

        return sum_of_losses

    def configure_optimizers(self):
        return torch.optim.Adam(self._optimized_image.parameters(), lr=self._learning_rate)

    def train_dataloader(self):
        """Configure dummy dataset with one empty tensor."""
        dummy_tensor = torch.empty((1, 1))
        dummy_tensor_dataset = TensorDataset(dummy_tensor)
        return DataLoader(dummy_tensor_dataset)

    @property
    def optimized_image(self) -> torch.Tensor:
        """Return a detached copy of optimized image tensor."""
        detached_image = self._optimized_image.detach().cpu().squeeze(0)
        return convert_image_dtype(detached_image, dtype=torch.uint8)

    def save_optimized_image(self, path: str) -> None:
        """
        Save optimized image at given path in jpg format.

        :param path: path save image at
        """
        write_jpeg(self.optimized_image, path)
