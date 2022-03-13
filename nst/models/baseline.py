from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import write_jpeg
import torchvision.io as io
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss
from nst.modules import FeatureExtractor


class Baseline(pl.LightningModule):
    def __init__(
        self,
        content_image: Union[torch.Tensor, str],
        style_image: Union[torch.Tensor, str],
        image_size: Tuple[int, int] = (225, 225),
        init_with_content: bool = True,
        learning_rate: float = 1e-1,
        content_weight: float = 0.4,
        style_weight: float = 0.4,
        total_variation_weight: float = 0.2,
        content_layers: List[str] = None,
        style_layers: List[str] = None,
    ):
        """
        Classic iterative style transfer algorithm.

        :param content_image: tensor with shape [3, image_height, image_width] and uint8 values or path to jpg file
        :param content_image: tensor with shape [3, image_height, image_width] and uint8 values or path to jpg file
        :param image_size: tuple representing output image size (at least 224x224)
        :param use_content_image: if true content image will be used to initialize training else noise
        :param learning_rate: learning rate passed to optimizer
        :param content_weight: content loss weight
        :param style_weight: style loss weight
        :param total_variation_weight: total variation loss weight
        :param content_layers: list of string names of vgg19 layers to use to compoute content feature maps
        :param style_layers: list of string names of vgg19 layers to use to compoute style feature maps
        """
        super().__init__()

        # load content image tensor
        if type(content_image) is str:
            content_image_tensor = io.read_image(content_image)
        else:
            content_image_tensor = content_image

        # load style image tensor
        if type(style_image) is str:
            style_image_tensor = io.read_image(style_image)
        else:
            style_image_tensor = style_image

        # preprocess style and content images
        preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Resize(image_size),
                T.Lambda(lambda x: x.unsqueeze(0)),
            ]
        )
        content_image_tensor = preprocess(content_image_tensor)
        style_image_tensor = preprocess(style_image_tensor)

        # initialize image to optimize
        if init_with_content:
            self._optimized_image = torch.nn.Parameter(content_image_tensor, requires_grad=True)
        else:
            random_image_tensor = torch.randn((1, 3, image_size[0], image_size[1]))
            self._optimized_image = torch.nn.Parameter(random_image_tensor, requires_grad=True)

        self._feature_extractor = FeatureExtractor(content_layers, style_layers)

        # extract feature maps from style and content image
        target_content_features_maps, _ = self._feature_extractor(content_image_tensor)
        _, target_style_features_maps = self._feature_extractor(style_image_tensor)

        # initialize loss functions and loss weights
        self._content_loss = ContentLoss(target_content_features_maps)
        self._style_loss = StyleLoss(target_style_features_maps)
        self._total_variation_loss = TotalVariationLoss()

        self._content_weight = content_weight
        self._style_weight = style_weight
        self._total_variation_weight = total_variation_weight

        self._learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        content_feature_maps, style_feature_maps = self._feature_extractor(self._optimized_image)

        content_loss = self._content_loss(content_feature_maps)
        style_loss = self._style_loss(style_feature_maps)

        total_variation_loss = self._total_variation_loss(self._optimized_image)
        sum_of_losses = (
            self._content_weight * content_loss
            + self._style_weight * style_loss
            + self._style_weight * total_variation_loss
        )

        writer = SummaryWriter()
        writer.add_image("Working Image", self._optimized_image.squeeze(), global_step=self.global_step)
        writer.add_scalar("Loss/content/", content_loss, global_step=self.global_step)
        writer.add_scalar("Loss/style/", style_loss, global_step=self.global_step)
        writer.add_scalar("Loss/total_variation/", total_variation_loss, global_step=self.global_step)
        writer.close()

        return sum_of_losses

    def configure_optimizers(self):
        return torch.optim.Adam([self._optimized_image], lr=self._learning_rate)

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
