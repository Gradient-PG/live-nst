from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import write_jpeg
import torchvision.io as io

from nst.losses import ContentLoss, StyleLoss, TotalVariationLoss
from nst.modules import FeatureExtractor, TransformerNetwork


class FeedForward(pl.LightningModule):
    def __init__(
        self,
        style_image: Union[torch.Tensor, str],
        image_size: Tuple[int, int] = (256, 256),
        learning_rate: float = 1e-1,
        content_weight: float = 1e1,
        style_weight: float = 1e7,
        total_variation_weight: float = 1e-1,
        content_layers: List[str] = None,
        style_layers: List[str] = None,
    ):
        """
        Classic iterative style transfer algorithm.

        :param style_image: tensor with shape [3, image_height, image_width] and uint8 values or path to jpg file
        :param image_size: tuple representing output image size (at least 224x224)
        :param learning_rate: learning rate passed to optimizer
        :param content_weight: content loss weight
        :param style_weight: style loss weight
        :param total_variation_weight: total variation loss weight
        :param content_layers: list of string names of vgg19 layers to use to compoute content feature maps
        :param style_layers: list of string names of vgg19 layers to use to compoute style feature maps
        """
        super().__init__()

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
        self._style_image_tensor = preprocess(style_image_tensor)

        self._model = TransformerNetwork()

        self._feature_extractor = FeatureExtractor(content_layers, style_layers)

        # extract feature maps from style image
        _, target_style_features_maps = self._feature_extractor(self._style_image_tensor)

        # compute and register target style gram matrices
        self._target_style_gram_matrices_names = []
        for idx, features_map in enumerate(target_style_features_maps):
            name = f"target_style_gram_matrix_{idx}"
            gram_matrix = StyleLoss.gram_matrix(features_map)
            self.register_buffer(name, gram_matrix)
            self._target_style_gram_matrices_names.append(name)

        # initialize loss functions and loss weights
        self._content_loss = ContentLoss(reduce_batch=True)
        self._style_loss = StyleLoss(reduce_batch=True)
        self._total_variation_loss = TotalVariationLoss(reduce_batch=True)

        self._content_weight = content_weight
        self._style_weight = style_weight
        self._total_variation_weight = total_variation_weight

        self._learning_rate = learning_rate

        # persist hyperparameters
        self.save_hyperparameters()

    def on_fit_start(self):
        # list target style feature maps attributes
        self._target_style_gram_matrices = []
        for name in self._target_style_gram_matrices_names:
            self._target_style_gram_matrices.append(getattr(self, name))

        # log style target
        self.logger.experiment.add_image("style_target", self._style_image_tensor.squeeze(0))

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        content_feature_maps, _ = self._feature_extractor(batch)
        optimized_image_batch = self._model(batch)

        with torch.no_grad():
            optimized_image_batch[:] = optimized_image_batch.clamp(0, 1)

        optimized_content_feature_maps, optimized_style_feature_maps = self._feature_extractor(optimized_image_batch)

        content_loss = self._content_loss(optimized_content_feature_maps, content_feature_maps)
        style_loss = self._style_loss(optimized_style_feature_maps, self._target_style_gram_matrices)
        tv_loss = self._total_variation_loss(optimized_image_batch)

        weighted_content_loss = self._content_weight * content_loss
        weighted_style_loss = self._style_weight * style_loss
        weighted_tv_loss = self._total_variation_weight * tv_loss
        loss = weighted_content_loss + weighted_style_loss + weighted_tv_loss

        self.log("loss/content", weighted_content_loss)
        self.log("loss/style", weighted_style_loss)
        self.log("loss/tv", weighted_tv_loss)
        self.log("loss", loss)

        # TODO move to val loop
        self.logger.experiment.add_image("result_image", optimized_image_batch[0].squeeze(0), step=self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
