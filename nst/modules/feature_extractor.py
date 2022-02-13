import torch
import torch.nn as nn
from typing import List, Tuple


class FeatureExtractor(nn.Module):
    def __init__(self, content_layers: List[str], style_layers: List[str]):
        """
        Module responsible for extracting feature maps from vgg19 network.

        :param content_layers: list of string names of vgg19 layers to use to compoute content feature maps
        :param style_layers: list of string names of vgg19 layers to use to compoute style feature maps
        """
        super().__init__()
        # TODO download pretrained vgg19 model without dense layers
        # TODO put model into eval mode and set require grad to false
        pass

    def forward(self, x: torch.tensor) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        """
        Pass image through vgg19 and return listed activations.

        :param x: input image batch of shape [batch, 3, height, width]
        :returns: first element of tuple is list of content feature maps and second is list of style feature maps (feature map shape [batch, channels, height, width])
        """
        # TODO pass images through every layer of vgg separately
        # TODO collect listed style and content activations
        pass
