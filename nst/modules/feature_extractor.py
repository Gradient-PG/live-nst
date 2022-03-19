import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import List, Tuple


class FeatureExtractor(nn.Module):
    def __init__(self, content_layers: List[str] = None, style_layers: List[str] = None, use_relu: bool = True):
        """
        Module responsible for extracting feature maps from vgg19 network.

        :param content_layers: list of string names of vgg19 layers to use to compute content feature maps
        :param style_layers: list of string names of vgg19 layers to use to compute style feature maps
        :param use_relu: if true extractor will return features after relu activation
        """
        super().__init__()

        # if layers not define use default from original publication
        if content_layers is None:
            content_layers = ["conv4_2"]
        if style_layers is None:
            style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        # written out manually from the vgg19 architecture
        # fmt: off
        vgg19_layer_indices = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34,
        }
        # fmt: on

        if use_relu:
            # add one to each index since relu follows conv layers
            self._content_layers_indices = [vgg19_layer_indices[layer] + 1 for layer in content_layers]
            self._style_layers_indices = [vgg19_layer_indices[layer] + 1 for layer in style_layers]
        else:
            self._content_layers_indices = [vgg19_layer_indices[layer] for layer in content_layers]
            self._style_layers_indices = [vgg19_layer_indices[layer] for layer in style_layers]

        self._vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True).features.eval()

        # change max pooling layers to average pooling layers
        for idx, layer in enumerate(self._vgg.children()):
            if isinstance(layer, nn.MaxPool2d):
                self._vgg[idx] = nn.AvgPool2d(kernel_size=2, stride=2)
            if isinstance(layer, nn.ReLU):
                self._vgg[idx] = nn.ReLU(inplace=False)

        for param in self._vgg.parameters():
            param.requires_grad = False

        # input image needs to be normalized using vgg's dataset mean and std
        self._preprocessor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Pass image through vgg19 and return listed activations.

        :param x: input image batch of shape [batch, 3, height, width]
        :returns: first element of tuple is list of content feature maps and second is list of style feature maps
         (feature map shape [batch, channels, height, width])
        """
        x = self._preprocessor(x)

        content_features = []
        style_features = []

        for idx, layer in enumerate(self._vgg.children()):
            x = layer(x)
            if idx in self._content_layers_indices:
                content_features.append(x)
            if idx in self._style_layers_indices:
                style_features.append(x)

        return content_features, style_features
