import torch
import torch.nn as nn
from typing import List, Tuple


class FeatureExtractor(nn.Module):
    def __init__(self, content_layers: List[str], style_layers: List[str]):
        """
        Module responsible for extracting feature maps from vgg19 network.

        :param content_layers: list of string names of vgg19 layers to use to compute content feature maps
        :param style_layers: list of string names of vgg19 layers to use to compute style feature maps
        """
        super().__init__()
        trained_vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
        # written out manually from the vgg19 architecture
        vgg19_indexes = {
            'conv1_1': 0,
            'conv1_2': 2,
            'conv2_1': 5,
            'conv2_2': 7,
            'conv3_1': 10,
            'conv3_2': 12,
            'conv3_3': 14,
            'conv3_4': 16,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv4_3': 23,
            'conv4_4': 25,
            'conv5_1': 28,
            'conv5_2': 30,
            'conv5_3': 32,
            'conv5_4': 34,
        }
        coded_content_layers = [(x, "content") for x in content_layers]
        coded_style_layers = [(x, "style") for x in style_layers]
        all_layers = sorted(coded_style_layers + coded_content_layers)

        self.features = []
        last_index = 0

        for i in range(len(all_layers)):
            # add each needed layer and its type
            self.features.append((torch.nn.Sequential(), all_layers[i][1]))
            # get index
            layer_index = vgg19_indexes[all_layers[i][0]]

            for j in range(last_index, layer_index + 1):
                # adding all layers between last layer up to current one
                self.features[i][0].training = False
                self.features[i][0].add_module(str(j), trained_vgg19[j])
            last_index = layer_index + 1

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.tensor) -> Tuple[List[torch.tensor], List[torch.tensor]]:
        """
        Pass image through vgg19 and return listed activations.

        :param x: input image batch of shape [batch, 3, height, width]
        :returns: first element of tuple is list of content feature maps and second is list of style feature maps (feature map shape [batch, channels, height, width])
        """
        styles = []
        contents = []
        for feature in self.features:
            x = feature[0](x)
            if feature[1] == 'style':
                styles.append(x)
            else:
                contents.append(x)

        return contents, styles
        # TODO pass images through every layer of vgg separately
        # TODO collect listed style and content activations
        pass
