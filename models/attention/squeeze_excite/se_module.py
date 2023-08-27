import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F, init
import matplotlib.pyplot as plt
import os

from models.utilities.utils import net_init_weights_normal


class ClassRelatedAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ClassRelatedAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Convolutional layers
        self.conv_Wg = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.conv_Wk = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)

        # Batch normalization layers
        self.bn_Wg = nn.BatchNorm2d(in_channels // reduction)
        self.bn_Wk = nn.BatchNorm2d(in_channels // reduction)

        # Fully connected layers for excitation operation
        self.fc1 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

        # Activation and sigmoid
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def save_attention_map_as_image(self, attention_masks, save_dir="attention_maps"):
        """
        Save a batch of attention maps as images.

        :param attention_masks: numpy array of shape (B, 1, H, W)
        :param save_dir: directory to save the attention maps
        """
        # Ensure the save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, mask in enumerate(attention_masks):
            # Create an RGB image initialized to white (255, 255, 255)
            img = np.ones((mask.shape[1], mask.shape[2], 3), dtype=np.uint8) * 255

            # Where the mask is 1, set the color to red (255, 0, 0)
            img[mask[0] == 1] = [255, 0, 0]

            # Save the image
            plt.imsave(os.path.join(save_dir, f"attention_map_{idx}.png"), img)

    def forward(self, x):
        b, c, h, w = x.size()

        # Non-local operation with batch normalization
        fg = self.bn_Wg(self.conv_Wg(x))
        fk = self.bn_Wk(self.conv_Wk(x))

        # Softmax normalization over the channel dimension
        fk = F.softmax(fk.view(b, self.in_channels // self.reduction, -1), dim=1).view(b,
                                                                                       self.in_channels // self.reduction,
                                                                                       h, w)

        non_local_op = fg * fk
        non_local_op = non_local_op.sum(dim=[2, 3])

        # Excitation operation
        y = self.fc1(non_local_op.unsqueeze(-1).unsqueeze(-1))
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Residual connection
        weighted_x = x + x * y

        # Sum the features of the scene-related channels
        scene_related_features = weighted_x.sum(dim=1, keepdim=True)

        # Obtain the scene-class-related attention feature map
        attention_feature_map = self.sigmoid(scene_related_features)

        return attention_feature_map
