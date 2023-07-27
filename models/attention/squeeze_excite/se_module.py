from torch import nn as nn
from torch.nn import functional as F


class ClassRelatedAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ClassRelatedAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.conv_Wg = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.conv_Wk = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc1 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1,
                             bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Non-local operation
        fg = self.conv_Wg(x)
        fk = self.conv_Wk(x)
        fk = F.softmax(fk.view(b, self.in_channels // self.reduction, -1), dim=-1).view(b,
                                                                                        self.in_channels // self.reduction,
                                                                                        h, w)
        non_local_op = fg * fk
        non_local_op = non_local_op.sum(dim=[2, 3])

        # Excitation operation
        y = self.fc1(non_local_op.unsqueeze(-1).unsqueeze(-1))
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Apply the weight vector to the input feature map
        weighted_x = x * y

        # Sum the features of the scene-related channels
        scene_related_features = weighted_x.sum(dim=1, keepdim=True)

        # Obtain the scene-class-related attention feature map
        attention_feature_map = self.sigmoid(scene_related_features)

        return attention_feature_map
