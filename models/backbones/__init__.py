from models.backbones.cnn.dn4.dn4_cnn import FourLayer_64F
from models.backbones.cnn.dn7.dn7_cnn import SevenLayer_64F
from models.backbones.resnet.resnet_256 import ResNetLike

__all__ = {
    'DN4_CNN2d': FourLayer_64F,
    'DN7_CNN2d': FourLayer_64F,
    'ResNet2d': ResNetLike
}


