from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.backbones.cnn.dn4.dn4am_cnn import DN4_AM
from models.backbones.cnn.dn7.dn7_cnn import SevenLayer_64F
from models.backbones.cnn.siamese_net.sn4_knn import DN4_MCNP
from models.backbones.cnn.siamese_net.sn4am_fc import B_4L64F_AM_MCNP
from models.backbones.resnet.resnet_256 import ResNetLike
from models.backbones.cnn.siamese_net.sn4_fc import B_4L64F_MCNP

__all__ = {
    'DN4_CNN2d': BaselineBackbone2d,
    'DN7_CNN2d': SevenLayer_64F,
    'DN4_AM_CNN2d': DN4_AM,
    'ResNet2d': ResNetLike,
    'B_4L64F_MCNP': B_4L64F_MCNP,
    'DN4_MCNP': DN4_MCNP,
    'B_4L64F_AM_MCNP': B_4L64F_AM_MCNP
}
