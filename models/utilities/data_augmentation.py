from kornia.augmentation import RandomCutMixV2
from torch import nn


class BaseBatchAugmentation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._f = nn.Identity()

    def forward(self, x, y):
        return self._f(x), y



class CutMix(BaseBatchAugmentation):
    r"""
    CutMix augmentation from https://arxiv.org/abs/1905.04899
    """
    def __init__(self, **kwargs):
        super().__init__()
        self._f = RandomCutMixV2(data_keys=['input', 'class'], **kwargs)

    def forward(self, x, y):
        x_cutmix, gt_permuted = self._f(x, y)
        return x_cutmix, gt_permuted[0]
