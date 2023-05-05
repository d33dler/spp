import dataclasses
import os
from pathlib import Path
from typing import Union

import torch
from torch import nn

from dataset.datasets_csv import CSVLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.utilities.utils import load_config, config_exchange

"""
Augmentation functions mappings from torchvision.transforms"""
TRANSFORM_MAP = {
    "RESIZE": transforms.Resize,
    "RANDOM_CROP": transforms.RandomCrop,
    "RANDOM_HORIZONTAL_FLIP": transforms.RandomHorizontalFlip,
    "RANDOM_VERTICAL_FLIP": transforms.RandomVerticalFlip,
    "RANDOM_RESIZED_CROP": transforms.RandomResizedCrop,
    "COLOR_JITTER": transforms.ColorJitter,
    "RANDOM_ROTATION": transforms.RandomRotation,
    "RANDOM_ERASING": transforms.RandomErasing,
    "RANDOM_PERSPECTIVE": transforms.RandomPerspective,
    "RANDOM_EQUALIZE": transforms.RandomEqualize,
    "RANDOM_ADJUST_SHARPNESS": transforms.RandomAdjustSharpness,
    "RANDOM_GAUSSIAN_BLUR": transforms.GaussianBlur,
    "TO_TENSOR": transforms.ToTensor,
    "NORMALIZE": transforms.Normalize,
    "GAUSSIAN_BLUR": transforms.GaussianBlur,
}


@dataclasses.dataclass
class Parameters:
    """
    DataLoader parameters (extracted from model root config @see models/architectures/configs)
    """
    shot_num: int
    way_num: int
    query_num: int
    episode_train_num: int
    episode_test_num: int
    episode_val_num: int
    outf: str
    workers: int
    episodeSize: int
    test_ep_size: int
    batch_sz: int


@dataclasses.dataclass
class Loaders:
    train_loader: Union[DataLoader, None]
    val_loader: Union[DataLoader, None]
    test_loader: DataLoader


class DatasetLoader:
    """
    Class used for data preparation (episode construction, pre-processing, augmentation)
    """

    def __init__(self, cfg, params: Parameters) -> None:
        self.params = params
        self.cfg = load_config(Path(Path(__file__).parent / "config.yaml"))
        self.cfg.AUGMENTOR = config_exchange(self.cfg.AUGMENTOR, cfg)

    def _read_transforms(self, cfg, cfg_aug):
        transform_ls = []
        for TF in cfg_aug:
            if TF.NAME in cfg.DISABLE:
                continue
            _t = TRANSFORM_MAP[TF.NAME]
            if TF.ARGS:
                if isinstance(TF.ARGS, dict):
                    _t = _t(**TF.ARGS)
                else:
                    _t = _t(*tuple(TF.ARGS))
            else:
                _t = _t()
            transform_ls.append(_t)
        return transform_ls

    def load_data(self, mode, dataset_directory, F_txt):
        # ======================================= Folder of Datasets =======================================
        # image transform & normalization
        dataset_dir = dataset_directory
        shot_num = self.params.shot_num
        way_num = self.params.way_num
        query_num = self.params.query_num
        episode_train_num = self.params.episode_train_num
        episode_val_num = self.params.episode_val_num
        episode_test_num = self.params.episode_test_num
        transform_ls = []

        cfg_aug = self.cfg.AUGMENTOR
        pre_process = transforms.Compose(self._read_transforms(cfg_aug, cfg_aug.PRE_PROCESS))
        augmentation = self._read_transforms(cfg_aug, cfg_aug.AUGMENTATION)
        post_process = transforms.Compose(self._read_transforms(cfg_aug, cfg_aug.POST_PROCESS))
        av_num = cfg_aug.AV_NUM

        if mode == 'train':
            trainset = CSVLoader(
                data_dir=dataset_dir, mode='train',
                pre_process=pre_process,
                augmentations=augmentation,
                post_process=post_process,
                episode_num=episode_train_num, way_num=way_num, shot_num=shot_num, query_num=query_num, av_num=av_num
            )
            valset = CSVLoader(
                data_dir=dataset_dir, mode='val',
                pre_process=pre_process,
                augmentations=None,
                post_process=post_process,
                episode_num=episode_val_num, way_num=way_num, shot_num=shot_num, query_num=query_num)
        testset = CSVLoader(
            data_dir=dataset_dir, mode='test',
            pre_process=pre_process,
            augmentations=None,
            post_process=post_process,
            episode_num=episode_test_num, way_num=way_num, shot_num=shot_num, query_num=query_num)
        if mode == 'train':
            print('Trainset: %d' % len(trainset))
            print('Trainset: %d' % len(trainset), file=F_txt)
            print('Valset: %d' % len(valset))
            print('Valset: %d' % len(valset), file=F_txt)
        print('Testset: %d' % len(testset))
        print('Testset: %d' % len(testset), file=F_txt)

        # ========================================== Load Datasets =========================================
        workers = self.params.workers
        train_loader = None
        val_loader = None
        if mode == 'train':
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params.episodeSize, shuffle=True,
                num_workers=int(workers), drop_last=True, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                valset, batch_size=self.params.test_ep_size, shuffle=True,
                num_workers=int(workers), drop_last=True, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.params.test_ep_size, shuffle=True,
            num_workers=int(workers), drop_last=True, pin_memory=True
        )
        return Loaders(train_loader, val_loader, test_loader)
