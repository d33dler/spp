import dataclasses
import os
from pathlib import Path

import torch
from torch import nn

from dataset.datasets_csv import CSVLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.utilities.utils import load_config, config_exchange


@dataclasses.dataclass
class Parameters:
    """
    DataLoader parameters (extracted from model root config @see models/architectures/configs)
    """
    img_sz: int
    dataset_dir: str
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
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class DatasetLoader:
    """
    Class used for data preparation (episode construction, pre-processing, augmentation)
    """

    def __init__(self, cfg, params: Parameters) -> None:
        self.transforms_ls = []
        self.params = params
        self.cfg = load_config(Path(Path(__file__).parent / "config.yaml"))
        self.cfg.AUGMENTOR = config_exchange(self.cfg.AUGMENTOR, cfg)

    def augment(self, transform: nn.Module):
        self.transforms_ls = [transform] + self.transforms_ls

    def load_data(self, mode, F_txt, dataset_directory=None):
        # ======================================= Folder of Datasets =======================================
        # image transform & normalization
        dataset_dir = dataset_directory or self.params.dataset_dir
        img_sz = self.params.img_sz
        shot_num = self.params.shot_num
        way_num = self.params.way_num
        query_num = self.params.query_num
        episode_train_num = self.params.episode_train_num
        episode_val_num = self.params.episode_val_num
        episode_test_num = self.params.episode_test_num
        transform_ls = []
        cfg_aug = self.cfg.AUGMENTOR
        for TF in cfg_aug.TRANSFORMS:
            if TF.NAME in cfg_aug.DISABLE:
                continue
            t = getattr(transforms, TF.NAME)
            if TF.ARGS:
                if isinstance(TF.ARGS, dict):
                    t = t(**TF.ARGS)
                else:
                    t = t(*tuple(TF.ARGS))
            else:
                t = t()
            transform_ls.append(t)
        transform_ls += [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ImgTransform = transforms.Compose(transform_ls)
        self.transforms_ls = transform_ls
        trainset = CSVLoader(
            data_dir=dataset_dir, mode=mode, image_size=img_sz, transform=ImgTransform,
            episode_num=episode_train_num, way_num=way_num, shot_num=shot_num, query_num=query_num
        )
        valset = CSVLoader(
            data_dir=dataset_dir, mode='val', image_size=img_sz, transform=ImgTransform,
            episode_num=episode_val_num, way_num=way_num, shot_num=shot_num, query_num=query_num
        )
        testset = CSVLoader(
            data_dir=dataset_dir, mode='test', image_size=img_sz, transform=ImgTransform,
            episode_num=episode_test_num, way_num=way_num, shot_num=shot_num, query_num=query_num
        )

        print('Trainset: %d' % len(trainset))
        print('Valset: %d' % len(valset))
        print('Testset: %d' % len(testset))
        print('Trainset: %d' % len(trainset), file=F_txt)
        print('Valset: %d' % len(valset), file=F_txt)
        print('Testset: %d' % len(testset), file=F_txt)

        # ========================================== Load Datasets =========================================
        workers = self.params.workers
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
