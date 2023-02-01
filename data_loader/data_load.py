import dataclasses
import os

import torch
from torch import nn

from dataset.datasets_csv import CSVLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


@dataclasses.dataclass
class Parameters:
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

    def __init__(self, cfg, params: Parameters) -> None:
        self.transforms_ls = []
        self.params = params
        self.cfg = cfg

    def augment(self, transform: nn.Module):
        self.transforms_ls = [transform] + self.transforms_ls

    def load_data(self, mode, F_txt):
        # ======================================= Folder of Datasets =======================================
        # image transform & normalization
        dataset_dir = self.params.dataset_dir
        img_sz = self.params.img_sz
        dataset_dir = dataset_dir
        shot_num = self.params.shot_num
        way_num = self.params.way_num
        query_num = self.params.query_num
        episode_train_num = self.params.episode_train_num
        episode_val_num = self.params.episode_val_num
        episode_test_num = self.params.episode_test_num
        transform_ls = []

        for TF in self.cfg.TRANSFORMS:
            if not TF.ENABLE:
                continue
            t = getattr(transforms, TF.NAME)
            if TF.ARGS:
                if isinstance(TF.ARGS, dict):
                    t = t(**TF.ARGS)
                else:
                    t = t(*tuple(TF.ARGS))
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
