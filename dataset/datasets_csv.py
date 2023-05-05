import csv
import dataclasses
import os
import os.path as path
import random
import sys
from typing import List

import numpy as np
from PIL import Image
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataclasses import field

sys.dont_write_bytecode = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def gray_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def default_loader(path):
    return pil_loader(path)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


class CSVLoader(Dataset):
    """
       Imagefolder for miniImageNet--ravi, StanfordDog, StanfordCar and CubBird datasets.
       Images are stored in the folder of "images";
       Indexes are stored in the CSV files.
    """

    class Batch:
        # Light dataclass for storing episode data using __slots__ and compare=False
        __slots__ = ['query_images', 'query_targets', 'support_images', 'support_targets']

        def __init__(self, query_images: List[Tensor], query_targets: List[Tensor],
                     support_images: List[List[Tensor]], support_targets: List[Tensor]):
            self.query_images = query_images
            self.query_targets = query_targets
            self.support_images = support_images
            self.support_targets = support_targets

    def __init__(self, data_dir="", mode="train",
                 pre_process: T.Compose = None,
                 augmentations: List[nn.Module] = None,
                 post_process: T.Compose = None,
                 loader=default_loader,
                 _gray_loader=gray_loader,
                 episode_num=1000, way_num=5, shot_num=5, query_num=5, av_num=1):

        super(CSVLoader, self).__init__()

        # set the paths of the csv files
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')
        data_map = {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv
        }
        data_list = []
        e = 0

        # store all the classes and images into a dict
        class_img_dict = {}
        with open(data_map[mode]) as f_csv:
            f_train = csv.reader(f_csv, delimiter=',')
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                img_name, img_class = row

                if img_class in class_img_dict:
                    class_img_dict[img_class].append(img_name)
                else:
                    class_img_dict[img_class] = []
                    class_img_dict[img_class].append(img_name)
        f_csv.close()
        class_list = class_img_dict.keys()

        while e < episode_num:

            # construct each episode
            episode = []
            e += 1
            temp_list = random.sample(class_list, way_num)
            label_num = -1

            for item in temp_list:
                label_num += 1
                imgs_set = class_img_dict[item]
                support_imgs = random.sample(imgs_set, shot_num)
                query_imgs = [val for val in imgs_set if val not in support_imgs]

                if query_num < len(query_imgs):
                    query_imgs = random.sample(query_imgs, query_num)

                # the dir of support set
                query_dir = [path.join(data_dir, 'images', i) for i in query_imgs]
                support_dir = [path.join(data_dir, 'images', i) for i in support_imgs]

                data_files = {
                    "query_img": query_dir,
                    "support_set": support_dir,
                    "target": label_num
                }
                episode.append(data_files)
            data_list.append(episode)

        self.data_list = data_list
        self.pre_process = pre_process
        self.post_process = post_process
        self.augmentations = augmentations
        self.av_num = av_num
        self.loader = loader
        self.gray_loader = _gray_loader

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query
        '''
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for i in range(len(episode_files)):
            data_files = episode_files[i]

            # load query images
            query_dir = data_files['query_img']

            # Randomly select a subset of augmentations to apply per episode
            augment = [None]
            if self.augmentations is not None:
                augment = [T.Compose(random.sample(self.augmentations, self.av_num)) for _ in range(self.av_num)]

            for j in range(len(query_dir)):
                temp_img = self.loader(query_dir[j])
                # Process the image
                temp_img = [self._process_img(aug, temp_img) for aug in augment]
                query_images += temp_img

            # load support images
            temp_support = []
            support_dir = data_files['support_set']

            for j in range(len(support_dir)):
                temp_img = self.loader(support_dir[j])

                # Process the image
                temp_img = self._process_img(augment[0], temp_img)
                temp_support.append(temp_img)

            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        return query_images, query_targets, support_images, support_targets

    def _process_img(self, augment, temp_img):
        if self.pre_process is not None:
            temp_img = self.pre_process(temp_img)
        # Normalization
        if augment is not None:
            temp_img = augment(temp_img)
        # Post-process
        if self.post_process is not None:
            temp_img = self.post_process(temp_img)
        return temp_img
