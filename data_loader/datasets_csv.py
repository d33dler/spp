import csv
import os
import os.path as path
import random
import sys
from functools import lru_cache
from typing import List, Union
import numpy as np
import torch
from PIL import Image
from easydict import EasyDict
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from models.utilities.utils import identity, DataHolder

sys.dont_write_bytecode = True

maxsize = int(os.getenv("LRU_CACHE_SIZE", 1000))  # Default value is 1000 if the environment variable is not set


@lru_cache(maxsize=maxsize)
def pil_loader(_path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


@lru_cache(maxsize=maxsize)
def gray_loader(_path):
    with open(_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


class BatchFactory(Dataset):
    """
       Imagefolder for miniImageNet--ravi, StanfordDog, StanfordCar and CubBird datasets.
       Images are stored in the folder of "images";
       Indexes are stored in the CSV files.
    """

    class AbstractBuilder:
        def build(self):
            raise NotImplementedError("build() not implemented")

        def get_item(self, index):
            raise NotImplementedError("get_item() not implemented")

    # TODO move parameters to a parameter class
    def __init__(self,
                 builder: Union[str, AbstractBuilder] = "image_to_class",
                 data_dir="",
                 mode="train",
                 pre_process: T.Compose = None,
                 augmentations: List[nn.Module] = None,
                 post_process: T.Compose = None,
                 loader=None,
                 _gray_loader=None,
                 episode_num=10000,
                 way_num=5, shot_num=5, query_num=5, qav_num=None, sav_num=None, aug_num=None, strategy: str = None,
                 is_random_aug: bool = False,
                 train_class_num: int = 15
                 ):
        """
        :param builder: the builder to build the dataset
        :param data_dir: the root directory of the dataset
        :param mode: the mode of the dataset, ["train", "val", "test"]
        :param pre_process: the pre-process of the dataset
        :param augmentations: the augmentations of the dataset
        :param post_process: the post-process of the dataset
        :param loader: the loader of the dataset
        :param _gray_loader: the gray_loader of the dataset
        :param episode_num: the number of episodes
        :param way_num: the number of classes in one episode
        :param shot_num: the number of support samples in one class
        :param query_num: the number of query samples in one class
        :param qav_num: the number of augmentations for each sample
        :param aug_num: the number of augmentations for each sample
        :param strategy: the strategy of the dataset [None, '1:1', '1:N'], '1:1' = 1 AV vs 1 support class AV-subset,
         '1:N' - 1 query-AV vs all samples of a support class
        """
        super(BatchFactory, self).__init__()

        # set the paths of the csv files
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')
        data_map = {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv
        }
        builder_map = {
            "image_to_class": ImageToClassBuilder,
            "npair_mc": NPairMCBuilder
        }
        print(f"Batch construction: {builder or 'image_to_class'}")
        if isinstance(builder, str):
            builder = builder.lower()
            self.builder = builder_map[builder](self) if builder in builder_map else ImageToClassBuilder(self)
        else:
            self.builder: BatchFactory.AbstractBuilder = ImageToClassBuilder(self) if builder or not isinstance(
                builder, BatchFactory.AbstractBuilder) else builder
        data_list = []
        # store all the classes and images into a dict
        class_img_dict = {}
        with open(data_map[mode]) as f_csv:
            f_train = csv.reader(f_csv, delimiter=',')
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                img_name, img_class = row
                if img_class in class_img_dict:
                    class_img_dict[img_class].append(path.join(data_dir, 'images', img_name))
                else:
                    class_img_dict[img_class] = [path.join(data_dir, 'images', img_name)]

        class_list = list(class_img_dict.keys())
        self.episode_num = episode_num
        self.way_num = way_num
        self.train_class_num = train_class_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.class_list = class_list
        self.class_img_dict = class_img_dict
        self.data_list = data_list
        self.pre_process = identity if pre_process is None else pre_process
        self.post_process = identity if post_process is None else post_process
        self.augmentations = augmentations
        self.qav_num = qav_num
        self.sav_num = sav_num
        self.aug_num = aug_num
        self.loader = pil_loader if loader is None else loader
        self.gray_loader = gray_loader if _gray_loader is None else _gray_loader
        self.strategy = strategy
        self.mode = mode
        self.is_random_aug = is_random_aug
        self.use_augmentation = len({qav_num, sav_num, aug_num}.intersection({0, None})) == 0
        # Build the dataset
        self.builder.build()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.builder.get_item(index)

    def process_img(self, augment, temp_img):
        return self.post_process(augment(self.pre_process(temp_img)))


class ImageToClassBuilder(BatchFactory.AbstractBuilder):

    def __init__(self, factory: BatchFactory):
        self.factory = factory

    def build(self):
        # assign all values from self
        factory = self.factory
        episode_num = factory.episode_num
        way_num = factory.way_num
        shot_num = factory.shot_num
        query_num = factory.query_num
        class_list = factory.class_list
        class_img_dict = factory.class_img_dict
        data_list = factory.data_list
        for _ in range(episode_num):

            # construct each episode
            episode = []
            temp_list = random.sample(class_list, way_num)

            for cls, item in enumerate(temp_list):  # for each class
                imgs_set = class_img_dict[item]
                random.shuffle(imgs_set)  # shuffle the images

                # split the images into support and query sets
                support_imgs = imgs_set[:shot_num]
                query_imgs = imgs_set[shot_num:shot_num + query_num]

                cls_subset = {
                    "query_img": query_imgs,  # query_num - query images for `cls`, default(15)
                    "support_set": support_imgs,  # SHOT - support images for `cls`, default(5)
                    "target": cls
                }
                episode.append(cls_subset)  # (WAY, QUERY (query_num) + SHOT, 3, x, x)

            data_list.append(episode)
        if factory.use_augmentation:
            factory.sav_num += 1  # +1 for original sample
            factory.qav_num += 1
        else:
            factory.sav_num,  factory.qav_num = 1, 1
        print("===========================================")
        print(f"Augmentation : {'ON' if factory.use_augmentation else 'OFF'}")
        print(f"Query   AV ({factory.mode}) : ", factory.qav_num)
        print(f"Support AV ({factory.mode}) : ", factory.sav_num)
        print("===========================================")

    def get_item(self, index):
        """Load an episode each time, including C-way K-shot and Q-query"""
        factory = self.factory
        episode_files = factory.data_list[index]
        loader = factory.loader
        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for cls_subset in episode_files:
            # Randomly select a subset of augmentations to apply per episode
            if factory.use_augmentation:
                av_num = factory.qav_num - 1
                sav_num = factory.sav_num - 1
                Q_augment = [
                    T.Compose(random.sample(factory.augmentations, min(factory.aug_num, len(factory.augmentations)))
                              if factory.is_random_aug
                              else factory.augmentations[:factory.aug_num]) for _ in range(av_num)]
                Q_augment += [identity]  # introduce original sample as well
                S_augment = [
                    T.Compose(random.sample(factory.augmentations, min(factory.aug_num, len(factory.augmentations)))
                              if factory.is_random_aug
                              else factory.augmentations[:factory.aug_num]) for _ in range(sav_num)]
                S_augment += [identity]  # introduce original sample as well
            else:
                Q_augment = [identity]
                S_augment = [identity]

            # load QUERY images, use the cached loader function
            query_dir = cls_subset['query_img']
            temp_imgs = [Image.fromarray(loader(temp_img)) for temp_img in query_dir]
            query_images += [factory.process_img(aug, temp_img) for aug in Q_augment for temp_img in temp_imgs]

            # load SUPPORT images, use the cached loader function
            support_dir = cls_subset['support_set']
            temp_imgs = [Image.fromarray(loader(temp_img)) for temp_img in support_dir]
            if factory.strategy is None or factory.strategy == 'N:1':
                temp_support = [factory.process_img(aug, temp_img).unsqueeze(0) for aug in S_augment for
                                temp_img in temp_imgs]  # Use the cached loader function
                support_images.append(torch.cat(temp_support, 0))
            elif factory.strategy:
                for av in range(factory.qav_num):
                    support_images.append([factory.process_img(aug, img) for aug in S_augment for img in temp_imgs])

            # read the label
            target = cls_subset['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))
        return query_images, query_targets, support_images, support_targets


class NPairMCBuilder(BatchFactory.AbstractBuilder):

    def __init__(self, factory: BatchFactory):
        self.factory = factory
        self.val_builder = ImageToClassBuilder(factory)

    def get_item(self, index):
        """Load an episode each time, including C-way K-shot and Q-query"""
        # write to log file the factory mode

        if self.factory.mode != 'train':
            return self.val_builder.get_item(index)
        factory = self.factory
        episode_files = factory.data_list[index]
        loader = factory.loader
        query_images = []
        targets = []
        positives = []
        for cls_subset in episode_files:
            augment = [identity]
            # Randomly select a subset of augmentations to apply per episode
            if None not in [factory.qav_num, factory.aug_num]:
                augment = [
                    T.Compose(random.sample(factory.augmentations, min(factory.aug_num, len(factory.augmentations))))
                    for _ in range(factory.qav_num)]
                augment += [identity]  # introduce original sample as well

            # load query images
            temp_q = Image.fromarray(loader(cls_subset['q']))
            query_images += [factory.process_img(aug, temp_q) for aug in augment]  # Use the cached loader function

            # load support images

            temp_support = [Image.fromarray(loader(pos_img)) for pos_img in cls_subset['+']]
            if factory.mode == 'train' and factory.shot_num > 1:
                augment = [
                    T.Compose(random.sample(factory.augmentations, min(factory.aug_num, len(factory.augmentations)))
                              if factory.is_random_aug else factory.augmentations[:factory.aug_num])]
            positives += [factory.process_img(aug, temp_img) for aug in augment for temp_img in
                          temp_support]  # Use the cached loader function

            # read the label
            targets.append(cls_subset['target'])
        return query_images, positives, targets

    def build(self):
        # assign all values from self
        if self.factory.mode != 'train':
            self.val_builder.build()
            return
        builder = self.factory
        episode_num = builder.episode_num
        way_num = builder.way_num
        # shot_num = builder.shot_num
        # query_num = builder.query_num
        class_list = builder.class_list
        class_img_dict = builder.class_img_dict
        data_list = builder.data_list

        for _ in range(episode_num):
            # construct each episode
            episode = []
            temp_list = random.sample(class_list, self.factory.train_class_num)
            for cls, item in enumerate(temp_list):  # for each class
                imgs_set = class_img_dict[item]
                # split the images into support and query sets
                query, *positives = random.sample(imgs_set, 1 + self.factory.shot_num)
                cls_subset = {
                    "q": query,  # query_num - query images for `cls`, default(15)
                    "+": positives,
                    "target": cls
                }
                episode.append(cls_subset)  # (WAY, QUERY (query_num) + SHOT, 3, x, x)
            data_list.append(episode)
