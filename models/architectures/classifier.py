from collections import OrderedDict
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader

from models import backbones, classifiers, dt_heads
from models.dt_heads.dtree import DTree
from models.model_utils.utils import load_config, DataHolder


class ClassifierModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg  # main cfg! (architecture cfg)
        self.num_classes = model_cfg.NUM_CLASSES
        self.k_neighbors = model_cfg.K_NEIGHBORS
        self.module_topology = ['backbone2d', 'knn_head', 'dt_head']
        self.data = DataHolder(model_cfg)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build(self):
        for module_name in self.module_topology:
            module = getattr(self, '_build_%s' % module_name)()
            self.add_module(module_name, module)
        self.backbone2d.cuda()

    def forward(self):
        raise NotImplementedError

    def _build_backbone2d(self):
        if self.model_cfg.get("BACKBONE_2D", None) is None:
            raise ValueError('Missing specification of backbone to use')
        m = backbones.__all__[self.model_cfg.BACKBONE_2D]()
        self.data.module_list.append(m)
        return m

    def _build_knn_head(self):
        if self.model_cfg.get("KNN", None) is None:
            return None
        m = classifiers.__all__[self.model_cfg.KNN](self.k_neighbors)
        self.data.module_list.append(m)
        return m

    def _build_dt_head(self):
        if self.model_cfg.get("DT", None) is None:
            return None
        m = dt_heads.__all__[self.model_cfg.DT](self.num_classes)
        self.data.module_list.append(m)
        return m

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        self.backbone2d.load_state_dict(state_dict=state_dict)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.backbone2d.parameters(recurse)

    def train(self, **kwargs):
        self.backbone2d.train(**kwargs)
        self.knn_head.train(**kwargs)

    def eval(self, **kwargs):
        self.backbone2d.eval(**kwargs)
        self.knn_head.eval(**kwargs)

    def fit_tree_episodes(self, train_set: DataLoader):
        """

        :param train_set:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        # create empty dataframe
        batch_sz = self.model_cfg.BATCH_SIZE
        dt_head: DTree = self.dt_head
        X_len = len(train_set) * batch_sz
        col_len = self.num_classes
        cls_labels = [f"cls_{i}" for i in range(0, self.num_classes)]
        tree_df = DataFrame(np.zeros(shape=(X_len, col_len), dtype=float),
                            columns=cls_labels)
        tree_df["y"] = pd.Series(np.zeros(shape=X_len), dtype=int)
        ix = 0

        print("--- Beginning inference step for tree fitting ---")
        print(tree_df.info(verbose=True))
        self.eval()
        cls_col_ix = tree_df.columns.get_indexer(cls_labels)
        y_col_ix = tree_df.columns.get_indexer(["y"])
        with torch.no_grad():
            for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_set):

                print("> Running episode: ", episode_index)
                # Convert query and support images
                query_images = torch.cat(query_images, 0)
                input_var1 = query_images.cuda()

                input_var2 = []
                for i in range(len(support_images)):
                    temp_support = support_images[i]
                    temp_support = torch.cat(temp_support, 0)
                    temp_support = temp_support.cuda()
                    input_var2.append(temp_support)
                target = torch.cat(query_targets, 0)
                # target = target.cuda()
                self.data.q_in, self.data.S_in = input_var1, input_var2
                # Calculate the output
                output = self.forward()
                output = np.array([dt_head.normalize(r) for r in output.cpu()])
                tree_df.iloc[ix:ix + batch_sz, cls_col_ix] = output
                tree_df.iloc[ix:ix + batch_sz, y_col_ix] = target.numpy()
                ix += batch_sz
            # add measurements and target value to dataframe
        self.data.X = tree_df[cls_labels]
        self.data.y = tree_df[['y']]
        print("Finished inference, fitting tree...")
        print(tree_df.head(5))
        print(tree_df.tail(5))
        dt_head.fit(self.data.X, self.data.y, self.data.eval_set)

    def get_tree(self, module_name)->DTree:
        return self.dt_head

    @staticmethod
    def load_config(path):
        return load_config(path)
