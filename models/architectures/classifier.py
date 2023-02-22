import os
from collections import OrderedDict
from datetime import datetime
from typing import Iterator, List, Dict

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.decomposition import KernelPCA, PCA
from torch import nn, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader as TorchDataLoader

from data_loader.data_load import Parameters, DatasetLoader
from models import backbones, clustering, dt_heads, necks
from models.dt_heads.dtree import DTree
from models.interfaces.arch_module import ArchM
from models.utilities.utils import load_config, DataHolder, save_checkpoint
import torch.nn.functional as F


class ClassifierModel(ArchM):
    arch = 'Missing'

    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        self.loaders = None
        self.criterion = None
        self.optimizer = None
        model_cfg = self.model_cfg
        self.num_classes = model_cfg.NUM_CLASSES
        self.k_neighbors = model_cfg.K_NEIGHBORS
        self.data = DataHolder(model_cfg)
        self.build()
        self._set_modules_mode()

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def load_data(self, mode, output_file, dataset_dir=None):
        self.loaders = self.data_loader.load_data(mode, output_file, dataset_dir)

    def build(self):
        for module_name in self.module_topology.keys():
            module = getattr(self, '_build_%s' % module_name)()
            self.add_module(module_name, module)

    def forward(self):
        raise NotImplementedError

    def _build_BACKBONE_2D(self):
        if self.model_cfg.get("BACKBONE_2D", None) is None:
            raise ValueError('Missing specification of backbone to use')
        m = backbones.__all__[self.model_cfg.BACKBONE_2D.NAME](self.data)
        m.cuda() if self.model_cfg.BACKBONE_2D.CUDA else False  # TODO may yield err?
        self.module_topology['BACKBONE_2D'] = m
        self.data.module_list.append(m)
        return m

    def _build_ENCODER(self):  # ENCODER | _
        if self.model_cfg.get("ENCODER", None) is None:
            raise ValueError('Missing specification of encoder to use')
        m = necks.__all__[self.model_cfg.ENCODER.NAME](self.data)
        m.cuda() if self.model_cfg.ENCODER.CUDA else False
        self.module_topology['ENCODER'] = m
        self.data.module_list.append(m)
        return m

    def _build_DT(self):
        if self.model_cfg.get("DT", None) is None:
            return None
        m = dt_heads.__all__[self.model_cfg.DT.NAME](self.data)
        m.cuda() if self.model_cfg.DT.CUDA else False
        self.module_topology['DT'] = m
        self.data.module_list.append(m)
        return m

    def sub_parameters(self, network: str, recurse: bool = True) -> Iterator[Parameter]:
        sub_net = getattr(self, network, None)
        return sub_net.parameters(recurse)  # TODO handle encoder's subnets

    def train(self, training: bool = True) -> None:
        [sub_mod.train(training) for sub_mod in self.module_topology.values() if isinstance(sub_mod, ArchM.Child)]

    def run_epoch(self, output_file):
        raise NotImplementedError

    def set_loss(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def fit_tree_episodes(self, train_set: TorchDataLoader):
        """

        :param train_set:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        # create empty dataframe
        batch_sz = self.model_cfg.BATCH_SIZE
        dt_head: DTree = self.DT
        X_len = self.model_cfg.DT.EPISODE_TRAIN_NUM* batch_sz
        ft_engine = ['max', 'mean', 'std']
        cls_labels = [f"cls_{i}" for i in range(0, self.num_classes)]
        ranks = [f"rank_{i}" for i in range(0, self.k_neighbors)]
        sim_topK = [f"DLD_sim_{i}" for i in range(0, self.num_classes * self.k_neighbors)]
        all_columns = cls_labels + ft_engine + ranks + sim_topK
        col_len = len(all_columns)
        dt_head.all_features = all_columns
        dt_head.base_features = cls_labels
        dt_head.deep_features = sim_topK
        dt_head.ranking_features = ranks
        if self.model_cfg.DATASET is None:
            tree_df = DataFrame(np.zeros(shape=(X_len, col_len), dtype=float), columns=all_columns)
            tree_df["y"] = pd.Series(np.zeros(shape=X_len), dtype=int)
            ix = 0

            print("--- Beginning inference step for tree fitting ---")
            print(tree_df.info(verbose=True))
            print("TRAIN SET SIZE: ", len(train_set))
            self.eval()
            cls_col_ix = tree_df.columns.get_indexer(cls_labels)
            DLD_topK = tree_df.columns.get_indexer(sim_topK)
            ranks_ix = tree_df.columns.get_indexer(ranks)
            max_ix = tree_df.columns.get_loc('max')
            # min_ix = tree_df.columns.get_loc('min')
            std_ix = tree_df.columns.get_loc('std')
            mean_ix = tree_df.columns.get_loc('mean')

            dt_head.max_ix = max_ix
            dt_head.mean_ix = mean_ix
            dt_head.std_ix = std_ix
            dt_head.ranks_ix = ranks_ix

            y_col_ix = tree_df.columns.get_indexer(["y"])
            out_bank = np.empty(shape=(0, 5))
            with torch.no_grad():
                for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(
                        train_set):
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
                    target: Tensor = torch.cat(query_targets, 0)

                    self.data.q_in, self.data.S_in = input_var1, input_var2
                    # Obtain and normalize the output
                    self.forward()
                    out = np.asarray(
                        [dt_head.normalize(x) for x in self.data.sim_list_BACKBONE2D.detach().cpu().numpy()])
                    out_bank = np.concatenate([out_bank, out], axis=0)
                    target: np.ndarray = target.numpy()
                    # add measurements and target value to dataframe
                    out_rows = out.shape[0]
                    tree_df.iloc[ix:ix + out_rows, cls_col_ix] = out
                    tree_df.iloc[ix:ix + out_rows, y_col_ix] = target
                    tree_df.iloc[ix:ix + out_rows, max_ix] = out.max(axis=1)
                    tree_df.iloc[ix:ix + out_rows, mean_ix] = out.mean(axis=1)
                    tree_df.iloc[ix:ix + out_rows, std_ix] = out.std(axis=1)

                    top_ranks = np.argpartition(-out, kth=self.k_neighbors, axis=1)[:, :self.k_neighbors]
                    tree_df.iloc[ix:ix + out_rows, ranks_ix] = out[np.arange(out.shape[0])[:, None], top_ranks]
                    tree_df.iloc[ix:ix + out_rows, DLD_topK] = self.data.DLD_topk

                    # tree_df.iloc[ix:ix + out_rows, deep_local_ix_S] = [sup_t.detach().cpu().view(sup_t.size()[0], -1) for sup_t in self.data.S_raw]
                    ix += out_rows
                    if episode_index == self.model_cfg.DT.EPISODE_TRAIN_NUM - 1: break
                print(out_bank.std(axis=1).mean())

        else:
            tree_df = pd.read_csv(self.model_cfg.DATASET, header=0)
        self.data.X = tree_df[all_columns]
        self.data.y = tree_df[['y']]
        print("Finished inference, fitting tree...")
        print(self.data.X.head(5))
        print(self.data.X.tail(5))
        if self.model_cfg.DATASET is None:
            tree_df.to_csv(f"tree_dataset{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv", index=False)

        dt_head.fit(self.data.X, self.data.y, self.data.eval_set)

    def feature_engine(self, tree_df: DataFrame, base_features: List[str], deep_features_Q: List[str]):
        base_features_ix = tree_df.index.get_indexer(base_features)

        # DEEP LOCAL DESCRIPTORS
        # queries_dld = self.normalizer(self.data.q.cpu()).detach().numpy().reshape(len(tree_df), 64 * 21 * 21)
        # tree_df[deep_features_Q] = self.pca_n.fit_transform(queries_dld)

        # MIN MAX MEAN STD
        # tree_df['min'] = tree_df.iloc[:, base_features_ix].min(axis=1)
        tree_df['max'] = tree_df.iloc[:, base_features_ix].max(axis=1)
        tree_df['mean'] = tree_df.iloc[:, base_features_ix].mean(axis=1)
        tree_df['std'] = tree_df.iloc[:, base_features_ix].std(axis=1)

        # RANKS
        cls_vals = tree_df.iloc[:, base_features_ix].to_numpy()
        top_k = np.argpartition(-cls_vals, kth=self.k_neighbors, axis=1)[:, :self.k_neighbors]
        tree_df[self.DT.ranking_features] = cls_vals[np.arange(cls_vals.shape[0])[:, None], top_k]

        tree_df[deep_features_Q] = self.data.DLD_topk.detach().numpy()
        return tree_df

    def get_tree(self, module_name) -> DTree:
        return self.DT
