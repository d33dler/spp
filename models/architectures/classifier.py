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
from models import backbones, clustering, dt_heads
from models.dt_heads.dtree import DTree
from models.utilities.utils import load_config, DataHolder
import torch.nn.functional as F


class ClassifierModel(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        self.criterion = None
        self.optimizer = None
        self.model_cfg = model_cfg  # main cfg! (architecture cfg)
        self.num_classes = model_cfg.NUM_CLASSES
        self.k_neighbors = model_cfg.K_NEIGHBORS
        self.module_topology: Dict[str, nn.Module] = {_: None for _ in model_cfg.TOPOLOGY}
        self.data = DataHolder(model_cfg)
        self.normalizer = torch.nn.BatchNorm2d(64)
        c = model_cfg
        p = Parameters(c.IMG_SIZE, c.DATASET_DIR, c.SHOT_NUM, c.WAY_NUM, c.QUERY_NUM, c.EPISODE_TRAIN_NUM,
                       c.EPISODE_TEST_NUM, c.EPISODE_VAL_NUM, c.OUTF, c.WORKERS, c.EPISODE_SIZE,
                       c.TEST_EPISODE_SIZE)
        self.data_loader = DatasetLoader(p)
        self._set_mode()

    def _set_mode(self):
        for k, m in self.module_topology.items():
            m.train(self.model_cfg[k].MODE == 'TRAIN')

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def load_data(self, mode, output_file):
        raise NotImplementedError

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

    def _build_ENCODER(self):  # encoder | _
        if self.model_cfg.get("ENCODER", None) is None:
            raise ValueError('Missing specification of encoder to use')
        m = backbones.__all__[self.model_cfg.ENCODER.NAME](self.data)
        m.cuda() if self.model_cfg.ENCODER.CUDA else False
        self.module_topology['neck'] = m
        self.data.module_list.append(m)
        return m

    def _build_KNN(self):
        if self.model_cfg.get("KNN", None) is None:
            return None
        m = clustering.__all__[self.model_cfg.KNN.NAME](self.data)
        m.cuda() if self.model_cfg.KNN.CUDA else False
        self.module_topology['KNN'] = m
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

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        self.backbone2d.load_state_dict(state_dict=state_dict)  # TODO add load_state dict neck

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.backbone2d.parameters(recurse)

    def train(self, **kwargs):
        self.backbone2d.train(**kwargs)
        self.knn_head.train(**kwargs)

    def run_epoch(self, epoch_index, output_file):
        raise NotImplementedError

    def eval(self, **kwargs):
        self.backbone2d.eval(**kwargs)
        self.knn_head.eval(**kwargs)

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
        dt_head: DTree = self.dt_head
        X_len = len(train_set) * batch_sz
        ft_engine = ['max', 'mean', 'std']
        cls_labels = [f"cls_{i}" for i in range(0, self.num_classes)]
        ranks = [f"rank_{i}" for i in range(0, self.k_neighbors)]
        deep_local_lbs_Q = [f"fQ_{i}" for i in range(0, 16)]
        deep_local_lbs_S = []
        for r in ranks:
            deep_local_lbs_S += [f"{r}_fS{i}" for i in range(0, 16)]
        all_columns = cls_labels + ft_engine + ranks
        col_len = len(all_columns)
        dt_head.all_features = all_columns
        dt_head.base_features = cls_labels
        # dt_head.deep_features = deep_local_lbs_Q
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
            deep_local_ix_Q = tree_df.columns.get_indexer(deep_local_lbs_Q)
            deep_local_ix_S = tree_df.columns.get_indexer(deep_local_lbs_S)
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
            pca_n = PCA(n_components=16)
            self.pca_n = pca_n
            normalizer = torch.nn.BatchNorm2d(64)

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
                    output = self.forward()
                    output = np.asarray([dt_head.normalize(x) for x in output.detach().cpu().numpy()])
                    target: np.ndarray = target.numpy()
                    # add measurements and target value to dataframe
                    tree_df.iloc[ix:ix + batch_sz, cls_col_ix] = output
                    tree_df.iloc[ix:ix + batch_sz, y_col_ix] = target
                    tree_df.iloc[ix:ix + batch_sz, max_ix] = output.max(axis=1)
                    tree_df.iloc[ix:ix + batch_sz, mean_ix] = output.mean(axis=1)
                    tree_df.iloc[ix:ix + batch_sz, std_ix] = output.std(axis=1)
                    top_k = np.argpartition(-output, kth=self.k_neighbors, axis=1)[:, :self.k_neighbors]
                    tree_df.iloc[ix:ix + batch_sz, ranks_ix] = output[np.arange(output.shape[0])[:, None], top_k]
                    # queries_norm = normalizer(self.data.q.cpu()).numpy().reshape(batch_sz, 64 * 21 * 21)
                    # tree_df.iloc[ix:ix + batch_sz, deep_local_ix_Q] = pca_n.fit_transform(queries_norm)
                    ix += batch_sz

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
        tree_df[self.dt_head.ranking_features] = cls_vals[np.arange(cls_vals.shape[0])[:, None], top_k]
        return tree_df

    def get_tree(self, module_name) -> DTree:
        return self.dt_head

    @staticmethod
    def load_config(path):
        return load_config(path)

    def adjust_learning_rate(self, epoch_num):
        """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
        lr = self.model_cfg.LEARNING_RATE * (0.05 ** (epoch_num // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
