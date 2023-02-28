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
from models import backbones, clustering, de_heads, necks
from models.de_heads.dengine import DecisionEngine
from models.de_heads.dtree import DTree
from models.interfaces.arch_module import ARCH
from models.utilities.utils import load_config, DataHolder, save_checkpoint, config_exchange
import torch.nn.functional as F


class DEModel(ARCH):
    """
    DEModel (Decision-Engine model)
    Provides model building & decision-engine fitting functionality.
    Can be used as a generic model as well.
    """
    arch = 'Missing'
    _DE: DecisionEngine
    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        self.loaders = None
        self.criterion = None
        self.optimizer = None
        model_cfg = self.root_cfg
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

    def _build_BACKBONE(self):
        if self.root_cfg.get("BACKBONE", None) is None:
            raise ValueError('Missing specification of backbone to use')
        m: ARCH.Child = backbones.__all__[self.root_cfg.BACKBONE.NAME](self.data)
        m.cuda() if self.root_cfg.BACKBONE.CUDA else False  # TODO may yield err?
        self.module_topology['BACKBONE'] = m
        self.data.module_list.append(m)
        return m

    def _build_ENCODER(self):  # ENCODER | _
        if self.root_cfg.get("ENCODER", None) is None:
            raise ValueError('Missing specification of encoder to use')
        m = necks.__all__[self.root_cfg.ENCODER.NAME]
        m = m(self.override_child_cfg(m.get_config(), "ENCODER"))
        m.cuda() if self.root_cfg.ENCODER.CUDA else False
        self.module_topology['ENCODER'] = m
        self.data.module_list.append(m)
        return m

    def _build_DE(self):
        if self.root_cfg.get('DE', None) is None:
            return None
        de: DecisionEngine = de_heads.__all__[self.root_cfg.DE.NAME]
        de = de(config_exchange(de.get_config(), self.root_cfg['DE']))
        de.cuda() if self.root_cfg.DE.CUDA else False
        self._DE = de
        return de

    def sub_parameters(self, network: str, recurse: bool = True) -> Iterator[Parameter]:
        sub_net = getattr(self, network, None)
        return sub_net.parameters(recurse)  # TODO handle encoder's subnets

    def verify_module(self, module):
        if not isinstance(module, ARCH.Child):
            raise ValueError(
                "[CFG_OVERRIDE] Cannot override child module config. Child module doesn't subclass ARCH.Child!")

    def train(self, training: bool = True) -> None:
        [sub_mod.train(training) for sub_mod in self.module_topology.values() if isinstance(sub_mod, ARCH.Child)]

    def run_epoch(self, output_file):
        raise NotImplementedError

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def save_model(self, filename=None):
        self.state.update({'DE': self._DE.dump()})
        super().save_model(filename)

    def load_model(self, path, txt_file=None):
        checkpoint = super().load_model(path, txt_file)
        if 'DE' in checkpoint and 'DE' in self.root_cfg:
            self._DE.load(checkpoint['DE'])

    def enable_decision_engine(self, train_set: TorchDataLoader = None, refit=False):
        """
        Enables the DE for inference calls (otherwise DE is not used).

        :param train_set: data loader hosting same data used for training the model
        :type train_set: TorchDataLoader
        :param refit: force re-fit a model that already contains a fitted DE
        :type refit: bool
        :return: None
        """
        if 'DE' not in self.root_cfg:
            raise AttributeError("Not able to enable decision engine - Missing DE module specification in config!")
        dengine = self._DE
        if not isinstance(dengine, DecisionEngine):
            raise ValueError(f"Wrong class type for Decision-engine module, expected DecisionEngine(ABC, ARCH.Child), "
                             f"got: : {type(dengine)}")
        if not dengine.is_fit or refit:
            self._fit_DE(train_set)
        dengine.enabled = True

    def _fit_DE(self, train_set):
        """
        Identify DE type and run the appropriate fitting function
        :param train_set:
        :type train_set:
        :return:
        :rtype:
        """
        if self.root_cfg.DE.ENGINE == "TREE":
            self._fit_tree_episodes(train_set)
        else:
            raise ValueError(f"Decision Engine type not supported from choices [TREE] , got: {self.root_cfg.DE.ENGINE}")

        self.save_model()

    def _fit_tree_episodes(self, train_set: TorchDataLoader):  # TODO move to DTree?
        """
        Create the inference dataset containing the feature set F_T for the DE and fit the DE.
        :param train_set: data loader hosting same data used for training the model
        :return: None
        """
        # create empty dataframe
        batch_sz = self.root_cfg.BATCH_SIZE
        dt: DTree = self._DE
        X_len = self.root_cfg.DE.EPISODE_TRAIN_NUM * batch_sz
        ft_engine = ['max', 'mean', 'std']
        cls_labels = [f"cls_{i}" for i in range(0, self.num_classes)]
        ranks_len = dt.ranks
        ranks = [f"rank_{i}" for i in range(0, ranks_len)]
        sim_topK = [f"DLD_sim_{i}" for i in range(0, self.num_classes * self.k_neighbors)]
        all_columns = cls_labels + ft_engine + ranks + sim_topK
        col_len = len(all_columns)
        # Add column IDs for the DE (required by DE for creating the inputs in inference step)
        dt.features[dt.ALL_FT] = all_columns
        dt.features[dt.BASE_FT] = cls_labels
        dt.features[dt.MISC_FT] = sim_topK
        dt.features[dt.RANK_FT] = ranks
        if self.root_cfg.DE.DATASET is None:
            tree_df = DataFrame(np.zeros(shape=(X_len, col_len), dtype=float), columns=all_columns)
            tree_df["y"] = pd.Series(np.zeros(shape=X_len), dtype=int)
            ix = 0

            print("--- Beginning training dataset creation for DE ---")
            print(tree_df.info(verbose=True))
            print("TRAIN SET SIZE: ", len(train_set))
            self.eval()
            cls_col_ix = tree_df.columns.get_indexer(cls_labels)
            DLD_topK = tree_df.columns.get_indexer(sim_topK)
            ranks_ix = tree_df.columns.get_indexer(ranks)
            max_ix = tree_df.columns.get_loc('max')
            std_ix = tree_df.columns.get_loc('std')
            mean_ix = tree_df.columns.get_loc('mean')

            y_col_ix = tree_df.columns.get_indexer(["y"])
            out_bank = np.empty(shape=(0, self.num_classes))
            with torch.no_grad():
                for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(
                        train_set):
                    if episode_index % 100 == 0:
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
                        [dt.normalize(x) for x in self.data.sim_list_BACKBONE2D.detach().cpu().numpy()])
                    out_bank = np.concatenate([out_bank, out], axis=0)
                    target: np.ndarray = target.numpy()
                    # add measurements and target value to dataframe
                    out_rows = out.shape[0]
                    tree_df.iloc[ix:ix + out_rows, cls_col_ix] = out
                    tree_df.iloc[ix:ix + out_rows, y_col_ix] = target
                    tree_df.iloc[ix:ix + out_rows, max_ix] = out.max(axis=1)
                    tree_df.iloc[ix:ix + out_rows, mean_ix] = out.mean(axis=1)
                    tree_df.iloc[ix:ix + out_rows, std_ix] = out.std(axis=1)

                    top_ranks = np.argpartition(-out, kth=ranks_len, axis=1)[:, :ranks_len]
                    tree_df.iloc[ix:ix + out_rows, ranks_ix] = out[np.arange(out.shape[0])[:, None], top_ranks]
                    tree_df.iloc[ix:ix + out_rows, DLD_topK] = self.data.DLD_topk

                    # tree_df.iloc[ix:ix + out_rows, deep_local_ix_S] = [sup_t.detach().cpu().view(sup_t.size()[0], -1) for sup_t in self.data.S_raw]
                    ix += out_rows
                    if episode_index == self.root_cfg.DE.EPISODE_TRAIN_NUM - 1:
                        break
                print("STD:", out_bank.std(axis=1).mean())

        else:
            tree_df = pd.read_csv(self.root_cfg.DE.DATASET, header=0)
        self.data.X = tree_df[all_columns]
        self.data.y = tree_df[['y']]
        print(self.data.X.head(5))
        print(self.data.X.tail(5))
        print("Finished inference, fitting tree...")
        if self.root_cfg.DE.DATASET is None:
            tree_df.to_csv(
                f"tree_dataset_W{self.root_cfg.WAY_NUM}_S{self.root_cfg.SHOT_NUM}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
                index=False)

        dt.fit(self.data.X, self.data.y, self.data.eval_set)

    def get_DEngine(self) -> DecisionEngine:
        return self._DE
