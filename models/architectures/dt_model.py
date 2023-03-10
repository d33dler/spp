import os
from collections import OrderedDict
from datetime import datetime
from typing import Iterator, List, Dict
from skimage.color import rgb2hsv
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pandas.io.formats.style import color
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader as TorchDataLoader

from data_loader.data_load import Parameters, DatasetLoader
from dataset.datasets_csv import CSVLoader
from models import backbones, clustering, de_heads, necks
from models.de_heads.dengine import DecisionEngine
from models.de_heads.dtree import DTree
from models.interfaces.arch_module import ARCH
from models.utilities.utils import load_config, DataHolder, save_checkpoint, config_exchange, create_confusion_matrix
import torch.nn.functional as F


class DEModel(ARCH):
    """
    DEModel (Decision-Engine model)
    Provides model building & decision-engine fitting functionality.
    Can be used as a generic model as well.
    """
    arch = 'Missing'
    DE: DecisionEngine

    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        self.loaders = None
        self.criterion = None
        self.optimizer = None
        model_cfg = self.root_cfg
        self.num_classes = model_cfg.WAY_NUM
        self.k_neighbors = model_cfg.K_NEIGHBORS
        self.data = DataHolder(model_cfg)
        self.build()
        self._set_modules_mode()

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def load_data(self, mode, output_file, dataset_dir=None):
        self.loaders = self.data_loader.load_data(mode, dataset_dir, output_file)

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
        self.module_topology['DE'] = de
        de.cuda() if self.root_cfg.DE.CUDA else False
        self.DE = de
        return de

    def sub_parameters(self, network: str, recurse: bool = True) -> Iterator[Parameter]:
        sub_net = getattr(self, network, None)
        return sub_net.parameters(recurse)

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
        if self.DE.is_fit:
            self.state.update({'DE': self.DE.dump()})
        super().save_model(filename)

    def load_model(self, path, txt_file=None):
        checkpoint = super().load_model(path, txt_file)
        if 'DE' in checkpoint and 'DE' in self.root_cfg:
            self.DE.load(checkpoint['DE'])

    def enable_decision_engine(self, refit=False, filename=None):
        """
        Enables the DE for inference calls (otherwise DE is not used).

        :param filename:
        :param train_set: data loader hosting same data used for training the model
        :type train_set: TorchDataLoader
        :param refit: force re-fit a model that already contains a fitted DE
        :type refit: bool
        :return: None
        """
        if 'DE' not in self.root_cfg:
            raise AttributeError("Not able to enable decision engine - Missing DE module specification in config!")
        dengine = self.DE
        if not isinstance(dengine, DecisionEngine):
            raise ValueError(f"Wrong class type for Decision-engine module, expected DecisionEngine(ABC, ARCH.Child), "
                             f"got: : {type(dengine)}")
        if not dengine.is_fit or refit:
            self._fit_DE()
            self.save_model(filename)
        dengine.enabled = True

    def _fit_DE(self):
        """
        Identify DE type and run the appropriate fitting function
        :return:
        :rtype:
        """
        if self.root_cfg.DE.ENGINE == "TREE":
            self._fit_tree_episodes()
        else:
            raise ValueError(f"Decision Engine type not supported from choices [TREE] , got: {self.root_cfg.DE.ENGINE}")

    def _create_df(self, dataset, length):
        dt = self.DE
        bins = dt.bins
        # to copilot: rewrite this the other way around
        all_columns = dt.features[dt.ALL_FT]
        cos_sim = dt.features[dt.BASE_FT]
        col_hist = dt.features[dt.MISC_FT]
        ranks = dt.features[dt.RANK_FT]

        scaler = StandardScaler()
        tree_df = DataFrame(np.zeros(shape=(length, len(all_columns)), dtype=float), columns=all_columns)
        tree_df["y"] = 0
        tree_df["y"] = tree_df["y"].astype(int)

        print("--- Beginning training dataset creation for DE ---")
        print(tree_df.info(verbose=True))
        print("TRAIN SET SIZE: ", len(dataset))

        cls = tree_df.columns.get_indexer(cos_sim)
        histix = tree_df.columns.get_indexer(col_hist)
        ranks_ix = tree_df.columns.get_indexer(ranks)
        max_ix = tree_df.columns.get_loc('max')
        std_ix = tree_df.columns.get_loc('std')
        mean_ix = tree_df.columns.get_loc('mean')

        y_col_ix = tree_df.columns.get_indexer(["y"])
        out_bank = np.empty(shape=(0, self.num_classes))
        target_bank = np.empty(shape=0, dtype=int)

        ix = 0
        self.eval()
        try:
            with torch.no_grad():
                for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(
                        dataset):
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
                    out = self.forward()
                    if isinstance(out, torch.Tensor):
                        out = out.detach().cpu().numpy()
                    out = dt.normalize(out)
                    target: np.ndarray = target.numpy()
                    out_bank = np.concatenate([out_bank, out], axis=0)
                    target_bank = np.concatenate([target_bank, target], axis=0)
                    # add measurements and target value to dataframe

                    out_rows = out.shape[0]
                    rank_indices = np.argsort(-out, axis=1)

                    tree_df.iloc[ix:ix + out_rows, cls] = out
                    tree_df.iloc[ix:ix + out_rows, y_col_ix] = target

                    tree_df.iloc[ix:ix + out_rows, max_ix] = out.max(axis=1)
                    tree_df.iloc[ix:ix + out_rows, mean_ix] = out.mean(axis=1)
                    tree_df.iloc[ix:ix + out_rows, std_ix] = out.std(axis=1)

                    tree_df.iloc[ix:ix + out_rows, ranks_ix] = rank_indices

                    # assume image_tensor is a torch tensor of shape [50, 3, 100, 100]
                    batch_size, num_channels, height, width = query_images.shape
                    # reshape the tensor to [50, 3, 10000] to simplify computation
                    image_tensor_reshaped = query_images.view(batch_size, num_channels, height * width)
                    # convert the tensor to numpy array
                    image_array = image_tensor_reshaped.numpy()
                    # convert from channel-first to channel-last format
                    image_array = np.transpose(image_array, (0, 2, 1))
                    # convert the images from RGB to HSV color space
                    image_array = rgb2hsv(image_array)
                    # extract color histogram features for each image
                    histograms = []
                    for i in range(batch_size):
                        hist_r, _ = np.histogram(image_array[i, :, 0], bins=bins, range=(0, 1))
                        hist_g, _ = np.histogram(image_array[i, :, 1], bins=bins, range=(0, 1))
                        hist_b, _ = np.histogram(image_array[i, :, 2], bins=bins, range=(0, 1))
                        histograms.append(np.concatenate([hist_r, hist_g, hist_b]))

                    # scale the features
                    histograms = scaler.fit_transform(histograms)
                    # convert the list of histograms to a pandas DataFrame
                    tree_df.iloc[ix:ix + out_rows, histix] = histograms
                    ix += out_rows
                    if ix == length:
                        break
        except ValueError:
            tree_df.to_csv(
                f"{self.root_cfg.DATASET}_{self.root_cfg.NAME}_W{self.root_cfg.WAY_NUM}_"
                f"S{self.root_cfg.SHOT_NUM}K_{self.k_neighbors}_"
                f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_V2.csv",
                index=False)
            print("STD:", out_bank.std(axis=1).mean())
        return tree_df

    def _fit_tree_episodes(self):  # TODO move to DTree?
        """
        Create the inference dataset containing the feature set F_T for the DE and fit the DE.
        :return: None
        """
        # create empty dataframe

        batch_sz = self.data_loader.params.batch_sz
        dt: DTree = self.DE
        bins = dt.bins
        ranks_len = dt.ranks

        cos_sim = [f"cls_{i}" for i in range(0, self.num_classes)]
        col_hist = [f'bin_{i}' for i in range(bins * 3)]
        ranks = [f"rank_{i}" for i in range(0, ranks_len)]
        agg = ['max', 'mean', 'std']

        all_columns = cos_sim + col_hist + ranks + agg
        # Add column IDs for the DE (required by DE for creating the inputs in inference step)
        dt.features[dt.ALL_FT] = all_columns
        dt.features[dt.BASE_FT] = cos_sim
        dt.features[dt.MISC_FT] = col_hist
        dt.features[dt.RANK_FT] = ranks

        train_set = self.loaders.train_loader
        val_set = self.loaders.val_loader
        if self.root_cfg.DE.DATASET is None:
            tree_df = self._create_df(dataset=train_set, length=self.root_cfg.DE.EPISODE_TRAIN_NUM * batch_sz)
        else:
            tree_df = pd.read_csv(self.root_cfg.DE.DATASET, header=0)

        tree_df[ranks] = tree_df[ranks].astype('int')
        self.data.X = tree_df[all_columns]
        self.data.y = tree_df[['y']]
        print(self.data.X.head(5))
        print(self.data.X.tail(5))
        print("Finished inference, fitting tree...")
        if self.root_cfg.DE.DATASET is None:
            tree_df.to_csv(
                f"{self.root_cfg.DATASET}_{self.root_cfg.NAME}_W{self.root_cfg.WAY_NUM}_"
                f"S{self.root_cfg.SHOT_NUM}K_{self.k_neighbors}_"
                f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_V2.csv",
                index=False)
        val_df = self._create_df(dataset=val_set, length=600 * batch_sz)
        dt.fit(self.data.X, self.data.y, [(val_df[all_columns], val_df["y"])])

    def get_DEngine(self) -> DecisionEngine:
        return self.DE
