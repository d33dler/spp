import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn, optim, Tensor
from torch.nn.functional import one_hot

from dataset.datasets_csv import CSVLoader
from models.architectures.classifier import ClassifierModel
from models.dt_heads.dtree import DTree
from models.utilities.utils import AverageMeter, accuracy
from torch.utils.data import DataLoader as TorchDataLoader


class DN4_DTR(ClassifierModel):
    """
    DN4 DTR Model

    Structure:
    [Deep learning module] ⟶ [K-NN module] ⟶ [Decision Tree]
                          ↘  [Encoder-NN]  ↗
    """
    arch = 'DN4_DTR'

    def __init__(self):
        super().__init__(Path(__file__).parent / 'config.yaml')
        # self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self):
        self.BACKBONE_2D.forward()
        self.ENCODER.forward()

        dt_head: DTree = self.DT
        if dt_head.is_fit:
            _input = np.asarray([dt_head.normalize(x) for x in self.data.sim_list_REDUCED.detach().cpu().numpy()])
            self.data.X = self.feature_engine(dt_head.create_input(_input), dt_head.base_features,
                                              dt_head.deep_features)
            o = torch.from_numpy(dt_head.forward(self.data).astype(np.int64))
            o = one_hot(o, self.num_classes).float().cuda()
            self.data.tree_pred = o

    def run_epoch(self, output_file):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        train_loader = self.loaders.train_loader
        end = time.time()
        epochix = self.epochix
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):

            # Measure data loading time
            data_time.update(time.time() - end)

            # Convert query and support images
            query_images = torch.cat(query_images, 0)
            input_var1 = query_images.cuda()

            input_var2 = []
            for i in range(len(support_images)):
                temp_support = support_images[i]
                temp_support = torch.cat(temp_support, 0)
                temp_support = temp_support.cuda()
                input_var2.append(temp_support)

            # Deal with the targets
            target = torch.cat(query_targets, 0)
            target = target.cuda()
            self.data.targets = target
            self.data.q_in = input_var1
            self.data.S_in = input_var2

            # Calculate the output
            self.forward()

            loss = self.get_loss('ENCODER')
            # Measure accuracy and record loss
            prec1_smax = accuracy(self.data.q_smax, target)[0]
            prec1_red, _ = accuracy(self.data.sim_list_REDUCED, target, topk=(1, 3))

            losses.update(min(loss).item(), query_images.size(0))
            prec1 = max(prec1_red, prec1_smax)
            top1.update(prec1[0], query_images.size(0))
            [l.detach_().detach().cpu() for l in loss]

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # ============== print the intermediate results ==============#
            if episode_index % self.model_cfg.PRINT_FREQ == 0 and episode_index != 0:
                print(f'Eposide-({epochix}): [{episode_index}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Prec@RED {prec1_red.item():.3f}\t'
                      f'Prec@SMAX {prec1_smax.item():.3f}\t')

                print('Eposide-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epochix, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time,
                    loss=losses,
                    top1=top1), file=output_file)
            self.epochix += 1
        del losses
        self.data.clear()

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
        X_len = len(train_set) * batch_sz
        ft_engine = ['max', 'mean', 'std']
        cls_labels = [f"cls_{i}" for i in range(0, self.num_classes)]
        ranks = [f"rank_{i}" for i in range(0, self.k_neighbors)]
        deep_local_lbs_Q = [f"fQ_{i}" for i in range(0, 8 * 8 * 8)]
        deep_local_lbs_S = [f"fS_{i}" for i in range(0, 8 * 8 * 8 * 5)]
        all_columns = cls_labels + ft_engine + ranks + deep_local_lbs_S + deep_local_lbs_Q
        col_len = len(all_columns)
        dt_head.all_features = all_columns
        dt_head.base_features = cls_labels
        dt_head.deep_features = deep_local_lbs_Q
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
                    out_BB2D = np.asarray(
                        [dt_head.normalize(x) for x in self.data.sim_list_BACKBONE2D.detach().cpu().numpy()])
                    target: np.ndarray = target.numpy()
                    # add measurements and target value to dataframe
                    tree_df.iloc[ix:ix + batch_sz, cls_col_ix] = out_BB2D
                    tree_df.iloc[ix:ix + batch_sz, y_col_ix] = target
                    tree_df.iloc[ix:ix + batch_sz, max_ix] = out_BB2D.max(axis=1)
                    tree_df.iloc[ix:ix + batch_sz, mean_ix] = out_BB2D.mean(axis=1)
                    tree_df.iloc[ix:ix + batch_sz, std_ix] = out_BB2D.std(axis=1)

                    top_k = np.argpartition(-out_BB2D, kth=self.k_neighbors, axis=1)[:, :self.k_neighbors]
                    tree_df.iloc[ix:ix + batch_sz, ranks_ix] = out_BB2D[np.arange(out_BB2D.shape[0])[:, None], top_k]
                    print(self.data.q_reduced.size())
                    print(len(self.data.S_reduced))
                    print(self.data.S_reduced[0].size())
                    tree_df.iloc[ix:ix + batch_sz, deep_local_ix_Q] = self.data.q_reduced.reshape(batch_sz,
                                                                                                  -1).detach().cpu().numpy()

                    for interv in range(0, 10):
                        o =  self.data.S_reduced[interv].reshape(5, -1).detach().cpu().numpy()
                        print(o.shape)
                        tree_df.iloc[ix + interv * 5:ix + interv * 5 + 5, deep_local_ix_S] =o
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
