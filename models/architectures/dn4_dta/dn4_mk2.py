import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn, optim
from torch.nn.functional import one_hot

from dataset.datasets_csv import CSVLoader
from models.architectures.classifier import ClassifierModel
from models.dt_heads.dtree import DTree
from models.utilities.utils import AverageMeter, accuracy


class DN4_DTA(ClassifierModel):
    """
    DN4 MK2 Model

    Structure:
    [Deep learning module] ⟶ [K-NN module] ⟶ [Decision Tree]
                          ↘  [Encoder-NN]  ↗
    """
    normalizer = torch.nn.BatchNorm2d(64)

    def __init__(self):
        self.loaders = None
        model_cfg = self.load_config(Path(__file__).parent / 'config.yaml')
        super().__init__(model_cfg)
        self.build()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.parameters(), lr=model_cfg.LEARNING_RATE, betas=(model_cfg.BETA_ONE, 0.9))

    def forward(self):
        self.backbone2d(self.data)
        out = self.knn_head(self.data)

        dt_head: DTree = self.dt_head
        if dt_head.is_fit:
            _input = np.asarray([dt_head.normalize(x) for x in out.detach().cpu().numpy()])
            self.data.X = self.feature_engine(dt_head.create_input(_input), dt_head.base_features,
                                              dt_head.deep_features)
            o = torch.from_numpy(dt_head.forward(self.data).astype(np.int64))
            o = one_hot(o, self.num_classes).float().cuda()
            return o
        return out

    def run_epoch(self, epoch_index, output_file):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        train_loader = self.loaders.train_loader
        end = time.time()
        criterion = self.criterion
        optimizer = self.optimizer
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
            self.data.q_in = input_var1
            self.data.S_in = input_var2
            # Calculate the output
            output = self.forward()
            loss = criterion(output, target)

            # Compute gradients and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure accuracy and record loss
            prec1, _ = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), query_images.size(0))
            top1.update(prec1[0], query_images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # ============== print the intermediate results ==============#
            if episode_index % self.model_cfg.PRINT_FREQ == 0 and episode_index != 0:
                print('Eposide-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time,
                    loss=losses,
                    top1=top1))

                print('Eposide-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time,
                    loss=losses,
                    top1=top1), file=self.F_txt)

    def load_data(self, mode, f_txt):
        self.loaders = self.data_loader.load_data(mode, f_txt)
