import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader as TorchDataLoader

from models.architectures.classifier import ClassifierModel
from models.dt_heads.dtree import DTree
from models.utilities.utils import AverageMeter, accuracy


class DN7DA_DTR(ClassifierModel):
    """
    DN4 DTR Model

    Structure:
    [Deep learning module] ⟶ [K-NN module] ⟶ [Decision Tree]

    """
    arch = 'DN4DA_DTR'

    def __init__(self):
        super().__init__(Path(__file__).parent / 'config.yaml')
        # self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self):
        self.BACKBONE_2D.forward()

        dt_head: DTree = self.DT
        if dt_head.is_fit:
            _input = np.asarray([dt_head.normalize(x) for x in self.data.sim_list_BACKBONE2D.detach().cpu().numpy()])
            self.data.X = self.feature_engine(dt_head.create_input(_input), dt_head.base_features,
                                              dt_head.deep_features)
            o = torch.from_numpy(dt_head.forward(self.data.X).astype(np.int64))
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

            loss = self.get_loss('BACKBONE_2D')
            # Measure accuracy and record loss
            prec1, _ = accuracy(self.data.sim_list_BACKBONE2D, target, topk=(1, 3))

            losses.update(loss.item(), query_images.size(0))
            top1.update(prec1[0], query_images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # ============== print the intermediate results ==============#
            if episode_index % self.model_cfg.PRINT_FREQ == 0 and episode_index != 0:
                print(f'Eposide-({epochix}): [{episode_index}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      )

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
