import time
from pathlib import Path

import torch

from models.architectures.dt_model import DEModel
from models.utilities.utils import AverageMeter, accuracy


class SN_X(DEModel):
    """
    Siamese Network (4|7|X) Model
    Implements epoch run employing few-shot learning & performance tracking during training
    Structure:
    [Deep learning module] ⟶ [K-NN module] ⟶ [Decision Tree]

    """
    arch = 'SNX'

    def __init__(self, cfg_path):
        """
        Pass configuration file path to the superclass
        :param cfg_path: model root configuration file path to be loaded (should include sub-module specifications)
        :type cfg_path: Path | str
        """
        super().__init__(cfg_path)

    def forward(self):
        self.BACKBONE.forward()
        return self.data.sim_list

    def run_epoch(self, output_file):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        train_loader = self.loaders.train_loader
        end = time.time()
        epochix = self.get_epoch()

        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            # Convert query and support images
            query_images = torch.cat(query_images, 0)
            input_var1 = query_images.cuda()

            input_var2 = []

            for temp_support in support_images:
                temp_support = torch.cat(temp_support, 0)
                temp_support = temp_support.cuda()
                input_var2.append(temp_support)
            # Deal with the targets
            target = torch.cat(query_targets, 0).cuda()

            self.data.targets = target
            self.data.q_CPU = query_images
            self.data.q_in = input_var1
            self.data.S_in = input_var2

            # Calculate the output
            out = self.forward()
            self.backward()
            loss = self.get_loss()
            # Measure accuracy and record loss
            prec1, _ = accuracy(out, target, topk=(1, 3))

            losses.update(loss.item(), query_images.size(0))
            top1.update(prec1[0], query_images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # ============== print the intermediate results ==============#
            if episode_index % self.root_cfg.PRINT_FREQ == 0 and episode_index != 0:
                print(f'Eposide-({epochix}): [{episode_index}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      )

                print('Eposide-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epochix, episode_index, len(train_loader),
                                                                      batch_time=batch_time, data_time=data_time,
                                                                      loss=losses,
                                                                      top1=top1), file=output_file)
        self.incr_epoch()
        self.data.empty_cache()

    def backward(self):
        self.BACKBONE.backward()
