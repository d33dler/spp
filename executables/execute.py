from __future__ import print_function

import argparse
from typing import List

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
import os
import sys
import time

import numpy as np
import scipy as sp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import yaml
from PIL import ImageFile
from easydict import EasyDict
from torchvision.transforms import transforms

from data_loader.datasets_csv import BatchFactory
from models import architectures
from models.architectures.DN_X.dnx_arch import DN_X
from models.architectures.dt_model import DEModel
from models.utilities.utils import AverageMeter, create_confusion_matrix

sys.dont_write_bytecode = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cudnn.benchmark = True

"""
This is the main execution script. For training a model simply add the config path, architecture name, 
dataset path & data_name.

Example:
python exec.py --arch DN_X --config models/architectures/DN4_Vanilla --dataset_dir your/dataset/path --data_name aName \
--mode train --epochs 30 --dengine --refit_dengine
"""


class ExperimentManager:
    target_bank = np.empty(shape=0)

    def __init__(self):
        self.output_dir = None
        self._args = None
        self.out_bank = None
        self.k = None
        self.loss_tracker = AverageMeter()

    def mean_confidence_interval(self, data, confidence=0.95):
        a = [1.0 * np.array(data[i].cpu()) for i in range(len(data))]
        n = len(a)
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, h

    def write_losses_to_file(self, losses: List[float]):
        with open(os.path.join(self.output_dir, "LOSS_LOG.txt"), 'w') as _f:
            for loss in losses:
                _f.write(str(loss) + '\n')

    def validate(self, val_loader, model: DEModel, best_prec1, F_txt, store_output=False, store_target=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        accuracies = []

        end = time.time()
        model.data.training(False)
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):

            # Convert query and support images
            query_images = torch.cat(query_images, 0)
            input_var1 = query_images.cuda()

            input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
            input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))
            # Deal with the targets
            target = torch.cat(query_targets, 0).cuda()

            model.data.q_CPU = query_images
            model.data.q_in, model.data.S_in = input_var1, input_var2

            out = model.forward()
            loss = model.calculate_loss(out, target)

            # measure accuracy and record loss
            losses.update(loss.item(), query_images.size(0))

            prec1, _ = model.calculate_accuracy(out, target, topk=(1, 3))

            top1.update(prec1[0], query_images.size(0))
            accuracies.append(prec1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            if store_output:
                self.out_bank = np.concatenate([self.out_bank, out], axis=0)
            if store_target:
                self.target_bank = np.concatenate([self.target_bank, target.cpu().numpy()], axis=0)
            # ============== print the intermediate results ==============#
            if episode_index % self._args.PRINT_FREQ == 0 and episode_index != 0:
                print(f'Test-({model.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val} ({top1.avg})\t')

                F_txt.write(f'\nTest-({model.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                            f'Prec@1 {top1.val} ({top1.avg})\n')
        self.loss_tracker.loss_list = losses.loss_list
        self.write_losses_to_file(self.loss_tracker.get_loss_history())
        print(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')
        F_txt.write(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')

        return top1.avg, accuracies

    def test(self, model, F_txt):
        # ============================================ Testing phase ========================================
        print('\n............Start testing............')
        start_time = time.time()
        repeat_num = 5  # repeat running the testing code several times
        total_accuracy = 0.0
        total_h = np.zeros(repeat_num)
        total_accuracy_vector = []
        best_prec1 = 0
        params = model.ds_loader.params
        model.eval()
        model.data.training(False)
        for r in range(repeat_num):
            print('===================================== Round %d =====================================' % r)
            F_txt.write('===================================== Round %d =====================================\n' % r)

            # ======================================= Folder of Datasets =======================================

            # image transform & normalization
            ImgTransform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            testset = BatchFactory(
                data_dir=self._args.DATASET_DIR, mode='test', pre_process=ImgTransform,
                episode_num=params.episode_test_num, way_num=params.way_num, shot_num=params.shot_num,
                query_num=params.query_num
            )
            F_txt.write('Testset: %d-------------%d' % (len(testset), r))

            # ========================================== Load Datasets =========================================
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=params.test_ep_size, shuffle=True,
                num_workers=int(params.workers), drop_last=True, pin_memory=True
            )

            # =========================================== Evaluation ==========================================
            prec1, accuracies = self.validate(test_loader, model, best_prec1, F_txt, store_output=True,
                                              store_target=True)
            best_prec1 = max(prec1, best_prec1)
            test_accuracy, h = self.mean_confidence_interval(accuracies)
            print("Test accuracy=", test_accuracy, "h=", h[0])
            F_txt.write(f"Test accuracy= {test_accuracy} h= {h[0]}\n")
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[r] = h

        aver_accuracy, _ = self.mean_confidence_interval(total_accuracy_vector)
        print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean())
        F_txt.write(f"\nAver_accuracy= {aver_accuracy} Aver_h= {total_h.mean()}\n")
        F_txt.close()
        create_confusion_matrix(self.target_bank.astype(int), np.argmax(self.out_bank, axis=1))

        # ============================================== Testing end ==========================================

    def train(self, model: DEModel, F_txt):
        best_prec1 = model.best_prec1
        # ======================================== Training phase ===============================================
        print('\n............Start training............\n')
        epoch = model.get_epoch()

        for epoch_index in range(epoch, epoch + self._args.EPOCHS):
            print('===================================== Epoch %d =====================================' % epoch_index)
            F_txt.write(
                '===================================== Epoch %d =====================================\n' % epoch_index)
            # ================================= Set the model data to training mode ==============================
            model.data.training()
            # ======================================= Adjust learning rate =======================================

            # ======================================= Folder of Datasets =========================================
            model.load_data(self._args.MODE, F_txt, self._args.DATASET_DIR)
            loaders = model.loaders
            # ============================================ Training ==============================================
            model.train()
            # Freeze the parameters of Batch Normalization after X epochs (root configuration defined)
            model.freeze_auxiliary()
            # Train for 10000 episodes in each epoch
            model.run_epoch(F_txt)

            # torch.cuda.empty_cache()
            # =========================================== Evaluation ==========================================
            print('============ Validation on the val set ============')
            F_txt.write('============ Testing on the test set ============\n')
            try:

                prec1, _ = self.validate(loaders.val_loader, model, best_prec1, F_txt)
            except Exception as e:
                print("Encountered an exception while running val set validation!")
                print(e)
                prec1 = 0

            # record the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            model.best_prec1 = best_prec1
            # save the checkpoint
            if epoch_index % 2 == 0 or is_best:
                filename = os.path.join(self._args.OUTF, 'epoch_%d.pth.tar' % model.get_epoch())
                model.save_model(filename)

            # Testing Prase
            print('============ Testing on the test set ============')
            F_txt.write('============ Testing on the test set ============\n')
            try:
                prec1, _ = self.validate(loaders.test_loader, model, best_prec1, F_txt)
            except Exception as e:
                print("Encountered an exception while running the test set validation!")
                print(e)
        ###############################################################################
        F_txt.close()
        # save the last checkpoint
        filename = os.path.join(self._args.OUTF, 'epoch_%d.pth.tar' % model.get_epoch())
        model.save_model(filename)
        print('>>> Training and evaluation completed <<<')

    def run(self, _args):

        # ======================================== Settings of path ============================================
        self._args = _args
        ARCHITECTURE_MAP = architectures.__all__
        model = ARCHITECTURE_MAP[_args.ARCH](_args.PATH)
        PRMS = model.ds_loader.params
        # create path name for model checkpoints and log files
        _args.OUTF = PRMS.outf + '_'.join(
            [_args.ARCH, _args.BACKBONE.NAME, os.path.basename(_args.DATASET_DIR), str(model.arch), str(PRMS.way_num), 'Way', str(
                PRMS.shot_num), 'Shot', 'K' + str(model.root_cfg.K_NEIGHBORS),
             'QAV' + str(model.data.qav_num),
             'SAV' + str(model.data.sav_num),
             "AUG" + '_'.join([str(_aug.NAME) for _aug in model.root_cfg.AUGMENTOR.AUGMENTATION])])
        PRMS.outf = _args.OUTF
        self.output_dir = PRMS.outf
        if not os.path.exists(_args.OUTF):
            os.makedirs(_args.OUTF, exist_ok=True)

        # save the opt and results to a txt file
        txt_save_path = os.path.join(_args.OUTF, 'opt_results.txt')
        txt_file = open(txt_save_path, 'a+')
        txt_file.write(str(_args))

        # optionally resume from a checkpoint
        if _args.RESUME:
            model.load_model(_args.RESUME, txt_file)


        if _args.NGPU > 1:
            model: DN_X = nn.DataParallel(model, range(_args.NGPU))

        # Print & log the model architecture
        print(model)
        print(model, file=txt_file)
        self.k = model.k_neighbors
        self.out_bank = np.empty(shape=(0, model.num_classes))

        if _args.MODE == "test":
            if _args.DENGINE:
                model.load_data(_args.MODE, txt_file, _args.DATASET_DIR)
                model.enable_decision_engine(refit=_args.REFIT_DENGINE)
            self.test(model, F_txt=txt_file)
        else:
            self.train(model, F_txt=txt_file)
        txt_file.close()


# ============================================ Training End ============================================================

def launch_job(args):
    e = ExperimentManager()
    e.run(args)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', required=True, nargs='+', type=str, help='PATH(s) to the model config file(s)')
    arguments = parser.parse_args()
    print(arguments.jobs)
    proc_ls = []

    for a in arguments.jobs:
        with open(a, 'r') as f:
            job_args = (EasyDict(yaml.load(f, Loader=yaml.SafeLoader)),)
            job_args[0].PATH = a
            p = Process(target=launch_job, args=job_args, daemon=False)
            p.start()
            proc_ls.append(p)
    while 1:
        time.sleep(10)
        if all([not p.is_alive() for p in proc_ls]):
            break
