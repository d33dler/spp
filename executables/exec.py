from __future__ import print_function

import argparse
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
from PIL import ImageFile
from torchvision.transforms import transforms

from dataset.datasets_csv import CSVLoader
from models import architectures
from models.architectures.DN_X.dnx_arch import DN_X
from models.architectures.dt_model import DEModel
from models.utilities.utils import AverageMeter, create_confusion_matrix

sys.dont_write_bytecode = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=True, type=str, choices=architectures.__all__.keys(),
                    help='Model architecture ID')
parser.add_argument('--config', required=True, type=str, help='Model architecture YAML config file')
parser.add_argument('--dengine', action="store_true", help='Model architecture YAML config file')
parser.add_argument('--dataset_dir', default=None, help='/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--mode', default='train', choices=["train", "test"])
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--epochs', type=int, default=30, help='the total number of training epoch')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()
cudnn.benchmark = True

"""
This is the main execution script. For training a model simply add the config path, architecture name, 
dataset path & data_name.

Example:
python exec.py --arch DN_X --config models/architectures/DN4_Vanilla --dataset_dir your/dataset/path --data_name aName
"""


class Exec:
    out_bank = np.empty(shape=(0, 5))
    target_bank = np.empty(shape=0)

    def mean_confidence_interval(self, data, confidence=0.95):
        a = [1.0 * np.array(data[i].cpu()) for i in range(len(data))]
        n = len(a)
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, h

    def validate(self, val_loader, model: DEModel, best_prec1, F_txt, store_output=False, store_target=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.train(False)
        data = model.data
        accuracies = []

        end = time.time()

        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):

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
            if store_target:
                self.target_bank = np.concatenate([self.target_bank, target.numpy()], axis=0)
            target = target.cuda()

            model.data.q_in, model.data.S_in = input_var1, input_var2

            out = model.forward()
            if store_output:
                self.out_bank = np.concatenate([self.out_bank, out.cpu().numpy()], axis=0)
            loss = model.calculate_loss(target, data.sim_list_BACKBONE2D)

            # measure accuracy and record loss
            prec1 = model.calculate_accuracy(out, target, topk=(1, 3))
            losses.update(loss.item(), query_images.size(0))

            top1.update(prec1[0].item(), query_images.size(0))
            accuracies.append(prec1[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # ============== print the intermediate results ==============#
            if episode_index % opt.print_freq == 0 and episode_index != 0:
                print(f'Test-({model.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val} ({top1.avg})\t')

                print('Test-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val} ({top1.avg})'.format(model.get_epoch(), episode_index, len(val_loader),
                                                              batch_time=batch_time, loss=losses, top1=top1),
                      file=F_txt)

        print(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')
        print(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}', file=F_txt)

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
        params = model.data_loader.params

        for r in range(repeat_num):
            print('===================================== Round %d =====================================' % r)
            print('===================================== Round %d =====================================' % r,
                  file=F_txt)

            # ======================================= Folder of Datasets =======================================

            # image transform & normalization
            ImgTransform = transforms.Compose([
                transforms.Resize((params.img_sz, params.img_sz)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            testset = CSVLoader(
                data_dir=opt.dataset_dir, mode=opt.mode, image_size=params.img_sz, transform=ImgTransform,
                episode_num=params.episode_test_num, way_num=params.way_num, shot_num=params.shot_num,
                query_num=params.query_num
            )
            print('Testset: %d-------------%d' % (len(testset), r), file=F_txt)

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
            print("Test accuracy", test_accuracy, "h", h[0])
            print("Test accuracy", test_accuracy, "h", h[0], file=F_txt)
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[r] = h

        aver_accuracy, _ = self.mean_confidence_interval(total_accuracy_vector)
        print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean())
        print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean(), file=F_txt)
        F_txt.close()
        create_confusion_matrix(self.target_bank.astype(int), np.argmax(self.out_bank, axis=1))

        # ============================================== Testing end ==========================================

    def train(self, model: DEModel, F_txt):
        best_prec1 = 0
        # ======================================== Training phase ===============================================
        print('\n............Start training............\n')
        start_time = time.time()
        epoch = model.get_epoch()
        for epoch_index in range(epoch, epoch + opt.epochs):
            print('===================================== Epoch %d =====================================' % epoch_index)
            print('===================================== Epoch %d =====================================' % epoch_index,
                  file=F_txt)
            model.adjust_learning_rate(epoch_index)

            # ======================================= Folder of Datasets =======================================
            model.load_data(opt.mode, F_txt, opt.dataset_dir)
            loaders = model.loaders
            # ============================================ Training ===========================================
            # Freeze the parameters of Batch Normalization after 10000 episodes (1 epoch)
            if model.get_epoch() > 1:
                model.BACKBONE.freeze_layers()
            # Train for 10000 episodes in each epoch
            model.run_epoch(F_txt)

            # torch.cuda.empty_cache()
            # =========================================== Evaluation ==========================================
            print('============ Validation on the val set ============')
            print('============ validation on the val set ============', file=F_txt)
            prec1, _ = self.validate(loaders.val_loader, model, best_prec1, F_txt)

            # record the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            model.best_prec1 = best_prec1
            # save the checkpoint
            if is_best:
                filename = os.path.join(opt.outf, 'epoch_%d_best.pth.tar' % model.get_epoch())
                model.save_model(filename)

            if epoch_index % 5 == 0:
                filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' % model.get_epoch())
                model.save_model(filename)

            # Testing Prase
            print('============ Testing on the test set ============')
            print('============ Testing on the test set ============', file=F_txt)
            prec1, _ = self.validate(loaders.test_loader, model, best_prec1, F_txt)
            model.train(True)
        ###############################################################################

        loaders = model.data_loader.load_data(opt.mode, F_txt, opt.dataset_dir)
        if opt.dengine:
            model.enable_decision_engine(loaders.train_loader)
            prec1, _ = self.validate(loaders.val_loader, model, best_prec1, F_txt)

            print('============ Testing on the test set ============')
            print('============ Testing on the test set ============', file=F_txt)
            prec1, _ = self.validate(loaders.test_loader, model, best_prec1, F_txt)
            # record the best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)
            model.get_DEngine().plot_self()
        F_txt.close()
        print('Training and evaluation completed')

    def run(self):
        # ======================================== Settings of path ============================================
        ARCHITECTURE_MAP = architectures.__all__
        model = ARCHITECTURE_MAP[opt.arch](opt.config)
        p = model.data_loader.params
        opt.outf = '_'.join([p.outf, opt.arch, opt.data_name, str(model.arch), str(p.way_num), 'Way', str(
            p.shot_num), 'Shot', 'K' + str(model.root_cfg.K_NEIGHBORS)])
        p.outf = opt.outf

        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf, exist_ok=True)


        # save the opt and results to a txt file
        txt_save_path = os.path.join(opt.outf, 'opt_resutls.txt')
        txt_file = open(txt_save_path, 'a+')
        print(opt)
        print(opt, file=txt_file)

        # optionally resume from a checkpoint
        if opt.resume:
            model.load_model(opt.resume, txt_file)
        else:
            model.init_weights()

        if opt.ngpu > 1:
            model: DN_X = nn.DataParallel(model, range(opt.ngpu))

        # Print & log the model architecture
        print(model)
        print(model, file=txt_file)

        if opt.mode == "test":
            if opt.dengine:
                model.enable_decision_engine()
            self.test(model, F_txt=txt_file)
        else:
            self.train(model, F_txt=txt_file)


# ============================================ Training End ============================================================


if __name__ == '__main__':
    Exec().run()
