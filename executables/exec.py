from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import scipy as sp
import scipy.special
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from PIL import ImageFile
from sklearn.metrics import accuracy_score
from torchvision.transforms import transforms

from dataset.datasets_csv import CSVLoader
from models import architectures
from models.architectures.DN_X.dnx_arch import DN_X
from models.architectures.dt_model import DEModel
from models.utilities.utils import AverageMeter, create_confusion_matrix

from scipy.special import softmax

sys.dont_write_bytecode = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--arch', required=True, type=str, choices=architectures.__all__.keys(),
                    help='Model architecture ID')
parser.add_argument('--config', required=True, type=str, help='Model architecture YAML config file')
parser.add_argument('--dengine', action="store_true", help='Model architecture YAML config file')
parser.add_argument('--refit_dengine', action="store_true", help='Force refit the decision engine')

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

    target_bank = np.empty(shape=0)

    def __init__(self):
        self.out_bank = None
        self.k = None

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

            target = target.cuda()
            model.data.q_in_CPU = query_images
            model.data.q_in, model.data.S_in = input_var1, input_var2

            out = model.forward()
            loss = model.calculate_loss(target, out)

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
            if episode_index % opt.print_freq == 0 and episode_index != 0:
                print(f'Test-({model.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val} ({top1.avg})\t')

                F_txt.write(f'\nTest-({model.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                            f'Prec@1 {top1.val} ({top1.avg})\n')

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
        params = model.data_loader.params

        for r in range(repeat_num):
            print('===================================== Round %d =====================================' % r)
            F_txt.write('===================================== Round %d =====================================\n' % r)

            # ======================================= Folder of Datasets =======================================

            # image transform & normalization
            ImgTransform = transforms.Compose([
                transforms.Resize((params.img_sz, params.img_sz)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            testset = CSVLoader(
                data_dir=opt.dataset_dir, mode='train', image_size=params.img_sz, transform=ImgTransform,
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
        best_prec1 = 0
        # ======================================== Training phase ===============================================
        print('\n............Start training............\n')
        start_time = time.time()
        epoch = model.get_epoch()
        for epoch_index in range(epoch, epoch + opt.epochs):
            print('===================================== Epoch %d =====================================' % epoch_index)
            F_txt.write(
                '===================================== Epoch %d =====================================\n' % epoch_index)
            model.adjust_learning_rate(epoch_index)

            # ======================================= Folder of Datasets =======================================
            model.load_data(opt.mode, F_txt, opt.dataset_dir)
            loaders = model.loaders
            # ============================================ Training ===========================================
            # Freeze the parameters of Batch Normalization after 10000 episodes (1 epoch)
            if model.get_epoch() > 0:
                model.BACKBONE.freeze_layers()
            # Train for 10000 episodes in each epoch
            model.run_epoch(F_txt)

            # torch.cuda.empty_cache()
            # =========================================== Evaluation ==========================================
            print('============ Validation on the val set ============')
            F_txt.write('============ Testing on the test set ============\n')
            prec1, _ = self.validate(loaders.val_loader, model, best_prec1, F_txt)

            # record the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            model.best_prec1 = best_prec1
            # save the checkpoint
            if is_best:
                filename = os.path.join(opt.outf, 'epoch_%d_best.pth.tar' % model.get_epoch())
                model.save_model(filename)

            if epoch_index % 2 == 0:
                filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' % model.get_epoch())
                model.save_model(filename)

            # Testing Prase
            print('============ Testing on the test set ============')
            F_txt.write('============ Testing on the test set ============\n')
            prec1, _ = self.validate(loaders.test_loader, model, best_prec1, F_txt)
            model.train(True)
        ###############################################################################

        loaders = model.data_loader.load_data(opt.mode, opt.dataset_dir, F_txt)
        if opt.dengine:
            filename = os.path.join(opt.outf, 'epoch_%d_DE.pth.tar' % model.get_epoch())
            model.enable_decision_engine(loaders.train_loader, refit=True, filename=filename)
            prec1, _ = self.validate(loaders.val_loader, model, best_prec1, F_txt)

            print('============ Testing on the test set ============')
            F_txt.write('\n============ Testing on the test set ============\n')
            prec1, _ = self.validate(loaders.test_loader, model, best_prec1, F_txt)
            # record the best prec@1 and save checkpoint
            model.best_prec1 = max(prec1, best_prec1)
        F_txt.close()

        print('Training and evaluation completed')

    def run(self):
        # ======================================== Settings of path ============================================
        ARCHITECTURE_MAP = architectures.__all__
        model = ARCHITECTURE_MAP[opt.arch](opt.config)
        p = model.data_loader.params
        opt.outf = p.outf + '_'.join([opt.arch, opt.data_name, str(model.arch), str(p.way_num), 'Way', str(
            p.shot_num), 'Shot', 'K' + str(model.root_cfg.K_NEIGHBORS)])
        p.outf = opt.outf
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf, exist_ok=True)

        # save the opt and results to a txt file
        txt_save_path = os.path.join(opt.outf, 'opt_results.txt')
        txt_file = open(txt_save_path, 'a+')
        txt_file.write(str(opt))

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
        self.k = model.k_neighbors
        self.out_bank = np.empty(shape=(0, model.num_classes))

        if opt.mode == "test":
            if opt.dengine:
                model.load_data(opt.mode, txt_file, opt.dataset_dir)
                model.enable_decision_engine(refit=opt.refit_dengine)
            self.test(model, F_txt=txt_file)
        else:
            self.train(model, F_txt=txt_file)
        txt_file.close()


# ============================================ Training End ============================================================


if __name__ == '__main__':
    Exec().run()
