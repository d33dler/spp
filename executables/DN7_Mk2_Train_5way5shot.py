from __future__ import print_function

import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from PIL import ImageFile

from models.architectures.classifier import ClassifierModel
from models.architectures.dn4_dta.dn4_mk2 import DN4_DTR
from models.architectures.dn7_dta.dn7_mk2 import DN7_DTR
from models.architectures.dn7da_dta.dn7da_mk2 import DN7DA_DTR
from models.utilities.utils import AverageMeter, accuracy, save_checkpoint

sys.dont_write_bytecode = True

# ============================ Data & Networks =====================================

# ==================================================================================


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # TODO might be cause for issues?

parser = argparse.ArgumentParser()
parser.add_argument('--id', default='', type=str, help='Run ID')
parser.add_argument('--dataset_dir', default=None, help='/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--mode', default='train', help='train|val|test|')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
#  Few-shot parameters  #
parser.add_argument('--epochs', type=int, default=30, help='the total number of training epoch')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True


# ======================================= Define functions ============================================


def validate(val_loader, model: ClassifierModel, epoch_index, best_prec1, F_txt):
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

        model.data.q_in = input_var1
        model.data.S_in = input_var2
        model.data.targets = target

        model.forward()

        loss = model.module_topology['BACKBONE_2D'].calculate_loss(data.sim_list_BACKBONE2D, target)

        # measure accuracy and record loss
        prec1, _ = accuracy(model.data.sim_list_BACKBONE2D, target, topk=(1, 3))
        losses.update(loss.item(), query_images.size(0))

        top1.update(prec1[0], query_images.size(0))
        accuracies.append(prec1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print(f'Test-({epoch_index}): [{episode_index}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t')

            print('Test-({0}): [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1), file=F_txt)

    print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))
    print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1), file=F_txt)

    return top1.avg, accuracies


def run():
    # ======================================== Settings of path ============================================
    # saving path
    model = DN7_DTR() if opt.id == "DN7Vanilla" else DN7DA_DTR()
    p = model.data_loader.params
    opt.outf = '_'.join([p.outf, opt.id, opt.data_name, str(model.arch), str(p.way_num), 'Way', str(
        p.shot_num), 'Shot', 'K' + str(model.model_cfg.K_NEIGHBORS)])
    p.outf = opt.outf
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # save the opt and results to a txt file
    txt_save_path = os.path.join(opt.outf, 'opt_resutls.txt')
    txt_file = open(txt_save_path, 'a+')
    print(opt)
    print(opt, file=txt_file)

    # ========================================== Model Config ===============================================

    best_prec1 = 0

    # optionally resume from a checkpoint
    if opt.resume:
        model.load_model(opt.resume, txt_file)

    if opt.ngpu > 1:
        model: DN7_DTR = nn.DataParallel(model, range(opt.ngpu))

    # print the architecture of the network
    print(model)
    print(model, file=txt_file)
    # ======================================== Training phase ===============================================
    print('\n............Start training............\n')
    start_time = time.time()

    for epoch_index in range(opt.epochs):
        print('===================================== Epoch %d =====================================' % epoch_index)
        print('===================================== Epoch %d =====================================' % epoch_index,
              file=txt_file)
        model.adjust_learning_rate(epoch_index)

        # ======================================= Folder of Datasets =======================================
        model.load_data(opt.mode, txt_file, opt.dataset_dir)
        loaders = model.loaders
        # ============================================ Training ===========================================
        # Fix the parameters of Batch Normalization after 10000 episodes (1 epoch)
        if model.epochix > 1: model.BACKBONE_2D.freeze_layers()
        # Train for 10000 episodes in each epoch
        model.run_epoch(txt_file)

        # torch.cuda.empty_cache()
        # =========================================== Evaluation ==========================================
        print('============ Validation on the val set ============')
        print('============ validation on the val set ============', file=txt_file)
        prec1, _ = validate(loaders.val_loader, model, epoch_index, best_prec1, txt_file)

        # record the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        model.best_prec1 = best_prec1
        # save the checkpoint
        if is_best:
            filename = os.path.join(opt.outf, 'epoch_%d_best.pth.tar' % model.epochix)
            model.save_model(filename)

        if epoch_index % 10 == 0:
            filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' % model.epochix)
            model.save_model(filename)

        # Testing Prase
        print('============ Testing on the test set ============')
        print('============ Testing on the test set ============', file=txt_file)
        prec1, _ = validate(loaders.test_loader, model, epoch_index, best_prec1, txt_file)
        model.train(True)
    ###############################################################################
    # Fitting tree, now forward will provide tree output

    print('============ Validation on the val set ============')
    print('============ validation on the val set ============', file=txt_file)

    loaders = model.data_loader.load_data(opt.mode, txt_file, opt.dataset_dir)

    model.fit_tree_episodes(loaders.train_loader)

    prec1, _ = validate(loaders.val_loader, model, opt.epochs, best_prec1, txt_file)

    print('============ Testing on the test set ============')
    print('============ Testing on the test set ============', file=txt_file)
    prec1, _ = validate(loaders.test_loader, model, opt.epochs, best_prec1, txt_file)
    # record the best prec@1 and save checkpoint
    best_prec1 = max(prec1, best_prec1)
    model.get_tree('DT').plot_tree()
    txt_file.close()
    print('Training and evaluation completed')


# ============================================ Training End ============================================================


if __name__ == '__main__':
    run()

