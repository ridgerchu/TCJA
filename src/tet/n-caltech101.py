import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.parallel
import torch.optim
from spikingjelly.datasets.n_caltech101 import NCaltech101
from models.VGG_models import *
import torch.nn.functional as F
import data_loaders
from functions import TET_loss, seed_all, get_logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
import math

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=1024,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T',
                    '--time',
                    default=2,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=False,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=1e-3,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--resume',
                    default=False,
                    type=bool,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--dts_cache', type=str, default='./dts_cache')
parser.add_argument('--b', default=16, type=int, help='batch size')
parser.add_argument('--T', default=16, type=int, help='simulating time-steps')
parser.add_argument('--j', default=15, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        one_hot = F.one_hot(labels)
        images = images.to(device)
        images = images.type(torch.cuda.FloatTensor)
        outputs = model(images)
        mean_out = outputs.mean(1)
        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':
    seed_all(args.seed)
    train_dataset, val_dataset = data_loaders.build_dvscifar('/home/ridger/dataset/cifar-dvs')
    train_set_pth = os.path.join(args.dts_cache, f'ncaltech_train_set_{args.T}.pt')
    test_set_pth = os.path.join(args.dts_cache, f'ncaltech_test_set_{args.T}.pt')
    if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
        train_set = torch.load(train_set_pth)
        test_set = torch.load(test_set_pth)
    else:
        origin_set = NCaltech101(root='/home/ridger/datasets/NCAL101/', data_type='frame', frames_number=args.T, split_by='number',)

        train_set, test_set = split_to_train_test_set(0.9, origin_set, 101)
        if not os.path.exists(args.dts_cache):
            os.makedirs(args.dts_cache)
        torch.save(train_set, train_set_pth)
        torch.save(test_set, test_set_pth)

    train_loader = DataLoaderX(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)
    test_loader = DataLoaderX(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)

    model = VGGSNN_TCJA_NCAL()

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0
    log_name = './log/ncaltech/TET_log'
    logger = get_logger(log_name + '.log')
    logger.info('start training!')
    if args.resume:
        model.load_state_dict(torch.load('VGGSNN.pth', map_location='cpu'))

    for epoch in range(args.epochs):

        loss, acc = train(model, device, train_loader, criterion, optimizer, epoch, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        scheduler.step()
        facc = test(model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), log_name + '.pth')
        logger.info('Best acc={:.3f}'.format(best_acc))
        print('\n')
