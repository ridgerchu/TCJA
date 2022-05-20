import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import math

import spikingjelly.datasets
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from spikingjelly.datasets.n_caltech101 import NCaltech101
from spikingjelly.clock_driven.model import sew_resnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven.functional import temporal_efficient_training_cross_entropy as tmt_loss
import time
import os
import tqdm
import argparse
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
import transfroms

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

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

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class PythonNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(PythonNet.conv3x3(2, 128))
        conv.append(nn.MaxPool2d(2))
        conv.extend(PythonNet.conv3x3(128, 128))
        conv.append(nn.MaxPool2d(2))
        conv.extend(PythonNet.conv3x3(128, 128))
        conv.append(nn.MaxPool2d(2))
        conv.extend(PythonNet.conv3x3(128, 128))
        conv.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.8),
            nn.Linear(128*8*8, 512, bias=False),
            neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(), detach_reset=False),
            layer.Dropout(0.5),
            nn.Linear(512, 100, bias=False),
            neuron.LIFNode(tau=2.,surrogate_function=surrogate.ATan(), detach_reset=False),
        )
        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out = self.conv(x[0])
        out = self.fc(out)
        out_spikes = self.vote(out)
        for t in range(1, x.shape[0]):
            out_spikes += self.vote(self.fc(self.conv(x[t])))
        return out_spikes/x.shape[0]

    @staticmethod
    def conv3x3_iz(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.Izhikevich_test(surrogate_function=surrogate.ATan(), detach_reset=False)
        ]
    def conv3x3_ad(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.LIF_with_adaption(a=0.2,b=0.6,vr=-0.05,d=0.5,surrogate_function=surrogate.ATan(), detach_reset=False)
        ]
    def conv3x3(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.LIFNode(tau=2.,surrogate_function=surrogate.ATan(), detach_reset=False)
        ]

try:
    import cupy

    class CextNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            conv = []
            conv.extend(CextNet.conv3x3(2, 64))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(CextNet.conv3x3(64, 128))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(CextNet.conv3x3(128, 256))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(CextNet.conv3x3(256, 256))
            conv.append(layer.TCJA(4, 4, 14, 256))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(CextNet.conv3x3(256, 512))
            conv.append(layer.TCJA(2, 4, 14, 512))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))

            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(2),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(512 * 5 * 7, 1024, bias=False)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True,
                                        backend='cupy'),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(1024, 101, bias=False)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            )
            self.vote = VotingLayer(10)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4) # [N, T, 2, H, W] -> [T, N, 2, H, W]
            out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
            return out_spikes

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                ),

                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
            ]

        def conv3x3_iz(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                ),
                layer.MultiStepThresholdDependentBatchNorm2d(1.,1.,out_channels),
                neuron.MultiStepIzhikevichNode(tau=2.,tau_w=1/0.175, a=.6, b=0.5,w_rest=0., surrogate_function=surrogate.ATan(), detach_reset=False, backend='torch'),
            ]

    class VGGNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            conv = []
            conv.extend(VGGNet.conv3x3(2, 64))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(VGGNet.conv3x3(64, 128))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(VGGNet.conv3x3(128, 256))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(VGGNet.conv3x3(256, 256))
            conv.append(layer.TCJA(4, 20, 256))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            conv.extend(VGGNet.conv3x3(256, 512))
            conv.append(layer.TCJA(4, 20, 512))
            conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(2),
                layer.MultiStepDropout(0.8),
                layer.SeqToANNContainer(nn.Linear(512 * 4 * 4, 512, bias=False)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True,
                                        backend='cupy'),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(512, 100, bias=False)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            )
            self.vote = VotingLayer(10)

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
            out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
            return self.vote(out_spikes.mean(0))

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                ),
                layer.MultiStepThresholdDependentBatchNorm2d(0.707,1,out_channels),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
            ]

    class VGGNet_2(nn.Module):
        def __init__(self):
            super().__init__()
            conv = []
            #conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
            conv.extend(VGGNet_2.conv3x3(2, 64))
            conv.extend(VGGNet_2.conv3x3(64, 128))

            conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

            conv.extend(VGGNet_2.conv3x3(128, 256))
            conv.extend(VGGNet_2.conv3x3(256, 256))

            conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

            conv.extend(VGGNet_2.conv3x3(256, 512))
            conv.extend(VGGNet_2.conv3x3(512, 512))
            conv.append(layer.TCJA(4, 4, 14, 512))
            conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))


            conv.extend(VGGNet_2.conv3x3(512, 512))
            conv.extend(VGGNet_2.conv3x3(512, 512))
            conv.append(layer.TCJA(4, 4, 14, 512))
            conv.append(layer.SeqToANNContainer(nn.AvgPool2d(4, 4)))


            self.conv = nn.Sequential(*conv)
            self.fc = nn.Sequential(
                nn.Flatten(2),
                layer.SeqToANNContainer(nn.Linear(512*5*7, 101)),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True,
                                        backend='cupy'),
            )

        def forward(self, x: torch.Tensor):
            x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
            out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
            return out_spikes

        @staticmethod
        def conv3x3(in_channels: int, out_channels):
            return [
                layer.SeqToANNContainer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels)
                ),
                neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
            ]







except ImportError:
    print('Cupy is not installed.')


def main():
    # python classify_dvsg.py -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024
    '''
    * :ref:`API in English <classify_dvsg.__init__-en>`

    .. _classify_dvsg.__init__-cn:

    用于分类DVS128 Gesture数据集的代码样例。网络结构来自于 `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_。

    .. code:: bash

        usage: classify_dvsg.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-channels CHANNELS] [-data_dir DATA_DIR] [-out_dir OUT_DIR] [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-lr LR] [-momentum MOMENTUM] [-lr_scheduler LR_SCHEDULER] [-step_size STEP_SIZE] [-gamma GAMMA] [-T_max T_MAX]

        Classify DVS128 Gesture

        optional arguments:
          -h, --help            show this help message and exit
          -T T                  simulating time-steps
          -device DEVICE        device
          -b B                  batch size
          -epochs N             number of total epochs to run
          -j N                  number of data loading workers (default: 4)
          -channels CHANNELS    channels of Conv2d in SNN
          -data_dir DATA_DIR    root dir of DVS128 Gesture dataset
          -out_dir OUT_DIR      root dir for saving logs and checkpoint
          -resume RESUME        resume from the checkpoint path
          -amp                  automatic mixed precision training
          -cupy                 use CUDA neuron and multi-step forward mode
          -opt OPT              use which optimizer. SDG or Adam
          -lr LR                learning rate
          -momentum MOMENTUM    momentum for SGD
          -lr_scheduler LR_SCHEDULER
                                use which schedule. StepLR or CosALR
          -step_size STEP_SIZE  step_size for StepLR
          -gamma GAMMA          gamma for StepLR
          -T_max T_MAX          T_max for CosineAnnealingLR

    运行示例：

    .. code:: bash

        python -m spikingjelly.clock_driven.examples.classify_dvsg -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024

    阅读教程 :doc:`./clock_driven/14_classify_dvsg` 以获得更多信息。

    * :ref:`中文API <classify_dvsg.__init__-cn>`

    .. _classify_dvsg.__init__-en:

    The code example for classifying the DVS128 Gesture dataset. The network structure is from `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_.


    .. code:: bash

        usage: classify_dvsg.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-channels CHANNELS] [-data_dir DATA_DIR] [-out_dir OUT_DIR] [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-lr LR] [-momentum MOMENTUM] [-lr_scheduler LR_SCHEDULER] [-step_size STEP_SIZE] [-gamma GAMMA] [-T_max T_MAX]

        Classify DVS128 Gesture

        optional arguments:
          -h, --help            show this help message and exit
          -T T                  simulating time-steps
          -device DEVICE        device
          -b B                  batch size
          -epochs N             number of total epochs to run
          -j N                  number of data loading workers (default: 4)
          -channels CHANNELS    channels of Conv2d in SNN
          -data_dir DATA_DIR    root dir of DVS128 Gesture dataset
          -out_dir OUT_DIR      root dir for saving logs and checkpoint
          -resume RESUME        resume from the checkpoint path
          -amp                  automatic mixed precision training
          -cupy                 use CUDA neuron and multi-step forward mode
          -opt OPT              use which optimizer. SDG or Adam
          -lr LR                learning rate
          -momentum MOMENTUM    momentum for SGD
          -lr_scheduler LR_SCHEDULER
                                use which schedule. StepLR or CosALR
          -step_size STEP_SIZE  step_size for StepLR
          -gamma GAMMA          gamma for StepLR
          -T_max T_MAX          T_max for CosineAnnealingLR

    Running Example:

    .. code:: bash

        python -m spikingjelly.clock_driven.examples.classify_dvsg -data_dir /userhome/datasets/DVS128Gesture -out_dir ./logs -amp -opt Adam -device cuda:0 -lr_scheduler CosALR -T_max 64 -cupy -epochs 1024

    See the tutorial :doc:`./clock_driven_en/14_classify_dvsg` for more details.
    '''
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-channels', default=128, type=int, help='channels of Conv2d in SNN')
    parser.add_argument('-data_dir', type=str, help='root dir of DVS128 Gesture dataset')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use CUDA neuron and multi-step forward mode')

    parser.add_argument('-opt', default='Adam', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-vgg', action='store_true', help='use CUDA neuron and multi-step forward mode')
    parser.add_argument('-vgg_2', action='store_true', help='use CUDA neuron and multi-step forward mode')
    parser.add_argument('-mixup', action='store_true')
    parser.add_argument('-add_dict', type=str)
    parser.add_argument('-loss', type=str, default='mse')
    parser.add_argument('-dts_cache', type=str, default='./dts_cache')


    args = parser.parse_args()
    print(args)

    if args.cupy:
        net = CextNet(channels=args.channels)
    elif args.vgg:
        net = VGGNet(channels=args.channels)
    elif args.vgg_2:
        net = VGGNet_2()

    print(net)
    net.to(args.device)

    optimizer = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    train_set_pth = os.path.join(args.dts_cache, f'ncaltech_train_set_{args.T}.pt')
    test_set_pth = os.path.join(args.dts_cache, f'ncaltech_test_set_{args.T}.pt')
    if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
        train_set = torch.load(train_set_pth)
        test_set = torch.load(test_set_pth)
    else:
        origin_set = NCaltech101(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number',)

        train_set, test_set = split_to_train_test_set(0.9, origin_set, 101)
        if not os.path.exists(args.dts_cache):
            os.makedirs(args.dts_cache)
        torch.save(train_set, train_set_pth)
        torch.save(test_set, test_set_pth)

    train_data_loader = DataLoaderX(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)
    test_data_loader = DataLoaderX(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()


    out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_c_{args.channels}_{args.opt}_lr_{args.lr}_128')
    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'
    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'dvsg_logs'), purge_step=start_epoch)
    if args.loss == 'mse':
        criterion = F.mse_loss
    elif args.loss == 'ce':
        criterion = F.cross_entropy
    elif args.loss == 'tmt':
        criterion = tmt_loss

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in tqdm.tqdm(train_data_loader):
            optimizer.zero_grad()
            frame = frame.float().to(args.device)
            label = label.to(args.device)
            if args.mixup:
                frame, label_a, label_b, lam = transfroms.mixup_data(frame, label)
                if args.loss == 'mse':
                    label_a = F.one_hot(label_a, 101).float()
                    label_b = F.one_hot(label_b, 101).float()
            if args.loss == 'mse':
                label_loss = F.one_hot(label, 101).float()
            else:
                label_loss = label
            if args.amp:
                with amp.autocast():
                    out_fr = net(frame)
                    if args.loss != 'tmt':
                        out_fr = out_fr.mean(0)
                        if args.mixup:
                            loss_func = transfroms.mixup_criterion(label_a, label_b, lam)
                            loss = loss_func(criterion, out_fr)
                        else:
                            loss = criterion(out_fr, label_loss)

                    else:
                        if args.mixup:
                            loss_func = transfroms.mixup_criterion(label_a, label_b, lam)
                            loss = loss_func(criterion, out_fr)
                        else:
                            loss = criterion(out_fr, label_loss)
                        out_fr = out_fr.mean(0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                out_fr = net(frame)
                if args.loss != 'tmt':
                    out_fr = out_fr.mean(0)
                    if args.mixup:
                        loss_func = transfroms.mixup_criterion(label_a, label_b, lam)
                        loss = loss_func(criterion, out_fr)
                    else:
                        loss = criterion(out_fr, label_loss)

                else:
                    if args.mixup:
                        loss_func = transfroms.mixup_criterion(label_a, label_b, lam)
                        loss = loss_func(criterion, out_fr)
                    else:
                        loss = criterion(out_fr, label_loss)
                    out_fr = out_fr.mean(0)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 101).float()
                out_fr = net(frame).mean(0)
                # loss = F.mse_loss(out_fr, label_onehot)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        # print(mon.get_avg_firing_rate(mon.get_avg_firing_rate(all=False, module_name='conv.0')))

        print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')


if __name__ == '__main__':
    main()