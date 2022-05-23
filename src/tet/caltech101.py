import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import math

from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from spikingjelly.datasets.n_caltech101 import NCaltech101
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import tqdm
import argparse
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np

from src.layers import TCJA, VotingLayer


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


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
        conv.append(TCJA(4, 4, 14, 256))
        conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
        conv.extend(CextNet.conv3x3(256, 512))
        conv.append(TCJA(4, 4, 14, 512))
        conv.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(512 * 5 * 7, 1024, bias=False)),
            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True,
                                    backend='cupy'),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(1024, 101, bias=False)), )
        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
        return out_spikes

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),

            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True,
                                    backend='cupy'),
        ]


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

    parser.add_argument('-opt', default='Adam', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-mixup', action='store_true')
    parser.add_argument('-add_dict', type=str)
    parser.add_argument('-loss', type=str, default='mse')
    parser.add_argument('-dts_cache', type=str, default='./dts_cache')

    args = parser.parse_args()
    print(args)

    net = CextNet(channels=args.channels)

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
        origin_set = NCaltech101(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number', )

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

    out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_c_{args.channels}_{args.opt}_lr_{args.lr}_tmt')
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
    criterion = nn.CrossEntropyLoss().to(args.device)
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
            if args.amp:
                with amp.autocast():
                    out_fr = net(frame)
                    loss_fr = out_fr.permute(1, 0, 2)

                    loss = TET_loss(loss_fr, label, criterion, 1.0, 0.001)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            out_fr = out_fr.mean(0)
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
