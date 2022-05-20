import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from models.VGG_models import *
import torch.nn.functional as F
from functions import TET_loss, seed_all, get_logger
from tqdm import tqdm
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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
        outputs = model(images)
        mean_out = outputs.mean(1)
        if args.TET:
            loss = TET_loss(outputs,labels,criterion,args.means,args.lamb)
        else:
            loss = criterion(mean_out,labels)
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = VGGSNN_TCJA()

    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0
    log_name = 'TET_TCJA_128-before-pool-k2-1024-514'
    logger = get_logger(log_name+'.log')
    logger.info('start training!')
    if args.resume:
        model.load_state_dict(torch.load('TET_TCJA_128-before-pool-k2-1024.pth', map_location='cpu'))

    for i in range(514):
        scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    for epoch in range(args.epochs):
    
        loss, acc = train(model, device, train_loader, criterion, optimizer, epoch, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc ))
        scheduler.step()
        facc = test(model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch , args.epochs, facc ))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), log_name+'.pth')
        logger.info('Best acc={:.3f}'.format(best_acc ))
        print('\n')
