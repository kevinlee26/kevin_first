import argparse
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from architectures import ARCHITECTURES
from datasets import DATASETS
from train_utils import AverageMeter, accuracy, log, requires_grad_, test
from train_utils import prologue

from consistency import consistency_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS)
parser.add_argument('--arch', default='cifar_resnet110', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=50,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=1.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--id', default=None, type=int,
                    help='experiment id, `randint(10000)` if None')

#####################
# Options added by Salman et al. (2019)
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='logs_teacher/checkpoint.pth.tar',
                    help='Path to a pretrained model')

#####################
parser.add_argument('--num-noise-vec', default=2, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=10., type=float)
parser.add_argument('--eta', default=0.5, type=float)
parser.add_argument('--dis', default=0.1, type=float)

# Options when SmoothAdv is used (Salman et al., 2019)
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--warmup', default=10, type=int, help="Number of epochs over which "
                                                           "the maximum allowed perturbation increases linearly "
                                                           "from zero to args.epsilon.")

args = parser.parse_args()
args.outdir = f"logs"
args.epsilon /= 256.0


def main():
    train_loader, test_loader, criterion, model, model_t, optimizer, scheduler, \
    starting_epoch, logfilename, logfilename_1, model_path, device = prologue(args)


    for epoch in range(starting_epoch, args.epochs):

        before = time.time()
        train_loss, train_acc = train(train_loader, model, model_t, criterion, optimizer, epoch,
                                      args.noise_sd, device, logfilename_1)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, args.noise_sd, device, args.print_freq)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)


def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


def train(loader: DataLoader, model: torch.nn.Module, model_t: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,
          device: torch.device, logfilename_1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    requires_grad_(model, True)
    model_t.eval()
    requires_grad_(model_t, False)

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            noises = [torch.randn_like(inputs, device=device) * noise_sd
                      for _ in range(args.num_noise_vec)]

            # augment inputs with noise
            inputs_c = torch.cat([inputs + noise for noise in noises], dim=0)
            targets_c = targets.repeat(args.num_noise_vec)

            logits = model(inputs_c)
            logits_t = model_t(inputs_c)

            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            logits_t_chunk = torch.chunk(logits_t, args.num_noise_vec, dim=0)            
            loss_con = consistency_loss(logits_chunk, logits_t_chunk, logfilename_1, args.lbd, targets_c, epoch, args.dis, args.eta)

            loss_xent = criterion(logits, targets_c)
            loss = loss_xent + loss_con
            
            acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_reg.update(loss_con.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)

if __name__ == "__main__":
    main()
