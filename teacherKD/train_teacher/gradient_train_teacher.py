import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

sys.path.append('../')
from TS_model import squeezenet, squeezenet_bn
from TS_model.vgg import VGG
from TS_model.resnet import ResNet18, ResNet50
from tensorboardX import SummaryWriter
from dataset import dataloader



# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR training')

parser.add_argument('--mode', type=str, default='train',
                    help='[train, test]')
parser.add_argument('--arch', type=str, default=None,
                    help='[vgg, resnet, convnet, alexnet, squeezenet]')
parser.add_argument('--depth', default=None, type=int,
                    help='depth of the neural network, 16,19 for vgg; 18, 50 for resnet')
parser.add_argument('--j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr-decay', type=int, default=60, metavar='LR_decay',
                    help='how many every epoch before lr drop (default: 30)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

parser.add_argument('--lr-scheduler', type=str, default='default',
                    help='define lr scheduler')
parser.add_argument('--warmup', action='store_true', default=False,
                    help='warm-up scheduler')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M',
                    help='warmup-lr, smaller than original lr')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M',
                    help='number of epochs for lr warmup')
parser.add_argument('--mixup', action='store_true', default=False,
                    help='ce mixup')
parser.add_argument('--alpha', type=float, default=0, metavar='M',
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='lable smooth')
parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M',
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
parser.add_argument('--model', type=str, default=None,
                    help='model for init or test')
parser.add_argument('--save_root', type=str, default='./save_root',
                    help='ckpt to save')
parser.add_argument('--gpu',type=str,default='0',help='gpu id')
parser.add_argument('--ckpt',type=str,default='../TS_model/ckpt/cifar10_resnet50_acc_94.680_sgd.pt')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:"+ args.gpu)


'''
# dog vs cat
dataset = 'dog_vs_cat'
trainset_path = 'data/dogs-vs-cats/train'
valset_path = 'data/dogs-vs-cats/val'
num_classes = 2
'''
# cifar10
dataset = 'cifar10'
trainset_path = '../../data'
valset_path = '../../data'
num_classes = 10

kwargs = {'num_workers': args.j, 'pin_memory': True} if args.cuda else {}
train_loader = dataloader.trian_dataloader(dataset, trainset_path, batch_size=32, kwargs=kwargs)
test_loader = dataloader.test_dataloader(dataset, valset_path, batch_size=32, kwargs=kwargs)

tensorboard_dir = 'tensorboard'
writers_oneEpoch = [SummaryWriter(os.path.join(tensorboard_dir, 'layer%d'%i)) for i in range(4)]

if args.arch == "vgg":
    if args.depth == 16:
        model = VGG(depth=16, init_weights=True, cfg=None, num_classes=num_classes)
    elif args.depth == 19:
        model = VGG(depth=19, init_weights=True, cfg=None, num_classes=num_classes)
    else:
        sys.exit("vgg doesn't have those depth!")
elif args.arch == "resnet":
    if args.depth == 18:
        model = ResNet18()
    elif args.depth == 50:
        model = ResNet50()
    else:
        sys.exit("resnet doesn't implement those depth!")
elif args.arch == 'squeezenet':
    args.depth = 1
    model = squeezenet.squeezenet1_0(pretrained=False, progress=True, num_classes=num_classes)
elif args.arch == 'squeezenet_bn':
    args.depth = 1
    model = squeezenet_bn.squeezenet1_1(pretrained=False, progress=True, num_classes=num_classes)

if args.ckpt is not None:
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda:' + args.gpu), strict=False)
    print('===============================Load===================')
if args.cuda:
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
model.to(device)



#############
criterion = nn.CrossEntropyLoss().to(device)
# args.smooth = args.smooth_eps > 0.0
# args.mixup = config.alpha > 0.0

optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

optimizer = None
if(args.optmzr == 'sgd'):
#    optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr,momentum=0.9, weight_decay=1e-4)
    bias_p, other_p = [], []
    for name, p in model.named_parameters():
      if 'bias' in name:
        bias_p.append(p)
      else:
        other_p.append(p)
    optimizer = torch.optim.SGD([{'params': bias_p, 'weight_decay': 0},
                                 {'params': other_p, 'weight_decay': 1e-4}], 
                                optimizer_init_lr, momentum=0.9)
elif(args.optmzr =='adam'):
    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)



scheduler = None
if args.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=4e-08)
elif args.lr_scheduler == 'default':
    # my learning rate scheduler for cifar, following https://github.com/kuangliu/pytorch-cifar
    epoch_milestones = [65, 100, 130]

    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones
    """
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*len(train_loader) for i in epoch_milestones], gamma=0.5)
else:
    raise Exception("unknown lr scheduler")

teacher_output_idx = [1, 2, 3, 4]
#############

def train(train_loader,criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        scheduler.step()

#        input = input.cuda(non_blocking=True)
#        target = target.cuda(non_blocking=True)
        input = input.to(device)
        target = target.to(device)

        # compute output
        outputs = model(input, teacher_output_idx)
        output = outputs[-1]

#        ce_loss = criterion(output, target, smooth=args.smooth)
        ce_loss = criterion(output, target) # CrossEntropyLoss

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))

        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        ce_loss.backward()

        norm_gradients = []
        for i, layer in enumerate(outputs[:-1]):
            norm_gradient = torch.norm(layer.grad).data
            if i == 0:
                norm_gradient/=8
            elif i == 1:
                norm_gradient/=4
            elif i == 2:
                norm_gradient/=2
            norm_gradients.append(norm_gradient.item())
            writers_oneEpoch[i].add_scalar('gradients', norm_gradient, step)

        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
#                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(model, criterion, test_loader):
    model.eval()
    losses = AverageMeter()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[0]
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set loss: {:.4f},  * Acc@1: {}/{} ({:.2f}%)\n'.format(
        losses.avg, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))
    return losses.avg, (100. * float(correct) / float(len(test_loader.dataset)))


def main():
    all_acc = [0.000]
    save_root = args.save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if args.model is not None:
        state_dict = torch.load(args.model)
        state_dict.pop('classifier.1.bias')
        state_dict.pop('classifier.1.weight')
        model.load_state_dict(state_dict, strict=False)
    for epoch in range(0, args.epochs):
        # if epoch in [args.epochs * 0.26, args.epochs * 0.4, args.epochs * 0.6, args.epochs * 0.83]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        # if args.lr_scheduler == "default":
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * (0.5 ** (epoch // args.lr_decay))
        # elif args.lr_scheduler == "cosine":
        #     scheduler.step()

        train(train_loader,criterion, optimizer, epoch)
        _, prec1 = test(model, criterion, test_loader)

        if prec1 > max(all_acc):
            print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
            save_path = os.path.join(save_root, "{}_{}{}_acc_{:.3f}_{}.pt".format(dataset, args.arch, args.depth, prec1, args.optmzr))
            torch.save(model.state_dict(), save_path)
            all_acc.append(prec1)
            if len(all_acc) > 1:
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(min(all_acc)))
                rm_path = os.path.join(save_root, "{}_{}{}_acc_{:.3f}_{}.pt".format(dataset, args.arch, args.depth, min(all_acc), args.optmzr))
                all_acc.remove(min(all_acc))
                if os.path.exists(rm_path):
                  os.remove(rm_path)
        if prec1 >= 99.99:
            print("accuracy is high ENOUGH!")
            break


    print("Best accuracy: " + str(max(all_acc)))


if __name__ == '__main__':
    if args.mode == 'train':
        main()
    elif args.mode == 'test':
        if args.model is None:
            raise(ValueError("for test mode, you must provide --model"))
        state_dict = torch.load(args.model)
        model.load_state_dict(state_dict, strict=False)
        _, prec1 = test(model, criterion, test_loader)
        
    else:
        raise(ValueError("No mode implemented as %s"%args.mode))
