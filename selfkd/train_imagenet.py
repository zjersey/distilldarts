import os
import sys
import numpy as np
import time
import torch
import utils
import logging
import argparse
import torch.nn as nn
from utils import Genotype
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import json
from model import NetworkImageNet as Network

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/home/wangxiaoxing/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--gpus', type=str, default='0,1', help='gpu device ids')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--base_dir', type=str, help='model dir')
parser.add_argument('--specific_model', type=int, default=49, help='the number of model to train')
parser.add_argument('--ckpt_file', type=str, default=None)

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
if not os.path.exists(args.save):
    os.makedirs(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    device = None
    if args.parallel:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = torch.device("cuda:0")
    else:
        torch.cuda.set_device(args.gpu)
        logging.info('gpu device = %d' % args.gpu)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    genotype_path = os.path.join(args.base_dir, 'results_of_7q/genotype')
    genotype_file = os.path.join(genotype_path, '%d.txt' % args.specific_model)
    tmp_dict = json.load(open(genotype_file, 'r'))
    genotype = Genotype(**tmp_dict)
    print(genotype)

    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)

    checkpoint = None
    start_epoch = 0

    if args.ckpt_file is not None:
        logging.info('====== Load ckpt ======')
        checkpoint = torch.load(args.ckpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = int(checkpoint['epoch'])

    if args.parallel:
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpus.split()))))
        model.to(device)
    else:
        model = model.cuda()

    logging.info('start_epoch: %d' % start_epoch)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.ckpt_file is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma, last_epoch = start_epoch)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc_top1 = 0
    for epoch in range(start_epoch, args.epochs):
        current_lr = adjust_lr(optimizer, epoch)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        if args.parallel:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # scheduler.step()
        # logging.info('epoch %d lr %e', start_epoch+epoch, scheduler.get_lr()[0])

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('train_acc %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d loss: %e top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d loss: %e top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()