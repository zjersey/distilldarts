import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import json
import csv

import torch.nn as nn
from utils import Genotype
import compare_genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

sys.path.append('../')
import utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/workspace/ShareData/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--base_dir', type=str, help='model dir')
parser.add_argument('--max_num', type=int, default=12, help='max num of different genotypes')
parser.add_argument('--from_scratch', action='store_true', default=False, help='train from scratch')
parser.add_argument('--specific_model', type=int, default=49, help='the number of model to train')
parser.add_argument('--eval_dir', type=str, default=None, help='dir to save 600-epoch evaluation')
parser.add_argument('--ckpt_file', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--nclasses', type=int, default=10)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'eval_log-doubleSepConv_16C.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %.3f %.3f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %.3f %.3f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def train_from_scratch(args, train_queue, valid_queue, genotype):
    start_epoch = 0
    checkpoint = None
    model = Network(args.init_channels, args.nclasses, args.layers, args.auxiliary, genotype)
    if args.epochs==600 and args.ckpt_file is not None:
        logging.info('====== Load ckpt ======')
        checkpoint = torch.load(args.ckpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = int(checkpoint['epoch'])

    model = model.cuda()
    # clear unused gpu memory
    torch.cuda.empty_cache()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if args.epochs == 600:
        logging.info("eval_dir: %s", args.eval_dir)
        ckpt_dir = os.path.join(args.eval_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'initial_lr' : args.learning_rate}],
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, last_epoch=start_epoch-1)

    best_acc = 0
    best_loss = 0
    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %.3f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %.3f', valid_acc)

        if args.epochs == 600:
            utils.save_checkpoint({
                'epoch':  epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, save=ckpt_dir)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_loss = valid_obj
    return best_acc, best_loss


# def search_optim_arch(choose_type='fromfile'):
def search_optim_arch(base_dir, genotype_names):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        args.nclasses = 10
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        args.nclasses = 100
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)
        args.nclasses = 10

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
    # alpha_path = os.path.join(base_dir, 'results_of_7q/alpha')

    if args.epochs != 600:
        eval_dir = os.path.join(base_dir, 'hyperband/hyperband%d-%depochs' % (len(genotype_names), args.epochs))
    else:
        eval_dir = os.path.join(base_dir, 'hyperband/600epochs-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
        args.eval_dir = eval_dir

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    val_acc = []
    val_loss = []
    num_sample = 1
    for name in genotype_names:
        genotype_file = os.path.join(genotype_path, '%s.txt' % name)
        tmp_dict = json.load(open(genotype_file, 'r'))
        genotype = Genotype(**tmp_dict)
        print(genotype)
        best_acc, best_loss = train_from_scratch(args, train_queue, valid_queue, genotype)
        val_acc.append(best_acc)
        val_loss.append(best_loss)
        print('Best: %f / %f' % (best_loss, best_acc))

    # print best results
    for idx, res in enumerate(val_acc):
        print(genotype_names[int(idx / num_sample)], res)
    res = np.array(val_acc)
    sort_idx = np.argsort(res)[::-1]
    sort_names = []
    for idx in sort_idx:
        sort_names.append(genotype_names[int(idx / num_sample)])
        print(idx, genotype_names[int(idx / num_sample)], val_acc[idx])

    # save the results
    result_file = os.path.join(eval_dir, 'results.csv')
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        title = ['genotype_name', 'val_loss', 'val_acc']
        writer.writerow(title)
        for idx, loss in enumerate(val_loss):
            a = [genotype_names[int(idx / num_sample)], loss, val_acc[idx]]
            writer.writerow(a)


if __name__ == '__main__':
    if args.seed < 0:
        args.seed = np.random.randint(low=0, high=10000)
    base_dir = args.base_dir
    if args.from_scratch:
        num_search = [1]
        hyperband_epochs = [600]
        sorted_names = [args.specific_model]
    else:
        genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
        genotype_names = compare_genotypes.get_distinct_genotype(genotype_path, None)
        if len(genotype_names) > args.max_num:
            genotype_names = genotype_names[-args.max_num:]
        print('num of different genotypes:', len(genotype_names))
        num_search = [len(genotype_names), 6, 1]
        hyperband_epochs = [20, 60, 600]
        assert (len(num_search) == len(hyperband_epochs))
        sorted_names = genotype_names
    for i, epochs in enumerate(hyperband_epochs):
        sorted_names = sorted_names[:num_search[i]]
        args.epochs = epochs
        print("\n############")
        print("hyperband search genotype names: ", sorted_names)
        print("search for %d epochs " % args.epochs)
        print("############\n")
        sorted_names = search_optim_arch(base_dir, sorted_names)






