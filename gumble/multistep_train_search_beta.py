import os
import sys
import time
import glob
import numpy as np
import json
import csv

import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search_beta import Network
from architect import Architect
from spaces import spaces_dict

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/workspace/ShareData/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0., help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--space', type=str, default='darts', help='space index')
parser.add_argument('--test', action='store_true', default=False)
# save path
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_one_shot', action='store_true', default=False, help='whether save the one-shot model')
parser.add_argument('--ckpt_file', type=str, default=None)
# gumbel
parser.add_argument('--tau_max', type=float, default=10., help='temparature of gumbel softmax')
parser.add_argument('--tau_min', type=float, default=0.1, help='temparature of gumbel softmax')
parser.add_argument('--efficient', action='store_true', default=True, help='use memory efficient strategy')

parser.add_argument('--train_weight_portion', type=float, default=0.8,
                    help='portion of training weight')
parser.add_argument('--weight_step', type=int, default=1)
parser.add_argument('--alpha_step', type=int, default=1)

args = parser.parse_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if not args.test:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not args.test:
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
torch.backends.cudnn.deterministic = True


def main():
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

    primitives = spaces_dict[args.space]

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    start_epoch = 0
    checkpoint = None

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, primitives, efficient=args.efficient,
                    drop_path_prob=args.drop_path_prob)
    # load model from ckpt
    if args.ckpt_file is not None:
        logging.info('====== Load ckpt ======')
        checkpoint = torch.load(args.ckpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        model.arch_parameters()[0].data.copy_(checkpoint['alphas_normal'])
        model.arch_parameters()[1].data.copy_(checkpoint['alphas_reduce'])
        model.arch_parameters()[2].data.copy_(checkpoint['beta_normal'])
        model.arch_parameters()[3].data.copy_(checkpoint['beta_reduce'])
        start_epoch = int(checkpoint['epoch'])

    model = model.cuda()
    logging.info("model size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'initial_lr': args.learning_rate}],
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.ckpt_file is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)  # 50000
    indices = list(range(num_train))
    split = int(np.floor(args.train_weight_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min, last_epoch=start_epoch - 1)

    architect = Architect(model, args)

    ckpt_dir = os.path.join(args.save, 'ckpt')
    result_dir = os.path.join(args.save, 'results_of_7q')  # preserve the results
    genotype_dir = os.path.join(result_dir, 'genotype')  # preserve the argmax genotype for each epoch
    alpha_dir = os.path.join(result_dir, 'alpha')
    beta_dir = os.path.join(result_dir, 'beta')

    if not args.test:
      if not os.path.exists(ckpt_dir):
          os.makedirs(ckpt_dir)
      if not os.path.exists(result_dir):
          os.makedirs(result_dir)
      if not os.path.exists(genotype_dir):
          os.makedirs(genotype_dir)
      if not os.path.exists(alpha_dir):
          os.makedirs(alpha_dir)
      if not os.path.exists(beta_dir):
          os.makedirs(beta_dir)

      result_file = os.path.join(result_dir, 'results.csv')
      f = open(result_file, 'w')
      writer = csv.writer(f)
      title = ['epoch', 'val_acc', 'val_loss', 'train_acc', 'train_loss']
      writer.writerow(title)
      f.close()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]

        if args.drop_path_prob != 0:
            model.drop_path_prob = args.drop_path_prob * epoch / (args.epochs - 1)
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr: %e | drop_prob: %e | cutout_prob: %e', epoch, lr, model.drop_path_prob,
                     train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr: %e' , epoch, lr)
        model.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.epochs - 1))
        logging.info('tau %f', model.get_tau())

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, optimizer, lr)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(test_queue, model)
        logging.info('valid_acc %f', valid_acc)

        # save
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'arch_optimizer': architect.optimizer.state_dict(),
            'alphas_normal': model.alphas_normal.data,
            'alphas_reduce': model.alphas_reduce.data,
            'beta_normal': model.beta_normal.data,
            'beta_reduce': model.beta_reduce.data,
        }, False, ckpt_dir)

        genotype_file = os.path.join(genotype_dir, '%d.txt' % epoch)
        with open(genotype_file, 'w') as f:
            json.dump(genotype._asdict(), f)
            # to recover: genotype = genotype(**dict)
        # save the alpha weights
        alpha_file = os.path.join(alpha_dir, '%d.txt' % epoch)
        beta_file = os.path.join(beta_dir, '%d.txt' % epoch)
        alpha_weights = model.arch_parameters()
        alphas = {}
        betas = {}
        alphas['alphas_normal'] = F.softmax(alpha_weights[0], dim=-1).data.cpu().numpy().tolist()
        alphas['alphas_reduce'] = F.softmax(alpha_weights[1], dim=-1).data.cpu().numpy().tolist()
        betas['betas_normal'] = alpha_weights[2].data.cpu().numpy().tolist()
        betas['betas_reduce'] = alpha_weights[3].data.cpu().numpy().tolist()
        with open(alpha_file, 'w') as f:
            json.dump(alphas, f)
        with open(beta_file, 'w') as f:
            json.dump(betas, f)

        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            a = [epoch, valid_acc, valid_obj, train_acc, train_obj]
            writer.writerow(a)


def train(train_queue, valid_queue, model, architect, optimizer, lr):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()
    optimizer.zero_grad()

    for step, (input, target) in enumerate(train_queue):
      n = input.size(0)  # input.size :[32,3,32,32]

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      optimizer.zero_grad()
      loss = None
      logits = None
      for i in range(args.weight_step):
          loss, logits = model.compute_loss(input, target, return_logits=True)
          loss.backward()

      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()


      [prec1] = utils.accuracy(logits, target)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)

      if step % args.report_freq == 0:
        logging.info('train weight %03d  |Acc= %.3f  |loss= %e', step, top1.avg, objs.avg)


    for step, (input_search, target_search) in enumerate(valid_queue):
        n = input_search.size(0)

        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        architect.optimizer.zero_grad()
        loss = None
        logits = None
        for i in range(args.alpha_step):
            loss, logits = model.compute_loss(input_search, target_search, return_logits=True)
            loss /= args.alpha_step
            loss.backward()
        architect.optimizer.step()

        [prec1] = utils.accuracy(logits, target_search)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        if step % args.report_freq == 0:
            logging.info('train alpha %03d  |Acc= %.3f  |loss= %e', step, top1.avg, objs.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            loss, logits = model.compute_loss(input, target, return_logits=True)

            n = input.size(0)
            [prec1] = utils.accuracy(logits, target)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d  |Acc= %.3f  |loss= %e', step, top1.avg, objs.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

