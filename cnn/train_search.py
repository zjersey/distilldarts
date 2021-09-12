import os
import sys
import time
import itertools
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
import model_KD
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/zhangzhexi/darts-master/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# save path
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_one_shot', action='store_true', default=False, help='whether save the one-shot model')

# For Knowledge Distillation
parser.add_argument('--teacher', type=str, default='res50', help='name of teacher model')
parser.add_argument('--teacher_ckpt', type=str, default='TS_model/ckpt/cifar10_resnet50_acc_94.680_sgd.pt', help='the ckpt path of teacher')
parser.add_argument('--T', type=float, default=1., help='temparature of KD')
parser.add_argument('--kd_alpha', type=float, default=0.5, help='weight of KD_logits_Loss')
parser.add_argument('--kd_beta', type=float, default=0.5, help='weight of KD_FeatureMap_loss')
parser.add_argument('--use_kd', action='store_true', default=True, help='whether use kd loss')
parser.add_argument('--fm_method', type=str, default=None, help='method used to unify dimension in fmloss')


args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
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
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = model_KD.Network_KD(args.init_channels, CIFAR_CLASSES, args.layers, criterion,
                              teacher=args.teacher, teacher_ckpt=args.teacher_ckpt, gpuLocation=args.gpu,
                              use_kd=args.use_kd, fm_method=args.fm_method,
                              kd_alpha=args.kd_alpha, kd_beta=args.kd_beta)
  model = model.cuda()
  logging.info("student param size = %fMB", utils.count_parameters_in_MB(model.teacher_model))
  logging.info("teacher param size = %fMB", utils.count_parameters_in_MB(model.student_model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  # test teacher_model
  #valid_acc, valid_obj = teacher_infer(valid_queue, model, criterion)
  #logging.info('teacher valid_acc %f', valid_acc)


  results = {}
  results['val_loss'] = []
  results['val_acc'] = []

  ckpt_dir = os.path.join(args.save, 'ckpt')
  result_dir = os.path.join(args.save, 'results_of_7q') # preserve the results
  genotype_dir = os.path.join(result_dir, 'genotype') # preserve the argmax genotype for each epoch  - q3,5,7
  alpha_dir = os.path.join(result_dir, 'alpha')
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  if not os.path.exists(genotype_dir):
    os.makedirs(genotype_dir)
  if not os.path.exists(alpha_dir):
    os.makedirs(alpha_dir)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.student_model.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(student_model.alphas_normal, dim=-1))
    #print(F.softmax(student_model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc = infer(valid_queue, model)
    logging.info('valid_acc %f', valid_acc)

    if args.save_one_shot:
      utils.save(model, os.path.join(ckpt_dir, 'weights_%d.pt'%epoch))

    # for seven questions
    #q1 & q2
    results['val_acc'].append(valid_acc)
    #q3,5,7
    genotype_file = os.path.join(genotype_dir, '%d.txt'%epoch)
    with open(genotype_file, 'w') as f:
      json.dump(genotype._asdict(), f)
      # to recover: genotype = genotype(**dict)
    #q6: save the alpha weights
    alpha_file = os.path.join(alpha_dir, '%d.txt'%epoch)
    alpha_weights = model.student_model.arch_parameters()
    alphas = {}
    alphas['alphas_normal'] = F.softmax(alpha_weights[0], dim=-1).data.cpu().numpy().tolist()
    alphas['alphas_reduce'] = F.softmax(alpha_weights[1], dim=-1).data.cpu().numpy().tolist()
    with open(alpha_file, 'w') as f:
      json.dump(alphas, f)

  # save the results:
  """
  result_file = os.path.join(result_dir, 'results.csv')
  with open(result_file, 'w') as f:
    writer = csv.writer(f)
    title = ['epoch', 'val_loss', 'val_acc', 'argmax_edge_top1', 'argmax_edge_loss', 'sample_edge_top1', 'sample_edge_loss', 'argmax_operator_top1', 'argmax_operator_loss', 'sample_operator_top1', 'sample_operator_loss']
    writer.writerow(title)
    for epoch, val_loss in enumerate(results['val_loss']):
      a = [epoch, val_loss, results['val_acc'][epoch]]
      writer.writerow(a)
  """



def train(train_queue, valid_queue, model, architect, optimizer, lr):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)  #input.size :[32,3,32,32]

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()

    loss, logits = model.compute_loss(input, target, True) #student_outputs:[1,32,10]

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)


    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model):
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda()

      student_outputs, teacher_outputs = model(input)

    prec1, prec5 = utils.accuracy(student_outputs[-1], target, topk=(1, 5))
    n = input.size(0)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      #logging.info('valid %03d |CELoss= %.3e |KDLogitsLoss= %.3e |KDFeatureMapLoss= %.3e |Acc= %.3f@1 %.3f@5', step, objs.avg, objs_kd.avg, objs_kd_feature.avg, top1.avg, top5.avg)
      logging.info('valid %03d  |Acc= %.3f@1 %.3f@5', step, top1.avg, top5.avg)
  return top1.avg


"""
def teacher_infer(valid_queue, model, criterion):
  objs = utils.AverageMeter()
  objs_kd = utils.AverageMeter()
  #objs_kd_feature = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda()

      student_outputs, teacher_outputs = model(input)
      loss = criterion(teacher_outputs[-1], target)
      kd_logits_loss = model_KD.KD_logits_loss(student_outputs[-1], teacher_outputs[-1], T=args.T)
      kd_feature_map_loss = model_KD.KD_FeatureMap_loss(student_outputs[:-1], teacher_outputs[:-1])

    prec1, prec5 = utils.accuracy(teacher_outputs[-1], target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    objs_kd.update(kd_logits_loss.item(), n)
    #objs_kd_feature.update(kd_feature_map_loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      #logging.info('valid %03d |CELoss= %.3e |KDLogitsLoss= %.3e |KDFeatureMapLoss= %.3e |Acc= %.3f@1 %.3f@5', step, objs.avg, objs_kd.avg, objs_kd_feature.avg, top1.avg, top5.avg)
      logging.info('valid %03d |CELoss= %.3e |KDLogitsLoss= %.3e  |Acc= %.3f@1 %.3f@5', step, objs.avg, objs_kd.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg
"""

if __name__ == '__main__':
  main() 

