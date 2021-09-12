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
import model_KD
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/zhangzhexi/darts-master/data/', help='location of the data corpus')
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
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# save path
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--save_one_shot', action='store_true', default=False, help='whether save the one-shot model')

# For Knowledge Distillation
parser.add_argument('--teacher', type=str, default='res50', help='name of teacher model')
parser.add_argument('--teacher_ckpt', type=str, default='TS_model/ckpt/cifar10_resnet50_acc_94.680_sgd.pt', help='the ckpt path of teacher')
parser.add_argument('--T', type=float, default=1., help='temparature of KD')
parser.add_argument('--kd_alpha', type=float, default=0.5, help='weight of KDLoss')
parser.add_argument('--student_output_idx', type=list, nargs='*', default=[], help='list of the output cell index for one-shot model')

args = parser.parse_args()
CIFAR_CLASSES=10

def main():
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = model_KD.Network_KD(args.init_channels, CIFAR_CLASSES, args.layers, criterion, teacher=args.teacher,
                                teacher_ckpt=args.teacher_ckpt, student_output_idx=args.student_output_idx)
    model=model.cuda()
    teacher_model = model.teacher_model
    student_model = model.student_model

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

    for step, (input, target) in enumerate(train_queue):
        if step>0: break
        model.student_model.eval()
        model.teacher_model.eval()
        n = input.size(0)
        #print('input size:', input.size())
        input = Variable(input, requires_grad=False).cuda()
        student_outputs, teacher_outputs = model(input)
        print(len(teacher_outputs),teacher_outputs[0].shape)
        #print(len(student_outputs),len(student_outputs[0]))
if __name__ == '__main__':
    main()