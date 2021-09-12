import argparse
import json
import os
import sys
from utils import *
import genotypes 
from model import NetworkImageNet as Network
from torchstat import stat


parser = argparse.ArgumentParser("flop")
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--arch', type=str, default='DARTS_V1', help='which architecture to use')
parser.add_argument('--base_dir',type=str,help='model dir')
parser.add_argument('--specific_model',type=int, default=49, help='the number of model to train')
parser.add_argument('--input_size',type=int, default=32, help='the number of model to train')

args = parser.parse_args()

CLASSES = 1000

args = parser.parse_args()

def main():
  if args.base_dir is None:
    genotype = eval("genotypes.%s" % args.arch)
  else:
    genotype_path = os.path.join(args.base_dir, 'results_of_7q/genotype')
    genotype_file = os.path.join(genotype_path, '%s.txt'%args.specific_model)
    tmp_dict = json.load(open(genotype_file,'r'))
    genotype = genotypes.Genotype(**tmp_dict)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model.eval()
  model.drop_path_prob = 0.
  stat(model,(3, args.input_size, args.input_size))
if __name__ == '__main__':
  main() 

