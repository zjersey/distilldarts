import json
import csv
import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from architect import Architect
from genotypes import STEPS

import gc

import data
import model_search as model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint
import utils

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')

# my add
parser.add_argument('--save_one_shot', action='store_true', default=False, help='whether save the one-shot model')
# For Prune
parser.add_argument('--sal_type', type=str, default='task', help='type of saliency: task or naive')
parser.add_argument('--warmup', type=int, default=10, help='epochs of warm up before pruning')
parser.add_argument('--num_compare', type=int, default=1, help='The number of candidate ops before pruning (The top least important in saliency). We need to calculate the loss and acc of these candidates and prune the least important one')
parser.add_argument('--iter_compare', type=int, default=30, help='The iteration (batch) to run each comparison candidate')

# For regularization of pruned alphas
parser.add_argument('--reg_type', type=str, default='norm', help='type of regularization: norm, gini, or entropy')
parser.add_argument('--reg_alpha', type=float, default=0.1, help='weight of KD_logits_Loss')
parser.add_argument('--sal_second', action='store_true', default=False, help='keep second order of Taylor Expansion')
parser.add_argument('--no_restrict', action='store_true', default=False, help='use cutout')
parser.add_argument('--num_kept', type=int, default=8, help='The number of reserved ops after discretization or pruning')

args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
if args.seed is None: args.seed = -1
if args.seed < 0:
  args.seed = np.random.randint(low=0, high=10000)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, args)
search_data = batchify(corpus.valid, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


ntokens = len(corpus.dictionary)
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = model.RNNModelSearch(ntokens, args.emsize, args.nhid, args.nhidlast, 
                       args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute)

size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')
logging.info(model.genotype())

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model
architect = Architect(parallel_model, args)

total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))


def evaluate(data_source, model, batch_size=10, num_iter=-1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    with torch.no_grad():
      ntokens = len(corpus.dictionary)
      hidden = model.init_hidden(batch_size)
      if num_iter < 0: num_iter = int(len(data_source)/args.bptt)
      for i in range(0, num_iter*args.bptt, args.bptt):
          data, targets = get_batch(data_source, i, args, evaluation=True)
          targets = targets.view(-1)
  
          log_prob, hidden = model(data, hidden)
          loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
  
          total_loss += loss * len(data)
  
          hidden = repackage_hidden(hidden)
#    return total_loss[0] / len(data_source)
    return total_loss.item() / (num_iter * args.bptt)


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args)
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
            cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])
            hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

            model.update_softmax_arch_parameters()
            hidden_valid[s_id], grad_norm = architect.step(
                    hidden[s_id], cur_data, cur_targets,
                    hidden_valid[s_id], cur_data_valid, cur_targets_valid,
                    optimizer,
                    args.unrolled)
            model.update_softmax_arch_parameters()

            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
              loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            logging.info(parallel_model.genotype())
            print(F.softmax(parallel_model.weights, dim=-1))
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len

def compute_sal(search_queue, sal_batch_size, model, sal_type='task', sal_file=None, second_order=False):
  ###################
  # naive saliency
  ###################
  if sal_type == 'naive':
    n_saliencys = model.student_model.get_softmax_arch_parameters()
    print("Saliency")
    print(n_saliencys)
    return n_saliencys

#  model.eval()
  model.train()
  if second_order:
      non_zero = 0.
      non_zero += model.prune_mask.sum()
      num_iter = 500/non_zero
      print(non_zero, num_iter)
  else: num_iter = len(search_queue)
  num_iter = int(num_iter / args.bptt)

  t_saliencys = [0]
  ntokens = len(corpus.dictionary)
  hidden = model.init_hidden(sal_batch_size)
  num_iter = int(len(search_queue)/args.bptt)
  for i in range(0, num_iter*args.bptt, args.bptt):
      alphas = model.get_softmax_arch_parameters()
      data, targets = get_batch(search_queue, i, args)
      targets = targets.view(-1)

      hidden = repackage_hidden(hidden)
      log_prob, hidden = model(data, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

      if second_order:
        saliency_utils.zero_grads(model.parameters())
        saliency_utils.zero_grads(alphas)
        hessian = saliency_utils._hessian(loss, alphas, model.student_model.prune_masks)
#        print(hessian)
        diags = hessian.diag()
        diags = [diags[:alphas[0].numel()].view(alphas[0].shape), diags[alphas[0].numel():].view(alphas[1].shape)]

        grads_alphas = torch.autograd.grad(loss, alphas, grad_outputs=None,
                                      allow_unused=True,
                                      retain_graph=None,
                                      create_graph=False)
        t_saliencys = [t_saliencys[k]+grad*(-alpha.detach())+diag*alpha.detach().pow(2) for k, (grad, alpha, diag) in enumerate(zip(grads_alphas, alphas, diags))]
        
      else:
        grads_alphas = torch.autograd.grad(loss, alphas, grad_outputs=None,
                                      allow_unused=True,
                                      retain_graph=None,
                                      create_graph=False)
        t_saliencys = [t_saliencys[k]+grad*(-alpha.detach()) for k, (grad, alpha) in enumerate(zip(grads_alphas, alphas))]
#        print(grads_alphas)
#        # judge whether grads_alphas is correct
#        grads_arch_param = torch.autograd.grad(loss, model.student_model.arch_parameters(), grad_outputs=None,
#                                      allow_unused=True,
#                                      retain_graph=None,
#                                      create_graph=False)
#        correct_alphas = []
#        for i, g in enumerate(grads_alphas):
#          const = (g*alphas[i]).sum(dim=-1, keepdim=True)
#          compute_grad_arch_param = (g-const) * alphas[i]
#          diff = (compute_grad_arch_param - grads_arch_param[i]).abs()
#          print(diff.min(), diff.max(), diff.sum())
#        assert(0)

      if i % args.log_interval == 0:
        logging.info('saliency step %03d/%d', int(i/args.bptt), num_iter)

  step = int(i/args.bptt)
  saliency = [v/step for v in t_saliencys]
  print("Saliency")
  print(saliency)
  # save saliency
  if sal_file is not None:
    save_sal = {}
    save_sal['weights'] = saliency[0].data.cpu().numpy().tolist()
    with open(sal_file, 'w') as f:
      json.dump(save_sal, f)
  return saliency
   

def prune_by_mask(model, saliency, num_compare=1, optim_e=0, train_queue=None, valid_queue=None, no_restrict=False):

  def can_prune_pos(mask, pos_r, pos_c):
    if mask[pos_r, pos_c] == 0: return False
    if no_restrict: return True
    start = 0
    for i in range(STEPS):
      end = start + i + 1
      if start <= pos_r and end > pos_r:
        num_valid = mask[start:end].sum()
        if num_valid > 1: return True
        elif num_valid == 1: return False
        else: raise(ValueError("The code runs wrong, since there is no selected edge at %d row"%pos_r))
      start = end
    raise(ValueError("The code runs wrong, since there is no %d rows in the mask"%pos_r))

  def refine_sal(mask, sal, max_value):
    ''' Since DARTS add some restriction on the degree of each node, so if a node has and only has two edges, we cannot prune the edges to this node anymore'''
    num_valid = mask.sum()
    if num_valid > 1: return sal
    elif num_valid == 1: sal.fill_(max_value)
    else:
      raise(ValueError("The code runs wrong, since it cannot make sure each node wons at least 2 input edges"))
    return sal

  R,C = saliency[0].shape
  mask = model.prune_mask
  saliency = saliency[0].abs()
  max_sal = saliency.max()
  # do not prune the same edge repeatedly
  saliency = torch.where(mask==1, saliency, max_sal)
  if not no_restrict:
    # make sure each node owns at least 1 input edges
    start = 0
    for i in range(STEPS):
      end = start + i + 1
      saliency[start:end] = refine_sal(mask[start:end], saliency[start:end], max_sal)
      start = end

  alpha = model.get_softmax_arch_parameters()[0]
  if num_compare <= 1:
      # update mask_normal
#      tmp, pos_c = s_normal.min(dim=1)
#      pos_r = tmp.argmin(dim=0)
#      pos_c = pos_c[pos_r]
      saliency = saliency.view(-1).cpu().data.numpy()
      idx = saliency.argsort()[0]
      pos_r = int(idx / C)
      pos_c = int(idx - pos_r*C)

      mask[pos_r, pos_c] = 0
      sal_change = saliency[idx]
      print("Prune normal idx: ", pos_r, pos_c)
      print("Saliency and beta: ", saliency[idx], alpha[pos_r, pos_c])

  else:
      new_model = model.new().cuda()
      model_dict = model.state_dict()
      new_model.load_state_dict(model_dict, strict=True)

      new_model.update_softmax_arch_parameters()
      new_alpha = new_model.get_softmax_arch_parameters()[0]
      new_alpha.data.mul_(mask.float())

      saliency = saliency.view(-1).cpu().data.numpy()
      idxes = saliency.argsort()[:num_compare]
#      print('debug:', saliency[idxes])
      best_acc = 0.
      best_loss = 100.
      losses = []
      for idx in idxes:
        pos_r = int(idx / C)
        pos_c = int(idx - pos_r*C)
        if not can_prune_pos(mask, pos_r, pos_c): continue
#        if mask[pos_r, pos_c] == 0: continue
#        print('debug:', pos_r, pos_c, saliency[idx], saliency[0].abs()[pos_r, pos_c])
        tmp_alpha = new_alpha[pos_r, pos_c].item()
        new_alpha[pos_r, pos_c].fill_(0.)
        loss = evaluate(valid_queue, new_model, eval_batch_size, num_iter=args.iter_compare)
#        print('debug:', new_alpha)
        losses.append(loss)
        if loss < best_loss:
          best_pos_r = pos_r
          best_pos_c = pos_c
          best_loss = loss
        new_alpha[pos_r, pos_c].fill_(tmp_alpha)
#        print('debug:', new_alpha)
      mask[best_pos_r, best_pos_c] = 0
      sal_change = saliency[best_pos_r*C+best_pos_c]
      print("Prune idx: ", best_pos_r, best_pos_c)
      print('losses', losses)
      print("Saliency and beta and best loss: ", saliency[best_pos_r*C+best_pos_c], new_alpha[best_pos_r, best_pos_c], best_loss)
      del new_model
#      assert 0, "Need to debug"

  print("After pruning, saliency changed %f"%(sal_change))
  model.prune(mask)
  model.update_softmax_arch_parameters()
  if optim_e > 0:
    # fine-tune
    raise(NotImplementedError("fine-tune has not been completed"))
#    print("Fine-tune begin")
#    fine_tune(model, optim_e, train_queue, valid_queue)

  return mask


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

ckpt_dir = os.path.join(args.save, 'ckpt')
result_dir = os.path.join(args.save, 'results_of_7q') # preserve the results
genotype_dir = os.path.join(result_dir, 'genotype') # preserve the argmax genotype for each epoch  - q3,5,7
alpha_dir = os.path.join(result_dir, 'alpha')
sal_dir = os.path.join(result_dir, 'sal')
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
if not os.path.exists(result_dir):
  os.makedirs(result_dir)
if not os.path.exists(genotype_dir):
  os.makedirs(genotype_dir)
if not os.path.exists(alpha_dir):
  os.makedirs(alpha_dir)
if not os.path.exists(sal_dir):
  os.makedirs(sal_dir)

results = {}
results['val_loss'] = []
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    print(model.get_softmax_arch_parameters())
    train()

    val_loss = evaluate(val_data, parallel_model, eval_batch_size)
    results['val_loss'].append(val_loss)
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    logging.info('-' * 89)

#    if val_loss < stored_loss:
#        save_checkpoint(model, optimizer, epoch, args.save)
#        logging.info('Saving Normal!')
#        stored_loss = val_loss
#    best_val_loss.append(val_loss)

    # compute saliency & prune
    prune = epoch >= args.warmup and model.has_redundent_op(no_restrict=args.no_restrict)
#    prune = False
    if prune:
      optim_e = 0
      sal_file = os.path.join(sal_dir, '%d.txt'%epoch)
#      torch.cuda.empty_cache()
      model.update_softmax_arch_parameters()
      saliency = compute_sal(search_data, args.batch_size, model, sal_type=args.sal_type, sal_file=sal_file, second_order=args.sal_second)
#      torch.cuda.empty_cache()

      prune_mask = prune_by_mask(model, saliency, args.num_compare, optim_e, train_data, val_data, no_restrict=args.no_restrict)
      model.update_softmax_arch_parameters()
      print("-"*10)
      print("prune_mask:")
      print(prune_mask)
      print("-"*10)

    genotype = model.genotype(args.no_restrict)
    logging.info('genotype = %s', genotype)

    if args.save_one_shot:
      utils.save_supernet(student_model, os.path.join(ckpt_dir, 'weights_%d.pt'%epoch))

    # for seven questions
    #q3,5,7
    genotype_file = os.path.join(genotype_dir, '%d.txt'%epoch)
    with open(genotype_file, 'w') as f:
      json.dump(genotype._asdict(), f)
      # to recover: genotype = genotype(**dict)
    #q6: save the alpha weights
    alpha_file = os.path.join(alpha_dir, '%d.txt'%epoch)
    alphas = {}
    alphas['weights'] = model.softmax_weights.data.cpu().numpy().tolist()
    with open(alpha_file, 'w') as f:
      json.dump(alphas, f)

# save the results:
result_file = os.path.join(result_dir, 'results.csv')
with open(result_file, 'w') as f:
  writer = csv.writer(f)
  title = ['epoch', 'val_loss']
  writer.writerow(title)
  for epoch, val_loss in enumerate(results['val_loss']):
    a = [epoch, val_loss, results['val_acc'][epoch]]
    writer.writerow(a)
