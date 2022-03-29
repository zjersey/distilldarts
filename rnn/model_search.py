import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
#    probs = F.softmax(self.weights, dim=-1)
    probs = self.weights

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none':
          continue
        if sum(self.prune_mask[offset:offset+i+1, k]) == 0:
          continue
        fn = self._get_activation(name)
        unweighted = states + c * (fn(h) - states)
        s += torch.sum(self.prune_mask[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1).float() * probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()
        self.num_kept = 8

    def new(self):
        model_new = RNNModelSearch(*self._args)
        model_new.prune_mask = self.prune_mask
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      k = sum(i for i in range(1, STEPS+1))
      weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)
      self.weights = Variable(weights_data.cuda(), requires_grad=True)
      self.softmax_weights = F.softmax(self.weights, dim=-1)
      self._arch_parameters = [self.weights]
      self.prune_mask = torch.ones_like(self.weights).byte() # 0-prune; 1-reserve
      for rnn in self.rnns:
        rnn.weights = self.softmax_weights
        rnn.prune_mask = self.prune_mask

    def arch_parameters(self):
      return self._arch_parameters

    def get_softmax_arch_parameters(self):
      return [self.softmax_weights]

    def update_softmax_arch_parameters(self, set_zero=False):
      if set_zero:
        self.softmax_weights_tmp = torch.zeros_like(self.weights)
    
        for i in range(self.weights.shape[0]):
          if self.prune_mask[i].sum() > 0:
            self.softmax_weights_tmp[i, self.prune_mask[i]] = F.softmax(self.weights[i, elf.prune_mask[i]], dim=-1)
    
        self.softmax_weights = self.softmax_weights_tmp
      else:
        self.softmax_weights = F.softmax(self.weights, dim=-1)

      for rnn in self.rnns:
        rnn.weights = self.softmax_weights
      return [self.softmax_weights]

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def genotype(self, no_restrict=False):

      def _parse(probs, prune_mask):
        gene = []
        start = 0
        try:
          none_idx = PRIMITIVES.index('none')
        except:
          none_idx = -1
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          W = W * prune_mask[start:end]
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != none_idx))[0]
          k_best = None
          for k in range(len(W[j])):
            if k != none_idx:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene

      gene = _parse(self.softmax_weights.data.cpu().numpy(), self.prune_mask.data.cpu().numpy())
      genotype = Genotype(recurrent=gene, concat=list(range(STEPS+1)[-CONCAT:]))
      return genotype

    def has_redundent_op(self, no_restrict=False):
      if no_restrict:
        if self.prune_mask.sum() > self.num_kept: return True
        return False

      start = 0
      for i in range(STEPS):
        end = start + i + 1
        tmp = self.prune_mask[start:end]
        if tmp.sum() > 1: return True
        start = end
      return False

    def prune(self, prune_mask):
        self.prune_mask = prune_mask
        for rnn in self.rnns:
          rnn.prune_mask = self.prune_mask
