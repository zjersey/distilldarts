import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from utils import drop_path, Genotype

"""
cell 0 input: torch.Size([32, 48, 32, 32]) torch.Size([32, 48, 32, 32])
cell 0 output: torch.Size([32, 64, 32, 32])
cell 1 input: torch.Size([32, 48, 32, 32]) torch.Size([32, 64, 32, 32])
cell 1 output: torch.Size([32, 64, 32, 32])
cell 2 input: torch.Size([32, 64, 32, 32]) torch.Size([32, 64, 32, 32])
cell 2 output: torch.Size([32, 128, 16, 16])
cell 3 input: torch.Size([32, 64, 32, 32]) torch.Size([32, 128, 16, 16])
cell 3 output: torch.Size([32, 128, 16, 16])
cell 4 input: torch.Size([32, 128, 16, 16]) torch.Size([32, 128, 16, 16])
cell 4 output: torch.Size([32, 128, 16, 16])
cell 5 input: torch.Size([32, 128, 16, 16]) torch.Size([32, 128, 16, 16])
cell 5 output: torch.Size([32, 256, 8, 8])
cell 6 input: torch.Size([32, 128, 16, 16]) torch.Size([32, 256, 8, 8])
cell 6 output: torch.Size([32, 256, 8, 8])
cell 7 input: torch.Size([32, 256, 8, 8]) torch.Size([32, 256, 8, 8])
cell 7 output: torch.Size([32, 256, 8, 8])
"""

class MixedOp(nn.Module):

  def __init__(self, C, stride, PRIMITIVES):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, efficient=False):
    if efficient:
      index = weights.max(-1, keepdim=True)[1].item()
      return weights[index] * self._ops[index](x)
    else:
      return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()

    edge_index = 0

    for i in range(self._steps):
      for j in range(2 + i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, self.primitives[edge_index])
        self._ops.append(op)
        edge_index += 1

  def forward(self, s0, s1, weights, drop_prob=0., efficient=False):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      if drop_prob > 0. and self.training:
          s = sum(drop_path(self._ops[offset + j](h, weights[offset + j], efficient), drop_prob) for j, h in enumerate(states))
      else:
        s = sum(self._ops[offset + j](h, weights[offset + j], efficient) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, primitives, efficient=False,
               steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.drop_path_prob = drop_path_prob
    self.efficient = efficient

    nn.Module.PRIMITIVES = primitives

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      while True:
        if cell.reduction:
          weights = F.gumbel_softmax(self.alphas_reduce, tau=self.tau, dim=-1) if not self.efficient \
               else F.gumbel_softmax(self.alphas_reduce, tau=self.tau, hard=True, dim=-1)
        else:
          weights = F.gumbel_softmax(self.alphas_normal, tau=self.tau, dim=-1) if not self.efficient \
               else F.gumbel_softmax(self.alphas_normal, tau=self.tau, hard=True, dim=-1)
        if (torch.isinf(weights).any()) or (torch.isnan(weights).any()): continue
        else: break
      s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob, self.efficient)

    out = self.global_pooling(s1)  # [32,256,1,1]
    logits = self.classifier(out.view(out.size(0), -1))
    return logits


  def compute_loss(self, input, target, return_logits=False):
    logits = self(input)
    loss = self._criterion(logits, target)

    if return_logits:
      return loss, logits
    else:
      return loss

  def set_tau(self, tau):
    self.tau = tau

  def get_tau(self):
    return self.tau

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, efficient=self.efficient,
                        drop_path_prob=self.drop_path_prob).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
      x.data.copy_(y.data)
    return model_new

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(self.PRIMITIVES['primitives_normal'][0])

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights, normal=True):
      PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']

      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()

        try:
          edges = sorted(range(i + 2), key=lambda x: -max(
            W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
        except ValueError:  # This error happens when the 'none' op is not present in the ops
          edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if 'none' in PRIMITIVES[j]:
              if k != PRIMITIVES[j].index('none'):
                if k_best is None or W[j][k] > W[j][k_best]:
                  k_best = k
            else:
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[start + j][k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

    concat = list(range(2+self._steps-self._multiplier, self._steps+2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype