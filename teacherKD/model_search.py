import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from utils import drop_path, Genotype
import numpy as np

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

  def forward(self, x, weights):
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

  def forward(self, s0, s1, weights, drop_prob=0.):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      if drop_prob > 0. and self.training:
        s = sum(drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))
      else:
        s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, primitives,
               steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0,
               fm_method=None, kd_alpha=0.5, kd_beta=1.0, T=1.0):

    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.drop_path_prob = drop_path_prob

    self.fm_method = fm_method
    self.kd_alpha = kd_alpha
    self.kd_beta = kd_beta
    self.T = T
    self.student_output_idx = [1, 3, 5, 7]
    #self.student_chs = [64, 128, 256, 256]
    self.teacher_output_idx = [1, 2, 3, 4]
    #self.teacher_chs = [256, 512, 1024, 2048]

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

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob,
                        fm_method=self.fm_method, kd_alpha=self.kd_alpha, kd_beta=self.kd_beta, T=self.T).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    outputs = []
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
      if i in self.student_output_idx:
        outputs.append(s1)

    out = self.global_pooling(s1)  #[32,256,1,1]
    logits = self.classifier(out.view(out.size(0),-1))
    outputs.append(logits)
    return outputs

  def compute_loss(self, input, target, teacher_model, return_logits=False):
    student_outputs = self(input)
    with torch.no_grad():
        teacher_outputs = teacher_model(input, teacher_output_idx=self.teacher_output_idx)
    ce_loss = self._criterion(student_outputs[-1], target)
    kl_loss = KD_logits_loss(student_outputs[-1], teacher_outputs[-1], T=self.T)
    fm_loss = KD_FeatureMap_loss(student_outputs[:-1],teacher_outputs[:-1], None, self.fm_method)
    loss = self.kd_alpha * ce_loss + (1 - self.kd_alpha) * kl_loss + self.kd_beta * fm_loss

    if return_logits:
      return loss, student_outputs[-1]
    else:
      return loss

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

    concat = list(range(2 + self._steps - self._multiplier, self._steps + 2))
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

def KD_logits_loss(student_logits, teacher_logits, T=1.):
  """
  INPUT:
      student_logics: output logics of student model
      teacher_logics: of output logics of teacher model
  OUTPUT:
      the loss of Knowledge Distillation
  """
  if len(teacher_logits) == 0 or len(student_logits) == 0:
    return 0.
  assert (len(teacher_logits) == len(student_logits))
  loss = 0.
  loss += nn.KLDivLoss()(F.log_softmax(student_logits / T, dim=1),
                         F.softmax(teacher_logits / T, dim=1)) * (T * T)

  return loss

def KD_FeatureMap_loss(student_outputs, teacher_outputs, transformer, fm_method):
  if fm_method == None or len(teacher_outputs) == 0 or len(student_outputs) == 0:
    return 0.
  assert (len(teacher_outputs) == len(student_outputs))
  loss = 0.
  for i in range(len(teacher_outputs)):
    teacher_feature = teacher_outputs[i]
    student_feature = student_outputs[i]
    if teacher_feature.shape[2] < student_feature.shape[2]:
      student_feature = F.avg_pool2d(student_feature, student_feature.shape[2] // teacher_feature.shape[2])
    elif teacher_feature.shape[2] > student_feature.shape[2]:
      teacher_feature = F.avg_pool2d(teacher_feature, teacher_feature.shape[2] // student_feature.shape[2])
    if fm_method == "conv":
      if teacher_feature.shape[1] != student_feature.shape[1]:
        student_feature = transformer[i](student_feature)
      loss += nn.MSELoss()(student_feature, teacher_feature)
    elif fm_method == "attn":
      loss += at_loss(student_feature, teacher_feature)
    else:
      raise (ValueError("No fm method is set."))

  #loss /= len(teacher_outputs)

  return loss


def at_loss(x, y):
  return (at(x) - at(y)).pow(2).mean()


def at(x):
  return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))