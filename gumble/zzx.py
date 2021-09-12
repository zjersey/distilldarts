import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import random
from collections import Counter, OrderedDict

"""
def gumbel_softmax_topk(logits, k=1, tau=1., hard=False, dim=-1):
  while True:
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if (torch.isinf(gumbels).any()) or (torch.isinf(y_soft).any()) or (torch.isnan(y_soft).any()):
      continue
    else:
      break

  index = None
  if hard:
    _, index = torch.topk(y_soft, k)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
  else:
    ret = y_soft
  return ret, index


def get_top2_prob(probs):
    res = []
    for i in range(len(probs)):
        for j in range(i+1, len(probs)):
            res.append(probs[i]*probs[j]/(1-probs[i]) + probs[j]* probs[i]/(1-probs[j]))
    return res

nums = 100000

logits = torch.tensor([-1., 0.3, 0.5, 1.5, -3.])
probs = F.softmax(logits,dim=-1)
print('probs:')
print(probs)
print('top2 probs: (without replacement)')
top2_prob = get_top2_prob(probs)
print(top2_prob)
sampleResults_gumbel = []
for i in range(nums):
    _, indexes = gumbel_softmax_topk(logits, k=2, tau=0.1, hard=True, dim=-1)
    sampleResults_gumbel.append('%d,%d' % (indexes.min(), indexes.max()))
gumbel_counter = Counter(sampleResults_gumbel)

index = 0
for i in range(len(probs)):
    for j in range(i + 1, len(probs)):
        s = '%d,%d' % (i, j)
        print( s+" : actual: ", gumbel_counter[s]/nums, "  theory: ", top2_prob[index].item())
        index += 1


"""
def sample_from_probs(probs):
    rand = random.random()
    if rand<probs[0]:
        return 0
    if rand<probs[0]+probs[1]:
        return 1
    return 2


#logits = torch.tensor([0.5,0.45,0.55])
logits = torch.tensor([-0.5,0.45,1.55])
probs = F.softmax(logits,dim=-1)
print('probs:')
print(probs)
num = 100000
sampleResults_probs = []
sampleResults_gumbel = []
for i in range(num):
    sampleResults_probs.append(sample_from_probs(probs))
    gumbel_distribution = F.gumbel_softmax(logits, tau=100., dim=-1, hard=True)
    sampleResults_gumbel.append(gumbel_distribution.max(-1)[1].item())
probs_counter = Counter(sampleResults_probs)
gumbel_counter = Counter(sampleResults_gumbel)
print('count for probs:')
print(probs_counter)
print('count for gumbel:')
print(gumbel_counter)
for i in range(3):
    print('%d' % i, 'sample: ', probs_counter[i]/num, 'gumbel: ', gumbel_counter[i]/num)
"""
def gumbel_softmax_topk(logits, k=1, tau=1. , hard=False, dim=-1):
  while True:
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if (torch.isinf(gumbels).any()) or (torch.isinf(y_soft).any()) or (torch.isnan(y_soft).any()): continue
    else: break

  index = None
  if hard:
    _ , index = torch.topk(y_soft, k)
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
  else:
    ret = y_soft
  return ret, index


input = Variable(torch.randn(2,1),requires_grad=False)
alpha = Variable(1e-3*torch.randn(4,3), requires_grad=True)
beta = Variable(1e-3*torch.randn(4), requires_grad=True)
W = Variable(torch.randn(4,3,2,2), requires_grad=True)
weights = F.gumbel_softmax(alpha,hard=True, tau=1., dim=-1)
weights_beta, edges = gumbel_softmax_topk(beta[:], 2, hard=True)
#_, edges= torch.topk(weights_beta, 2)
#weights_beta.scatter_(-1, edges, 1.0)
print('weights:')
print(weights)
print('weights_beta:')
print(weights_beta)
print('edges:')
print(edges)
output=[]
for edge in edges:
    index = weights[edge].max(-1, keepdim=True)[1].item()
    output.append(weights_beta[edge]*weights[edge][index]*W[edge][index].mm(input))
print('output_size:')
output = sum(output)
print(output.size())
loss = output.mean()
loss.backward()
print(beta.grad)
"""

