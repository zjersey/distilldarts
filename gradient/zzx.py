import os
import json
import torch



a=torch.tensor([-0.5, 0.3, 1.0, 0.6])
b=torch.nn.functional.softmax(a,dim=-1)
print(b)