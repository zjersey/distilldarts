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
                s = sum(
                    drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network_selfKD(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, primitives,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0,
                 alphaKD=0.5, betaKD=1.0, temperature=1.):
        super(Network_selfKD, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob
        #self.selfKD_output_idx = [0, 2, 4, 6, 7]
        self.selfKD_output_idx = [0, 1, 2, 3, 4, 5, 6]
        #self.selfKD_chs = [64, 128, 128, 256, 256]
        self.selfKD_chs = [64, 64, 128, 128, 128, 256, 256]
        self.alphaKD = alphaKD
        self.betaKD = betaKD
        self.temperature = temperature

        nn.Module.PRIMITIVES = primitives

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.external_classifiers=self.build_external_classifiers()
        self._initialize_alphas()

    def build_external_classifiers(self):
        classifiers=nn.ModuleList()
        for channel in self.selfKD_chs:
            classifier=nn.Linear(channel,self._num_classes)
            classifiers.append(classifier)
        return classifiers

    def new(self):
        model_new = Network_selfKD(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, drop_path_prob=self.drop_path_prob,
                                   alphaKD=self.alphaKD, betaKD=self.betaKD, temperature=self.temperature).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        featureMaps = []
        logits_set = []
        outputs = []
        s0 = s1 = self.stem(input)
        if -1 in self.selfKD_output_idx:
            featureMaps.append(s0)
        classifier_index=0
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if self.training:
                s1.retain_grad()
            outputs.append(s1)
            if i in self.selfKD_output_idx:
                featureMaps.append(s1)
                pooling_auxiliary=self.global_pooling(s1)
                logits_auxiliary=self.external_classifiers[classifier_index](pooling_auxiliary.view(pooling_auxiliary.size(0),-1))
                classifier_index+=1
                logits_set.append(logits_auxiliary)

        featureMaps.append(s1)
        out = self.global_pooling(s1)  # [32,256,1,1]
        logits = self.classifier(out.view(out.size(0), -1))
        logits_set.append(logits)
        return featureMaps , logits_set, outputs

    """
    def compute_loss(self, input, target, return_logits=False):
        featureMaps , logits_set, outputs = self(input)
        CEloss_LastLayer = self._criterion(logits_set[-1], target)
        CEloss = self._celoss(logits_set[:-1], target)
        KLloss = self._klloss(logits_set[-1].detach(), logits_set[:-1], self.temperature)
        FMloss = self._fmloss(featureMaps)
        loss = 1.5 * CEloss_LastLayer + self.alphaKD * CEloss + (1 - self.alphaKD) * KLloss+ self.betaKD * FMloss

        if not return_logits:
            return [loss, CEloss, KLloss, FMloss]
        else:
            return [loss, CEloss, KLloss, FMloss] , logits_set, outputs
    """

    def compute_loss(self, input, target, return_logits=False):
        featureMaps , logits_set, outputs = self(input)
        CEloss = self._celoss(logits_set, target)
        KLloss = self._klloss2(logits_set, self.temperature)
        FMloss = self._fmloss3(featureMaps)
        loss = self.alphaKD * CEloss + (1 - self.alphaKD) * KLloss + self.betaKD * FMloss

        if not return_logits:
            return [loss, CEloss, KLloss, FMloss]
        else:
            return [loss, CEloss, KLloss, FMloss], logits_set, outputs

    def _celoss(self, logits_set, target):
        loss=0.
        for logits in logits_set:
            loss += self._criterion(logits, target)
        loss /= len(logits_set)
        return loss

    def _klloss(self, logits1, selfKD_logits, T=1.):
        loss = 0.
        for logits2 in selfKD_logits:
            assert len(logits1) == len(logits2)
            loss += nn.KLDivLoss() (F.log_softmax(logits1/T, dim=1),
                             F.softmax(logits2/T, dim=1)) * (T * T)
        loss /= len(selfKD_logits)
        return loss

    def _klloss2(self, logits_set, T=1.):
        loss = 0.
        for i in range(1, len(logits_set)):
            loss += nn.KLDivLoss() (F.log_softmax(logits_set[i-1]/T, dim=1),
                                       F.softmax(logits_set[i]/T, dim=1)) * (T * T)
        loss /= (len(logits_set)-1)
        return loss


    def _fmloss(self, featureMaps):
        loss = 0.
        final_layer = featureMaps[-1].detach()
        for i in range(1,len(featureMaps[:-1])):
            featureMap = featureMaps[i]
            featureMap = F.avg_pool2d(featureMap, featureMap.shape[2] // final_layer.shape[2])
            loss += self._at_loss(final_layer, featureMap)
        loss /= (len(featureMaps)-2)
        return loss
        
    def _fmloss2(self, featureMaps):
        loss =0.
        final_layer = featureMaps[-1].detach()
        reduction_layer_2 = featureMaps[5].detach()
        reduction_layer_1 = featureMaps[2]
        reduction_layer_1 = F.avg_pool2d(reduction_layer_1, reduction_layer_1.shape[2] // reduction_layer_2.shape[2])
        loss += self._at_loss(reduction_layer_1,reduction_layer_2)
        for i in range(1, len(featureMaps[:-1])):
            if i==2 or i==5:
                continue
            featureMap = featureMaps[i]
            featureMap = F.avg_pool2d(featureMap, featureMap.shape[2] // final_layer.shape[2])
            loss += self._at_loss(final_layer, featureMap)
        loss /= (len(featureMaps)-3)
        return loss

    def _fmloss3(self, featureMaps):
        loss = 0.
        for  i in range(1, len(featureMaps)):
            fm1 = featureMaps[i-1]
            fm2 = featureMaps[i]
            if not fm1.shape[2] == fm2.shape[2]:
                fm1 = F.avg_pool2d(fm1, fm1.shape[2] // fm2.shape[2])
            loss += self._at_loss(fm1, fm2)
        loss /= (len(featureMaps) -1)
        return loss


    def _at_loss(self,x, y):
        return (self._at(x) - self._at(y)).pow(2).mean()

    def _at(self,x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
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


