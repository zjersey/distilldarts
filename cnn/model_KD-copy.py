import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import ReLUConvBN
import numpy as np

from model_search import Network
import TS_model


class Network_KD(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, alpha_weights=None, teacher=None, teacher_ckpt=None, gpuLocation=None, student_output_idx=[], use_kd=True, kd_alpha=0.5, kd_beta=1.0):
    """
    teacher: 'res50','res18','vgg16','googleNet'
    """
    super(Network_KD, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self.gpuLocation = gpuLocation
    self.teacher_model = TS_model.build_teacher(num_classes, teacher, teacher_ckpt, gpuLocation)
    self.student_model = Network(C, num_classes, layers, criterion, steps, multiplier, stem_multiplier, alpha_weights)
    self.teacher_output_idx = [0,1,2,3,4]
    self.teacher_chs = [64, 256, 512, 1024, 2048]
    self.student_output_idx = [0,2,4,6,7]
    self.student_chs = [64, 128, 128, 256, 256]
    self.transformer = self.build_KD_conv()
    self.use_kd=use_kd
    self.kd_alpha=kd_alpha
    self.kd_beta=kd_beta

  def build_KD_conv(self):
    transformer = nn.ModuleList()
    for t_c, s_c in zip(self.teacher_chs, self.student_chs):
        #conv = nn.Conv2d(student_feature.shape[1], teacher_feature.shape[1], kernel_size=1, stride=1, padding=0, bias=False)
        conv = ReLUConvBN(C_in=s_c, C_out=t_c, kernel_size=3, stride=1, padding=1)
        transformer.append(conv)
    return transformer

  def forward(self, x):
    student_outputs = self.student_model(x, student_output_idx=self.student_output_idx)
    teacher_outputs = []
    if self.teacher_model is not None:
        with torch.no_grad():
            teacher_outputs = self.teacher_model(x, teacher_output_idx=self.teacher_output_idx)
    return student_outputs, teacher_outputs

  def new_transformer(self):
    transformer_new = self.build_KD_conv()
    return transformer_new

  def _loss(self, input, target, T=1):
    student_outputs, teacher_outputs = self(input)
    ce_loss = self.student_model._criterion(student_outputs[-1], target)
    if self.use_kd:
      kd_loss = KD_logits_loss(student_outputs[-1], teacher_outputs[-1], T=T)
      kd_fm_loss=KD_FeatureMap_loss(student_outputs[:-1],teacher_outputs[:-1], self.transformer)
      loss = self.kd_alpha*kd_loss + (1-self.kd_alpha)*ce_loss + self.kd_beta*kd_fm_loss
    else:
      loss = ce_loss
    return  loss


def KD_logits_loss(student_logits, teacher_logits, T=1):
    """
    INPUT:
        student_logics: output logics of student model
        teacher_logics: of output logics of teacher model
    OUTPUT:
        the loss of Knowledge Distillation
    """
    if len(teacher_logits)==0 or len(student_logits)==0:
        return 0.
    assert(len(teacher_logits) == len(student_logits))
    loss = 0.
    loss += nn.KLDivLoss()(F.log_softmax(student_logits/T, dim=1),
                             F.softmax(teacher_logits/T, dim=1)) * (T * T)

    return loss

def KD_FeatureMap_loss(student_outputs, teacher_outputs, transformer):
    if len(teacher_outputs)==0 or len(student_outputs)==0:
        return 0.
    assert(len(teacher_outputs) == len(student_outputs))
    loss = 0.
    for i in range(len(teacher_outputs)):
        teacher_feature = teacher_outputs[i]
        student_feature = student_outputs[i]
        if teacher_feature.shape[2] < student_feature.shape[2]:
            student_feature = F.avg_pool2d(student_feature,student_feature.shape[2]//teacher_feature.shape[2])
        elif teacher_feature.shape[2] > student_feature.shape[2]:
            teacher_feature = F.avg_pool2d(teacher_feature, teacher_feature.shape[2]//student_feature.shape[2])
        if teacher_feature.shape[1] != student_feature.shape[1] :
            student_feature=transformer[i](student_feature)
        loss += nn.MSELoss()(student_feature,teacher_feature)
        #loss += at_loss(student_feature,teacher_feature)
        
    loss/=len(teacher_outputs)

    return loss
    
def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))