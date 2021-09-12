import torch
from TS_model.resnet import ResNet50, ResNet18
from TS_model.vgg import vgg16, vgg16_bn
from TS_model.googlenet import googlenet

teacher_models = {
'res18': ResNet18,
'res50': ResNet50,
'vgg16': vgg16,
'vgg16_bn': vgg16_bn,
'googlenet': googlenet,
}

teacher_models_path = {
'res18': 'ckpt/teacher/res18_cifar10.pt',
'res50': 'TS_model/ckpt/cifar10_resnet50_acc_94.680_sgd.pt',
'vgg16': None,
'vgg16_bn': None,
'googlenet': None,
}

def build_teacher(num_classes, teacher=None, teacher_ckpt=None ,gpulocation=None):
    """
    INPUT:
      teacher_ckpt: the path of teacher ckpt, if None, then use the default path (teacher_models_path)

    OUTPUT:
      teacher model, nn.Module
    """
    if teacher is None: return None
    if teacher not in teacher_models.keys():
        raise(ValueError("No teacher called: %s"%teacher))
    if teacher_ckpt is None:
        teacher_ckpt = teacher_models_path[teacher]
    model = teacher_models[teacher](num_classes)
    print(teacher_ckpt)
    model.load_state_dict(torch.load(teacher_ckpt,map_location='cuda:'+str(gpulocation)),strict=False)

    return model
