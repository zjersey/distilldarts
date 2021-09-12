import torch
import torch.nn as nn
import torch.nn.init as init

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = BasicConv2d(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = BasicConv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand3x3 = BasicConv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', embedding_size=128, num_classes=1000,  drop_probability=0.5):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        if version == '1_0':
            '''
            # slim 0.5
            self.base = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 32, 32),
                Fire(64, 16, 32, 32),
                Fire(64, 32, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 64, 64),
                Fire(128, 48, 96, 96),
                Fire(192, 48, 96, 96),
                Fire(192, 64, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 64, 128, 128),
            )
            ''' 
            #ori
            self.base = nn.Sequential(
                BaiscConv2d(3, 96, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )

        elif version == '1_1':
            self.base = nn.Sequential(
                BasicConv2d(3, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))
        self.fc = nn.Sequential(
                nn.Dropout(p=drop_probability),
                nn.Conv2d(512, 960, kernel_size=1, bias=False),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(960, embedding_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(embedding_size)
        )
        self.classifier = nn.Linear(embedding_size, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier:
#                    init.normal_(m.weight, mean=0.0, std=0.01)
                    init.kaiming_normal_(m.weight)
                else:
#                    init.kaiming_uniform_(m.weight)
                    init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
#                    init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        self.features = x
        x = self.classifier(x)
        return x


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)

