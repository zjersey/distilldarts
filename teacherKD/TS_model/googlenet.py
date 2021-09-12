'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class Inception(nn.Module):
    def __init__(self, in_planes,
                 n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red, 
                                    kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3, 
                                    kernel_size=3, padding=1)

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red, 
                                    kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5, 
                                    kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5, 
                                    kernel_size=3, padding=1)

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv2d(in_planes, pool_planes, 
                                  kernel_size=1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, cifar=True):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        pre_layers = []
        if cifar:
            pre_layers.append(BasicConv2d(3, 192, kernel_size=3, padding=1))
            pre_layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        else:
            pre_layers.append(BasicConv2d(3, 64, kernel_size=7, stride=2, padding=1))
            pre_layers.append(nn.MaxPool2d(3, stride=2, padding=1))
            pre_layers.append(BasicConv2d(64, 64, kernel_size=1))
            pre_layers.append(BasicConv2d(64, 192, kernel_size=3, padding=1))
            pre_layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.pre_layers = nn.Sequential(*pre_layers)
           

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool_3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

#        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(1024, self.num_classes)

    def forward(self, x, output_idx=[]):
        """
        INPUT:
        	output_idx: the index of the cell, whose output will be add to outputs. '0' means the output of stem

        OUTPUT:
        	a list of feature maps, with len(output_idx)+1 items, the last item is the final output before the softmax.
        """
        outputs = []
        out = self.pre_layers(x)
        if -1 in output_idx: outputs.append(out)
        out = self.a3(out)
        out = self.b3(out)
        if 1 in output_idx: outputs.append(out)
        out = self.maxpool_3(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        if 2 in output_idx: outputs.append(out)
        out = self.maxpool_4(out)
        out = self.a5(out)
        out = self.b5(out)
        if 3 in output_idx: outputs.append(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        outputs.append(out)
        return outputs

def googlenet(num_classes=10, cifar=True):
    return GoogLeNet(num_classes=num_classes, cifar=cifar)

def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    y = net(x)

if __name__ == "__main__":
    test()

