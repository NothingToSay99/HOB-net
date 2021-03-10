# encoding: utf-8


import math

import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#####  W2_1  *  W2_2
class HOB1(nn.Module):
    def __init__(self, in_channels,inter_channels,embedings,class_num,order=1):
        super(HOB1, self).__init__()
        self.order = order
        self.inter_channels = inter_channels
        self.embedings = embedings
        self.class_num = class_num
        # for j in range(self.order):
        #     name = 'order' + str(self.order) + '_' + str(j + 1)
        #     setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, (1,1), padding=0, bias=False)))
        for j in range(self.order):
            for i in range(j+1):
                name = 'order' + str(self.order) + '_' + str(j+1) + '_' + str(i+1)
                setattr(self, name, nn.Sequential(nn.Conv2d(in_channels, self.inter_channels, 1, padding=0, bias=False)))

        for j in range(self.order):
            name = 'GAP' + str(self.order) + '_' + str(j + 1)
            setattr(self, name, nn.Sequential(nn.AdaptiveAvgPool2d(1)))

        name = 'fc_norm'
        setattr(self, name, nn.Sequential(nn.Linear(self.inter_channels, self.embedings, bias=False)))

        for i in range(self.order):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(self.embedings, self.class_num))

    def forward(self, x):
        y = []
        for j in range(self.order):
            for i in range(j + 1):
                name = 'order' + str(self.order) + '_' + str(j + 1) + '_' + str(i + 1)
                layer = getattr(self, name)
                y.append(layer(x))

        y_ = []
        cnt = 0
        for j in range(self.order):
            y_temp = 1
            for i in range(j + 1):
                y_temp = y_temp * y[cnt]
                cnt += 1
            y_.append(y_temp)

        y__= []
        for j in range(self.order):
            name = 'GAP' + str(self.order) + '_' + str(j + 1)
            # name = 'GAP'
            layer = getattr(self, name)
            y1 = layer(y_[j])
            y1 = y1.view(y1.size(0), -1)
            y__.append(y1)

        y___ = []
        for j in range(1,self.order):
            name = 'fc_norm'
            layer = getattr(self, name)
            y_n = layer(y__[j])
            y_n = F.normalize(y_n)* 20
            y___.append(y_n)

        predict = []
        for i in range(self.order - 1):
            name = 'classifier' + str(i)
            classifier = getattr(self, name)
            predict.append(classifier(y__[i]))

        return y___, predict


class ClassBlock(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassBlock, self).__init__()
        add_block = []
        bottleblock = nn.BatchNorm1d(input_dim)
        bottleblock.bias.requires_grad_(False)
        add_block += [bottleblock]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(input_dim, num_classes, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)