import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import random
import time
# from Models import *
from torchvision import models
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, track_running_stats=False, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'pixelwise':
        norm_layer = PixelNorm
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.001)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_C(nf, which_model_netC, init_type='normal', norm='batch', gpu_ids=[]):
    netC = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netC == 'resnet_34':
        netC = Resnet34(60)
    elif which_model_netC == 'vgg_16':
        netC = VGG(60)
    elif which_model_netC == 'alexnet':
        netC = AlexNet(60)
    else:
        raise NotImplementedError('Classification model name [%s] is not recognized' % which_model_netC)
    if len(gpu_ids) > 0:
        netC.cuda(gpu_ids[0])
    # init_weights(netC, init_type=init_type)
    return netC


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class ClassificationLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(ClassificationLoss, self).__init__()
        self.Tensor = tensor
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class PixelNorm(nn.Module):
    def __init__(self, c, momentum=0.0):
        super(PixelNorm, self).__init__()
        self.pixelnorm = functools.partial(nn.functional.normalize, p=2, dim=1)

    def forward(self, input):
        return self.pixelnorm(input) 

    def __repr__(self):
        return self.__class__.__name__ + '(p = %s, dim = %s)' % (2, 1)

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
        )
        self._initialize_weights()


    def forward(self, x):
        x = self.net.features(x)
        x = x.view(x.size(0), -1)
        x = self.net.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.net.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__()
        self.net = models.resnet34( pretrained=True )
        self.net.fc = nn.Linear(512 * 1, num_classes)

    def forward(self, x):
        return self.net(x)

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.net = models.alexnet(pretrained=True)
        self.net.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.net.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.net.classifier(x)
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, input_nc, inner_nc, output_nc, norm_layer, restype='self'):
        super(ResnetBlock, self).__init__()
        self.restype = restype
        self.conv_block = self.build_conv_block(input_nc, inner_nc, output_nc, norm_layer, restype)
        if restype=='resize' or restype=='init':
            self.self_block = self.build_self_block(input_nc, output_nc, norm_layer, restype)

    def build_conv_block(self, input_nc, inner_nc, output_nc, norm_layer, restype):
        s=1
        if restype=='resize':
            s=2
        conv_block  = [nn.Conv2d(input_nc, inner_nc, kernel_size=1, stride=s, padding=0, bias=False),
                       norm_layer(inner_nc),
                       nn.ReLU(True)
                      ]

        conv_block += [nn.Conv2d(inner_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=False),
                       norm_layer(inner_nc),
                       nn.ReLU(True),
                       nn.Conv2d(inner_nc, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
                       norm_layer(output_nc),
                       nn.ReLU(True)
                      ]
        return nn.Sequential(*conv_block)


    def build_self_block(self, input_nc, output_nc, norm_layer, restype):
        s=1
        if restype=='resize':
            s=2
        self_block  = [nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=s, padding=0, bias=False),
                       norm_layer(output_nc),
                       nn.ReLU(True)
                      ]
        return nn.Sequential(*self_block)

    def forward(self, x):
        if self.restype=='self':
            out = x + self.conv_block(x)
        else:
            out = self.self_block(x) + self.conv_block(x)
        return out



