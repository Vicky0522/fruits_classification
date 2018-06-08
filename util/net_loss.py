import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from caffe2pytorch.caffenet import *
import scipy.io as sio
import numpy as np
from torchvision import models

class NetLoss(object):
    def __init__(self, opt):
        super(NetLoss, self).__init__()
        self.opt = opt
        if opt.which_loss_net=='SSL':
            self.loss_net = SSL_Loss('/mnt/tencent/vicky/models/SSL_Model/caffe/train.prototxt',
                                     '/mnt/tencent/vicky/models/SSL_Model/caffe/attention+ssl.caffemodel',
                                     (321, 321), (122.675, 116.669, 104.008), 'fc8_mask_1st', gpu_ids=opt.gpu_ids)
        elif opt.which_loss_net=='SSL_l1':
            self.loss_net = SSL_L1Loss('/mnt/tencent/vicky/models/SSL_Model/caffe/deploy.prototxt',
                                       '/mnt/tencent/vicky/models/SSL_Model/caffe/attention+ssl.caffemodel',
                                       (321, 321), (122.675, 116.669, 104.008), gpu_ids=opt.gpu_ids)
        elif opt.which_loss_net=='VGG_CNN_S':
            self.interp = Interp(pad_beg_=0, pad_end_=0, height=224, width=224)
            self.loss = nn.L1Loss()
            self.vgg = models.vgg16( pretrained=True)
            self.vgg = self.vgg.cuda()
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.loss_net = nn.Sequential(*list(self.vgg.features.children())[0:14])

        elif opt.which_loss_net=='MaskLoss':
            self.loss_net = Mask_Loss('/mnt/tencent/vicky/models/SSL_Model/caffe/train.prototxt',
                                      '/mnt/tencent/vicky/models/SSL_Model/caffe/deploy.prototxt',
                                      '/mnt/tencent/vicky/models/SSL_Model/caffe/attention+ssl.caffemodel',
                                      (321, 321), (122.675, 116.669, 104.008), 'fc8_mask_1st', 'fc8_mask', gpu_ids=opt.gpu_ids)
        else:
            raise NotImplementedError('loss_net [%s] is not implemented' % opt.which_loss_net)


    def forward(self, input, target_m = None, target_i = None):
        if self.opt.which_loss_net=='SSL':
            return self.loss_net.forward(input, target_m)
        elif self.opt.which_loss_net=='VGG_CNN_S':
            input = self.interp(input)
            target_i = self.interp(target_i)
            return self.loss(self.loss_net(input), self.loss_net(target_i))
        else:
            return self.loss_net.forward(input, target_i)

    def get_output(self):
        assert(self.opt.which_loss_net!='MaskLoss')
        return self.loss_net.get_output()

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)

class Mask_Loss(nn.Module):
    def __init__(self, train_protofile, deploy_protofile,
                 weightfile, size, mean, fake_output_name, real_output_name, gpu_ids=[]):
        super(Mask_Loss, self).__init__()
        self.net_pred = CaffeNet(deploy_protofile, data_width=size[1], data_height=size[0], 
                                 label_width = size[1], label_height = size[0],
                                 omit_data_layer=True, require_label=False, phase='TEST')
        self.net_loss = CaffeNet(train_protofile, data_width=size[1], data_height=size[0],
                                 label_width = size[1], label_height = size[0],
                                 omit_data_layer=True, require_label=True)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.net_pred.cuda(gpu_ids[0])
            self.net_loss.cuda(gpu_ids[0])

        self.net_pred.load_weights(weightfile)
        self.net_loss.load_weights(weightfile)

        self.net_pred.set_verbose(False)
        self.net_pred.eval()
        self.net_loss.set_verbose(False)

        self.size = size
        self.mean = mean
        self.fake_output_name = fake_output_name
        self.real_output_name = real_output_name

        # register hook
        #for key in self.net.models:
        #    self.net.models[key].register_backward_hook(printgradnorm)

        # some modules to preprocess input
        self.slice = Slice(1, (1,2,3))
        self.concat = Concat(1)
        self.interp = Interp(pad_beg_=0, pad_end_=0, height=size[0], width=size[1])

    def preprocess(self, input):
        # some transforms on input
        output = torch.mul(torch.add(input, 1.0), 0.5*255)
        outr, outg, outb = self.slice(output)
        outr = torch.add(outr, -self.mean[0])
        outg = torch.add(outg, -self.mean[1])
        outb = torch.add(outb, -self.mean[2])
        output = self.concat(outb, outg, outr)
        output = self.interp(output)
        return output

    def forward(self, input, targets):
        input = self.preprocess(input)
        targets = self.preprocess(targets)
        #print(output)
        #print self.net.models['conv1_1'].weight
        self.net_pred.forward(targets)
        targets = self.net_pred.blobs['fc8_mask'].detach()
        self.net_loss.forward(input, targets)
        return self.net_loss.get_loss()

    def get_fake_output(self):
        return self.net_loss.blobs[self.fake_output_name]

    def get_real_output(self):
        return self.net_pred.blobs[self.real_output_name]


class SSL_Loss(nn.Module):
    def __init__(self, protofile, weightfile, size, mean, output_name, gpu_ids=[]):
        super(SSL_Loss, self).__init__()
        self.net = CaffeNet(protofile, data_width=size[1], data_height=size[0], 
                            label_width = size[1], label_height = size[0],
                            omit_data_layer=True, require_label=True)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.net.cuda(gpu_ids[0])
        self.net.load_weights(weightfile)
        self.net.set_verbose(False)
        self.size = size
        self.mean = mean
        self.output_name = output_name

        # register hook
        #for key in self.net.models:
        #    self.net.models[key].register_backward_hook(printgradnorm)

        # some modules to preprocess input
        self.slice = Slice(1, (1,2,3))
        self.concat = Concat(1)
        self.interp = Interp(pad_beg_=0, pad_end_=0, height=size[0], width=size[1])

    def forward(self, input, targets):
        # some transforms on input
        output = torch.mul(torch.add(input, 1.0), 0.5*255)
        outr, outg, outb = self.slice(output)
        outr = torch.add(outr, -self.mean[0])
        outg = torch.add(outg, -self.mean[1])
        outb = torch.add(outb, -self.mean[2])
        output = self.concat(outb, outg, outr)
        output = self.interp(output)
        targets = self.interp(targets)
        #print(output)
        #print self.net.models['conv1_1'].weight
        self.net.forward(output, targets)
        return self.net.get_loss()

    def get_output(self):
        return self.net.blobs[self.output_name]

class SSL_L1Loss(nn.Module):
    def __init__(self, protofile, weightfile, size, mean, blob_name='fc8_fusion', gpu_ids=[]):
        super(SSL_L1Loss, self).__init__()
        self.net = CaffeNet(protofile, data_width=size[1], data_height=size[0], 
                            label_width = size[1], label_height = size[0],
                            omit_data_layer=True, require_label=False)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.net.cuda(gpu_ids[0])
        self.net.load_weights(weightfile)
        self.net.set_verbose(False)
        self.size = size
        self.mean = mean
        self.blob_name = blob_name

        # register hook
        #for key in self.net.models:
        #    self.net.models[key].register_backward_hook(printgradnorm)

        # some modules to preprocess input
        self.slice = Slice(1, (1,2,3))
        self.concat = Concat(1)
        self.interp = Interp(pad_beg_=0, pad_end_=0, height=size[0], width=size[1])

        self.loss = nn.L1Loss()

    def transform_data(self, input):
        # some transforms on input
        output = torch.mul(torch.add(input, 1.0), 0.5*255)
        outr, outg, outb = self.slice(output)
        if type(self.mean)==np.ndarray:
            output = self.interp(self.concat(outb, outg, outr)) - torch.from_numpy(self.mean).unsqueeze(0).cuda()
        else:
            outr = torch.add(outr, -self.mean[0])
            outg = torch.add(outg, -self.mean[1])
            outb = torch.add(outb, -self.mean[2])
            output = self.concat(outb, outg, outr)
            output = self.interp(output)
        return output

    def forward(self, input, targets):
        input = self.transform_data(input)
        targets = self.transform_data(targets)
        #print(output)
        #print self.net.models['conv1_1'].weight
        with torch.no_grad():
            self.net.forward(targets)
        output2 = self.net.blobs[self.blob_name]
        self.net.forward(input)
        output1 = self.net.blobs[self.blob_name]
        return self.loss(output1, output2)

    def get_output(self):
        return self.net.blobs['fc8_mask']

