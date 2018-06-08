import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.caffe2pytorch.caffenet import Accuracy


class ClassifyModel(BaseModel):
    def name(self):
        return 'ClassifyModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_I = self.FloatTensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_C = self.LongTensor(opt.batchSize, 1)

        # load/define networks
        self.netC = networks.define_C(opt.nf, opt.which_model_netC, opt.init_type, opt.norm, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            print("load successfully")
            self.load_network(self.netC, 'C', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionClassify = networks.ClassificationLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_C = torch.optim.SGD(self.netC.parameters(),
                                               lr=opt.lr, momentum=0.9, weight_decay=5e-4)
            # self.optimizer_C = torch.optim.Adam(self.netC.parameters(),
            #                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_C)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        self.criterionClassify = networks.ClassificationLoss()
        self.accuracy = Accuracy()
        self.test_count = 0
        self.test_acc = 0
        self.test_loss = 0

        print('---------- Networks initialized -------------')
        networks.print_network(self.netC)

    def set_input(self, input):
        input_I = input['I']
        input_C = input['C']
        self.input_I.resize_(input_I.size()).copy_(input_I)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths = input['I_paths']

    def forward(self):
        self.I = Variable(self.input_I)
        self.C = Variable(self.input_C)
        self.P = self.netC(self.I)
        self.acc_C = self.accuracy(self.P, self.C)

    # no backprop gradients
    def test(self):
        self.test_count += 1
        with torch.no_grad():
            self.I = Variable(self.input_I)
            self.C = Variable(self.input_C)
            self.P = self.netC(self.I)
            self.loss_C_test = self.criterionClassify(self.P, self.C)
            self.acc_C_test = self.accuracy(self.P, self.C)
            # print(self.loss_C_test)
            self.test_acc += self.acc_C_test
            self.test_loss += self.loss_C_test

    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_C(self):
        self.loss_C = self.criterionClassify(self.P, self.C)

        self.loss_C.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()

    def get_current_errors(self):
        errors = {}
        if self.isTrain:
            errors = OrderedDict([('C_loss', self.loss_C.item()),
                                  ('C_acc', self.acc_C.item())])
        if self.test_count>0:
            self.test_loss /= self.test_count
            self.test_acc /= self.test_count
            if not self.isTrain:
                print("Test total %d cases. Average loss = %.2f. Average accuracy = %.2f" % \
                      (self.test_count, self.test_loss, self.test_acc))
            errors.update({'test_loss': self.test_loss,
                           'test_acc': self.test_acc
                          })
            self.test_count=0
            self.test_loss=0
            self.test_acc=0

        return errors

    def save(self, label):
        self.save_network(self.netC, 'C', label, self.gpu_ids)
