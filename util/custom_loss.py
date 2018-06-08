import numpy as np
import cv2
import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn as nn
from caffe2pytorch.caffenet import Slice, Concat

class MixedLoss(nn.Module):
    def __init__(self, ratio=0.5):
        super(MixedLoss, self).__init__()
        self.ratio = ratio
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, input, targets):
        return self.l1loss(input, targets) * self.ratio + \
               self.l2loss(input, targets) * (1-self.ratio)


class SeparateLoss(nn.Module):
    def __init__(self, ratio=1.0):
        super(SeparateLoss, self).__init__()
        self.ratio = ratio
        self.loss = nn.L1Loss()
        self.indices_body = Variable(torch.LongTensor([2, 5 ,6]))
        self.indices_cloth = Variable(torch.LongTensor([0,1,3,4,7,8,9,10,11,12,13,14,15,16,17]))

        self.slice = Slice(1, (1,2,3))
        self.concat = Concat(axis=1)

    def forward(self, input, targets, input_k):
        self.indices_body = self.indices_body.cuda() if input_k.is_cuda else self.indices_body
        self.indices_cloth = self.indices_cloth.cuda() if input_k.is_cuda else self.indices_cloth

        input_k_body = torch.index_select(input_k, 1, self.indices_body)
        input_k_cloth = torch.index_select(input_k, 1, self.indices_cloth)

        input_k_body_0_1_mask = torch.sum(torch.mul(torch.add(input_k_body, 1.0), 0.5), 1, keepdim=True)
        input_k_cloth_0_1_mask = torch.sum(torch.mul(torch.add(input_k_cloth, 1.0), 0.5), 1, keepdim=True)

        input_body = torch.add(torch.mul(torch.add(input, 1), input_k_body_0_1_mask), -1)
        input_cloth = torch.add(torch.mul(torch.add(input, 1), input_k_cloth_0_1_mask), -1)
        targets_body = torch.add(torch.mul(torch.add(targets, 1), input_k_body_0_1_mask), -1)
        targets_cloth = torch.add(torch.mul(torch.add(targets, 1), input_k_cloth_0_1_mask), -1)

        loss_body = self.loss(input_body, targets_body)
        loss_cloth = self.loss(input_cloth, targets_cloth)

        return (loss_body * self.ratio + loss_cloth * (1-self.ratio)) * 2



class LaplaceLoss(nn.Module):
    def __init__(self, opt):
        super(LaplaceLoss, self).__init__()
        self.slice = Slice(1, (1,2,3))
        self.laplace = nn.Conv2d(1, 1, 3, bias=False)
        laplace = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
        self.laplace.weight.data.copy_(laplace)
        self.laplace = self.laplace.cuda() if len(opt.gpu_ids)>0 else self.laplace
        self.laplace.weight.requires_grad = False

        if opt.laplace_use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, input, targets):
        input_r, input_g, input_b = self.slice(input)
        targets_r, targets_g, targets_b = self.slice(targets)
        return self.loss(self.laplace(input_r), self.laplace(targets_r)) + \
               self.loss(self.laplace(input_g), self.laplace(targets_g)) + \
               self.loss(self.laplace(input_b), self.laplace(targets_b))

class GaussLoss(nn.Module):
    def __init__(self, opt):
        super(GaussLoss, self).__init__()
        self.slice = Slice(1, (1,2,3))

        self.gauss = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        gauss = torch.mul(torch.Tensor([[1,2,1],[2,4,2],[1,2,1]]), 1/16.0)
        self.gauss.weight.data.copy_(gauss)
        self.gauss = self.gauss.cuda() if len(opt.gpu_ids)>0 else self.gauss
        self.gauss.weight.requires_grad = False

        self.laplace = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        laplace = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
        self.laplace.weight.data.copy_(laplace)
        self.laplace = self.laplace.cuda() if len(opt.gpu_ids)>0 else self.laplace
        self.laplace.weight.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, input, targets):
        input_r, input_g, input_b = self.slice(input)
        targets_r, targets_g, targets_b = self.slice(targets)
        return self.loss(self.laplace(input_r), self.laplace(self.gauss(targets_r))) + \
               self.loss(self.laplace(input_g), self.laplace(self.gauss(targets_g))) + \
               self.loss(self.laplace(input_b), self.laplace(self.gauss(targets_b)))


class LabelLoss(nn.Module):
    def __init__(self, opt):
        super(LabelLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.l2loss = nn.MSELoss()
        index = torch.LongTensor([1,3,16,17,4,7,5,6])
        index = Variable(index)
        index = index.cuda(opt.gpu_ids[0]) if len(opt.gpu_ids)>0 else index
        self.index = index

    def forward(self, input, targets):
        input = torch.index_select(input, 1, self.index)
        out = torch.sum(torch.sum(input, dim=3, keepdim=True), dim=2, keepdim=True)
        out = self.sigmoid(torch.add(torch.mul(out, 0.1), -5))
        return self.l2loss(out, targets)

def latent_kl(mr, mf, vr, vf):
    mean1 = q
    mean2 = 0

    kl = 0.5 * torch.pow(mean2 - mean1, 2)
    kl = torch.sum(torch.sum(torch.sum(kl, dim=3), dim=2), dim=1)
    kl = torch.mean(kl)
    return kl

class SigmaLoss(nn.Module):
    def __init__(self, lambda_sigma):
        super(SigmaLoss, self).__init__()
        self.lambda_sigma = lambda_sigma
        self.mr = Variable(torch.Tensor([0.2947, 0.2856, 0.2849]).cuda())
        self.vr = Variable(torch.Tensor([0.098952, 0.092775, 0.092406]).cuda())

    def step_lambda(self):
        pass

    def kl_loss(self, mf, vf):
        kl = - 0.5 * (torch.log(vf/self.vr) + 1 - 1/self.vr * (vf + (self.mr - mf)**2))
        return torch.mean(kl)

    def minibatch_analysis(self, input, sort=True):
        mean = torch.mean(torch.mean(torch.mean(input, dim=3), dim=2), dim=0)
        vars = (torch.mean(torch.mean(input, dim=3), dim=2) - mean) ** 2
        if sort:
            unnormal = torch.sum(vars, dim=1)
            var, id = torch.sort(unnormal, descending=True)

        out_var = torch.mean(vars, dim=0)
        return (mean, out_var, id.data[0:5]) if sort else (mean, out_var)

    def forward(self, input_f, input_f_z1, input_m = None):
        if input_m is not None:
            input_f_z1 = torch.clamp(input_f + 1 + 2 * (1 - input_m.clamp(0,1)), 0, 2) - 1
        mf, vf= self.minibatch_analysis(input_f, False)
        mf_z1, vf_z1, id = self.minibatch_analysis(input_f_z1)
        print('mf')
        print(mf, mf_z1)
        print('vf')
        print(vf, vf_z1)
        return self.kl_loss(mf, vf) * self.lambda_sigma[0], \
               self.kl_loss(mf_z1, vf_z1) * self.lambda_sigma[1], \
               Variable(id)



