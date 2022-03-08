# -*- coding: utf-8 -*-
# 2019 1127 Modified   by S. Liu
# 2020 0321 ReModified by S. Liu
# 2022 0117 Re S. Liu
# 2022 0308 Re S. Liu

import argparse
import re
import os
import glob
import datetime
import time
import numpy as np
import torch
import h5py
import torch.nn as nn
from scipy import io
from utils import cov


class ComplexBN(torch.nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(ComplexBN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.num_features = num_features
        self.upper = True
        # self.batchNorm2dF = torch.nn.BatchNorm2d(num_features,
        #                                          affine=affine).to(self.device)

    def forward(self, x):  # shpae of x : [batch,2,channel,axis1,axis2]
        # divide dim=1 to 2 parts -> real and imag
        # real/imag = [batch, channel, axis1, axis2]

        real = x[:, 0]
        imag = x[:, 1]

        realVec = torch.flatten(real)
        imagVec = torch.flatten(imag)
        re_im_stack = torch.stack((realVec, imagVec), dim=1)
        covMat = cov(re_im_stack)
        # e, v = torch.symeig(covMat, True)
        e, v = torch.linalg.eigh(covMat, UPLO='U' if self.upper else 'L')
        covMat_sq2 = torch.mm(torch.mm(v, torch.diag(torch.pow(e, -0.5))),
                              v.t())
        data = torch.stack((realVec - real.mean(), imagVec - imag.mean()),
                           dim=1).t()
        whitenData = torch.mm(covMat_sq2, data)
        real_data = whitenData[0, :].reshape(real.shape[0], real.shape[1],
                                             real.shape[2], real.shape[3])
        imag_data = whitenData[1, :].reshape(real.shape[0], real.shape[1],
                                             real.shape[2], real.shape[3])
        output = torch.stack((real_data, imag_data), dim=1)
        return output


class ComplexConv2D(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(ComplexConv2D, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        self.paddingF = nn.ZeroPad2d(1)
        # Model components
        # define complex conv
        self.conv_re = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias).to(self.device)
        self.conv_im = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias).to(self.device)
        self.weight1 = self.conv_re.weight
        self.weight2 = self.conv_im.weight
        self.bias1 = self.conv_re.bias
        self.bias2 = self.conv_im.bias

    def forward(self, x):
        
        # print(x.shape)
        r = self.paddingF(x[:, 0])  # NCHW
        # print(r.shape)
        i = self.paddingF(x[:, 1])
        # print(r.shape)
        # New 20191102
        r[:, :, 0, :], i[:, :, 0, :] = r[:, :, -2, :], i[:, :, -2, :]
        r[:, :, -1, :], i[:, :, -1, :] = r[:, :, 1, :], i[:, :, 1, :]
        r[:, :, :, 0], i[:, :, :, 0] = r[:, :, :, 2], i[:, :, :, 2]
        r[:, :, :, -1], i[:, :, :, -1] = r[:, :, :, 1], i[:, :, :, 1]
        # NEW END
        real = self.conv_re(r) - self.conv_im(i)
        # print(real.shape)
        imaginary = self.conv_re(i) + self.conv_im(r)
        # stack real and imag part together @ dim=1
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexReLU(torch.nn.Module):

    def __init__(self, inplace=False):
        super(ComplexReLU, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.relu = nn.ReLU()
        # self.relu_re = nn.ReLU(inplace=inplace).to(self.device)
        # self.relu_im = nn.ReLU(inplace=inplace).to(self.device)

    def forward(self, x):
        # output = torch.stack(
        #     (self.relu_re(x[:, 0]), self.relu_im(x[:, 1])), dim=1).to(self.device)
        return self.relu(x)


class ComplexDnCNN(torch.nn.Module):

    def __init__(self,
                 depth=17,
                 n_channels=64,
                 image_channels=1,
                 use_bnorm=True,
                 kernel_size=3):
        super(ComplexDnCNN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # kernel_size = 3
        padding = 0
        layers = []
        # 1. Conv2d and ReLU
        layers.append(
            ComplexConv2D(in_channels=image_channels,
                          out_channels=n_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=True))
        layers.append(ComplexReLU(inplace=False))
        # 2. 15 * (Conv2d + BN + ReLU)
        for _ in range(depth - 2):
            layers.append(
                ComplexConv2D(in_channels=n_channels,
                              out_channels=n_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=False))
            '''layers.append(torch.nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.95).to(device=self.device))'''
            layers.append(ComplexBN(n_channels, eps=0.0001, momentum=0.95))
            layers.append(ComplexReLU(inplace=False))
        # 3. conv2d
        layers.append(
            ComplexConv2D(in_channels=n_channels,
                          out_channels=image_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False))
        self.dncnn = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out
