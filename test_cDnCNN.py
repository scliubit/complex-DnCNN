# -*- coding: utf-8 -*-

import glob
import re
import argparse
import h5py
import os
import time
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from utils import cov


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data',
                        default='channel',
                        type=str,
                        help='path of test data')
    parser.add_argument('--snr', default=10, type=int, help='noise level')
    parser.add_argument('--model',
                        default=os.path.join(
                            'models', 'ComplexDnCNNcc'),
                        help='directory of the model')
    parser.add_argument('--model_name',
                        default='model_150.pth',
                        type=str,
                        help='the model name')
    parser.add_argument('--channel_clusters', default=6,
                        type=int, help='clusters of channel')
    parser.add_argument('--paths_per_cluster', default=1,
                        type=int, help='paths per cluster')
    parser.add_argument('--angle_spread', default=0,
                        type=int, help='angle spread')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args,
          **kwargs)


class ComplexBN(torch.nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(ComplexBN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps
        self.num_features = num_features
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
        e, v = torch.symeig(covMat, True)
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

        # Model components
        # define complex conv
        self.conv_re = torch.nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias).to(self.device)
        self.conv_im = torch.nn.Conv2d(in_channels,
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
        paddingF = torch.nn.ZeroPad2d(1)
        # print(x.shape)
        r = paddingF(x[:, 0])  # NCHW
        i = paddingF(x[:, 1])
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
        self.relu_re = torch.nn.ReLU(inplace=inplace).to(self.device)
        self.relu_im = torch.nn.ReLU(inplace=inplace).to(self.device)

    def forward(self, x):
        output = torch.stack(
            (self.relu_re(x[:, 0]), self.relu_im(x[:, 1])), dim=1).to(self.device)
        return output


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


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


if __name__ == '__main__':

    args = parse_args()
    snr = args.snr
    save_dir = os.path.join(args.model + '_' + 'snr' + str(snr))
    print(save_dir)
    clusters = args.channel_clusters
    paths = args.paths_per_cluster
    AS = args.angle_spread
    model = ComplexDnCNN().cuda()
    # device_ids = [0, 1]
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    checkPoint = findLastCheckpoint(save_dir=save_dir)
    if checkPoint > 0:
        print('resuming by loading epoch %03d' % checkPoint)
        model.load_state_dict(
            torch.load(os.path.join(save_dir,
                                    'model_%03d.pth' % checkPoint)))
    print(">>> Building Model Finished")
    model.eval()  # evaluation mode
    snr = 5

    test_est = args.test_data + '/' + str(snr)+'dB/testChannel'+str(snr) + \
        '_C'+str(clusters)+'P'+str(paths)+'_AS'+str(AS) + '.mat'
    test_true = args.test_data + '/' + str(snr)+'dB/trueTestChannel'+str(snr) + \
        '_C'+str(clusters)+'P'+str(paths)+'_AS'+str(AS) + '.mat'

    # print(test_est)
    test_est_mat = h5py.File(test_est, 'r')
    x_test = test_est_mat['testChannel']
    x_test = np.transpose(x_test, [3, 2, 1, 0])
    print(x_test.shape)
    print('x_test set over')

    test_true_mat = h5py.File(test_true, 'r')
    y_test = test_true_mat['trueTestChannel']
    y_test = np.transpose(y_test, [3, 2, 1, 0])
    print('y_test set over')

    x_test = torch.from_numpy(x_test).float().reshape(
        [x_test.shape[0], x_test.shape[1], 1, x_test.shape[2], x_test.shape[3]])
    x_test = x_test[-1000:, :, :, :, :]
    y_test = torch.from_numpy(y_test).float().reshape(
        [y_test.shape[0], y_test.shape[1], 1, y_test.shape[2], y_test.shape[3]])
    y_test = y_test[-1000:, :, :, :, :]

    print(x_test.shape)
    batch_size = 1
    predict = torch.zeros([batch_size, 2, 256, 256])
    NMSE = 0

    for i in range(int(x_test.shape[0] / batch_size)):
        data = x_test[i * batch_size:(i + 1) * batch_size, :, :, :]
        groud_truth = y_test[i * batch_size:(i + 1) * batch_size, :, :, :]
        predict = model(data)
        predict = predict.cpu().detach().numpy()

        groud_truth = groud_truth.cpu().detach().numpy()

        predict_real = np.reshape(predict[:, 0], (len(predict), -1))
        predict_imag = np.reshape(predict[:, 1], (len(predict), -1))
        predict_C = predict_real + 1j * predict_imag

        groud_truth_real = np.reshape(groud_truth[:, 0],
                                      (len(groud_truth), -1))
        groud_truth_imag = np.reshape(groud_truth[:, 1],
                                      (len(groud_truth), -1))
        groud_truth_C = groud_truth_real + 1j * groud_truth_imag

        NMSE += np.mean(
            np.sum(abs(groud_truth_C - predict_C)**2, axis=1) /
            np.sum(abs(groud_truth_C)**2, axis=1))
        print(i, NMSE, end='\r')
    print('\nNMSE = ', 10 * np.log10(NMSE /
                                     int(x_test.shape[0] / batch_size)))
