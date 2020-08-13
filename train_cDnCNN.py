# -*- coding: utf-8 -*-
# 2019 1127 Modified   by S. Liu
# 2020 0321 ReModified by S. Liu

import argparse
import re
import os
import glob
import datetime
import time
import numpy as np
import torch
import h5py
from scipy import io
from utils import cov
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset


class MyDenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, xs, ys):
        super(MyDenoisingDataset, self).__init__()
        self.xs = xs
        self.ys = ys

    def __getitem__(self, index):
        batch_x = self.xs[index]  # ground truth
        batch_y = self.ys[index]  # noisy image
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


# Params
parser = argparse.ArgumentParser(description='PyTorch Complex DnCNN')
parser.add_argument(
    '--model',
    default='cDnCNN',
    type=str,
    help='choose a type of model')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--train_data',
                    default='channel',
                    type=str,
                    help='path of train data')
parser.add_argument('--snr', default=5, type=int, help='noise level')
parser.add_argument('--channel_clusters', default=6,
                    type=int, help='clusters of channel')
parser.add_argument('--paths_per_cluster', default=8,
                    type=int, help='paths per cluster')
parser.add_argument('--angle_spread', default=7.5,
                    type=int, help='angle spread')
parser.add_argument('--epoch',
                    default=150,
                    type=int,
                    help='number of train epoches')
parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='initial learning rate for Adam')
args = parser.parse_args()
clusters = args.channel_clusters
paths = args.paths_per_cluster
AS = args.angle_spread
batch_size = args.batch_size
train_data = args.train_data
cuda = torch.cuda.is_available()
print(cuda)
n_epoch = args.epoch
snr = args.snr
# snr = 10
save_dir = os.path.join('models', args.model + '_' + 'snr' + str(snr))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


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


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args,
          **kwargs)


if __name__ == '__main__':

    print('>>> Building Model')
    model = ComplexDnCNN().cuda()
    # uncomment to use dataparallel (unstable)
    # device_ids = [0, 1]
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(
            torch.load(os.path.join(save_dir,
                                    'model_%03d.pth' % initial_epoch)))
    print(">>> Building Model Finished")
    model.train()  # Enable BN and Dropout
    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 120],
                            gamma=0.5)  # learning rates
    print("Loading Data")
    # should be generated by yourself
    train_est = train_data + '/' + \
        str(snr)+'dB/trainingChannel'+str(snr) + \
        '_C' + str(clusters) + 'P' + str(paths) + '_AS' + str(AS) + '.mat'

    train_true = train_data + '/' + \
        str(snr)+'dB/trueTrainingChannel'+str(snr) + \
        '_C' + str(clusters) + 'P' + str(paths) + '_AS' + str(AS) + '.mat'

    train_est_mat = h5py.File(train_est, mode='r')
    # print(train_est_mat.keys())
    x_train = train_est_mat['trainingChannel']
    x_train = np.transpose(x_train, [3, 2, 1, 0])
    print('>>> training Set setup complete')

    # ground truth
    train_true_mat = h5py.File(train_true, mode='r')
    # y_train = train_true_mat['trueTrainingChannel']
    y_train = train_true_mat['trueTrainingChannel']
    y_train = np.transpose(y_train, [3, 2, 1, 0])
    print('>>> groundTruth Set setup complete')

    x_train = torch.from_numpy(x_train).float().reshape(
        [x_train.shape[0], x_train.shape[1], 1, x_train.shape[2], x_train.shape[3]])
    # x_train = x_train[0:2000, :]
    print(x_train.shape)
    y_train = torch.from_numpy(y_train).float().reshape(
        [y_train.shape[0], y_train.shape[1], 1, y_train.shape[2], y_train.shape[3]])
    print(y_train.shape)

    for epoch in range(initial_epoch, n_epoch):

        DDataset = MyDenoisingDataset(y_train, x_train)
        DLoader = DataLoader(dataset=DDataset,
                             num_workers=4,
                             drop_last=True,
                             batch_size=batch_size,
                             shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
            loss = criterion(model(batch_y), batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)  # step to the learning rate in this epoch
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f\t average Loss:%2.4f' %
                      (epoch + 1, n_count, x_train.size(0) // batch_size,
                       loss.item() / batch_size, avgLoss[epoch]))
        elapsed_time = time.time() - start_time
        log('epoch = %4d , loss = %4.4f , time = %4.2f s' %
            (epoch + 1, epoch_loss / n_count, elapsed_time))
        torch.save(model.state_dict(),
                   os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
