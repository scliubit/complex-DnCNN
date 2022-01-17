import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import glob
import datetime
import re


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


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


if __name__ == "__main__":
    r = False
    x = np.random.randn(30, 2)
    xt = torch.from_numpy(x).type(torch.double)
    np_c = np.cov(x, rowvar=r)
    our_c = cov(xt, rowvar=r).numpy()
    print(np.allclose(np_c, our_c))
    print(x, '\n\n\n', our_c, '\n\n\n', torch.var(xt))
