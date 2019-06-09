import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import conv_init

class STUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 dropout=0.5,
                 mask_learning=False,
                 dataset='NTU'):
        super(STUnit, self).__init__()

        self.dataset = dataset
        # Number of joints (or nodes in the graph)
        self.V = A.size()[-1]

        # The adjacency matrices of the graph (different node-node similarities)
        self.A = Variable(
            A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of adjacency matrices (number of partitions)
        self.num_A = self.A.size()[0]

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mask_learning = mask_learning

        # if true, each node have specific parameters of batch normalization layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn

        # Convolutions for each parition (different similarities => different weights)
        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                padding=0,
                stride=1) for i in range(self.num_A)
        ])

        if use_local_bn:
            self.local_bn = nn.BatchNorm1d(self.out_channels * self.V)

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.temporal_conv = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=(kernel_size, 1),
            padding=(int((kernel_size - 1)/2), 0),
            stride=(stride, 1),
            bias=True
        )

        self.bn2 = nn.BatchNorm2d(self.out_channels)

        # initialize
        for conv in self.conv_list:
            conv_init(conv)
        conv_init(self.temporal_conv)

    def forward(self, x):
        A = self.A.cuda(x.get_device())

        N, C, T, V = x.size()

        # reweight adjacency matrix
        if self.mask_learning:
            A = A * self.mask

        # graph convolution
        for i, a in enumerate(A):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)

            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y + self.conv_list[i](xa)

        # batch normalization
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.local_bn(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn1(y)

        # nonlinearity
        y = self.relu(y)

        # Temporal convolution
        y = self.drop(y)
        y = self.temporal_conv(y)
        y = self.bn2(y)

        # Added on last day
        y = self.relu(y)

        return y


class STResidualUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 dropout=0.5,
                 mask_learning=False,
                 multiscale=False,
                 dataset='NTU'):
        super(STResidualUnit, self).__init__()

        self.dataset = dataset
        if multiscale:
            self.stUnit = nn.Sequential(
                STUnit(in_channels, out_channels/2, A,
                    use_local_bn, kernel_size, stride, dropout, mask_learning, dataset),
                STUnit(in_channels, out_channels-out_channels/2, A,
                    use_local_bn, kernel_size*2-1, stride, dropout, mask_learning, dataset))
        else:
            self.stUnit = nn.Sequential(
                STUnit(in_channels, out_channels, A,
                    use_local_bn, kernel_size, stride, dropout, mask_learning, dataset))

        self.relu = nn.ReLU()

        if (in_channels != out_channels) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=(stride, 1),
                    bias=False
                ), nn.BatchNorm2d(out_channels)
            )
            conv_init(list(self.downsample.children())[0])
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.stUnit(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
