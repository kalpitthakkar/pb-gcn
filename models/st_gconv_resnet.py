import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from . import graph as graf
from .utils import conv_init
from .st_residual_unit import STResidualUnit, STUnit

class STGraphConvResnet(nn.Module):
    """ Spatio-Temporal Graph Convolutional Residual Network
        for skeleton action recognition
    """

    def __init__(self,
                 channel,
                 num_class,
                 window_size,
                 num_joints,
                 num_actors=1,
                 use_data_bn=False,
                 layers_config=None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5,
                 dataset='NTU'):
        super(STGraphConvResnet, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = getattr(graf, graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A).float().cuda(0)

        self.dataset = dataset
        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale

        # Different bodies share batch norm parameters or not
        self.M_dim_bn = True

        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_joints * num_actors)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_joints)

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size,
            dataset=dataset)

        self.layers = nn.ModuleList([
            STResidualUnit(in_c, out_c, stride=stride, multiscale=self.multiscale, **kwargs)
            for in_c, out_c, stride in layers_config
        ])
        net_inp_channels = layers_config[0][0]
        net_out_channels = layers_config[-1][1]
        net_out_frames = window_size

        # head
        self.head = STUnit(
            channel,
            net_inp_channels,
            **kwargs
        )

        # tail
        self.fcn = nn.Conv1d(net_out_channels, num_class, kernel_size=1)
        conv_init(self.fcn)

    def forward(self, x):

        N, C, T, V, M = x.size()

        # data bn
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)

            x = self.data_bn(x)
            # to (N*M, C, T, V)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # model
        x = self.head(x)

        for layer in self.layers:
            x = layer(x)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)

        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])

        # C fcn
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.size()[2:])
        x = x.view(N, self.num_class)

        return x
