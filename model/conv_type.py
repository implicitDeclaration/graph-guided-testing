import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init


import networkx as nx
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.autograd import Function

from itertools import repeat
from networkx.utils import py_random_state
# from pycls.datasets.load_graph import load_graph
import pdb
import time
import random

# import different types of cage_graph
import model
from model import Heawood,McGee,Pentatope,Petersen,Ultility,Regular_Graph,Gaussian_Optimal,PowLaw
from args import args


def compute_count(channel, group):
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    return out

def compute_size(channel, group, seed=1):
    np.random.seed(seed)
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    out = np.random.permutation(out)
    return out

def compute_densemask(in_channels, out_channels, group_num, edge_index):
    repeat_in = compute_size(in_channels, group_num)
    repeat_out = compute_size(out_channels, group_num)
    mask = np.zeros((group_num, group_num))
    mask[edge_index[:, 0], edge_index[:, 1]] = 1
    mask = np.repeat(mask, repeat_out, axis=0)
    mask = np.repeat(mask, repeat_in, axis=1)
    # print('mask',mask.shape)
    # print('mask',np.sum(mask))
    # import sys
    # sys.exit()
    return mask

def generate_index_and_GroupNum(message_type,self_loop=False,directed=False):
    # print('message_type',message_type)
    if message_type == 'Heawood':
        edge_index = Heawood.Heawood_Graph()
    elif message_type == 'McGee':
        edge_index = McGee.McGee_Graph()
    elif message_type == 'Pentatope':
        edge_index = Pentatope.Pentatope_Graph()
    elif message_type == 'Petersen':
        edge_index = Petersen.Petersen_Graph()
    elif message_type == 'Ultility':
        edge_index = Ultility.Utility_Graph()
    elif message_type == 'Regular':
        edge_index = Regular_Graph.Generate_Optimal_Symmetric_Network(args.neighbors,args.nodes)
    elif message_type == 'Guassian':
        edge_index = Gaussian_Optimal.Generate_Optimal_Guassian_Network(args.neighbors,args.nodes)
    elif message_type == 'PowLaw':
        edge_index = PowLaw.Generate_Optimal_ScaleFree_Network(args.neighbors,args.nodes)
        # edge_index = PowLaw.GenPowLawDisGraph(args.neighbors, args.nodes)

    else:
        edge_index = np.array([[0,0],[0,1],[1,0],[1,1]])
    group_num = len(set(np.reshape(edge_index,-1)))
    if not directed:
        edge_index = np.concatenate([edge_index,edge_index[:,::-1]],axis=0)
    if self_loop:
        loop = np.array([[i,i] for i in range(group_num)])
        edge_index = np.concatenate([edge_index,loop],axis=0)
    return edge_index,group_num

def get_mask(in_channels, out_channels, directed=False,self_loop=False):



    # edge_index_high,group_num = generate_index_and_GroupNum(message_type=message_type,self_loop=self_loop,directed=directed)
    # high-level graph edge index
    group_num = args.group_num
    edge_index = args.edge_index
    # print(group_num,in_channels,out_channels )
    assert group_num <= in_channels and group_num <= out_channels  
    # get in/out size for each high-level node
    in_sizes = compute_size(in_channels, group_num)
    out_sizes = compute_size(out_channels, group_num)
    # decide low-level node num
    group_num_low = int(min(np.min(in_sizes), np.min(out_sizes)))
    # decide how to fill each node
    mask_high = compute_densemask(in_channels, out_channels, group_num, edge_index)
    # print("mask_hign.shape",np.sum(mask_high),mask_high.shape)
    return mask_high




############## Conv model

class GraphMapConv2d(_ConvNd):
    '''Relational graph version of Conv2d. Neurons "talk" according to the graph structure'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False, message_type=None, directed=False, agg='sum',
                 sparsity=0.5, p=0.2, talk_mode='dense', seed=None):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GraphMapConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation,
            False, _pair(0), 1, bias, 'zeros')
        self.mask = get_mask(in_channels, out_channels, directed=directed)
        nonzero = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask[:, :, np.newaxis, np.newaxis]).float()

        self.init_scale = torch.sqrt(out_channels / torch.sum(self.mask.cpu(), dim=0, keepdim=True))
        self.flops_scale = nonzero / (in_channels * out_channels)
        self.params_scale = self.flops_scale
        self.weight_global = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.weight_global, a=math.sqrt(5))
    def forward(self, input):

        # if True:
        weight = self.weight * self.mask.cuda()  # 子网络


            # print('==>using the global network')
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, 1)

########### Other OPs

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish activation fun."""

    def __init__(self, in_w, se_w, act_fun):
        super(SE, self).__init__()
        self._construct_class(in_w, se_w, act_fun)

    def _construct_class(self, in_w, se_w, act_fun):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Swish, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(in_w, se_w, kernel_size=1, bias=True),
            act_fun(),
            nn.Conv2d(se_w, in_w, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))




############## Linear model

class GraphMapLinear(nn.Linear):
    '''Relational graph version of Linear. Neurons "talk" according to the graph structure'''

    def __init__(self, in_channels, out_channels, bias=False,directed=False):
        # group_num_max = min(in_channels, out_channels)
        # if group_num > group_num_max:
        #     group_num = group_num_max
        # print(group_num, in_channels, out_channels, kernel_size, stride)
        super(GraphMapLinear, self).__init__(
            in_channels, out_channels, bias)

        self.mask = get_mask(in_channels, out_channels, directed=directed)
        nonzero = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask).float()

        self.init_scale = torch.sqrt(out_channels / torch.sum(self.mask.cpu(), dim=0, keepdim=True))
        self.flops_scale = nonzero / (in_channels * out_channels)
        self.params_scale = self.flops_scale
    def forward(self, x):
        weight = self.weight * self.mask.cuda()
        # pdb.set_trace()
        return F.linear(x, weight, self.bias)