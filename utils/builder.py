import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import model
from model import conv_type 
from model.conv_type import GraphMapConv2d
import utils
from utils import initialization
from args import args

def GraphMapConv2dBuilder(inplanes, outplanes, kernel_size, stride, bias, group_num, index, padding=0):

    GraphMapConv = GraphMapConv2d(inplanes,outplanes,kernel_size=kernel_size,
                                  stride=stride,padding=padding,bias=bias)
                                  
    GraphMapConv.weight = initialization.ConvWeights_Initialization(args,GraphMapConv)

    return GraphMapConv
