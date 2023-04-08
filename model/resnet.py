#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from model import conv_type 
from model.conv_type import GraphMapConv2d
import utils 
from utils import initialization
from utils.builder import GraphMapConv2dBuilder
from model.conv_type import generate_index_and_GroupNum
from args import args
import os
import numpy as np



class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,group_num=None,index=None):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = GraphMapConv2dBuilder(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False,group_num=group_num,index=index)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = GraphMapConv2dBuilder(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False,group_num=group_num,index=index)

        if stride != 1 or in_planes != self.expansion*planes:
            # self.shortcut = nn.Sequential(
            #     GraphMapConv2dBuilder(in_planes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False,message_type=message_type)
            # )
            self.shortcut = GraphMapConv2dBuilder(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False,group_num=group_num,index=index)
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_num=None,index=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = GraphMapConv2dBuilder(in_planes, planes, kernel_size=1, 
                                stride=1,padding=0,bias=False, group_num=group_num, index=index)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = GraphMapConv2dBuilder(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, group_num=group_num, index=index)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = GraphMapConv2dBuilder(planes, self.expansion*planes,kernel_size=1,
                                stride=1,padding=0,bias=False, group_num=group_num, index=index)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                GraphMapConv2dBuilder(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False,group_num=group_num,index=index))
            

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,  group_num=None,index=None):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, group_num=group_num, index=index)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, group_num=group_num, index=index)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, group_num=group_num, index=index)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, group_num=group_num, index=index)
        self.avg_pool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear1 = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride, group_num, index):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,  group_num=group_num, index=index))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # print(self.conv1.weight[0,0,0,0])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        # if not args.GlobalMap:
        #     out = self.linear(out)
        # else:
        #     out = self.linear1(out)
        out = self.linear(out)
        return out


def GraphMapResNet18(num_classes=10,message_type=None,index=None, *arg):
    print(f'The graph method is {message_type}')
    if args.pretrained:
        args.edge_index = np.load(os.path.join(os.path.dirname(args.pretrained),'edge_index.npy'))
        unique = np.unique(args.edge_index)
        args.group_num = len(unique)
    else:
        args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)
        unique = np.unique(args.edge_index)
        args.group_num = len(unique)
    return PreActResNet(PreActBlock, [2,2,2,2],num_classes=num_classes, index=args.edge_index, group_num=args.group_num)

def GraphMapResNet34(num_classes=10, message_type=None, *arg):
    print(f'The graph method is {message_type}')
    args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)
    return PreActResNet(PreActBlock, [3,4,6,3],num_classes=num_classes, index=args.edge_index, group_num=args.group_num)

def GraphMapResNet50(num_classes=10,message_type=None, *arg):
    print(f'The graph method is {message_type}')
    args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes=num_classes, index=args.edge_index, group_num=args.group_num)

def GraphMapResNet101(num_classes=10, message_type=None, *arg):
    print(f'The graph method is {message_type}')
    args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)
    return PreActResNet(PreActBottleneck, [3,4,23,3],num_classes=num_classes, index=args.edge_index, group_num=args.group_num)

def GraphMapResNet152(num_classes=10, message_type=None,index=None, *arg):
    print(f'The graph method is {message_type}')
    if args.pretrained:
        args.edge_index = np.load(os.path.join(os.path.dirname(args.pretrained),'edge_index.npy'))
        unique = np.unique(args.edge_index)
        args.group_num = len(unique)
    else:
        args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)

    return PreActResNet(PreActBottleneck, [3,8,36,3],num_classes=num_classes, index=args.edge_index, group_num=args.group_num)


