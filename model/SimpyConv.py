import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import model
from model import conv_type 
from model.conv_type import GraphMapConv2d
from utils.builder import GraphMapConv2dBuilder
from model.conv_type import generate_index_and_GroupNum
from args import args


class SimpleConv(nn.Module):
    def __init__(self, num_classes, index, group_num, in_planes=3, out_planes=512, dropRate=0.0,stride=2, *args):
        super(SimpleConv, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes//2, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = GraphMapConv2dBuilder(out_planes//2, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False,group_num=group_num,index=index)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = GraphMapConv2dBuilder(out_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False,group_num=group_num,index=index)
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(out_planes,num_classes)
    def forward(self, x):

        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = self.conv3(self.relu3(self.bn3(x)))
        x = self.avg(x).view(x.size(0), -1)
        x = self.fc(x)
        return x



def SimpleVGG(num_classes=10, message_type=None,**kwargs):
    print(f'The graph method is {message_type}')
    args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)
    model = SimpleConv(num_classes=num_classes, index=args.edge_index, group_num=args.group_num)
    return model


