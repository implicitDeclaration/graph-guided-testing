import torch.nn as nn
import numpy as np
import os
from args import args
from utils.builder import GraphMapConv2dBuilder
from model.conv_type import generate_index_and_GroupNum


class VGG16(nn.Module):
    def __init__(self, num_classes=10, group_num=None, index=None):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            GraphMapConv2dBuilder(inplanes=64, outplanes=64, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            GraphMapConv2dBuilder(inplanes=64, outplanes=128, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            GraphMapConv2dBuilder(inplanes=128, outplanes=128, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            GraphMapConv2dBuilder(inplanes=128, outplanes=256, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            GraphMapConv2dBuilder(inplanes=256, outplanes=256, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            GraphMapConv2dBuilder(inplanes=256, outplanes=256, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            GraphMapConv2dBuilder(inplanes=256, outplanes=512, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            GraphMapConv2dBuilder(inplanes=512, outplanes=512, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            GraphMapConv2dBuilder(inplanes=512, outplanes=512, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            GraphMapConv2dBuilder(inplanes=512, outplanes=512, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            GraphMapConv2dBuilder(inplanes=512, outplanes=512, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            GraphMapConv2dBuilder(inplanes=512, outplanes=512, kernel_size=3, stride=1, padding=1, bias=False,
                                  group_num=group_num, index=index),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )

        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def GraphMapVGG16(num_classes=10, message_type=None,index=None, *arg):
    print(f'The graph method is {args.message_type}')
    if args.pretrained:
        args.edge_index = np.load(os.path.join(os.path.dirname(args.pretrained), 'edge_index.npy'))
        unique = np.unique(args.edge_index)
        args.group_num = len(unique)
    else:
        args.edge_index, args.group_num = generate_index_and_GroupNum(message_type=message_type)

    return VGG16(num_classes=num_classes, index=args.edge_index,group_num=args.group_num)
