from args import args
import math

import torch
import torch.nn as nn

def ConvWeights_Initialization(args,conv):

    if args.init == "signed_constant":

        fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
        if args.scale_fan:
            fan = fan * (1 - args.prune_rate)
        gain = nn.init.calculate_gain(args.nonlinearity)
        std = gain / math.sqrt(fan)
        conv.weight.data = conv.weight.data.sign() * std

    elif args.init == "unsigned_constant":

        fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
        if args.scale_fan:
            fan = fan * (1 - args.prune_rate)

        gain = nn.init.calculate_gain(args.nonlinearity)
        std = gain / math.sqrt(fan)
        conv.weight.data = torch.ones_like(conv.weight.data) * std

    elif args.init == "kaiming_normal":

        if args.scale_fan:
            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            fan = fan * (1 - args.prune_rate)
            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                conv.weight.data.normal_(0, std)
        else:
            nn.init.kaiming_normal_(
                conv.weight, mode=args.mode, nonlinearity=args.nonlinearity
            )

    elif args.init == "kaiming_uniform":
        nn.init.kaiming_uniform_(
            conv.weight, mode=args.mode, nonlinearity=args.nonlinearity
        )
    elif args.init == "xavier_normal":
        nn.init.xavier_normal_(conv.weight)
    elif args.init == "xavier_constant":

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        conv.weight.data = conv.weight.data.sign() * std

    elif args.init == "standard":

        nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
    
    return conv.weight


