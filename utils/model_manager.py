import torch

import sys
sys.path.append('../')
import numpy as np

import torch

import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from model.resnet import *
from model.VGG16 import *
from model.vgg19 import *
from args import args



def set_gpu(args, model):
    

    # DataParallel will divide and allocate batch_size to all available GPUs
    print(f"=> Parallelizing on {args.multigpu} gpus")
    torch.cuda.set_device(args.multigpu[0])

    model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def fetch_models(path, device,model_num):
    '''
    :param models_folder:
    :param num_models: the number of models to be load
    :param start_no: the start serial number from which loading  "num_models" models. 1-index
    :return: the top [num_models] models in models_folder
    '''
    target_models = []
    #  gpu memory is not enough

    for j in range(max(model_num-5, 0), model_num):  ### >>> for large model

        args.edge_index = np.load(path + '{}'.format(j) + '/checkpoints/edge_index.npy')
    
        unique = np.unique(args.edge_index)
        args.group_num = len(unique)
        model1 = path + '{}'.format(j) + '/checkpoints/model_best.pth'
        
        args.pretrained = model1
        if args.set == 'cifar10':
            model = PreActResNet(PreActBlock, [2,2,2,2],num_classes=10, index=args.edge_index, group_num=args.group_num)
        elif args.set == 'svhn':
            model = VGG16(num_classes=10, index=args.edge_index, group_num=args.group_num)
        else:
            model = VGG19(num_classes=10, index=args.edge_index, group_num=args.group_num)
        model = set_gpu(args, model)

        pretrained = torch.load(
            model1,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]
        print('===> successful loading the pretrained')
        model_state_dict = model.state_dict()

        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }

        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

        target_models.append(model)
    return target_models



