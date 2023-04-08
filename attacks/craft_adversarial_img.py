from __future__ import print_function
import sys
sys.path.append('../')
from attacks.attack_type.fgsm import FGSM
from attacks.attack_type.carlinl2 import CarliniL2
from attacks.attack_type.jsma import *
from attacks.attack_type.deepfool import DeepFool
from attacks.attack_type.onepixel import *
from utils.data_manger import *
from attacks.attack import ILA,ifgsm
import os
from advertorch.attacks import LocalSearchAttack
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.net_utils import (
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)

from args import args

import model as net
import numpy as np

sys.path.append('/public/mount_data/data/xjy/GraphMapNetwork/')


'''
This component has three levels

The genereate_?_samples is the kernel generator which calls a specific adversary to generate the adversarial samples by 
batch or by single sample.

do_craft_? is designed to do attack for multil-models.

The two levels are not bound with certain data

The level, like craft_mnist,craft_cifar10,craft_imagenet,is the application level,which is bound with certain level.

'''
def genereate_fgsm_samples(model, source_data, save_path, eps, is_exclude_wr=True,
                           data_type=DATA_MNIST, device="cpu"):
    '''

    :param model_path:
    :param source_data:
    :param save_path:
    :param eps:
    :param is_save:
    :param is_exclude_wr:  exclude the wrong labeled or not
    :return:
    '''

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    test_data = exclude_wrong_labeled(model, test_data, device)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    fgsm = FGSM(model, eps=eps, device=device)
    adv_samples, y = fgsm.do_craft_batch(test_loader)
    adv_loader = DataLoader(TensorDataset(adv_samples, y), batch_size=1, shuffle=False)
    succeed_adv_samples = samples_filter(model, adv_loader, "Eps={}".format(eps), device=device)
    num_adv_samples = len(succeed_adv_samples)
    print('successful samples', num_adv_samples)
    if is_save:
        save_imgs(succeed_adv_samples, TensorDataset(adv_samples, y), save_path, 'fgsm', channel)
    print('Done!')


def genereate_cw_samples(target_model, source_data, save_path,c=0.8, iter=10000, batch_size=1,
                         data_type=DATA_MNIST, device='cpu'):
    # at present, only  cuda0 suopport

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    test_data = exclude_wrong_labeled(target_model, test_data, device=device)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    l2Attack = CarliniL2(target_model=target_model, max_iter=iter, c=c, k=0, device=device, targeted=False)
    print("Generating adversarial sampels...")
    for i, data_pair in enumerate(test_loader):
        i += 1
        data, real_label = data_pair
        data, real_label = data.to(device), real_label.to(device)
        scores = target_model(data)
        normal_preidct = torch.argmax(scores, dim=1, keepdim=True)
        adv_samples = l2Attack.do_craft(data, normal_preidct)
        success_samples, normal_labels, adv_label = l2Attack.check_adversarial_samples(l2Attack.target_model,
                                                                                       adv_samples, normal_preidct)
        if is_save:
            save_imgs_tensor(success_samples, normal_labels, adv_label, save_path, 'cw', no_batch=i,
                             batch_size=batch_size, channels=3)
        logging.info('batch:{}'.format(i))
        if i > 1500:
            break


def genereate_jsma_samples(model, source_data, save_path,max_distortion=0.12, dim_features=784,
                           num_out=10, data_type=DATA_MNIST, img_shape={'C': 3, 'H': 32, 'W': 32}, device='cpu'):

    # train_data, _ =  load_data_set(data_type=data_type,source_data=source_data,train=True)
    # complete_data = ConcatDataset([test_data,train_data])

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)

    jsma = JSMA(model, max_distortion, dim_features, num_out=num_out, theta=1, optimal=True, verbose=False,
                device=device, shape=img_shape)
    success = 0
    progress = 0

    all_lables = range(num_out)
    for data, label in test_data_laoder:
        data, label = data.to(device), label.to(device)
        target_label = jsma.uniform_smaple(label, all_lables)
        adv_sample, normal_predit, adv_label = jsma.do_craft(data, target_label)
        if adv_label == target_label:
            success += 1
            if is_save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_imgs_tensor([adv_sample.to('cpu')], [label], [adv_label], save_path, 'jsma', no_batch=success,
                                 channels=3)

        progress += 1
        sys.stdout.write(
            '\rprogress:{:.2f}%,success:{:.2f}%'.format(100. * progress / len(test_data_laoder),
                                                        100. * success / progress))
        sys.stdout.flush()

        if success > 5000:
            break

    print(success * 1. / progress)


def genereate_deepfool_samples(model, source_data, save_path=None, overshoot=0.02, num_out=10, max_iter=50,
                               data_type=DATA_MNIST, device="cpu"):
    '''
    Single data only!Do not Support batch
    :return:
    '''
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)
    deepfool = DeepFool(target_model=model, num_classes=num_out, overshoot=overshoot, max_iter=max_iter, device=device)
    count = 1
    for data, label in test_data_laoder:
        data = data.squeeze(0)
        data, label = data.to(device), label.to(device)
        adv_img, normal_label, adv_label = deepfool.do_craft(data)
        assert adv_label != normal_label
        assert label.item() == normal_label
        if is_save:
            save_imgs_tensor([adv_img.to('cpu')], [label], [adv_label], save_path, 'deepfool', no_batch=count,
                             channels=3)
        logging.info('{}th success!'.format(count))
        if count > 4000:
            break
        count += 1

def generate_ILA_samples(model, source_data,save_path,data_type,device,batch_size=1):
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)
    i = 0
    for data, label in test_data_laoder:
        data, label = data.to(device), label.to(device)
        scores = model(data)
        normal_preidct = torch.argmax(scores, dim=1, keepdim=True)
        if normal_preidct != label:
            continue
        adversarial_xs = ifgsm(model, data, label, niters= 10)

        
        for layer in model.modules():
            
            if(isinstance(layer,nn.Conv2d)):
                ILA_adversarial_xs = ILA(model, data, X_attack=adversarial_xs, y=label, feature_layer=layer)

                scores = model(ILA_adversarial_xs)
                adv_preidct = scores.data.max(1, keepdim=True)[1]
           
                if adv_preidct != label:
                    i += 1
                    if is_save:
                        save_imgs_tensor([ILA_adversarial_xs.to('cpu')], [label], [adv_preidct], save_path, 'ILA', no_batch=i,
                                        channels=3)
            
            if i > 1500:
                break


mid_output = None


def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1:
        abs_sum=torch.sum(torch.abs(grad),(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = torch.sum(grad.pow(2),(1,2,3),keepdims=True)
        nor_grad=grad/torch.sqrt(square)
    return nor_grad


def generate_FIA_samples(model, source_data,save_path,data_type,device,batch_size=1):
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True


    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    test_data = exclude_wrong_labeled(model, test_data, device=device)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
    
    
    count = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        
        X = data.detach()
        X_pert = torch.zeros(X.size()).to(device)
        X_pert.copy_(X).detach()
        X_pert.requires_grad = True
    
        labels= F.one_hot(label,10)

        for layer in model.modules():
            if(isinstance(layer,nn.Conv2d)):
                h = layer.register_forward_hook(get_mid_output)
                gt = []
                
                for i in range(10):
                    if i == 0:
                        out = model(data)
                        weight_np = torch.zeros(mid_output.size()).to(device)
                        for l in range(30):
                            mask = torch.from_numpy(np.random.binomial(1, 0.7, size=(1,3,32,32))).to(device)
                           
                            images_tmp2 = data * mask
                                             
                            out = model(images_tmp2)
                   
                            mid_output.retain_grad()
                            
                            logit = out*labels
                                                       
                            logit.backward(torch.ones_like(logit),retain_graph=True)
                            weight_tensor = mid_output.grad.data
                                            
                            weight_np += weight_tensor
                            
                        weight_np = -normalize(weight_np, 2)
                        mid_output.grad.data.zero_()
          
                    out_ori = model(X_pert)
                    mid_original = torch.zeros(mid_output.size()).to(device)
                    mid_original.copy_(mid_output)

                    
                    loss_fia = torch.sum(mid_original*weight_np) /  mid_original.numel()
                    fia_grad = torch.autograd.grad(loss_fia, X_pert, retain_graph=True)[0]
               
                    
                    noise = fia_grad / torch.mean(torch.abs(fia_grad),[1,2,3],keepdim=True)                  
                    gt.append(noise)
                    if i == 0:
                        noise = fia_grad
                    else:
                        noise = fia_grad + gt[i-1]
                    eps= 0.03
                    X_pert = torch.clamp(X_pert + torch.sign(noise),-eps,eps)
                scores = model(X_pert)
                adv_preidct = scores.data.max(1, keepdim=True)[1]
                h.remove()
                if adv_preidct != label:
                    count += 1
                    if is_save:
                        save_imgs_tensor([X_pert.to('cpu')], [label], [adv_preidct], save_path, 'FIA', no_batch=count,
                                        channels=3)
            
        if count > 1500:
            break

def genereate_local_search_samples(model, source_data, save_path,
                           data_type=DATA_MNIST, device="cpu"):
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)


    test_data = exclude_wrong_labeled(model, test_data, device)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    count = 0

    for batch_idx, (input, target) in enumerate(test_loader):
        adv = LocalSearchAttack(model,clip_max=1.0,clip_min=0.0,p=0.6,r=1,loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                d=3, t=3, k=1, round_ub=10, seed_ratio=0.1, max_nb_seeds=128, comply_with_foolbox=False,
                                targeted=False)
        adv_untargeted = adv.perturb(input.to(device), target.to(device))
        output = model(adv_untargeted)
        pred = output.data.max(1, keepdim=True)[1]
        if pred != target.to(device) and is_save:
            count+=1
            save_imgs_tensor([adv_untargeted.to('cpu')], [target], [pred], save_path, 'local_search', no_batch=count,
                             channels=3)
        if count > 20000:
            break

def genereate_one_pixel_samples(model, source_data, save_path,
                           data_type=DATA_MNIST, device="cpu"):

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_CIFAR10:
        test_data, channel = load_data_set(data_type, source_data, train=False)
    elif data_type == DATA_svhn:
        _,test_data = load_svhn(source_data,split=True,normalize=normalize_svhn)

    test_data = exclude_wrong_labeled(model, test_data, device)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    OnePixel = onepixel(model,pixels=3,maxiter=100,popsize=200,target=False,verbose=True,device=device)
    count = 0
    
    for batch_idx, (input, target) in enumerate(test_loader):


        targets = [None]

        for target_calss in targets:
            if (targets):
                if (target_calss == target[0]):
                    continue

            flag, x,y = OnePixel.attack(input.to(device), target[0])
            if flag == 0:
                continue
            count += flag
            if is_save:
                save_imgs_tensor([x.to('cpu')], [target], [y], save_path, 'onepixel', no_batch=count,
                                 channels=3)
        if count > 2000:
            break

def get_model(args):

    print("=> Creating model '{}'".format(args.arch))
    model = net.__dict__[args.arch](num_classes=args.num_classes,message_type=args.message_type)
    # applying sparsity to the network

    if args.freeze_weights:
        freeze_model_weights(model)
        print("=> freeze model weights {}'".format(args.arch))
    return model

def set_gpu(args, model):

    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    print(f"=> Parallelizing on {args.multigpu} gpus")
    torch.cuda.set_device(args.multigpu[0])
    args.gpu = args.multigpu[0]
    model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

def pretrained(args, model):

    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

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

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))
        import sys
        sys.exit()


def run():

    model = get_model(args)
    # print(model)
    model = set_gpu(args, model)
    if args.pretrained:
        pretrained(args, model
        )

    device= args.multigpu[0]

    if args.set == 'cifar10':
        data_type = DATA_CIFAR10
        sourceDataPath = '../dataset/cifar10'
        dim_features = 3 * 32 * 32
        num_out = 10
        img_shape = {'C': 3, 'H': 32, 'W': 32}
    elif args.set == 'svhn':
        data_type = DATA_svhn
        sourceDataPath = '../dataset/svhn'
        dim_features = 3 * 32 * 32
        num_out = 10
        img_shape = {'C': 3, 'H': 32, 'W': 32}
    else:
        data_type = DATA_flower
        dim_features = 3 * 224 * 224
        num_out = 102
        img_shape = {'C': 3, 'H': 224, 'W': 224}
    

    if args.attackType == 'fgsm':
        genereate_fgsm_samples(model, sourceDataPath, args.savePath, data_type=data_type,
                               device=device, eps=0.03, is_exclude_wr=True)

    elif args.attackType == 'jsma':
        genereate_jsma_samples(model, sourceDataPath, args.savePath, data_type=data_type,
                               max_distortion=0.12, dim_features=dim_features,
                               num_out=num_out,device=device,img_shape=img_shape)

    elif args.attackType == 'cw':
        genereate_cw_samples(model, sourceDataPath, args.savePath, data_type=data_type, device=device,
                           c=0.6, iter=1000, batch_size=1)

    elif args.attackType == 'deepfool':
        genereate_deepfool_samples(model, sourceDataPath, args.savePath, data_type=data_type, overshoot=0.02,
                                   num_out=num_out,max_iter=50,device=device)
    
    elif args.attackType == 'localsearch':
        genereate_local_search_samples(model, sourceDataPath, args.savePath,data_type=data_type, device=device)
    
    elif args.attackType == 'onepixle':
        genereate_one_pixel_samples(model, sourceDataPath, args.savePath, data_type=data_type, device=device)
    
    elif args.attackType == 'ILA':
        generate_ILA_samples(model,sourceDataPath,args.savePath,data_type=data_type,device=device)
    
    elif args.attackType == 'FIA':
        generate_FIA_samples(model, sourceDataPath, args.savePath, data_type=data_type, device=device)
        
    else:
        raise Exception("{} is not supported".format(args.attackType))

    ########
    # remove the saved deflected adversarial samples
    ########

    rename_advlabel_deflected_img(model, args.savePath, data_description='attack-{}'.format('{}'.format(args.attackType)), img_mode=None, device='cuda',
                                  data_type=data_type)

    print("Adversarial samples are saved in {}".format(args.savePath))

if __name__ == '__main__':
    run()




