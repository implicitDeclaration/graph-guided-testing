import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy

from args import args
import importlib

import data
import model as net

import numpy as np
import networkx as nx


def main():
    print(args)
   
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    for inter in range(args.iter_num):
        main_worker(args)


def main_worker(args):
    args.gpu = None
    train, validate, modifier = get_trainer(args)

    if args.gpu is not None:
       
        print("Use GPU: {} for training".format(args.gpu))
    
    # create model and optimizer
    model = get_model(args)
    # print(model)
    model = set_gpu(args, model)
 
    if args.pretrained:
        pretrained(args, model
                   )
 
    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)
 

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
 
    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        return
 
    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

 
    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None
 

    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )
 
    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        current_learning_rate = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)
     
  
        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )
   
        writer.add_scalar("test/lr", current_learning_rate, epoch)
        end_epoch = time.time()
    average_shortest_path_length, average_clustering, diameter, transitivity, density = \
        Calculate_Information_of_graph(args)
    write_result_to_csv(
        set=args.set,
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
        message_type=args.message_type,
        nodes_num=args.group_num,
        average_shortest_path_length=average_shortest_path_length,
        average_clustering=average_clustering,
        diameter=diameter,
        transitivity=transitivity,
        density=density,
        neighbors_num=args.neighbors
    )
    np.save(ckpt_base_dir / 'edge_index.npy', args.edge_index)
    print(f"==> edge_index has been save in {ckpt_base_dir}")




def Calculate_Information_of_graph(args):

    graph = nx.Graph()
    graph.add_edges_from(args.edge_index)


    average_shortest_path_length = nx.average_shortest_path_length(graph)

    average_clustering = nx.average_clustering(graph)

    diameter = nx.diameter(graph)

    transitivity = nx.transitivity(graph)

    density = nx.density(graph)

    return average_shortest_path_length, average_clustering, diameter, transitivity, density



def get_trainer(args):
    
    print(f"=> Using trainer from utils.{args.trainer}")
    trainer = importlib.import_module(f"utils.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier



def set_gpu(args, model):

    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model




def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")



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




def get_dataset(args):
    print(f"=> Getting {args.set} dataset")

    dataset = getattr(data, args.set)(args)

    return dataset



def get_model(args):
    
    print("=> Creating model '{}'".format(args.arch))
    model = net.__dict__[args.arch](num_classes=args.num_classes, message_type=args.message_type)
    # applying sparsity to the network

    if args.freeze_weights:
        freeze_model_weights(model)
        print("=> freeze model weights {}'".format(args.arch))
    return model




def get_optimizer(args, model):
    
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer



def _run_dir_exists(run_base_dir):
 
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()



def get_directories(args):
   

 
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")
  
    config = pathlib.Path(args.config).stem

 
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.set}/{args.name}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{args.set}/{config}/{args.name}"
        )
   
    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

 
    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir




def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"
    if not results.exists():
        results.write_text(
            "Date Finished, "
            'DataSet, '
            "Base Config, "
            "Name, "
            "Message Type, "
            "nodes_num, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "average_shortest_path_length, "
            "average_clustering, "
            "diameter, "
            "transitivity, "
            "density, "
            "neighbors_num\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                '{set}, '
                "{base_config}, "
                "{name}, "
                "{message_type}, "
                "{nodes_num}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{average_shortest_path_length}, "
                "{average_clustering}, "
                "{diameter}, "
                "{transitivity}, "
                "{density}, "
                "{neighbors_num}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()
