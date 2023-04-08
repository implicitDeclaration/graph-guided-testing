import argparse
import sys
import yaml
from config import parser as _parser
args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch GraphMapNetwork detecting adversarial samples",epilog="End of Parameters")

    # Parameters for training
    parser.add_argument(
        "--search_direction", help="shortest average path search direction", default='min', type=str
    )       
    parser.add_argument(
        "--need_min", help="whether need min", action="store_false",
    )      
    parser.add_argument(
        "--whether_search", help="whether use the algorithm to decrease graph's shortest average path", action="store_true",
    )   

    parser.add_argument(
        "--iter_num", help="iter_num used in optimal graph training times during the running ", default=1, type=int
    )   
    parser.add_argument(
        "--nodes", help="used in the optimal regular graph ", default=None,type=int
    )   

    parser.add_argument(
        "--neighbors", help="used in the optimal regular graph ", default=None,type=int
    )
    parser.add_argument(
        "--group_num", help="the num of nodes ", default=None
    )
    parser.add_argument(
        "--edge_index", help="the edge relationship of the graph ", default=None,type=list
    )   
    parser.add_argument(
        "--message_type", help="the type of graph ", default=None
    )
  
    parser.add_argument(
        "--data", help="path to dataset base directory", default="./dataset"
    )
  
    parser.add_argument("--optimizer", help="Which optimizer to use", default="adam")

    parser.add_argument("--set", help="name of dataset", type=str, default="cifar10")

    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="GraphMapResNet18", help="model architecture"
    )

    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )

    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )    
    
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
   
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
 
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
  
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
 
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
   
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--no_bn_decay", default=True, help="Number of warmup iterations"
    )

  
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
  
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
  
    parser.add_argument("--num-classes", default=10, type=int)

 
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
  
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--model_num", default=10, type=int, help="number of used models for calculating lcr. "
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
  
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
 
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
 
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
  
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )
   
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
  
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
  
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
  
    parser.add_argument(
        "--trainer", type=str, default="default", help="standard training"
    )

    parser.add_argument(
        "--freeze_weights",action="store_true", help="freeze_weights"
    )
   
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--prune-rate",
        default=0.5,
        help="Amount of pruning to do during sparse training",
        type=float,
    )

    # Parameters for generating adversarial samples
    parser.add_argument(
        "--savePath", help="The path where the adversarial samples to be stored", type=str
    )

    parser.add_argument("--attackType", type=str,
                        help="four attacks are available: fgsm, jsma, deepfool, cw, localsearch, ILA, FIA"
    )

    # Parameters for calculating lcr
    parser.add_argument("--testType", type=str,
                        help="Tree types are available: [adv], advesarial data; [normal], test on normal data; [wl],test on wrong labeled data",
    )


    parser.add_argument("--prunedModelsPath", type=str,
                        help="The path of pruned models",
    )

    parser.add_argument("--testSamplesPath", type=str,
                        help="The path of adversarial samples",
    )

    parser.add_argument("--logPath", type=str, help="The files path of batch testing results")

    parser.add_argument("--maxModelsUsed", type=int,
                        help="Total mutated models are used to yield the label change rate(lcr)")

    parser.add_argument("--isAdv", type=str, help="True if the samples are adversarial, otherwise,false")

    parser.add_argument("--nrLcrPath", type=str,
                        help="The lcr list of normal samples. This is just for the auc computing")

    parser.add_argument("--lcrSavePath", type=str, help="The path to save the lcr list")
    
    # Parameter for detecting
    parser.add_argument("--threshold", type=float,
                        help="The lcr_auc of normal samples. The value is equal to: avg+99%confidence.")

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
