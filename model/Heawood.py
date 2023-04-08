
# generate McGee Graph
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
import matplotlib 
from matplotlib import pyplot
import matplotlib.pyplot as plt


def Heawood_Graph():
    
    G = nx.heawood_graph()
    edges = np.array(G.edges())
    # nx.draw(G,with_labels=True)
    # plt.show()
    return edges



# edges = Heawood_Graph()
# print(edges)
#     nx.draw(G,with_labels=True)
#     plt.show()
# McGee_Graph()


