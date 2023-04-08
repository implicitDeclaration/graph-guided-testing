
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


def Petersen_Graph():
    
    nodes = 10

    edges_list = []
    for node in range(nodes//2):
        edges_list.append([node,(node+1)%(nodes//2)])
    edges_list.append([5,7])
    edges_list.append([6,8])
    edges_list.append([7,9])
    edges_list.append([8,5])
    edges_list.append([9,6])
    for node in range(nodes//2):
        edges_list.append([node,node+5])
    G = nx.Graph()
    for edge in edges_list:
        G.add_edge(edge[0],edge[1])       
    edges = np.array(G.edges())
    # nx.draw(G,with_labels=True)
    # plt.show()
    return edges


# edges = Petersen_Graph()

# print(edges.shape)
# print(edges)
#     nx.draw(G,with_labels=True)
#     plt.show()
# McGee_Graph()


