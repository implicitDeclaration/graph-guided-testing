
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


def McGee_Graph():
    
    nodes = 24
    edges_num = 36
    around = 7
    nodes_list = [node for node in range(nodes)]
    edges_list = []
    G = nx.Graph()
    for node in range(0,nodes//2,nodes//8):
        # range(0,12,3)
        edges_list.append([node,node+nodes//2])
    # print(edges_list)
    for node in range(0,nodes,nodes//8):
        edges_list.append([node+1,(node+1+around)%nodes])
        edges_list.append([node+2,(node+2-around)%nodes])    
    for edge in edges_list:
        G.add_edge(edge[0],edge[1])
    for i in range(nodes):
        G.add_edge(i,(i+1)%nodes)
    edges = np.array(G.edges())
    assert len(edges)==edges_num,'McGee_Graph Wrong'
    
    return edges



# edges = McGee_Graph()
# print(edges.shape)
#     nx.draw(G,with_labels=True)
#     plt.show()
# McGee_Graph()


