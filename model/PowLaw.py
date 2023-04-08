#%%
import networkx as nx
import matplotlib.pyplot as plt
import random
from random import choice
import numpy as np
import tqdm
from args import args
import scipy
from scipy import stats


# #%%
# sequence = np.ceil(scipy.stats.powerlaw.rvs(0.5,loc=4,scale=10.5,size=128))
# np.mean(sequence)
#%%
def GenPowlawDisbute(size,mean,x=0.5,loc=1,scale=8):

    sequence = 0
    if mean==3:
        sequence = np.ceil(scipy.stats.powerlaw.rvs(x,loc=loc,scale=scale,size=size))
    elif mean==8:
        sequence = np.ceil(scipy.stats.powerlaw.rvs(x,loc=4,scale=10.5,size=size))
    while np.mean(sequence)!=mean:
        # print(np.mean(sequence))
        index = random.randint(0,size-1)
        if np.mean(sequence)>mean:
            if sequence[index]==1:
                pass
            else:
                sequence[index] = sequence[index] - 1
        else:
            sequence[index] = sequence[index] + 1
    
    return sequence
#%%
def compute_average_degree(G):
    degree = nx.degree_histogram(G)
    def ave(degree):
        s_um = 0
        for i in range(len(degree)):
            s_um =s_um+i*degree[i]
        return s_um/nx.number_of_nodes(G)
    path_len = ave(degree)
    return path_len
#%%
def GenPowLawDisGraph(neighbors,nodes):
    while True:
        try:
            print('==>begin to generate PowLawDistribution Graph')

            sequence = GenPowlawDisbute(size=nodes,mean=neighbors)
            # print(np.mean(sequence),sequence.shape)
   
            RG = nx.random_degree_sequence_graph(sequence,tries=1000000)
            # print(nx.is_connected(RG))
            if  nx.is_connected(RG):
                edges = np.array(RG.edges())
                infor = Calculate_Information_of_graph(edges)
                print('==>success to generate PowLawDistribution Graph')
                return RG

        except:
            print('==>loss to generate PowLawDistribution Graph')


# def GenPowLawDisGraph(neighbors,nodes):
#
#     print('==>begin to generate PowLawDistribution Graph')
#
#     sequence = GenPowlawDisbute(size=nodes,mean=neighbors)
#     # print(np.mean(sequence),sequence.shape)
#     # 确保度值平均度值是mean
#     RG = nx.random_degree_sequence_graph(sequence,tries=1000000)
#     # print(nx.is_connected(RG))
#     if  nx.is_connected(RG):
#         edges = np.array(RG.edges())
#         infor = Calculate_Information_of_graph(edges)
#         print('==>success to generate PowLawDistribution Graph')
#         return RG


#%%
   
# RG = Gaussion_Distribution(3,3)
#%%
# judge the choice condition
def Initialize_Judge(source1,source2,target1,target2):

    if source1 == source2:
        return False
    if target1 == target2:
        return False
    if source1 == target2:
        return False
    if source2 == target1:
        return False
    return True
# generate a regular graphy
def Generate_Optimal_ScaleFree_Network(neighbors, nodes):

    if not args.whether_search:
        RG = GenPowLawDisGraph(neighbors, nodes)
        edges = np.array(RG.edges())
        average_shortest_path_length,average_clustering,diameter,transitivity,density = \
                            Calculate_Information_of_graph(edges)               
        print("average_shortest_path_length :{:<10}".format(average_shortest_path_length))
        print("average_clustering           :{:<10}".format(average_clustering))
        print("diameter                     :{:<10}".format(diameter))
        print("transitivity                 :{:<10}".format(transitivity))
        print("density                      :{:<10}".format(density))
        return edges
    if args.edge_index is None:
        RG = GenPowLawDisGraph(neighbors, nodes)
    else:
 
        RG = nx.Graph()
        RG.add_edges_from(args.edge_index)
   
    path_begin =  nx.average_shortest_path_length(RG)
    # degree_before = compute_average_degree(RG)

    # pbar = range(nodes*(nodes-1)*neighbors*neighbors//2)
    pbar = range(nodes*(nodes-1)*neighbors)
    num = 0
    origin_path =  nx.average_shortest_path_length(RG)
    for index in tqdm.tqdm(enumerate(pbar), ascii=True, total=len(pbar)):
    
        num = num +1 
        path_before = nx.average_shortest_path_length(RG)
        
        if num%100==0:
            print(f'===>path_length:{path_before}')
        source1 = choice(np.array(RG.nodes()))
        source2 = choice(np.array(RG.nodes()))
        target1 = choice(np.array(RG[source1]))
        target2 = choice(np.array(RG[source2]))
        state_judge = Initialize_Judge(source1,source2,target1,target2)
        # print(f'source1: {source1} source2: {source2}')
        # print(f'target1: {target1} target2: {target2}')
        # print(f'state: {state}')
        if not state_judge:
            continue
        if RG.has_edge(source1,target2) or RG.has_edge(source2,target1):
            continue

        Temp_RG = RG.copy()

        RG.remove_edges_from([[source1,target1],[source2,target2]])
        RG.add_edges_from([[source1,target2],[source2,target1]])

        if nx.is_connected(RG):
            path_after = nx.average_shortest_path_length(RG)
            # print(f'path_after: {nx.average_shortest_path_length(Temp_RG)}')

            if args.search_direction == 'min':
                if path_after > path_before:
                    RG = Temp_RG
                # elif np.abs(path_before - path_after)>0.05:
                if np.abs(path_before - path_after)>0.05 and path_after < path_before:
                    edges = np.array(RG.edges())
                    if not args.need_min:
                        average_shortest_path_length,average_clustering,diameter,transitivity,density = \
                                            Calculate_Information_of_graph(edges)               
                        print("average_shortest_path_length :{:<10}".format(average_shortest_path_length))
                        print("average_clustering           :{:<10}".format(average_clustering))
                        print("diameter                     :{:<10}".format(diameter))
                        print("transitivity                 :{:<10}".format(transitivity))
                        print("density                      :{:<10}".format(density))
                        return edges
            if args.search_direction == 'max':
                if path_after < path_before:
                    RG = Temp_RG
                if np.abs(origin_path - path_after)>0.05 and path_after >= path_before:
                    edges = np.array(RG.edges())
                    # if not args.need_min:
                    average_shortest_path_length,average_clustering,diameter,transitivity,density = \
                                        Calculate_Information_of_graph(edges)
                    print("average_shortest_path_length :{:<10}".format(average_shortest_path_length))
                    print("average_clustering           :{:<10}".format(average_clustering))
                    print("diameter                     :{:<10}".format(diameter))
                    print("transitivity                 :{:<10}".format(transitivity))
                    print("density                      :{:<10}".format(density))
                    return edges
        else:
      
            RG.add_edges_from([[source1,target1],[source2,target2]])
            RG.remove_edges_from([[source1,target2],[source2,target1]])
             
        # print(50*'==')
        # pos = nx.spectral_layout(RG)
        # # draw the regular graphy
        # nx.draw(RG, pos, with_labels = True, node_size = 30)
        # plt.show()
        # plt.close()
        edges = np.array(RG.edges())
    return edges

def Calculate_Information_of_graph(edge_index):
    
   
    graph = nx.Graph()
    graph.add_edges_from(edge_index)


    average_shortest_path_length = nx.average_shortest_path_length(graph)
 
    average_clustering = nx.average_clustering(graph)

    diameter = nx.diameter(graph) 

    transitivity = nx.transitivity(graph)

    density = nx.density(graph)

    return average_shortest_path_length,average_clustering,diameter,transitivity,density
#%%