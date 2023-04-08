#%%
import networkx as nx
import matplotlib.pyplot as plt
import random
from random import choice
import numpy as np
import tqdm
from args import args
#%%

def compute_average_degree(G):
    degree = np.array(nx.degree(G))
    ave_degree = 2 * np.sum(degree[:,-1])/(degree.shape[0] * (degree.shape[0]-1))
    return ave_degree
def Gaussion_Distribution(neighbors,nodes):
    while True:
        try:
            print('==>begin to generate Gaussion_Distribution Graph')
            sequence = np.random.normal(loc=neighbors,size=nodes)
            sequence = np.round(neighbors / np.mean(sequence) * sequence)
    # 确保度值平均度值是3
            while np.mean(sequence) !=neighbors:
                index = random.randint(0,nodes-1)
                if np.mean(sequence)<neighbors:           
                    sequence[index] = sequence[index] + 1
                else:
                    sequence[index] = sequence[index] - 1
            RG = nx.random_degree_sequence_graph(sequence,tries=1000000)            
            if  nx.is_connected(RG):
                edges = np.array(RG.edges())
                infor = Calculate_Information_of_graph(edges)
                print('==>success to generate Gaussion_Distribution Graph')
                # print(np.mean(np.array(nx.degree(RG))[:,1]))
                return RG
        except:
            print('==>loss to generate Gaussion_Distribution Graph')
        # 死循环 不能生成这个图        

   
# RG = Gaussion_Distribution(3,3)
#%%
# judge the choice condition
def Initialize_Judge(source1,source2,target1,target2):
    # 判断一下四个节点是不相同的
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
def Generate_Optimal_Guassian_Network(neighbors, nodes):
  
    if not args.whether_search:
        RG = Gaussion_Distribution(neighbors, nodes)
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
        RG = Gaussion_Distribution(neighbors, nodes)
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
                if np.abs(origin_path - path_after)>0.1 and path_after >= path_before:
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
# graph = Gaussion_Distribution(3,128)

#%%
# regular graphy
# generate a regular graph which has 20 nodes & each node has 3 neghbour nodes.
# neighbors = 3
# nodes = 98
# path_begin,path_end,RG = Generate_Optimal_Symmetric_Network(neighbors, nodes)
# print('path_begin: {}'.format(path_begin))
# print('path_end: {}'.format(path_end))
# the spectral layout
# pos = nx.spectral_layout(RG)
# draw the regular graphy
# nx.draw(RG, pos, with_labels = True, node_size = 30)
# nx.draw(RG, with_labels = True, node_size = 30)
# plt.show()