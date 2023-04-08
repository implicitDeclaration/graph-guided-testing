#%%
import networkx as nx
import matplotlib.pyplot as plt
import random
from random import choice
import numpy as np
import tqdm
# from args import args
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
            sequence = sequence + 1
            RG = nx.random_degree_sequence_graph(sequence,tries=100,seed=1) 
            print(nx.is_connected(RG))           
            if nx.is_connected(RG):
                edges = np.array(RG.edges())
                infor = Calculate_Information_of_graph(edges)
                print('==>success to generate Gaussion_Distribution Graph')
                # print(np.mean(np.array(nx.degree(RG))[:,1]))
                return RG
        except:
            print('==>loss to generate Gaussion_Distribution Graph')
        # 死循环 不能生成这个图
# RG = Gaussion_Distribution(3,64)

#%%



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
def Generate_Optimal_Guassian_Network(neighbors, nodes,whether_search=True,edge_index=None,need_min=True):

    if not whether_search:
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

    if edge_index is None:
        RG = Gaussion_Distribution(neighbors, nodes)
    else:

        RG = nx.Graph()
        RG.add_edges_from(edge_index)
   
    path_begin =  nx.average_shortest_path_length(RG)
    # degree_before = compute_average_degree(RG)

    # pbar = range(nodes*(nodes-1)*neighbors*neighbors//2)
    pbar = range(nodes*(nodes-1))
    for index in tqdm.tqdm(enumerate(pbar), ascii=True, total=len(pbar)):
 
        path_before = nx.average_shortest_path_length(RG)
        try:
            source1 = choice(np.array(RG.nodes()))
            source2 = choice(np.array(RG.nodes()))
            # print('edges:',nx.edges(RG))
            # print('degree:',nx.degree(RG))       
            # print('source1:',source1)
            # print('11111',np.array(RG[source1]))
            # print('target1:',target1)
            # print('source2:',source2)        
            # print('22222',np.array(RG[source2]))
            # print('target2:',target2)
            target1 = choice(np.array(RG[source1]))
            target2 = choice(np.array(RG[source2]))

        finally:
            print(source1,source2)
            print(nx.degree(RG))
            print(nx.edges(RG))
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

            if path_after > path_before:
                RG = Temp_RG
            else:
                edges = np.array(RG.edges())
                if not need_min:
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

G = Generate_Optimal_Guassian_Network(3,64)
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