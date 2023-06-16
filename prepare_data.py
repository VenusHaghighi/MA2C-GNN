from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from collections import defaultdict
import random


class Dataset:
    def __init__(self, name='amazon'):
        self.name = name
        graph = None
        
        if name == 'yelp':
            
                dataset = FraudYelpDataset()
                graph = dataset[0]
            
                graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
                graph.ndata['feature'] = graph.ndata['feature'].float()
                print(graph)

                feature = graph.ndata['feature']
                labels = graph.ndata['label']
                train_mask = graph.ndata['train_mask']
                test_mask = graph.ndata['test_mask']
                valid_mask = graph.ndata['val_mask']
                
                src_rsr, dst_rsr = graph.adj_sparse('coo', etype = 'net_rsr')
                src_rtr, dst_rtr = graph.adj_sparse('coo', etype = 'net_rtr')
                src_rur, dst_rur = graph.adj_sparse('coo', etype = 'net_rur')
                src_homo = torch.cat([src_rsr, src_rtr, src_rur])
                dst_homo = torch.cat([dst_rsr, dst_rtr, dst_rur])
                
                yelp_graph = dgl.heterograph({('r', 'homo', 'r'): (src_homo, dst_homo), ('r', 's', 'r'): (src_rsr, dst_rsr), ('r', 't', 'r'): (src_rtr, dst_rtr), ('r', 'u', 'r'): (src_rur, dst_rur)})
                
                yelp_graph.ndata['feature'] = feature
                yelp_graph.ndata['label'] = labels
                yelp_graph.ndata['train_mask'] = train_mask
                yelp_graph.ndata['test_mask'] = test_mask
                yelp_graph.ndata['valid_mask'] = valid_mask
                
                
                index = list(range(len(labels)))

                idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                        train_size=0.4,
                                                                        random_state=2, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                        test_size=0.67,
                                                                        random_state=2, shuffle=True)
                train_mask = torch.zeros([len(labels)]).bool()
                valid_mask = torch.zeros([len(labels)]).bool()
                test_mask = torch.zeros([len(labels)]).bool()

                train_mask[idx_train] = 1
                valid_mask[idx_valid] = 1
                test_mask[idx_test] = 1
            

                graph = generate_edge_labels(yelp_graph, idx_train, idx_test, idx_valid) 
                
                dgl.save_graphs('yelp.dgl', graph)
                
        elif name == 'amazon':
            
                dataset = FraudAmazonDataset()
                graph = dataset[0]
            
                
                graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
                graph.ndata['feature'] = graph.ndata['feature'].float()
                print(graph)
                
                feature = graph.ndata['feature']
                labels = graph.ndata['label']
                train_mask = graph.ndata['train_mask']
                test_mask = graph.ndata['test_mask']
                valid_mask = graph.ndata['val_mask']
                
                src_upu, dst_upu = graph.adj_sparse('coo', etype = 'net_upu')
                src_usu, dst_usu = graph.adj_sparse('coo', etype = 'net_usu')
                src_uvu, dst_uvu = graph.adj_sparse('coo', etype = 'net_uvu')
                src_homo = torch.cat([src_upu, src_usu, src_uvu])
                dst_homo = torch.cat([dst_upu, dst_usu, dst_uvu])
                
                amazon_graph = dgl.heterograph({('r', 'homo', 'r'): (src_homo, dst_homo), ('r', 'p', 'r'): (src_upu, dst_upu), ('r', 's', 'r'): (src_usu, dst_usu), ('r', 'v', 'r'): (src_uvu, dst_uvu)})
                
                amazon_graph.ndata['feature'] = feature
                amazon_graph.ndata['label'] = labels
                amazon_graph.ndata['train_mask'] = train_mask
                amazon_graph.ndata['test_mask'] = test_mask
                amazon_graph.ndata['valid_mask'] = valid_mask
                
                
                
                index = list(range(3305, len(labels)))
                
                idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=0.4, random_state=2, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=0.67, random_state=2, shuffle=True)
                train_mask = torch.zeros([len(labels)]).bool()
                valid_mask = torch.zeros([len(labels)]).bool()
                test_mask = torch.zeros([len(labels)]).bool()

                train_mask[idx_train] = 1
                valid_mask[idx_valid] = 1
                test_mask[idx_test] = 1
                

                graph = generate_edge_labels(amazon_graph, idx_train, idx_test, idx_valid) 
             
                dgl.save_graphs('amazon.dgl', graph)
        
        else:
            print('no such dataset')
            exit(1)

        self.graph = graph

def generate_edge_labels(graph, idx_train, idx_test, idx_valid):
    
    labels = graph.ndata['label']
    src, dst = graph.edges(etype = 'homo')
    edge_labels = []
    edge_train_mask = []
    edge_valid_mask = []
    edge_test_mask = []
    
    
    for i, j in zip(src, dst):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(-1)
        
        if i in idx_train and j in idx_train:
            edge_train_mask.append(1)
        else:
             edge_train_mask.append(0)
             
        if i in idx_valid and j in idx_valid:
            edge_valid_mask.append(1)
        else:
             edge_valid_mask.append(0)
             
        if i in idx_test and j in idx_test:
            edge_test_mask.append(1)
        else:
             edge_test_mask.append(0)

    
    edge_labels = torch.Tensor(edge_labels).long()
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    edge_valid_mask = torch.Tensor(edge_valid_mask).bool()
    edge_test_mask = torch.Tensor(edge_test_mask).bool()
    
    graph.edges['homo'].data['edge_labels'] = edge_labels
    graph.edges['homo'].data['edge_train_mask'] = edge_train_mask
    graph.edges['homo'].data['edge_test_mask'] = edge_test_mask
    graph.edges['homo'].data['edge_valid_mask'] = edge_valid_mask
    
    return graph



def sparse_to_adjlist(sp_matrix):

	"""Transfer sparse matrix to adjacency list"""

	#add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	#creat adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	adj_lists = {keya:random.sample(adj_lists[keya],10) if len(adj_lists[keya])>=10 else adj_lists[keya] for i, keya in enumerate(adj_lists)}

	return adj_lists


def connection_domination(graph, adj_list):
    
    """ determine which types of connections are dominated in a given neighborhood """
    src, dst = graph.edges(etype = 'homo')
    edgeType = graph.edges['homo'].data['edgeType']
    k = graph.number_of_nodes()
    h2_rate = []
    for i in range(0, k):
        temp  = torch.cat([src.view(-1,1), dst.view(-1,1), edgeType.view(-1,1)], dim = -1)
        temp = temp.type(torch.int64)
        #print("i:", i)
        mask = temp[:, 0] == i
        #print("mask:", mask)
        temp = temp[mask]
        #print("temp:", temp)
        result = temp[:, -1]
        #print("result:", result)
        homo_rate = np.count_nonzero(result == 1)
        hetero_rate = np.count_nonzero(result == -1)
        if (homo_rate > hetero_rate):
            rate = 1
        elif(hetero_rate > homo_rate):
            rate = -1
        h2_rate.append(rate)
        
    #print("h2_rate:", h2_rate, len(h2_rate))
    h2_rate = torch.tensor(h2_rate)
    #print("h2_rate:", h2_rate, h2_rate.size())
    graph.ndata['h2_rate'] = h2_rate
    return h2_rate, graph

