import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from prepare_data import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from model import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from prepare_data import *


def edgeLabelling_train(data, features):
    
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[edge_train_mask], edge_labels_float[edge_train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)
    
    print('\nLabelling edges...')
    input_dim = np.shape(features)[1]
    output_dim = 2
    dropout = 0.1
    lr = 0.1
    weight_decay = 5e-05
    epochs = 100
    model = Edge_Labelling(input_dim, output_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
       
    # Edge labelling   
    edges = data.edges['homo']
    edge_labels = edges.data['edge_labels']
    edge_train_mask = edges.data['edge_train_mask'].bool()
    edge_valid_mask = edges.data['edge_valid_mask'].bool()
    edge_test_mask = edges.data['edge_test_mask'].bool()
    edge_labels_float = edge_labels.type(torch.FloatTensor)
    
            
    # print(f"\nNumber of training edges: {edges.data['train_mask'].sum()}")
    # print(f"Number of validation edges: {edges.data['valid_mask'].sum()}")
    # print(f"Number of test edges: {edges.data['test_mask'].sum()}")
    
    
    # Traing and Testing
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_acc = final_test_acc = 0
    for epoch in tqdm(range(1, 100)):
        loss = train()
        model.eval()
        pred = model(data).detach()
        accs = []
        
        for mask in [edge_train_mask, edge_valid_mask, edge_test_mask]:        
            result = accuracy_score(edge_labels[mask], torch.sign(pred[mask]))
            accs.append(result)
            
        train_acc, val_acc, tmp_test_acc = accs
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(tmp_test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            
        else:
            test_acc = tmp_test_acc
    
    signed = torch.sign(pred)
    
    # Evaluation
    model.eval()
    correct = (edge_labels[edge_test_mask] == signed[edge_test_mask]).sum()
    acc = int(correct) / int(edge_test_mask.sum())
    print(f'\nLabelling edges is done!')
    print(f'\nAccuracy: {acc:.4f}')
    
    print("Number of predicted homo connection: ", np.count_nonzero(signed ==1))
    print("Number of predicted hetro connection: ",  np.count_nonzero(signed ==-1))

    edge_labels_ = edge_labels.numpy()
    print("Number of actual homo connection: ", np.count_nonzero(edge_labels_ ==1))
    print("Number of actual hetro connection: ",  np.count_nonzero(edge_labels_ ==-1))
    
    return signed, edge_test_mask, edge_labels

### training and testing model
def train_model(model, graph, args):
    

        features = graph.ndata['feature']
        labels = graph.ndata['label']
        index = list(range(len(labels)))
        if dataset_name == 'amazon':
            index = list(range(3305, len(labels)))

        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=args.train_ratio, random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=0.67, random_state=2, shuffle=True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1
        print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

        weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print('cross entropy weight: ', weight)
        time_start = time.time()
        order = args.order
        print("order:", order)
        
       
        
        for e in range(1000):
            
            model.train()
            logits = model(graph, features, order)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            probs = logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = numpy.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1
            trec = recall_score(labels[test_mask], preds[test_mask])
            tpre = precision_score(labels[test_mask], preds[test_mask])
            tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
            tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
            print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

        time_end = time.time()
        print('time cost: ', time_end - time_start, 's')
        print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                         final_tpre*100, final_tmf1*100, final_tauc*100))
        return final_tmf1, final_tauc
    
    
    

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="yelp",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=3, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    dataset_path = 'C:/Users/46665056/Desktop/Fraud-aware-GNN-V5/'+ dataset_name + '.dgl'
    homo = args.homo
    order = args.order
    h_channels = args.hid_dim
    #graph = Dataset(dataset_name).graph
    graph = dgl.load_graphs(dataset_path)[0][0]
    in_channels = graph.ndata['feature'].shape[1]
    num_classes = 2
    feature = graph.ndata['feature']
    
    
    
    #Train the edge labelling module to get the edgeType 
    # edgeType, edge_test_mask, edge_labels = edgeLabelling_train(graph, feature) 
    # graph.edges['homo'].data['edgeType'] = edgeType
    
    # ### get sparse adjacency matrix of dgl graph
    # sparse_adj = graph.adj(scipy_fmt='coo', etype='homo')
    
    # ### get the adjacency list for all nodes in graph
    # adj_list = sparse_to_adjlist(sparse_adj)
    
    # ### get the connection domination for each node in graph
    # h2_rate, graph = connection_domination(graph, adj_list)
    
    
    
        
        

    if args.run == 1:
        
        model = BWGNN(in_channels, h_channels, num_classes, graph, order)
        
        train_model(model, graph, args)
        
        
        

    else:
        
        
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
