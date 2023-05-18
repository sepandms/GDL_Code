######################################################
####### Please define Parameters ########################
dataset_path = '../dataset/'
save_plots = True
save_embedding = True
# device = 'cpu' # {'cuda','cpu'}
num_workers = None #{None, 1, 2, 3, 4,...}

cora_params      = {'embedding_dim':128 , 'batch_size':128, 'walk_length':12 , 'context_size':12 ,'walks_per_node':10, 'lr':0.01,'num_negative_samples':1,'p':1,'q':1 ,'epochs':40,'log_steps':100}
# cora_params      = {'embedding_dim':128 , 'batch_size':512, 'walk_length':20 , 'context_size':15 ,'walks_per_node':15, 'lr':0.001,'num_negative_samples':3,'p':0.9,'q':0.9 ,'epochs':40,'log_steps':100}

citeseer_params  = {'embedding_dim':128 , 'batch_size':128, 'walk_length':12 , 'context_size':12 ,'walks_per_node':10, 'lr':0.01,'num_negative_samples':1,'p':1,'q':1 ,'epochs':40,'log_steps':100}
pubmed_params    = {'embedding_dim':128 , 'batch_size':128, 'walk_length':15 , 'context_size':15 ,'walks_per_node':10, 'lr':0.01,'num_negative_samples':1,'p':1,'q':1 ,'epochs':40,'log_steps':100}
arxiv_params     = {'embedding_dim':128 , 'batch_size':512, 'walk_length':20 , 'context_size':15 ,'walks_per_node':10, 'lr':0.001,'num_negative_samples':2,'p':1,'q':1 ,'epochs':40,'log_steps':100}
reddit_params    = {'embedding_dim':128 , 'batch_size':512, 'walk_length':15 , 'context_size':15 ,'walks_per_node':15, 'lr':0.02,'num_negative_samples':3,'p':1,'q':1 ,'epochs':40,'log_steps':100}
products_params  = {'embedding_dim':128 , 'batch_size':256, 'walk_length':15 , 'context_size':15 ,'walks_per_node':15, 'lr':0.01,'num_negative_samples':1,'p':1,'q':1 ,'epochs':40,'log_steps':100}


####### Setting up Parameters ########################
import sys
import os
dataset_name = sys.argv[1]

model_params = {}
if str.lower(dataset_name) == 'cora':
    model_params = cora_params
elif str.lower(dataset_name) == 'citeseer':
    model_params = citeseer_params
elif str.lower(dataset_name) == 'pubmed':
    model_params = pubmed_params
elif str.lower(dataset_name) == 'ogbn-arxiv':
    model_params = arxiv_params
elif str.lower(dataset_name) == 'reddit':
    model_params = reddit_params
elif str.lower(dataset_name) == 'ogbn-products':
    model_params = products_params
else:
    print('Please define the right dataset included in the list')
    exit()

embedding_dim = model_params['embedding_dim']
batch_size = model_params['batch_size']
walk_length = model_params['walk_length']
context_size = model_params['context_size']
walks_per_node = model_params['walks_per_node']
lr = model_params['lr']
epochs = model_params['epochs']
log_steps = model_params['log_steps']
num_negative_samples = model_params['num_negative_samples']
p = model_params['p']
q = model_params['q']

embedding_path = '../Embeddings_Saved/'
embedding_store = f'N2V_embedding_{dataset_name}.pt'

if os.path.exists(dataset_path):
    None
else:
    os.mkdir(dataset_path)

if os.path.exists('./plots/'):
    None
else:
    os.mkdir('./plots/')

if os.path.exists(embedding_path):
    None
else:
    os.mkdir(embedding_path)

####### End of Setting Parameters ####################
######################################################

print(f'###### Node2Vec: {dataset_name} #########')

import os.path as osp
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from clustering_metric import clustering_metrics
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from ogb.nodeproppred import PygNodePropPredDataset
from load_data_graph_saint import load_data

try: 
    import warnings
    warnings.filterwarnings('ignore')
except: None


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if str.lower(dataset_name) in ['cora','citeseer','pubmed']:

    dataset = Planetoid(root=dataset_path, name=dataset_name)
    data = dataset[0].to(device)
    y = data.y.view(-1)
    edge_index = data.edge_index
    val_idx = data.val_mask.nonzero().view(-1).tolist()
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes
    train_idx = data.train_mask
    test_idx = data.test_mask
    del dataset, data

elif str.lower(dataset_name) in ['ogbn-arxiv','ogbn-products']:

    dataset = PygNodePropPredDataset(name = dataset_name, root=dataset_path)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0].to(device)
    y = data.y.view(-1)
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    del dataset, data

elif str.lower(dataset_name) in ['reddit']:
    A,_,X,label,split = load_data(dataset_path + dataset_name)
    val_idx = split['va'] 
    train_idx = split['tr']
    test_idx = split['te']
    y = label.view(-1).cpu()
    A = A.to(device)
    edge_index = A.coalesce().indices()
    # edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
    del X, A, split

model = Node2Vec(
    edge_index,
    embedding_dim=embedding_dim,
    walk_length=walk_length,
    context_size=context_size,
    walks_per_node=walks_per_node,
    num_negative_samples=num_negative_samples,
    p=p,
    q=q,
    sparse=True,).to(device)

loader = model.loader(batch_size=batch_size, shuffle=True)

num_workers = 0 if sys.platform.startswith('win') else 4
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

def save_embedding(embedding, save_embedding):
    torch.save(embedding.cpu(), save_embedding)


def main():
    start_train=time.time()
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[train_idx], y[train_idx],
                            z[test_idx], y[test_idx],
                            max_iter=150)
        return acc

    details_val = pd.DataFrame(columns=['Epoch','Accuracy','NMI','CS','F1','ARI','loss']) 
    details_train = pd.DataFrame(columns=['Epoch','Accuracy','NMI','CS','F1','ARI','loss']) 
    max_nmi = -1

    for epoch in range(1, epochs):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

        model.eval()
        embedding = model()
        kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
        y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
        cm = clustering_metrics(y[val_idx].cpu().numpy(), y_pred)
        acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()
        print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

        new_row = {'Epoch':epoch,'Accuracy':acc,'NMI':nmi,'CS':cs,'F1':f1_macro,'ARI':adjscore}
        # details_val = details_val.append(new_row,ignore_index=True)
        details_val = pd.concat([details_val, pd.DataFrame([new_row])], ignore_index=True)


        if nmi > max_nmi:
            max_nmi = nmi
            if save_embedding:
                save_embedding(embedding, embedding_path + embedding_store)
        del embedding, y_pred

        embedding = model()
        kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
        y_pred = kmeans.fit_predict(embedding.detach().cpu().numpy())
        cm = clustering_metrics(y.cpu().numpy(), y_pred)
        acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()

        new_row = {'Epoch':epoch,'Accuracy':acc,'NMI':nmi,'CS':cs,'F1':f1_macro,'ARI':adjscore}
        # details_train = details_train.append(new_row,ignore_index=True)
        details_train = pd.concat([details_train, pd.DataFrame([new_row])], ignore_index=True)



    print("Time taken for training: " + str(time.time()-start_train))
    print("######################### END of Training ####################")
    embedding = torch.load(embedding_path + embedding_store)
    y_pred = kmeans.fit_predict(embedding.detach().cpu().numpy())
    cm = clustering_metrics(y.cpu().numpy(), y_pred)
    acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()
    print(' ############### Performance Results on  whole dataset ###############')
    print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

    y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
    cm = clustering_metrics(y[val_idx].cpu().numpy(), y_pred)
    acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()
    print(' ############### Performance Results on  Valid dataset ###############')
    print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

    # os.remove(embedding_store)


    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(num_nodes, device=device))
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
        y = y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700']
    # plot_points(colors)


    fig, ax = plt.subplots(figsize=(6,4))
    KPIs = ['Accuracy','NMI','CS','F1','ARI']
    plot_data = details_val[KPIs]
    epochs_X = details_val['Epoch']
    # plt.subplots(figsize=(6,4))
    for m_ in KPIs:
            plt.plot(epochs_X, plot_data[m_ ], 'o-', label=m_)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Metrichs", fontsize = 12)
    plt.grid()
    plt.legend(loc='lower center',ncol=5)
    plt.title(f'Node2Vec: Performance Metrichs ({dataset_name}: Valid)')
    plot_name = f'Node2Vec_{dataset_name}_Valid_1.png'
    if save_plots:
        plt.savefig('./plots/'+plot_name)
    # plt.show()

    fig, ax = plt.subplots(figsize=(6,4))
    KPIs = ['Accuracy','NMI','CS','F1','ARI']
    plot_data = details_train[KPIs]
    epochs_X = details_train['Epoch']
    # plt.subplots(figsize=(6,4))
    for m_ in KPIs:
            plt.plot(epochs_X, plot_data[m_ ], 'o-', label=m_)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Metrichs", fontsize = 12)
    plt.grid()
    plt.legend(loc='lower center',ncol=5)
    plt.title(f'Node2Vec: Performance Metrichs ({dataset_name})')
    plot_name = f'Node2Vec_{dataset_name}_Train_1.png'
    if save_plots:
        plt.savefig('./plots/'+plot_name)
    # plt.show()

if __name__ == "__main__":
    main()