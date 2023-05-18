######################################################
####### Please define Parameters ########################
num_workers = None #{None, 1, 2, 3, 4,...}
dataset_path = '../dataset/'
save_embedding = True
save_plots = True


import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans as KMEANS
from tqdm import tqdm
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from sklearn.base import clone
from ogb.nodeproppred import PygNodePropPredDataset
from load_data_graph_saint import load_data
from clustering_metric import clustering_metrics




cora_params     = {'embedding_dims':512, 'num_neighbors':[25, 20, 10] , 'batch_size':256 , 'lr':0.001,'epochs':40}
citeseer_params = {'embedding_dims':512, 'num_neighbors':[25, 20, 10] , 'batch_size':128 , 'lr':0.0001,'epochs':40}
pubmed_params   = {'embedding_dims':512, 'num_neighbors':[25, 20, 10] , 'batch_size':256 , 'lr':0.0001,'epochs':40}
arxiv_params    = {'embedding_dims':512, 'num_neighbors':[25, 20, 10] , 'batch_size':128 , 'lr':0.0001,'epochs':40}
reddit_params   = {'embedding_dims':512, 'num_neighbors':[25, 20, 10] , 'batch_size':128 , 'lr':0.0001,'epochs':40}
products_params = {'embedding_dims':512, 'num_neighbors':[25, 20, 10] , 'batch_size':128 , 'lr':0.0001,'epochs':40}


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


embedding_dims = model_params['embedding_dims']
num_neighbors = model_params['num_neighbors']
batch_size = model_params['batch_size']
lr = model_params['lr']
epochs = model_params['epochs']
embedding_path = '../Embeddings_Saved/'
embedding_store = f'DGI_embedding_{dataset_name}.pt'


import sys
import os
dataset_name = sys.argv[1]

embedding_path = '../Embeddings_Saved/'
embedding_store = f'DGI_embedding_{dataset_name}.pt'

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

def save_embedding(embedding, save_embedding):
    torch.save(embedding.cpu(), save_embedding)


def Kmeans_(x, K=-1, Niter=10, verbose=False):
    # start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    x_temp = x.detach()

    temp = set()
    while len(temp) < K:
        temp.add(np.random.randint(0, N))
    c = x_temp[list(temp), :].clone()

    x_i = x_temp.view(N, 1, D)  # (N, 1, D) samples
    cutoff = 1
    if K > cutoff:
        c_j = []
        niter = K // cutoff
        rem = K % cutoff
        if rem > 0:
            rem = 1
        for i in range(niter + rem):
            c_j.append(c[i * cutoff:min(K, (i + 1) * cutoff), :].view(1, min(K, (i + 1) * cutoff) - (i * cutoff), D))
    else:
        c_j = c.view(1, K, D)  # (1, K, D) centroids

    for i in range(Niter):
        # print("iteration: " + str(i))

        # E step: assign points to the closest cluster -------------------------
        if K > cutoff:
            for j in range(len(c_j)):
                if j == 0:
                    D_ij = ((x_i - c_j[j]) ** 2).sum(-1)
                else:
                    D_ij = torch.cat((D_ij, ((x_i - c_j[j]) ** 2).sum(-1)), dim=-1)
                    # D_ij += ((x_i - c_j[j]) ** 2).sum(-1)
        else:
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        assert D_ij.size(1) == K
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x_temp)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        # print(Ncl[:10])
        Ncl += 0.00000000001
        c /= Ncl  # in-place division to compute the average
    return cl, c


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if str.lower(dataset_name) in ['cora','citeseer','pubmed']:
    dataset = Planetoid(root=dataset_path, name=dataset_name)
    data = dataset[0].to(device)
    val_idx = data.val_mask.nonzero().view(-1).tolist()
    edge_index = data.edge_index
    Node_Attr = data.x

elif str.lower(dataset_name) in ['ogbn-arxiv','ogbn-products']:
    dataset = PygNodePropPredDataset(name = dataset_name, root=dataset_path)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0].to(device)
    edge_index = data.edge_index
    Node_Attr = data.x

elif str.lower(dataset_name) in ['reddit']:
    A,_,X,label,split = load_data(dataset_path + dataset_name)
    val_idx = split['va']
    y = label.view(-1).cpu()
    A = A.to(device)
    Node_Attr = X.float().to(device)
    edge_index = A.coalesce().indices()




print("###### Graph Structure and Node Attricutions size  ##############")
print('Node_Attr.shape: ',Node_Attr.shape)
print('edge_index.shape: ',edge_index.shape)


if num_workers:
    data_loader = NeighborSampler(edge_index, node_idx=None,sizes=num_neighbors, batch_size=batch_size,shuffle=False, num_workers=num_workers)
else:
    data_loader = NeighborSampler(edge_index, node_idx=None,sizes=num_neighbors, batch_size=batch_size,shuffle=False)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x
    


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


model = DeepGraphInfomax(
    hidden_channels=embedding_dims, encoder=Encoder(dataset.num_features, embedding_dims),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption).to(device)




model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

x, y = Node_Attr.to(device), data.y.view(-1).to(device)


# def test():
#     model.eval()
#     zs = []
#     for i, (batch_size, n_id, adjs) in enumerate(data_loader):
#         adjs = [adj.to(device) for adj in adjs]
#         zs.append(model(x[n_id], adjs)[0])
#     z = torch.cat(zs, dim=0)

#     train_val_mask = val_idx

#     clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000) \
#         .fit(z[train_val_mask].detach().cpu().numpy(), data.y[train_val_mask].detach().cpu().numpy())

#     y_pred = clf.predict(z[data.test_mask].detach().cpu().numpy())
#     acc = metrics.accuracy_score(data.y[data.test_mask].detach().cpu().numpy(), y_pred)
#     f1 = metrics.f1_score(data.y[data.test_mask].detach().cpu().numpy(), y_pred, average='macro')

#     return acc, f1


max_nmi = -1
best_kmeans = None

details = pd.DataFrame(columns=['Epoch','Accuracy','NMI','CS','F1','ARI','loss']) 

for epoch in range(1, epochs+1):
    zs_tr = []
    model.train()

    total_loss = total_examples = 0
    it = 0
    for batch_size, n_id, adjs in tqdm(data_loader,
                                       desc=f'Epoch {epoch:02d}'):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        pos_z, neg_z, summary = model(x[n_id], adjs)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_z.size(0)
        total_examples += pos_z.size(0)
        it += 1
    #         if it==10:
    #             break
    
    with torch.no_grad():
        model.eval()
        for batch_size, n_id, adjs in tqdm(data_loader, desc=f'Test Epoch {epoch:02d}'):
            adjs = [adj.to(device) for adj in adjs]
            zs_tr.append(model(x[n_id], adjs)[0])
        zs_tr = torch.cat(zs_tr, dim=0)
        zs_tr = torch.nn.functional.normalize(zs_tr)
        y_pred, _ = Kmeans_(zs_tr.detach(), y.max() + 1)

    
    loss_ = total_loss / total_examples

    # test_acc, f1 = test()

    embedding = zs_tr.clone()
    kmeans = KMEANS(n_clusters=y.max().item()+1, n_init=10)

    y_pred = kmeans.fit_predict(embedding.detach().cpu().numpy())
    cm = clustering_metrics(y.cpu().numpy(), y_pred)
    acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()

    print('##############  Evaluation on all embedding for each epoch  ############')
    print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

    new_row = {'Epoch':epoch,'Accuracy':acc,'NMI':nmi,'CS':cs,'F1':f1_macro,'ARI':adjscore,'loss':loss_}
    details = pd.concat([details, pd.DataFrame([new_row])], ignore_index=True)



    if nmi > max_nmi:
        max_nmi = nmi
        if save_embedding:
            save_embedding(embedding, embedding_path + embedding_store)
            best_kmeans = clone(kmeans)
    del embedding, y_pred

    print(f'Epoch {epoch:02d}, Loss: {loss:.4f} ')




# @torch.no_grad()
model.eval()
print('############## Final results  ############')
embedding = torch.load(embedding_path + embedding_store)
# kmeans = KMEANS(n_clusters=y.max().item()+1, n_init=10)
kmeans = best_kmeans
y_pred = kmeans.fit_predict(embedding.detach().cpu().numpy())
cm = clustering_metrics(y.cpu().numpy(), y_pred)
acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()
print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)



fig, ax = plt.subplots(figsize=(6,4))
KPIs = ['Accuracy','NMI','CS','F1','ARI']
plot_data = details[KPIs]
epochs_X = details['Epoch']
# plt.subplots(figsize=(6,4))
for m_ in KPIs:
        plt.plot(epochs_X, plot_data[m_ ], 'o-', label=m_)
plt.xlabel("Epochs", fontsize = 12)
plt.ylabel("Metrics", fontsize = 12)
plt.grid()
plt.legend(loc='lower center',ncol=5)
plt.title(f'DGI: Performance Metrics ({dataset_name})')
plot_name = f'DGI_{dataset_name}.png'
if save_plots:
    plt.savefig('./plots/'+plot_name)
# plt.show()
