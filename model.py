import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from graph import h_hop_subgraph, generate_full_adj

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.elu = nn.ELU(inplace=True)
        self.W = nn.Parameter(torch.zeros(size=(in_feats, out_feats, )))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, adj, input):
        h = torch.matmul(input, self.W)
        h = torch.matmul(adj, h)
        h = self.elu(h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, n_layers):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.layers = nn.ModuleList([GCNLayer(in_feats, h_feats), ] + [GCNLayer(h_feats, h_feats) for _ in range(n_layers-1)])
    
    def adj_helper(self, adj):
        n = adj.size()[0]
        a = torch.eye(n).to(adj.device)
        adj = adj + a
        degree = adj.sum(dim=0).unsqueeze(1)
        adj = adj / degree
        return adj

    def forward(self, adj, input):
        z = input
        a = self.adj_helper(adj)
        for gcnlayer in self.layers:
            z = gcnlayer(a, z)
        return z

class PredictModel(nn.Module):
    def __init__(self, in_feats, h_feats, h_hops, n_layers, dropout):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.h_hops = h_hops
        self.elu = nn.ELU(inplace=True)

        self.GNN = GCN(in_feats, h_feats, n_layers)
        self.out1 = nn.Linear(2 * h_feats, h_feats)
        self.out2 = nn.Linear(h_feats, 1)

    def forward(self, src, dst, labels, adj, x):
        m = len(src)
        a = torch.tensor(adj).to(x.device)
        h = self.GNN(a, x)

        src = torch.LongTensor(src).to(x.device)
        dst = torch.LongTensor(dst).to(x.device)
        src_h = torch.index_select(h, 0, src)
        dst_h = torch.index_select(h, 0, dst)
        # z = src_h * dst_h
        z = torch.cat([src_h, dst_h], dim=1)

        z = self.dropout(z)
        z = self.elu(self.out1(z))
        z = self.dropout(z)
        logits = torch.squeeze(self.out2(z), dim=1)
        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        predict = (logits >= 0)
        return (loss, logits, predict)