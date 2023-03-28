import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from graph import h_hop_subgraph, generate_full_adj

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.W = nn.Parameter(torch.zeros(size=(in_feats, out_feats, )))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, adj, input):
        h = torch.matmul(input, self.W)
        h = torch.matmul(adj, h)
        return h

class SIGN(nn.Module):
    def __init__(self, in_feats, out_feats, n_heuristic):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_heuristic = n_heuristic
        self.elu = nn.ELU(inplace=True)
        self.operators = nn.ModuleList([GCNLayer(in_feats, out_feats) for _ in range(n_heuristic)])
    
    def adj_helper(self, p, adj):
        n = adj.size()[0]
        a = torch.eye(n).to(adj.device)
        for _ in range(p):
            a = a.matmul(adj)
        return a

    def forward(self, adj, input):
        z = torch.cat([op(self.adj_helper(p, adj), input) for p, op in enumerate(self.operators)], dim=1)
        z = self.elu(z)
        return z

class PredictModel(nn.Module):
    def __init__(self, in_feats, h_feats, h_hops, n_heuristic, dropout):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_heuristic = n_heuristic
        self.dropout = nn.Dropout(dropout)
        self.h_hops = h_hops
        self.elu = nn.ELU(inplace=True)

        self.GNN = SIGN(in_feats, h_feats, n_heuristic)
        self.out1 = nn.Linear(h_feats * n_heuristic, h_feats)
        self.out2 = nn.Linear(h_feats, 1)
    
    def gnn_helper(self, src, dst, adj, x):
        sub_x, sub_adj = h_hop_subgraph(src, dst, self.h_hops, adj, x)
        sub_adj = torch.FloatTensor(sub_adj).to(x.device)
        n = sub_adj.size()[0]
        sub_I = torch.eye(n).to(x.device)
        sub_adj = sub_adj.add(sub_I)
        sub_adj_ = sub_adj / sub_adj.sum(dim=1)

        z = self.GNN(sub_adj_, sub_x)
        pooled = z[0] * z[1]
        return pooled

    def forward(self, src, dst, labels, adj, x):
        n = len(src)
        z = torch.stack([self.gnn_helper(src[i], dst[i], adj, x) for i in range(n)], dim=0)
        logits = z.sum(dim=-1)
        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        predict = (logits >= 0)
        return (loss, logits, predict)

        z = self.dropout(z)
        
        z = self.elu(self.out1(z))
        z = self.dropout(z)
        logits = torch.squeeze(self.out2(z), dim=1)
        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        predict = (logits >= 0)
        return (loss, logits, predict)