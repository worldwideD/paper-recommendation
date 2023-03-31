import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
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
    def __init__(self, in_feats, h_feats, n_layers, dropout):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GCNLayer(in_feats, h_feats), ] + [GCNLayer(h_feats, h_feats) for _ in range(n_layers-1)])
    
    def adj_helper(self, adj):
        n = adj.size()[0]
        i = torch.eye(n).to(adj.device)
        adj = adj + i
        degree = adj.sum(dim=0).unsqueeze(1)
        adj = adj / degree
        return adj

    def forward(self, adj, input):
        z = input
        a = self.adj_helper(adj)
        for l, gcnlayer in enumerate(self.layers):
            if l > 0:
                z = self.dropout(z)
            z = gcnlayer(a, z)
        return z

class PredictModel(nn.Module):
    def __init__(self, in_feats, h_feats, h_hops, n_layers, dropout, top_k):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.h_hops = h_hops
        self.elu = nn.ELU(inplace=True)
        self.top_k = top_k

        self.GNN1 = GCN(in_feats, h_feats, n_layers, dropout)
        self.W = nn.Parameter(torch.zeros(size=(in_feats, h_feats, )))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(h_feats, 1, )))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(h_feats, 1, )))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)    
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.GNN2 = GCN(in_feats, h_feats, n_layers, dropout)

        self.bilinear = nn.Bilinear(in_feats, h_feats * 2, 1)
    
    def get_sim_graph(self, h):
        n = h.size()[0]
        h = torch.matmul(h, self.W)
        h1 = torch.matmul(h, self.a1).repeat(1, n)
        h2 = torch.matmul(h, self.a2).view(1, n).repeat(n, 1)
        i = torch.eye(n).to(h.device)
        a = h1 + h2 - i * 1e15
        a = F.softmax(self.leaky_relu(a), dim=1)
        adj = a >= torch.topk(a, self.top_k, dim=1, largest=True, sorted=True)[0][..., -1, None]
        adj = adj | adj.transpose(0, 1)
        adj = adj.type(torch.float)
        return adj

    def forward(self, src, dst, labels, adj, x):
        m = len(src)
        a = torch.FloatTensor(adj).to(x.device)
        h = self.GNN1(a, x)
        sim_adj = self.get_sim_graph(x)
        h_ = self.GNN2(sim_adj, x)
        h = torch.cat([h, h_], dim=1)
        # h = h_

        src = torch.LongTensor(src).to(x.device)
        dst = torch.LongTensor(dst).to(x.device)
        src_h = torch.index_select(x, 0, src)
        dst_h = torch.index_select(h, 0, dst)
        
        # z = src_h * dst_h
        # logits = z.sum(dim=-1)
        logits = self.bilinear(src_h, dst_h)
        logits = logits.squeeze(1)
        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        predict = (logits >= 0)
        return (loss, logits, predict)
        
        z = torch.cat([src_h, dst_h], dim=1)
        z = self.dropout(z)
        z = self.elu(self.out1(z))
        z = self.dropout(z)
        logits = torch.squeeze(self.out2(z), dim=1)
        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        predict = (logits >= 0)
        return (loss, logits, predict)