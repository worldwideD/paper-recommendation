import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

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
        h = F.elu(h)
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

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, alpha):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        self.W = nn.Parameter(torch.zeros(size=(in_feats, out_feats)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_feats, 1, )))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_feats, 1, )))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)    
        self.leaky_relu = nn.LeakyReLU(alpha)
    
    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        n = h.size()[0]
        h1 = torch.matmul(h, self.a1).repeat(1, n)
        h2 = torch.matmul(h, self.a2).view(1, n).repeat(n, 1)
        a = F.softmax(self.leaky_relu(h1 + h2), dim=1)
        zero_vec = -1e15 * torch.ones_like(a)
        att = torch.where(adj > 0, a, zero_vec)
        att = F.softmax(a, dim=1)
        h = torch.matmul(att, h)
        return h

class MultiHeadLayer(nn.Module):
    def __init__(self, in_feats, out_feats, nheads, is_concat, alpha):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.is_concat = is_concat
        self.nheads = nheads

        self.attns = nn.ModuleList([GATLayer(in_feats, out_feats, alpha) for _ in range(nheads)])
    
    def forward(self, x, adj):
        if self.is_concat:
            h = torch.cat([attn(x, adj) for attn in self.attns], dim=1)
        else:
            h = torch.mean(torch.stack([attn(x, adj) for attn in self.attns], dim=0), dim=0)
        h = F.elu(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, nlayers, nheads, alpha, dropout):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.nlayers = nlayers
        self.nheads = nheads
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        if nlayers == 1:
            self.GATLayers = nn.ModuleList([MultiHeadLayer(in_feats, h_feats, nheads, False, alpha)])
        else:
            self.GATLayers = nn.ModuleList([MultiHeadLayer(in_feats, h_feats, nheads, True, alpha),] + 
                                            [MultiHeadLayer(h_feats * nheads, h_feats, nheads, True, alpha) for _ in range(nlayers-2)] + 
                                            [MultiHeadLayer(h_feats * nheads, h_feats, nheads, False, alpha),])

    def forward(self, adj, x):
        h = x
        for l, GATLayer in enumerate(self.GATLayers):
            if l > 0:
                h = self.dropout(h)
            h = GATLayer(h, adj)
        return h

class PredictModel(nn.Module):
    def __init__(self, in_feats, h_feats, h_hops, n_layers1, n_layers2, nheads, dropout, top_k):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers1 = n_layers1
        self.n_layers2 = n_layers2
        self.dropout = nn.Dropout(dropout)
        self.h_hops = h_hops
        self.top_k = top_k

        self.GNN1 = GCN(in_feats, h_feats, n_layers1, dropout)
        self.W = nn.Parameter(torch.zeros(size=(in_feats, h_feats, )))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(h_feats, 1, )))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(h_feats, 1, )))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)    
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.GNN2 = GCN(in_feats, h_feats, n_layers2, dropout)
        # self.GNN2 = GAT(in_feats, h_feats, n_layers2, nheads, 0.2, dropout)

        # self.bilinear = nn.Bilinear(in_feats, h_feats * 2, 1)
        self.PredW = nn.Parameter(torch.zeros(size=(in_feats, h_feats * 2, )))
        nn.init.xavier_uniform_(self.PredW.data, gain=1.414)
        self.PredB = nn.Parameter(torch.zeros(size=(1, 1, )))
        nn.init.xavier_uniform_(self.PredB.data, gain=1.414)
    
    def get_sim_graph(self, h):
        '''
        n = h.size()[0]
        h_ = h.transpose(0, 1)
        dot_pro = torch.matmul(h, h_)
        len = torch.sqrt(torch.sum(h * h, dim=-1))
        len_sq = torch.matmul(len.unsqueeze(0), len.unsqueeze(1))
        sim = dot_pro / len_sq
        i = torch.eye(n).to(h)
        sim = sim - i * 9e15
        top = torch.topk(sim, self.top_k, dim=1, largest=True, sorted=True)[0][..., -1, None]
        e = sim >= top
        e = e | e.transpose(0, 1)
        e = e.type(torch.float)
        adj = sim * e
        return adj
        '''
        n = h.size()[0]
        h = torch.matmul(h, self.W)
        h1 = torch.matmul(h, self.a1).repeat(1, n)
        h2 = torch.matmul(h, self.a2).view(1, n).repeat(n, 1)
        i = torch.eye(n).to(h.device)
        a = h1 + h2 - i * 9e15
        a = F.softmax(self.leaky_relu(a), dim=1)
        # a = F.softmax(a, dim=1)
        top = torch.topk(a, self.top_k, dim=1, largest=True, sorted=True)[0][..., -1, None]
        e = a >= top
        # e = e | e.transpose(0, 1)
        e = e.type(torch.float)
        a = F.softmax(self.leaky_relu(h1 + h2 - (1 - e) * 9e15), dim=1)
        # a = F.softmax(h1 + h2 - (1 - e) * 9e15, dim=1)
        adj = e * a
        return adj

    def forward(self, src, dst, labels, adj, x, mode):
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
        if mode == "train":
            # logits = self.bilinear(src_h, dst_h)
            src_h = src_h.unsqueeze(1)
            dst_h = dst_h.unsqueeze(2)
            logits = src_h.matmul(self.PredW).matmul(dst_h).add(self.PredB).view(-1)
        else:
            dst_h = dst_h.transpose(0, 1)
            logits = src_h.matmul(self.PredW).matmul(dst_h).add(self.PredB).view(-1)

        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        return (loss, logits)