import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

def adj_helper(adj):
    n = adj.size()[0]
    i = torch.eye(n).to(adj.device)
    adj = adj + i
    d1 = adj.sum(dim=-1).sqrt().unsqueeze(1)
    d2 = adj.sum(dim=0).sqrt().unsqueeze(0)
    adj = adj / d1 / d2
    return adj

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

    def forward(self, adj, input):
        z = input
        a = adj_helper(adj)
        for l, gcnlayer in enumerate(self.layers):
            if l > 0:
                z = self.dropout(z)
            z = gcnlayer(a, z)
        return z

class SAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.W = nn.Parameter(torch.zeros(size=(in_feats + out_feats, out_feats, )))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.Wp = nn.Parameter(torch.zeros(size=(in_feats, out_feats, )))
        nn.init.xavier_uniform_(self.Wp.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_feats, )))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, adj, input):
        h = torch.matmul(input, self.Wp)
        h = h + self.b
        h = F.elu(h)
        h = torch.matmul(adj, h)
        h = torch.cat([input, h], dim=1)
        h = torch.matmul(h, self.W)
        h = F.elu(h)
        return h

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, n_layers, dropout):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([SAGELayer(in_feats, h_feats), ] + [SAGELayer(h_feats, h_feats) for _ in range(n_layers-1)])

    def forward(self, adj, input):
        z = input
        degree = adj.sum(dim=-1).unsqueeze(1)
        degree = torch.clamp(degree, min=1e-5)
        adj = adj / degree
        for l, gcnlayer in enumerate(self.layers):
            if l > 0:
                z = self.dropout(z)
            z = gcnlayer(adj, z)
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
        a = self.leaky_relu(h1 + h2)
        zero_vec = -9e15 * torch.ones_like(a)
        att = torch.where(adj > 0, a, zero_vec)
        att = F.softmax(att, dim=1)
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
        adj = adj_helper(adj)
        for l, GATLayer in enumerate(self.GATLayers):
            if l > 0:
                h = self.dropout(h)
            h = GATLayer(h, adj)
        return h

class PredictModel(nn.Module):
    def __init__(self, GNN1, GNN2, sim, in_feats, h_feats, n_layers1, n_layers2, nheads, dropout, top_k):
        super().__init__()
        assert sim == "cos" or sim == "attn", "similarity calc should be cos or attn"
        self.sim = sim
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_layers1 = n_layers1
        self.n_layers2 = n_layers2
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k

        if GNN1 == "GCN":
            self.GNN1 = GCN(in_feats, h_feats, n_layers1, dropout)
        elif GNN1 == "GraphSAGE":
            self.GNN1 = SAGE(in_feats, h_feats, n_layers1, dropout)
        elif GNN1 == "GAT":
            self.GNN1 = GAT(in_feats, h_feats, n_layers1, nheads, 0.2, dropout)
        else:
            assert 1 == 2, "GNN1 name error or not implemented"
        self.W1 = nn.Parameter(torch.zeros(size=(in_feats, h_feats, )))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.zeros(size=(in_feats, h_feats, )))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(h_feats, 1, )))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(h_feats, 1, )))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)    
        self.leaky_relu = nn.LeakyReLU(0.2)
        if GNN2 == "GraphSAGE":
            self.GNN2 = SAGE(in_feats, h_feats, n_layers2, dropout)
        elif GNN2 == "GCN":
            self.GNN2 = GCN(in_feats, h_feats, n_layers2, dropout)
        else:
            assert 1 == 2, "GNN2 name error or not implemented"

        self.PredW = nn.Parameter(torch.zeros(size=(h_feats * 2, h_feats * 2, )))
        nn.init.xavier_uniform_(self.PredW.data, gain=1.414)
        self.PredB = nn.Parameter(torch.zeros(size=(1, 1, )))
        nn.init.xavier_uniform_(self.PredB.data, gain=1.414)
    
    def get_sim_graph(self, h):
        if self.sim == "cos":
            n = h.size()[0]
            h_ = h.transpose(0, 1)
            dot_pro = torch.matmul(h, h_)

            len = torch.sqrt(torch.sum(h * h, dim=-1))
            len_sq = torch.matmul(len.unsqueeze(1), len.unsqueeze(0))

            sim = dot_pro / len_sq
            i = torch.eye(n).to(h)
            sim = sim - i * 9e15
            val, ids = torch.topk(sim, self.top_k, dim=1, largest=True, sorted=True)
            p = torch.arange(0, n).to(sim).unsqueeze(1).repeat(1, self.top_k).view(-1)
            p = p.type(torch.long)
            q = ids.view(-1)
            v = val.view(-1)
            z = torch.zeros_like(sim).to(sim)
            adj = z.index_put((p, q), v)
        
            return adj

        n = h.size()[0]
        h1 = torch.tanh(torch.matmul(h, self.W1))
        h2 = torch.tanh(torch.matmul(h, self.W2))
        h1 = torch.matmul(h1, self.a1).repeat(1, n)
        h2 = torch.matmul(h2, self.a2).view(1, n).repeat(n, 1)
        i = torch.eye(n).to(h.device)
        a = h1 + h2 - i * 9e15
        a = F.softmax(a, dim=1)
        val, ids = torch.topk(a, self.top_k, dim=1, largest=True, sorted=True)
        p = torch.arange(0, n).to(a).unsqueeze(1).repeat(1, self.top_k).view(-1)
        p = p.type(torch.long)
        q = ids.view(-1)
        v = val.view(-1)
        z = torch.zeros_like(a).to(a)
        adj = z.index_put((p, q), v)
        return adj

    def forward(self, src, dst, labels, adj, x, mode):
        a = torch.FloatTensor(adj).to(x.device)
        h = self.GNN1(a, x)
        sim_adj = self.get_sim_graph(x)
        h_ = self.GNN2(sim_adj, x)
        h = torch.cat([h, h_], dim=1)
        
        src = torch.LongTensor(src).to(x.device)
        dst = torch.LongTensor(dst).to(x.device)
        src_h = torch.index_select(h, 0, src)
        dst_h = torch.index_select(h, 0, dst)
        if mode == "train":
            src_h = src_h.unsqueeze(1)
            dst_h = dst_h.unsqueeze(2)
            logits = src_h.matmul(self.PredW).matmul(dst_h).add(self.PredB).view(-1)
        else:
            dst_h = dst_h.transpose(0, 1)
            logits = src_h.matmul(self.PredW).matmul(dst_h).add(self.PredB).view(-1)

        labels = torch.tensor(labels).to(logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        return (loss, logits)