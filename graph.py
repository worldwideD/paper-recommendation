import numpy as np
import torch

def arrange_id(nodes):
    ndict = dict()
    for x, y in enumerate(nodes):
        ndict[y] = x
    return ndict

def generate_adj(edges, n):
    if len(edges) == 0:
        return [] * n
    undir_edges = []
    for edge in edges:
        undir_edges.append([edge[0], edge[1]])
        undir_edges.append([edge[1], edge[0]])
    
    undir_edges = np.array(undir_edges)
    sorted_edges = undir_edges[np.lexsort(undir_edges[:, ::-1].T)]

    adj, j, m = [], 0, len(sorted_edges)
    for i in range(n):
        neighbors = []
        while j < m and sorted_edges[j][0] < i:
            j += 1
        while j < m and sorted_edges[j][0] == i:
            neighbors.append(sorted_edges[j][1])
            j += 1
        adj.append(neighbors)
    
    return adj

def generate_full_adj(edges, ndict):
    n = len(ndict)
    adj = np.zeros(shape=(n, n))
    
    for edge in edges:
        adj[ndict[edge[0]], ndict[edge[1]]] = 1
        adj[ndict[edge[1]], ndict[edge[0]]] = 1
    
    return adj


def neighbors(fringe, adj):
    res = set()
    for n in fringe:
        res = res | set(adj[n])
    return res

def h_hop_subgraph(src, dst, hops, adj, x):
    visited = set([src, dst])
    fringe = set([dst])
    nodes = [src, dst]
    subgraph_dict = dict()
    subgraph_dict[src] = 0
    subgraph_dict[dst] = 1
    n = 2
    edges = []
    for dist in range(1, hops+1):
        next = neighbors(fringe, adj)
        next = next - visited
        for node in next:
            subgraph_dict[node] = n
            nodes.append(node)
            n += 1
        for fr in fringe:
            for to in adj[fr]:
                if to in next:
                    edges.append((subgraph_dict[fr], subgraph_dict[to]))
        fringe = next
        visited = visited | fringe
    
    subgraph_adj = generate_full_adj(edges, n)
    nodes = torch.LongTensor(nodes).to(x.device)
    subgraph_x = torch.index_select(x, 0, nodes)
    return subgraph_x, subgraph_adj