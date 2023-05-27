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
