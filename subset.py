from prepro import read_graphdata, read_metadata, wry
import random

def generate_subset(t, n, path):

    metapath = 'release/2014/acl-metadata.txt'
    datadict, metadata = read_metadata(metapath)
    graphpath = 'graph.txt'
    graphdata = read_graphdata(graphpath)
    
    test_list = []
    for p in metadata:
        if p['year'] == 2014:
            cnt = 0
            for e in graphdata:
                if e['from'] == p['id']:
                    cnt += 1
            if cnt > 1:
                test_list.append(p['id'])
    random.shuffle(test_list)
    print(len(test_list))

    node = t
    node_list = test_list[0:t]

    curn, cure, l, ee = 0, 0, len(graphdata), 0
    while node < n and curn < node:
        while ee < 5 and cure < l and graphdata[cure]['from'] != node_list[curn] and graphdata[cure]['to'] != node_list[curn]:
            cure += 1
        if cure == l or ee == 5:
            cure, ee = 0, 0
            curn += 1
        else:
            if not graphdata[cure]['to'] in node_list:
                node_list.append(graphdata[cure]['to'])
                node += 1
            elif not graphdata[cure]['from'] in node_list and not graphdata[cure]['from'] in test_list:
                node_list.append(graphdata[cure]['from'])
                node += 1
            cure += 1
            ee += 1
    
    print(node)
    subgraph = []
    te = 0
    out = ''
    for edge in graphdata:
        if edge['from'] in node_list and edge['to'] in node_list:
            subgraph.append(edge)
            out = out + edge['from'] + " ==> " + edge['to'] + '\n'
            if edge['from'] in test_list:
                te += 1
    print(len(subgraph))
    print(te)
    wry(out, path)

generate_subset(100, 1500, 'subgraph.txt')