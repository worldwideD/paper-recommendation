from prepro import read_graphdata, read_metadata, get_paperlist
import codecs
import numpy as np
import random
from tqdm import tqdm
import nltk.tokenize as tk
import nltk.stem.porter as pt

val_devide_year = 2013  # years before it are train set
test_devide_year = 2014  # years before it are train & val sets
text_keys = 20  # max key phrases count for text

# generate train negatives
def generate_neg(edges, nodes):
    train_neg_cnt = len(edges)
    nodes_cnt = len(nodes)
    edge_set = set(edges)
    neg = []
    while train_neg_cnt > 0:
        f = np.random.randint(0, nodes_cnt, size=train_neg_cnt)
        t = np.random.randint(0, nodes_cnt, size=train_neg_cnt)
        for x, y in zip(f, t):
            fr = nodes[x]
            to = nodes[y]
            if fr != to and not (fr, to) in edge_set and not (to, fr) in edge_set:
                edge_set.add((fr, to))
                neg.append([fr, to, 0])
                train_neg_cnt -= 1
    neg = np.array(neg)
    return neg

# generate train, val & test sets
def devide_edges(id_edges, datadict):
    # get nodes
    test_node_list, val_node_list, train_node_list = [], [], []
    for n in datadict.values():
        if n['metadata']['year'] >= test_devide_year:
            test_node_list.append(n['id'])
        elif n['metadata']['year'] >= val_devide_year:
            val_node_list.append(n['id'])
        else:
            train_node_list.append(n['id'])
    
    test_node = len(test_node_list)
    train_node = len(train_node_list)
    val_node = len(val_node_list)

    train_edges, train_neg, val_pos, val_neg, test_pos, test_neg = [], [], [], [], [], []
    # generate positive samples
    for edge in id_edges:
        if edge[0] in test_node_list:
            if not edge[1] in test_node_list:
                test_pos.append(edge)
        elif edge[0] in val_node_list:
            if not edge[1] in val_node_list:
                val_pos.append(edge)
        else:
            train_edges.append(edge)
    
    total = len(train_edges)
    p = np.random.permutation(total)
    msg_edges, train_pos = [], []
    for i in range(total):
        if i < total // 2:
            train_pos.append(train_edges[p[i]])
        else:
            msg_edges.append(train_edges[p[i]])

    edge_set = set(id_edges)
    
    # generate negative samples
    train, val, test = len(train_pos), len(val_pos), len(test_pos)
    
    # val negative
    val_neg_cnt = val
    while val_neg_cnt > 0:
        f = np.random.randint(0, val_node, size=val_neg_cnt)
        t = np.random.randint(0, train_node, size=val_neg_cnt)
        for x, y in zip(f, t):
            fr = val_node_list[x]
            to = train_node_list[y]
            if fr != to and not (fr, to) in edge_set and not (to, fr) in edge_set:
                edge_set.add((fr, to))
                val_neg.append((fr, to))
                val_neg_cnt -= 1
    # test negative
    test_neg_cnt = test
    while test_neg_cnt > 0:
        f = np.random.randint(0, test_node, size=test_neg_cnt)
        t = np.random.randint(0, train_node+val_node, size=test_neg_cnt)
        for x, y in zip(f, t):
            fr = test_node_list[x]
            if y < train_node:
                to = train_node_list[y]
            else:
                to = val_node_list[y-train_node]
            if fr != to and not (fr, to) in edge_set and not (to, fr) in edge_set:
                edge_set.add((fr, to))
                test_neg.append((fr, to))
                test_neg_cnt -= 1
    print('train size: {}\nval size: {}\ntest size: {}\n'.format(train*2, val*2, test*2))
    return msg_edges, train_pos, train_node_list, val_pos, val_neg, test_pos, test_neg

def read_data(graphpath, metapath, titlepath, keydirpath):
    alldatadict, allmetadata = read_metadata(metapath)
    graphdata = read_graphdata(graphpath)

    # index the nodes
    node = 0
    datadict = dict()
    id_edges = []
    for edge in graphdata:
        if not edge['from'] in datadict:
            datadict[edge['from']] = {'id': node, 'metadata': alldatadict[edge['from']]}
            node += 1
        if not edge['to'] in datadict:
            datadict[edge['to']] = {'id': node, 'metadata': alldatadict[edge['to']]}
            node += 1
        fr = datadict[edge['from']]['id']
        to = datadict[edge['to']]['id']
        if fr != to:
            id_edges.append((fr, to))
    
    print('{} number of nodes\n{} number of edges'.format(node, len(id_edges)))
    
    # get examples
    msg_edges, train_pos, train_nodes, val_pos, val_neg, test_pos, test_neg = devide_edges(id_edges, datadict)
    
    # get preprocessed title data
    title_dict = dict()
    title_cnt = 0
    with codecs.open(titlepath, 'r', 'utf-8', 'ignore') as f:
        lines = f.readlines()

        id, cur_keys, cur_v, title = -1, [], [], ''
        for line in lines:
            line = line[:-1]
            if line.find(' ') == -1: # id
                # last one
                if id != -1:
                    title_dict[id] = {'title': title, 'keyphrases': cur_keys, 'value': cur_v}
                    cur_keys, cur_v = [], []

                # current one
                if not line in datadict:
                    id = -1
                else:
                    title_cnt += 1
                    id = datadict[line]['id']
            elif line.find(' ') != 0: # title
                title = line
            elif id != -1: # key phrase
                sep = line.rfind(' ')
                keyphrase = line[1:sep]
                value = float(line[sep+1:])
                cur_keys.append(keyphrase)
                cur_v.append(value)
        if id != -1:
            title_dict[id] = {'title': title, 'keyphrases': cur_keys, 'value': cur_v}
            cur_keys, cur_v = [], []
    print('read {} titles\n'.format(title_cnt))

    # get preprocessed text data
    keypathlist, keylist = get_paperlist(datadict, keydirpath)
    text_dict = dict()
    text_cnt = 0
    stemmer = pt.PorterStemmer()
    for path, key in zip(keypathlist, keylist):
        text_cnt += 1
        cur_keys, cur_v = [], []
        with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
            keystr = 'keyphrasebytextrank: '
            valstr = 'valuebytextrank: '
            str = f.read()
            '''
            for _ in range(text_keys):
                k = str.find(keystr)
                if k == -1:
                    break
                
                v = str.find(valstr)
                keyphrase = str[k+21:v-1]
                str = str[v:]
                sep = str.find(' ', 17)
                value = str[:sep]
                value = value.strip(valstr).strip()
                value = float(value)
                cur_keys.append(keyphrase)
                cur_v.append(value)
                str = str[sep:]
            '''
            key_cnt = 0
            stemlist = []
            while key_cnt < text_keys:
                k = str.find(keystr)
                if k == -1:
                    break

                v = str.find(valstr)
                keyphrase = str[k+21:v-1]
                str = str[v:]
                sep = str.find(' ', 17)
                value = str[:sep]
                value = value.strip(valstr).strip()
                value = float(value)
                str = str[sep:]

                tokens = tk.word_tokenize(keyphrase)
                stems = [stemmer.stem(t) for t in tokens]
                if stems in stemlist:
                    continue
                cur_keys.append(keyphrase)
                cur_v.append(value)
                stemlist.append(stems)
                key_cnt += 1
        
        text_dict[datadict[key]['id']] = {'keyphrases': cur_keys, 'value': cur_v}
    
    print('read {} texts\n'.format(text_cnt))
    return msg_edges, train_pos, train_nodes, val_pos, val_neg, test_pos, test_neg, title_dict, text_dict