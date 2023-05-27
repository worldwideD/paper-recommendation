import codecs

def read_metadata(path):
    with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
        lines = f.readlines()

        datadict = dict()
        metadata = []

        for line in lines:
            if len(line) == 1:
                continue
            blank1 = line.find(' ')
            typ = line[0:blank1]
            blank2 = line.find(' ', blank1+1)
            cnt = line[blank2+2:-2]
            
            if typ == 'id':
                id = cnt
            elif typ == 'author':
                author = cnt
            elif typ == 'title':
                title = cnt
            elif typ == 'venue':
                venue = cnt
            elif typ == 'year':
                if id in datadict:
                    continue
                year = int(cnt)
                datadict[id] = {'author': author, 'title': title, 'venue': venue, 'year': year}
                metadata.append({'id': id, 'author': author, 'title': title, 'venue': venue, 'year': year})
            else:
                continue
    
    print('read metadata done!')
    return datadict, metadata

def befwry(lis):  # 写入预处理，将list转为string
    outall = ''
    for i in lis:
        ech = str(i).replace("('", '').replace("',", '').replace(')', '')
        outall = outall + ' ' + ech + '\n'
    return outall

def wry(txt, path):  # 写入txt文件
    f = codecs.open(path, 'w', 'utf-8')
    f.write(txt)
    f.close()
    return path

# 读入citation network
def read_graphdata(path):
    with codecs.open(path, 'r', 'utf-8', 'ignore') as f:
        lines = f.readlines()

        graphdata = []

        for line in lines:
            if len(line) == 1:
                continue
            line = line.split()
            graphdata.append({'from': line[0], 'to': line[2]})

    print('read graphdata done!')
    return graphdata

import os


# 获取论文txt文件
def get_paperlist(dict, path):
    paperpathlist, paperlist = [], []
    for root, dirs, files in os.walk(path):
        for fn in files:
            typ = os.path.splitext(fn)[-1]
            if typ == '.txt' and fn != '1.txt':
                n = os.path.splitext(fn)[0]
                if not n in dict:
                    continue
                paperpathlist.append(path + '/' + fn)
                paperlist.append(n)
    
    print('paper paths done!')
    return paperpathlist, paperlist
'''
from keybert import KeyBERT

model = KeyBERT('distilbert-base-nli-mean-tokens')

# 用keybert提取标题关键词
def keyword_title(metadata, wrypath):
    outall = ''

    for data in metadata:
        title = data['title']
        echout1 = model.extract_keywords(title, top_n=5, keyphrase_ngram_range=(1, 1), use_mmr=True)
        echout2 = model.extract_keywords(title, top_n=5, keyphrase_ngram_range=(2, 2), use_mmr=True)
        echout3 = model.extract_keywords(title, top_n=5, keyphrase_ngram_range=(3, 3), use_mmr=True)
        outall = outall + data['id'] + '\n' + title + '\n' + befwry(echout1) + befwry(echout2) + befwry(echout3)

    wry(outall, wrypath)
'''

import spacy
# 必须导入pytextrank，虽然表面上没用上，

'''
import pytextrank

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def read(path):  # 读取txt文件，并返回list
    with codecs.open(path, 'r', 'utf-8') as f:
        data = f.read()
    return data

def textrank(path):
    text = read(path)
    doc = nlp(text)
    out = ""
    cnt = 0
    for phrase in doc._.phrases:
        # 短语
        out = out + "keyphrasebytextrank: " + phrase.text + '\n'
        # 权重、词频
        out = out + "valuebytextrank: " + str(phrase.rank) + ' ' + str(phrase.count) + '\n'
        # 短语的列表
        # out = out + str(phrase.chunks) + '\n'
        cnt += 1
        if cnt == 300:
            break

    sep = path.find('/')
    wrypath = 'papers_keyword' + '/' + path[sep+1:]
    p = wry(out, wrypath)
    # print(wry(out, wrypath) + ' is ok!')
'''
'''
metapath = 'release/2014/acl-metadata.txt'

datadict, metadata = read_metadata(metapath)

graphpath = 'graph.txt'

graphdata = read_graphdata(graphpath)

train, test, val = 0, 0, 0

for edge in graphdata:
    if datadict[edge['from']]['year'] > 2012 and datadict[edge['from']]['year'] == datadict[edge['to']]['year']:
        continue
    if datadict[edge['from']]['year'] == 2014:
        test += 1
    elif datadict[edge['from']]['year'] > 2012:
        val += 1
    else:
        train += 1
print(train)
print(val)
print(test)

    
    # if datadict[edge['from']]['year'] == 2014 and datadict[edge['to']]['year'] < 2014:
        # test_cnt += 1
'''
'''
metapath = 'release/2014/acl-metadata.txt'

datadict, metadata = read_metadata(metapath)

# wrypath = "keyword_title.txt"

# keyword_title(metadata, wrypath)


paperpath = 'papers_text'

paperpathlist, paperlist = get_paperlist(datadict, paperpath)

graphpath = 'release/2014/acl.txt'

graphdata = read_graphdata(graphpath)

# test_cnt = 0

new_graphdata = []
wry_newgraph = ''

for edge in graphdata:
    if not edge['from'] in paperlist:
        continue
    if not edge['to'] in paperlist:
        continue
    if datadict[edge['from']]['year'] < datadict[edge['to']]['year']:
        print("from {} to {}.".format(edge['from'], edge['to']))
        continue
    new_graphdata.append(edge)
    wry_newgraph = wry_newgraph + edge['from'] + " ==> " + edge['to'] + '\n'
    # if datadict[edge['from']]['year'] == 2014 and datadict[edge['to']]['year'] < 2014:
        # test_cnt += 1

new_graphpath = 'graph.txt'
p = wry(wry_newgraph, new_graphpath)

##  calculate new graph stats
node_list, node_cnt, edge_cnt = [], 0, 0
for edge in new_graphdata:
    edge_cnt += 1
    if not edge['from'] in node_list:
        node_list.append(edge['from'])
        node_cnt += 1
    if not edge['to'] in node_list:
        node_list.append(edge['to'])
        node_cnt += 1
print("node numbers: {}\nedge numbers: {}\n".format(node_cnt, edge_cnt))


cnt = 0

for paper in paperpathlist:
    cnt = cnt + 1
    
    textrank(paper)
    if cnt % 100 == 0:
        print(str(cnt) + 'papers done!')
'''