import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, ndcg_score
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics import AUROC, Recall
import wandb

from reader import read_data, generate_neg
from sim import text2vec
from tqdm import tqdm
from graph import generate_adj, h_hop_subgraph, generate_full_adj, arrange_id
from utils import set_seed, collate_fn, gen_test_labels
from model import PredictModel

def train(args, model, train_pos, val_set, test_set, train_adj, val_adj, test_adj, msg_edges, train_x, val_x, test_x):
    total_steps = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_iterator = range(int(args.epochs))

    best, best_auc, best_ndcg, best_recall = 0., 0. ,0., 0.
    test_auc = 0.
    test_ndcg = 0.
    test_recall = 0.
    print("start training")
    for epoch in train_iterator:
        total_steps += 1
        # train neg
        train_neg = generate_neg(train_adj.shape[0], msg_edges, train_pos, train_pos.shape[0])
        train_set = np.concatenate([train_pos, train_neg], axis=0)

        # training
        model.train()
        inputs = {
            'src': train_set[:, 0],
            'dst': train_set[:, 1],
            'labels': train_set[:, 2],
            'adj': train_adj,
            'x': train_x,
            'mode': "train",
        }
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        wandb.log({"loss": loss.item()}, step = total_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch == args.epochs-1:
            auc, ndcg, recall = evaluate(args, model, val_set, val_adj, val_x, 1)
        auc, ndcg, recall = evaluate(args, model, val_set, val_adj, val_x, 0)
        wandb.log({"dev AUC": auc, "dev ndcg": ndcg, "dev recall": recall}, step = total_steps)
        if epoch >= args.epochs // 2 and ndcg > best:
            best = ndcg
            best_auc, best_ndcg, best_recall = auc, ndcg, recall
            test_auc, test_ndcg, test_recall = evaluate(args, model, test_set, test_adj, test_x, 0)

        if (epoch + 1) % 100 == 0:
            print("after {} epochs, loss: {}".format(epoch+1, loss.item()))
            print("validate AUC score: {}    ndcg: {}    recall: {}".format(auc, ndcg, recall))
    
    print("best validate AUC score: {}   best ndcg: {}   best recall: {}".format(best_auc, best_ndcg, best_recall))
    print("test AUC score: {}    test ndcg: {}    test recall: {}".format(test_auc, test_ndcg, test_recall))

def evaluate(args, model, data, adj, x, flag):
    score, labels = [], data[2]
    input_labels = labels.reshape(-1)
    model.eval()
    inputs = {
        'src': data[0],
        'dst': data[1],
        'labels': input_labels,
        'adj': adj,
        'x': x,
        'mode': "eval",
    }
    with torch.no_grad():
        _, logits = model(**inputs)
        score = torch.sigmoid(logits)
        score[torch.isnan(score)] = 0
    # calculate auc
    labels = torch.tensor(input_labels).to(score)
    AUC = AUROC(task="binary")
    auc = AUC(score, labels).item()
    # calculate ndcg
    n, m = data[0].shape[0], data[1].shape[0]
    labels = labels.view(n, m)
    score = score.view(n, m)
    top_n = args.ndcg_top_n
    thres = score.topk(top_n, largest=True, dim=1, sorted=True)[0][..., -1, None]
    preds = score >= thres
    preds = preds.to(labels)
    recall = Recall(task="binary").to(labels)
    rec = recall(preds, labels).item()
    # ids = torch.arange(n) * m
    # ids = ids.to(score).type(torch.long).unsqueeze(1).repeat(1, top_n)
    # index = (index + ids).view(-1)
    # l = labels.index_select(0, index).view(n, -1)
    # ndcg_list = [retrieval_normalized_dcg(sc[i], l[i]) for i in range(n)]
    ndcg_list = [retrieval_normalized_dcg(score[i], labels[i], top_n) for i in range(n)]
    ndcg = torch.mean(torch.stack(ndcg_list)).item()
    '''
    input_labels = input_labels.astype(np.float32)
    n, m = data[0].shape[0], data[1].shape[0]
    score = score.view(n, m)
    labels = torch.tensor(input_labels).to(score)
    top_n = 30
    sc, index = score.topk(top_n, largest=True)
    ids = torch.arange(n) * m
    ids = ids.to(score).type(torch.long).unsqueeze(1).repeat(1, top_n)
    index = (index + ids).view(-1)
    l = labels.index_select(0, index).view(n, -1)
    l = l.cpu().numpy()
    sc = sc.cpu().numpy()
    ndcg = ndcg_score(l, sc)
    score = score.view(-1).cpu().numpy()
    auc = roc_auc_score(input_labels, score)
    '''
    return auc, ndcg, rec

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph", default="graph.txt", type=str)
    parser.add_argument("--metadata", default="acl-metadata.txt", type=str)
    parser.add_argument("--title", default="keyword_title.txt", type=str)
    parser.add_argument("--textkey_dir", default="./papers_keyword", type=str)

    parser.add_argument("--hops", default=2, type=int, help="hops of subgraphs.")
    parser.add_argument("--layers", default=3, type=int, help="num of layers of gnn.")
    parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate.")
    parser.add_argument("--epochs", default=50.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=2333, help="random seed for initialization")
    parser.add_argument("--dropout", default=0.5, type=float, help="drop-out rate")
    parser.add_argument("--hidden_size", default=256, type=int, help="hidden state dimension")
    parser.add_argument("--top_k", default=5, type=int, help="top k similar nodes")
    parser.add_argument("--ndcg_top_n", default=50, type=int, help="n for NDCG@N")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--heads", default=4, type=int, help="number of attention heads for GAT")
    parser.add_argument("--gpu", default="0", type=str, help="gpu device number")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    wandb.init(project="paper_recommendation")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args)

    # load dataset
    msg_edges, train_pos, train_nodes, val_pos, val_neg, val_nodes, test_pos, test_neg, test_nodes, title_dict, text_dict = read_data(
        args.graph, args.metadata, args.title, args.textkey_dir)

    # get vectors
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    Bertmodel = AutoModel.from_pretrained('bert-base-cased')
    Bertmodel = Bertmodel.to(args.device)
    Bertmodel.eval()
    
    reps = []
    n = len(title_dict)
    for i in tqdm(range(n), desc="vecs"):
        text = [title_dict[i]['title']] + text_dict[i]['keyphrases']
        vec = text2vec(text, tokenizer, Bertmodel, device) # [texts, emb_size]
        vec[torch.isnan(vec)] = 0.
        values = torch.tensor(text_dict[i]['value'], dtype=torch.float).unsqueeze(0).to(device)
        
        text_rep = torch.mm(values, vec[1:])
        title_rep = vec[0].unsqueeze(0)
        v_total = values.sum()
        text_rep = text_rep / v_total
        if v_total.item() != 0:
            rep = title_rep * 0.5 + text_rep * 0.5
        else:
            rep = title_rep
        reps.append(rep)
    reps = torch.stack(reps, dim=0).squeeze(1)

    # get graphs
    # train set
    msg, train_set, val_set, test_set = [], [], [], []
    train_cnt, val_cnt, test_cnt = len(train_nodes), len(val_nodes), len(test_nodes)
    train_dict = arrange_id(train_nodes)
    for edge in train_pos:
        train_set.append([train_dict[edge[0]], train_dict[edge[1]], 1])
    for edge in msg_edges:
        msg.append([train_dict[edge[0]], train_dict[edge[1]]])
    train_adj = generate_full_adj(msg_edges, train_dict)
    ids = torch.zeros(len(train_dict)).type(torch.int).to(args.device)
    for k, v in train_dict.items():
        ids[v] = k
    train_x = torch.index_select(reps, 0, ids)

    # val set
    val_dict = arrange_id(val_nodes)
    for edge in val_pos:
        val_set.append([val_dict[edge[0]], val_dict[edge[1]]])
    val_adj = generate_full_adj(msg_edges + train_pos, val_dict)
    ids = torch.zeros(len(val_dict)).type(torch.int).to(args.device)
    for k, v in val_dict.items():
        ids[v] = k
    val_x = torch.index_select(reps, 0, ids)
    val_set = (np.arange(train_cnt, val_cnt), np.arange(0, train_cnt), gen_test_labels(val_cnt - train_cnt, train_cnt, val_set))

    # test set
    test_dict = arrange_id(test_nodes)
    for edge in test_pos:
        test_set.append([test_dict[edge[0]], test_dict[edge[1]]])
    test_adj = generate_full_adj(msg_edges + train_pos + val_pos, test_dict)
    ids = torch.zeros(len(test_dict)).type(torch.int).to(args.device)
    for k, v in test_dict.items():
        ids[v] = k
    test_x = torch.index_select(reps, 0, ids)
    test_set = (np.arange(val_cnt, test_cnt), np.arange(0, val_cnt), gen_test_labels(test_cnt - val_cnt, val_cnt, test_set))
    
    
    train_set = np.array(train_set)
    msg = np.array(msg)
    model = PredictModel(
        in_feats=768, h_feats=args.hidden_size, h_hops=args.hops, n_layers1=args.layers, n_layers2=args.layers, nheads=args.heads, dropout=args.dropout, top_k=args.top_k)
    model = model.to(device)

    train(args, model, train_set, val_set, test_set, train_adj, val_adj, test_adj, msg, train_x, val_x, test_x)

if __name__ == "__main__":
    main()