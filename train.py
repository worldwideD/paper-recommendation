import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score
import wandb

from reader import read_data, generate_neg
from sim import text2vec
from tqdm import tqdm
from graph import generate_adj, h_hop_subgraph, generate_full_adj, arrange_id
from utils import set_seed, collate_fn
from model import PredictModel

def train(args, model, train_pos, val_set, test_set, train_adj, val_adj, test_adj, msg_edges, train_x, val_x, test_x):
    total_steps = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_iterator = range(int(args.epochs))

    best_auc = 0.
    test_auc = 0.
    print("start training")
    for epoch in train_iterator:
        total_steps += 1
        # train neg
        train_neg = generate_neg(train_adj.shape[0], msg_edges, train_pos)
        train_set = np.concatenate([train_pos, train_neg], axis=0)

        # training
        model.train()
        inputs = {
            'src': train_set[:, 0],
            'dst': train_set[:, 1],
            'labels': train_set[:, 2],
            'adj': train_adj,
            'x': train_x,
        }
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        wandb.log({"loss": loss.item()}, step = total_steps)
        loss.backward()
        optimizer.step()
        auc = evaluate(args, model, val_set, val_adj, val_x)
        wandb.log({"dev AUC": auc}, step = total_steps)

        if (epoch + 1) % 100 == 0:
            '''
            logits = outputs[1]
            preds = torch.sigmoid(logits)
            preds = preds.cpu().detach().numpy()
            preds[np.isnan(preds)] = 0
            labels = train_set[:, 2].astype(np.float32)
            t_auc = roc_auc_score(labels, preds)
            print("train AUC score: {}".format(t_auc))
            '''
            print("after {} epochs, loss: {}".format(epoch+1, loss.item()))
            print("validate AUC score: {}".format(auc))
        
        if auc > best_auc:
            best_auc = auc
            test_auc = evaluate(args, model, test_set, test_adj, test_x)
    
    print("best validate AUC score: {}".format(best_auc))
    print("test AUC score: {}".format(test_auc))

def evaluate(args, model, data, adj, x):
    preds, labels = [], data[:, 2]
    model.eval()
    inputs = {
        'src': data[:, 0],
        'dst': data[:, 1],
        'labels': labels,
        'adj': adj,
        'x': x,
    }
    with torch.no_grad():
        _, logits, _ = model(**inputs)
        preds = torch.sigmoid(logits)
        preds = preds.cpu().numpy()
        preds[np.isnan(preds)] = 0
    labels = labels.astype(np.float32)
    auc = roc_auc_score(labels, preds)
    return auc

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
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--heads", default=4, type=int, help="number of attention heads for GAT")
    
    args = parser.parse_args()
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
        # text = [title_dict[i]['title']] + title_dict[i]['keyphrases'] + text_dict[i]['keyphrases']
        text = [title_dict[i]['title']] + text_dict[i]['keyphrases']
        vec = text2vec(text, tokenizer, Bertmodel, device) # [texts, emb_size]
        vec[torch.isnan(vec)] = 0.
        # sep = len(title_dict[i]['keyphrases'])+1
        # values = torch.tensor([1.] + title_dict[i]['value'] + text_dict[i]['value'], dtype=torch.float).unsqueeze(0).to(device)
        values = torch.tensor(text_dict[i]['value'], dtype=torch.float).unsqueeze(0).to(device)
        '''
        rep = torch.mm(values, vec)
        v_total = values.sum()
        rep = rep / v_total
        '''
        text_rep = torch.mm(values, vec[1:])
        title_rep = vec[0].unsqueeze(0)
        v_total = values.sum()
        text_rep = text_rep / v_total
        if v_total.item() != 0:
            rep = title_rep * 0.5 + text_rep * 0.5
        else:
            rep = title_rep
        reps.append(rep)
        # title_dict[i]['titlevec'] = vec[0]
        # title_dict[i]['keyvec'] = vec[1:sep]
        # text_dict[i]['vec'] = vec[sep:]
    reps = torch.stack(reps, dim=0).squeeze(1)

    # get graphs
    msg, train_set, val_set, test_set = [], [], [], []
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

    val_dict = arrange_id(val_nodes)
    for edge in val_pos:
        val_set.append([val_dict[edge[0]], val_dict[edge[1]], 1])
    for edge in val_neg:
        val_set.append([val_dict[edge[0]], val_dict[edge[1]], 0])
    val_adj = generate_full_adj(msg_edges + train_pos, val_dict)
    ids = torch.zeros(len(val_dict)).type(torch.int).to(args.device)
    for k, v in val_dict.items():
        ids[v] = k
    val_x = torch.index_select(reps, 0, ids)

    test_dict = arrange_id(test_nodes)
    for edge in test_pos:
        test_set.append([test_dict[edge[0]], test_dict[edge[1]], 1])
    for edge in test_neg:
        test_set.append([test_dict[edge[0]], test_dict[edge[1]], 0])
    test_adj = generate_full_adj(msg_edges + train_pos + val_pos, test_dict)
    ids = torch.zeros(len(test_dict)).type(torch.int).to(args.device)
    for k, v in test_dict.items():
        ids[v] = k
    test_x = torch.index_select(reps, 0, ids)
    
    train_set = np.array(train_set)
    val_set = np.array(val_set)
    test_set = np.array(test_set)
    msg = np.array(msg)
    model = PredictModel(
        in_feats=768, h_feats=args.hidden_size, h_hops=args.hops, n_layers1=args.layers, n_layers2=args.layers, nheads=args.heads, dropout=args.dropout, top_k=args.top_k)
    model = model.to(device)

    train(args, model, train_set, val_set, test_set, train_adj, val_adj, test_adj, msg, train_x, val_x, test_x)

if __name__ == "__main__":
    main()