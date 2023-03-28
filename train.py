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
from graph import generate_adj, h_hop_subgraph, generate_full_adj
from utils import set_seed, collate_fn
from model import PredictModel

def train(args, model, train_pos, val_set, test_set, train_adj, val_adj, test_adj, train_edges, train_nodes, x):
    total_steps = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_iterator = range(int(args.epochs))

    best_auc = 0.
    test_auc = 0.
    print("start training")
    for epoch in train_iterator:
        total_steps += 1
        # train neg
        train_neg = generate_neg(train_edges, train_nodes)
        train_set = np.concatenate([train_pos, train_neg], axis=0)

        # training
        model.train()
        inputs = {
            'src': train_set[:, 0],
            'dst': train_set[:, 1],
            'labels': train_set[:, 2],
            'adj': train_adj,
            'x': x,
        }
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        wandb.log({"loss": loss.item()}, step = total_steps)
        loss.backward()
        optimizer.step()
        auc = evaluate(args, model, val_set, val_adj, x)
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
            test_auc = evaluate(args, model, test_set, test_adj, x)
    
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

'''
def train(args, model, train_set, val_set, test_set, train_adj, val_adj, test_adj, x):
    total_steps = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_iterator = range(int(args.epochs))

    best_auc = 0.
    test_auc = 0.
    print("start training")
    for epoch in train_iterator:
        model.train()
        total_loss, steps = 0., 0
        for step, batch in enumerate(train_dataloader):
            total_steps += 1
            steps += 1
            inputs = {
                'src': batch[0],
                'dst': batch[1],
                'labels': batch[2],
                'adj': train_adj,
                'x': x,
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[0]
            # wandb.log({"loss": loss.item()}, step = total_steps)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / steps
        print("after {} epochs, avg loss: {}".format(epoch+1, avg_loss))

        auc = evaluate(args, model, val_set, val_adj, x)
        print("AUC score: {}".format(auc))
        if auc > best_auc:
            best_auc = auc
            test_auc = evaluate(args, model, test_set, test_adj, x)
    
    print("best validate AUC score: {}".format(best_auc))
    print("test AUC score: {}".format(test_auc))

def evaluate(args, model, data, adj, x):
    eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, labels = [], []
    model.eval()
    for batch in eval_dataloader:
        inputs = {
                'src': batch[0],
                'dst': batch[1],
                'labels': batch[2],
                'adj': adj,
                'x': x,
            }
        label = np.array(batch[2])
        labels.append(label)
        with torch.no_grad():
            _, logits, _ = model(**inputs)
            pred = torch.sigmoid(logits)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.float32)
    auc = roc_auc_score(labels, preds)
    return auc
'''
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
    
    args = parser.parse_args()
    wandb.init(project="paper_recommendation")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args)

    # load dataset
    msg_edges, train_pos, train_nodes, val_pos, val_neg, test_pos, test_neg, title_dict, text_dict = read_data(
        args.graph, args.metadata, args.title, args.textkey_dir)
    
    # get graphs
    n = len(title_dict)
    train_adj = generate_full_adj(msg_edges, n)
    val_adj = generate_full_adj(msg_edges + train_pos, n)
    test_adj = generate_full_adj(msg_edges + train_pos + val_pos, n)

    # get vectors
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    Bertmodel = AutoModel.from_pretrained('bert-base-cased')
    Bertmodel = Bertmodel.to(args.device)
    Bertmodel.eval()
    
    reps = []
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

    # get datas
    '''
    train_set, val_set, test_set = [], [], []
    for edge in train_pos:
        train_set.append({'src': edge[0], 'dst': edge[1], 'label': 1})
    for edge in train_neg:
        train_set.append({'src': edge[0], 'dst': edge[1], 'label': 0})
    
    for edge in val_pos:
        val_set.append({'src': edge[0], 'dst': edge[1], 'label': 1})
    for edge in val_neg:
        val_set.append({'src': edge[0], 'dst': edge[1], 'label': 0})
    
    for edge in test_pos:
        test_set.append({'src': edge[0], 'dst': edge[1], 'label': 1})
    for edge in test_neg:
        test_set.append({'src': edge[0], 'dst': edge[1], 'label': 0})
    '''
    train_set, val_set, test_set = [], [], []
    for edge in train_pos:
        train_set.append([edge[0], edge[1], 1])
    # for edge in train_neg:
    #     train_set.append([edge[0], edge[1], 0])
    
    for edge in val_pos:
        val_set.append([edge[0], edge[1], 1])
    for edge in val_neg:
        val_set.append([edge[0], edge[1], 0])
    
    for edge in test_pos:
        test_set.append([edge[0], edge[1], 1])
    for edge in test_neg:
        test_set.append([edge[0], edge[1], 0])
    
    train_set = np.array(train_set)
    val_set = np.array(val_set)
    test_set = np.array(test_set)
    model = PredictModel(in_feats=768, h_feats=args.hidden_size, h_hops=args.hops, n_layers=args.layers, dropout=args.dropout)
    model = model.to(device)

    train(args, model, train_set, val_set, test_set, train_adj, val_adj, test_adj, train_pos, train_nodes, reps)

if __name__ == "__main__":
    main()