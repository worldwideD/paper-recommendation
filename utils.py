import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def gen_test_labels(n, m, pos):
    l = np.zeros(shape=(n, m))
    for p in pos:
        l[p[0]-m, p[1]] = 1
    return l

def collate_fn(batch):
    src = [f['src'] for f in batch]
    dst = [f['dst'] for f in batch]
    labels = [f['label'] for f in batch]
    output = (src, dst, labels)
    return output