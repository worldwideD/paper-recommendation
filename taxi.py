from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, recall_score, ndcg_score, roc_auc_score

import codecs

import numpy as np

import nltk
import nltk.tokenize as tk
import nltk.stem.porter as pt
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics import AUROC, Recall

# nltk.download('punkt')
# a = np.array([[0.1, 0.8, 0.7, 0.3, 0.5, 0.34], [0.4, 0.6, 0.7, 0.3, 0.5, 0.55]])
# b = np.array([[1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1]])

a1 = np.array([0.1, 0.8, 0.7, 0.3, 0.5, 0.34])
a2 = np.array([0.6, 0.4, 0.41, 0.35, 0.51, 0.3])
a3 = np.array([0.4, 0.3, 0.5, 0.6, 0.31, 0.580])
b1 = np.array([1, 0, 0, 1, 1, 1])
b2 = np.array([1, 0, 1, 1, 0, 0])
b3 = np.array([0, 0, 1, 1, 1, 1])
print(ndcg_score(np.array([b1]), np.array([a1]), k=3))
print(ndcg_score(np.array([b2]), np.array([a2]), k=3))
print(ndcg_score(np.array([b3]), np.array([a3]), k=3))
print(ndcg_score(np.array([b1, b2, b3]), np.array([a1, a2, a3]), k=3))

a1 = torch.tensor(a1)
a2 = torch.tensor(a2)
a3 = torch.tensor(a3)
b1 = torch.LongTensor(b1)
b2 = torch.LongTensor(b2)
b3 = torch.LongTensor(b3)
print(retrieval_normalized_dcg(a1, b1, k=3))
print(retrieval_normalized_dcg(a2, b2, k=3))
print(retrieval_normalized_dcg(a3, b3, k=3))
preds = torch.stack([a1, a2, a3], 0)
target = torch.stack([b1, b2, b3], 0)
k = 3
# print(retrieval_normalized_dcg(preds, target, k))
# print(torch.mean(torch.stack([retrieval_normalized_dcg(a, b) for a, b in zip([a1, a2, a3], [b1, b2, b3])])))

recall = Recall(task="binary")
p = torch.LongTensor([[1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0]])
t = torch.LongTensor([[0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]])
print(recall(p, t))
'''
a = torch.tensor([0.93, 0.94, 0.91, 0.92, 0.6, 0.5, 0.8, 0.7])
b = torch.LongTensor([2, 3, 0, 3, 3, 0, 1, 2])
print(retrieval_normalized_dcg(a, b, 6))
'''
'''
with codecs.open('keyword_title.txt', 'r', 'utf-8', 'ignore') as f:
    lines = f.readlines()

    

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

text1 = "language information retrieval system"
text2 = "machine translation"

mask_text1 = '[CLS]' + text1 + '[SEP]'
mask_text2 = '[CLS]' + text2 + '[SEP]'

tokenized_text1 = tokenizer.tokenize(mask_text1)
tokenized_text2 = tokenizer.tokenize(mask_text2)

indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)	

# print(tokenized_text)

# for tup in zip(tokenized_text, indexed_tokens):
#     print(tup)

segments_ids1 = [1] * len(tokenized_text1)
segments_ids2 = [1] * len(tokenized_text2)

tokens_tensor1 = torch.tensor([indexed_tokens1, ])
segments_tensors1 = torch.tensor([segments_ids1, ])
tokens_tensor2 = torch.tensor([indexed_tokens2, ])
segments_tensors2 = torch.tensor([segments_ids2, ])


model = AutoModel.from_pretrained('bert-base-cased')
model.eval()

with torch.no_grad():
    output1 = model(input_ids=tokens_tensor1, attention_mask=segments_tensors1, output_attentions=True, )
    sequence_output1 = output1[0]

    output2 = model(input_ids=tokens_tensor2, attention_mask=segments_tensors2, output_attentions=True, )
    sequence_output2 = output2[0]
    print(torch.cosine_similarity(sequence_output1[0][1], sequence_output2[0][1], dim=0))
    '''