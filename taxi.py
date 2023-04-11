from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, recall_score, ndcg_score

import codecs

import numpy as np

import nltk
import nltk.tokenize as tk
import nltk.stem.porter as pt

a = torch.tensor([[1, 2, 3], [4, 7, 6]])
_, b = a.topk(2, dim=1, largest=True)
print(b)

# nltk.download('punkt')
# a = np.array([[0.1, 0.8, 0.7, 0.3, 0.5, 0.34], [0.4, 0.6, 0.7, 0.3, 0.5, 0.55]])
# b = np.array([[1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1]])
'''
a1 = np.array([0.1, 0.8, 0.7, 0.3, 0.5, 0.34])
a2 = np.array([0.6, 0.4, 0.41, 0.35, 0.51, 0.3])
a3 = np.array([0.4, 0.3, 0.5, 0.6, 0.31, 0.580])
b1 = np.array([1, 0, 0, 1, 1, 1])
b2 = np.array([1, 0, 1, 1, 0, 0])
b3 = np.array([0, 0, 1, 1, 1, 1])
print(ndcg_score(np.array([b1]), np.array([a1])))
print(ndcg_score(np.array([b2]), np.array([a2])))
print(ndcg_score(np.array([b3]), np.array([a3])))
print(ndcg_score(np.array([b1, b2, b3]), np.array([a1, a2, a3])))
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