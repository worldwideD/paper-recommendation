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

from sentence_transformers import SentenceTransformer

# nltk.download('punkt')
# a = np.array([[0.1, 0.8, 0.7, 0.3, 0.5, 0.34], [0.4, 0.6, 0.7, 0.3, 0.5, 0.55]])
# b = np.array([[1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1]])

x = torch.tensor([1, 2])
y = torch.tensor([3, 3])
z = torch.stack([x, y])
print(z)

a = torch.FloatTensor([[1, 2], [2, 3]])
b = torch.FloatTensor([[1, 2], [3, 3]])
c = torch.FloatTensor([[3, 3], [4, 4]])
l = torch.stack([a, b, c], dim=0)
l1 = torch.mean(l, dim=1)
print(l1)
l2 = l.view(3, -1)
print(l2)

'''
a = torch.tensor([0.93, 0.94, 0.91, 0.92, 0.6, 0.5, 0.8, 0.7])
b = torch.LongTensor([2, 3, 0, 3, 3, 0, 1, 2])
print(retrieval_normalized_dcg(a, b, 6))
'''
'''

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

text1 = "language information retrieval system"
text2 = "machine translation"
text3 = "performance * coreference * resolution"

mask_text1 = '[CLS]' + text1 + '[SEP]'
mask_text2 = '[CLS]' + text2 + '[SEP]'
mask_text3 = '[CLS]' + text3 + '[SEP]'
mask_t = '[CLS]' + text1 + '[SEP]' + '[CLS]' + text2 + '[SEP]'

model = SentenceTransformer('bert-base-nli-mean-tokens')
rep = model.encode([text1, text2, text3], convert_to_tensor=True)
print(torch.cosine_similarity(rep[0], rep[1], dim=0))

tokenized_text1 = tokenizer.tokenize(mask_text1)
tokenized_text2 = tokenizer.tokenize(mask_text2)
tokenized_text3 = tokenizer.tokenize(mask_text3)
print(tokenized_text3)
'''
exit()
tokenized_t = tokenizer.tokenize(mask_t)

indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
indexed_tokens3 = tokenizer.convert_tokens_to_ids(tokenized_text3)
indexed_t = tokenizer.convert_tokens_to_ids(tokenized_t)
tokenizer.build_inputs_with_special_tokens
segments_ids1 = [1] * len(tokenized_text1)
segments_ids2 = [1] * len(tokenized_text2)
segments_ids3 = [1] * len(tokenized_text3)
segments_ids = [0] * len(tokenized_text1) + [1] * (len(tokenized_text2) )

tokens_tensor1 = torch.tensor([indexed_tokens1, ])
segments_tensors1 = torch.tensor([segments_ids1, ])
tokens_tensor2 = torch.tensor([indexed_tokens2, ])
segments_tensors2 = torch.tensor([segments_ids2, ])
tokens_tensor3 = torch.tensor([indexed_tokens3, ])
segments_tensors3 = torch.tensor([segments_ids3, ])
tokens_t = torch.tensor([indexed_t, ])
segments_t = torch.tensor([segments_ids, ])

model = AutoModel.from_pretrained('bert-base-cased')
model.eval()

with torch.no_grad():
    output1 = model(input_ids=tokens_tensor1, attention_mask=segments_tensors1, output_attentions=True, )
    sequence_output1 = output1[0]

    output2 = model(input_ids=tokens_tensor2, attention_mask=segments_tensors2, output_attentions=True, )
    sequence_output2 = output2[0]

    output3 = model(input_ids=tokens_tensor3, attention_mask=segments_tensors3, output_attentions=True, )
    sequence_output3 = output3[0]

    output = model(input_ids=tokens_t, attention_mask=segments_t, output_attentions=True)
    sequence_output = output[0]
    print(torch.cosine_similarity(sequence_output1[0][1], sequence_output2[0][1], dim=0))
    print(torch.cosine_similarity(sequence_output1[0][1], sequence_output3[0][1], dim=0))
    print(torch.cosine_similarity(sequence_output2[0][1], sequence_output3[0][1], dim=0))
    print(torch.cosine_similarity(sequence_output[0][1], sequence_output[0][len(tokenized_text1)+1], dim=0))
