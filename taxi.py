from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

import codecs

import numpy as np

import nltk
import nltk.tokenize as tk
import nltk.stem.porter as pt

# nltk.download('punkt')

stemmer = pt.PorterStemmer()

str1 = "Ambiguity threshold"
str2 = "ambiguity instances"
str3 = "ambiguity thresholds"
tokens1 = tk.word_tokenize(str1)
tokens2 = tk.word_tokenize(str2)
tokens3 = tk.word_tokenize(str3)
# print([stemmer.stem(c) for c in tokens1])
stem1 = [stemmer.stem(c) for c in tokens1]
stem2 = [stemmer.stem(c) for c in tokens2]
stem3 = [stemmer.stem(c) for c in tokens3]
print(stem3 in [stem1, stem2])
print(stem2 in [stem1, stem3])

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