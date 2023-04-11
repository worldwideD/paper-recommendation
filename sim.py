import torch

def text2vec(texts, tokenizer, model, device):
    sum = len(texts)
    seqlen = 2
    for text in texts:
        mask_text = '[CLS]' + text + '[SEP]'
        tokenized_text = tokenizer.tokenize(mask_text)
        seqlen = max(seqlen, len(tokenized_text))
        
    lens, indexs, segs = [], [], []

    for text in texts:
        mask_text = '[CLS]' + text + '[SEP]'
        tokenized_text = tokenizer.tokenize(mask_text)
        
        l = len(tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) + [0] * (seqlen - l)
        segments_ids = [1.] * l + [0.] * (seqlen - l)


        lens.append(l)
        indexs.append(indexed_tokens)
        segs.append(segments_ids)
    
    indexs = torch.tensor(indexs, dtype=torch.long).to(device)
    segs = torch.tensor(segs, dtype=torch.float).to(device)

    vecs = []
    with torch.no_grad():
        output = model(input_ids=indexs, attention_mask=segs, output_attentions=True, )
        sequence_output = output[0]

        for i in range(sum):
            vec = torch.mean(sequence_output[i][1:lens[i]-1][:], dim=0)
            vecs.append(vec)
    
    vecs = torch.stack(vecs, dim=0).to(device)
    return vecs

def calc_similarity(h):
    n = h.shape[0]
    sim = [torch.cosine_similarity(h[i].unsqueeze(0), h) for i in range(n)]
    sim = torch.stack(sim, dim=0)
    return sim