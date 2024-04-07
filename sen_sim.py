import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def cal_sen_sim(sentences):
    embeddings = model.encode(sentences)
    embed1 = torch.from_numpy(embeddings[0])
    embed2 = torch.from_numpy(embeddings[1])
    sim = torch.nn.functional.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0))
    return sim

def cal_sens_sim(sentences):
    sentence1 = sentences[0]
    sentence2 = sentences[1:]
    sims = []
    for i, sen in enumerate(sentence2):
        sims.append([cal_sen_sim([sentence1, sen]), i])
    return sims

def top_sim(similarity):
    sorted_sim = sorted(similarity, key=lambda x: x[0], reverse=True)
    return sorted_sim[0][1]

