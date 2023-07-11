import numpy as np
import torch


folder2 = "weights/skipgram_WikiText2"
folder = "weights/cbow_WikiText2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(f"{folder}/model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")

# embedding from first model layer
embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()
print(embeddings.shape) # (vocab_size, 300)

# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
print(embeddings_norm.shape) # (vocab_size, 300)

# get token order
print(type(vocab.get_itos()), len(vocab.get_itos()), type(vocab.get_itos()[0])) # <class 'list'> 4099 <class 'str'>
print(vocab.get_itos())

# Find Similar Words
def get_top_similar(word: str, topN: int = 10):
    word_id = vocab[word]
    print(f"word_id: {word_id}")
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings_norm[word_id]
    # print(word_vec.shape) # (300,)
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    # print(word_vec.shape) # (300, 1)
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    # print(type(dists), dists.shape) # <class 'numpy.ndarray'> (4099,)
    topN_ids = np.argsort(-dists)[1 : topN + 1]
    # print(topN_ids)

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        topN_dict[sim_word] = dists[sim_word_id]
    
    print(topN_dict)
    return topN_dict


for word, sim in get_top_similar("father").items(): # gernamy
    print("{}: {:.3f}".format(word, sim))


# Vector Equations
print("\n\nVector Equations")
emb1 = embeddings[vocab["bigger"]]
emb2 = embeddings[vocab["big"]]
emb3 = embeddings[vocab["small"]]

emb4 = emb1 - emb2 + emb3
emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
emb4 = emb4 / emb4_norm

emb4 = np.reshape(emb4, (len(emb4), 1))
dists = np.matmul(embeddings_norm, emb4).flatten()

top5 = np.argsort(-dists)[:5]
print(top5)

for word_id in top5:
    print("{}: {:.3f}".format(vocab.lookup_token(word_id), dists[word_id]))
