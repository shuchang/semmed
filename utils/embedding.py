import numpy as np
from tqdm import tqdm

__all__ = ['make_embedding']


# def load_vectors(path, skip_head=False, add_special_tokens=None, random_state=0):
#     vocab = []
#     vectors = None
#     nrow = sum(1 for line in open(path, 'r', encoding='utf-8'))
#     with open(path, "r", encoding="utf8") as fin:
#         if skip_head:
#             fin.readline()
#         for i, line in tqdm(enumerate(fin), total=nrow):
#             elements = line.strip().split(" ")
#             word = elements[0].lower()
#             vec = np.array(elements[1:], dtype=float)
#             vocab.append(word)
#             if vectors is None:
#                 vectors = np.zeros((nrow, len(vec)), dtype=np.float64)
#             vectors[i] = vec

#     np.random.seed(random_state)
#     n_special = 0 if add_special_tokens is None else len(add_special_tokens)
#     add_vectors = np.random.normal(np.mean(vectors), np.std(vectors), size=(n_special, vectors.shape[1]))
#     vectors = np.concatenate((vectors, add_vectors), 0)
#     vocab += add_special_tokens
#     return vocab, vectors



# def load_vectors_from_npy_with_vocab(glove_npy_path, glove_vocab_path, vocab, verbose=True, save_path=None):
#     with open(glove_vocab_path, 'r') as fin:
#         glove_w2idx = {line.strip(): i for i, line in enumerate(fin)}
#     glove_emb = np.load(glove_npy_path)
#     vectors = np.zeros((len(vocab), glove_emb.shape[1]), dtype=float)
#     oov_cnt = 0
#     for i, word in enumerate(vocab):
#         if word in glove_w2idx:
#             vectors[i] = glove_emb[glove_w2idx[word]]
#         else:
#             oov_cnt += 1
#     if verbose:
#         print(len(vocab))
#         print('embedding oov rate: {:.4f}'.format(oov_cnt / len(vocab)))
#     if save_path is None:
#         return vectors
#     np.save(save_path, vectors)


# def load_pretrained_embeddings(glove_npy_path, glove_vocab_path, vocab_path, verbose=True, save_path=None):
#     vocab = []
#     with open(vocab_path, 'r') as fin:
#         for line in fin.readlines():
#             vocab.append(line.strip())
#     load_vectors_from_npy_with_vocab(glove_npy_path=glove_npy_path, glove_vocab_path=glove_vocab_path, vocab=vocab, verbose=verbose, save_path=save_path)


def make_embedding(vocab_path, embeding_size, save_path):
    with open(vocab_path, "r", encoding="utf-8") as fin:
        vocab = [w.strip() for w in fin]
        embedding = np.random.uniform(-1, 1, [len(vocab), embeding_size])
        pad = np.zeros([1, embeding_size])
        embedding = np.concatenate([embedding, pad])
        print(embedding.shape)
        np.save(save_path, embedding)


if __name__ == "__main__":
    make_embedding("../data/semmed/cui_vocab.txt", 300, "../data/semmed/cui_embedding.npy")