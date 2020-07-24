import json
import pickle as pkl
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


try:
    from .semmed import relations
except ModuleNotFoundError:
    from semmed import relations
try:
    from .utils import check_path
except:
    from utils import check_path


cui2idx = None
idx2cui = None
relation2idx = None
idx2relation = None


def load_resources(semmed_cui_path):
    global cui2idx, idx2cui, relation2idx, idx2relation

    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def generate_triples_from_adj(adj_pk_path, grounded_path, semmed_cui_path, triple_path):
    print(f"generating triples from {adj_pk_path}")

    global cui2idx, idx2cui, relation2idx, idx2relation
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)

    with open(grounded_path, "r") as fin:
        data = [json.loads(line) for line in fin]
    mentioned_cui = [([cui2idx[record] for record in item["record_cui"]] + \
                      [cui2idx[hf] for hf in item["hf_cui"]]) for item in data]

    with open(adj_pk_path, "rb") as fin:
        adj_cui_pairs = pkl.load(fin)

    nrow = len(adj_cui_pairs)
    triples = []
    mc_triple_num = []
    for idx, (adj_data, mc) in tqdm(enumerate(zip(adj_cui_pairs, mentioned_cui)),
                                    total=nrow, desc="loading adj matrices"):
        adj, cui, _, _ = adj_data
        mapping = {i: (cui[i]) for i in range(len(cui))} # index to corresponding grounded cui_idx
        ij = adj.row
        k = adj.col
        n_node = adj.shape[1]
        n_rel = 2 * adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node

        j = np.array([mapping[j[idx]] for idx in range(len(j))])
        k = np.array([mapping[k[idx]] for idx in range(len(k))])

        mc2mc_mask = np.isin(j, mc) & np.isin(k, mc)
        mc2nmc_mask = np.isin(j, mc) | np.isin(k, mc)
        others_mask = np.invert(mc2nmc_mask)
        mc2nmc_mask = ~mc2mc_mask & mc2nmc_mask
        mc2mc = i[mc2mc_mask], j[mc2mc_mask], k[mc2mc_mask]
        mc2nmc = i[mc2nmc_mask], j[mc2nmc_mask], k[mc2nmc_mask]
        others = i[others_mask], j[others_mask], k[others_mask]
        [i, j, k] = [np.concatenate((a, b, c), axis=-1) for (a, b, c) in zip(mc2mc, mc2nmc, others)]
        triples.append((i, j, k))
        mc_triple_num.append(len(mc2mc) + len(mc2nmc))

        # i, j, k = np.concatenate((i, i + n_rel // 2), 0), np.concatenate((j, k), 0), np.concatenate((k, j), 0)  # add inverse relations
        # mask = np.isin(j, mc)
        # inverted_mask = np.invert(mask)
        # masked = i[mask], j[mask], k[mask]
        # mc_triple_num.append(len(masked[0]))
        # remaining = i[inverted_mask], j[inverted_mask], k[inverted_mask]
        # [i, j, k] = [np.concatenate((m, r), axis=-1) for (m, r) in zip(masked, remaining)]
        # triples.append((i, j, k))  # i: relation, j: head, k: tail

    check_path(triple_path)
    with open(triple_path, 'wb') as fout:
        pkl.dump((triples, mc_triple_num), fout)

    print(f"Triples saved to {triple_path}")
    print()



if __name__ == "__main__":
    pass
