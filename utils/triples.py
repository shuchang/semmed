import json
import pickle as pkl
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import torch
from transformers import (BertConfig, BertModel, BertTokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer, XLNetConfig,
                          XLNetModel, XLNetTokenizer)

try:
    from .semmed import relations
except ModuleNotFoundError:
    from semmed import relations



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
        # mapping = {i: (cui[i] for i range(len(cui)))} # index to corresponding grounded cui_idx




if __name__ == "__main__":
    pass
