import json
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from tqdm import tqdm

try:
    from .semmed import relations
except ModuleNotFoundError:
    from semmed import relations


# (33 pos, 31 neg): no neg_compared_with, neg_prep
# relations = ['administered_to', 'affects', 'associated_with', 'augments', 'causes', 'coexists_with', 'compared_with', 'complicates',
#              'converts_to', 'diagnoses', 'disrupts', 'higher_than', 'inhibits', 'isa', 'interacts_with', 'location_of', 'lower_than',
#              'manifestation_of', 'measurement_of', 'measures', 'method_of', 'occurs_in', 'part_of', 'precedes', 'predisposes', 'prep',
#              'prevents', 'process_of', 'produces', 'same_as', 'stimulates', 'treats', 'uses',
#              'neg_administered_to', 'neg_affects', 'neg_associated_with', 'neg_augments', 'neg_causes', 'neg_coexists_with',
#              'neg_complicates', 'neg_converts_to', 'neg_diagnoses', 'neg_disrupts', 'neg_higher_than', 'neg_inhibits', 'neg_isa',
#              'neg_interacts_with', 'neg_location_of', 'neg_lower_than', 'neg_manifestation_of', 'neg_measurement_of', 'neg_measures',
#              'neg_method_of', 'neg_occurs_in', 'neg_part_of', 'neg_precedes', 'neg_predisposes', 'neg_prevents', 'neg_process_of',
#              'neg_produces', 'neg_same_as', 'neg_stimulates', 'neg_treats', 'neg_uses']


cui2idx = None
idx2cui = None
relation2idx = None
idx2relation = None

semmed = None
semmed_simple = None

cui_embs = None
relation_embs = None


def load_resources(semmed_cui_path):
    global cui2idx, idx2cui, relation2idx, idx2relation

    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}


def load_semmed(semmed_graph_path):
    global semmed, semmed_simple
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, _ in semmed.edges(data=True):
        w = 1.0 # initialize weight to 1.0
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)


##################### path finding #####################


def get_edge(src_idx: int, tgt_idx: int) -> list:
    """
    find edges between source cui and target cui

    `params`:
        src_idx: index of the single source cui
        tgt_idx: index of the single target cui
    `return`:
        res: list of all relations with no repetition
    """
    global semmed
    if semmed.has_edge(src_idx, tgt_idx):
        rel_list = semmed[src_idx][tgt_idx]
        rel_seen = set()
        res = [r["rel"] for r in rel_list.values() if r["rel"] not in rel_seen and (rel_seen.add(r["rel"]) or True)]
    else:
        res = -1
    return res


def find_paths_between_single_cui_pair(src_cui: str, tgt_cui: str) -> list:
    """
    find paths for a (single_record_cui, single_hf_cui) pair

    `params`:
        src_cui: single source cui
        tgt_cui: single target cui
    `return`:
        pf_res: list of dictionaries with the form of {"path": p, "rel": rl}
    """
    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    src_idx = cui2idx[src_cui]
    tgt_idx = cui2idx[tgt_cui]

    # if src_idx not in semmed_simple.nodes() or tgt_idx not in semmed_simple.nodes():
    #     return

    all_path = []
    try:
        for p in nx.shortest_simple_paths(semmed_simple, source=src_idx, target=tgt_idx):
            if len(p) > 5 or len(all_path) >= 100:
                break
            if len(p) >= 2: # skip paths of length 1 (self-loop?)
                all_path.append(p)
    except nx.exception.NetworkXNoPath:
        pass

    pf_res = []
    for p in all_path:
        rl = [] # list of rel_list of all pair in this path
        for src in range(len(p) - 1): # num of pair in the path
            s = p[src]
            t = p[src + 1]

            rel_list = get_edge(s, t)
            if rel_list == -1:
                rl = -1
                break
        else:
            rl.append(rel_list)

        if rl == -1:
            continue
        pf_res.append({"path": p, "rel": rl})
    return pf_res


def find_paths_between_pair(pair: list) -> list:
    """
    find paths between a pair of record_cui and hf_cui

    `param`:
        pair: list of a record_cui and hf_cui pair
    `return`:
        pfr_pair: list of dictionaries with the form of
        {"record_cui": record_cui, "hf_cui": hf_cui, "pf_res": pf_res}
    """
    record_cui, hf_cui = pair
    pfr_pair = []
    for single_hf_cui in hf_cui:
        for single_record_cui in record_cui:
            pf_res = find_paths_between_single_cui_pair(single_record_cui, single_hf_cui)
            pfr_pair.append({"record_cui": record_cui, "hf_cui": hf_cui, "pf_res": pf_res})
    return pfr_pair


##################### path scoring #####################

# TODO: no pretrained embedding with TransE

# def score_triple(h, t, r, flag):

# def score_triples(cui_idx, rel_idx, debug=False):
#     global cui_embs, relation_embs, idx2cui, idx2relation
#     cui = cui_embs[cui_idx]
#     relation = []
#     flag = []
#     for i in range(len(rel_idx))
#     embs = []
#     l_flag = []


# def score_paths_between_pair(pfr_pair):
#     """
#     score paths between a pair of record_cui and hf_cui
#     """
#     # between a pair of cui, there are several paths
#     pair_scores = []
#     for pf_res in pfr_pair:
#         pair_paths = pf_res["pf_res"]
#         if pair_paths is not None:
#             path_scores = []
#             for path in pair_paths


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def find_paths(grounded_path, semmed_cui_path, semmed_graph_path, output_path, num_processes=8, random_state=0, debug=True):
    print(f'generating paths for {grounded_path}...')

    random.seed(random_state)
    np.random.seed(random_state)

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    with open(grounded_path, "r") as fin:
        data = [json.loads(line) for line in fin]

    if debug:
        data = data[0:8]

    data = [[item["record_cui"], item["hf_cui"]] for item in data]
    nrow = len(data)

    with Pool(num_processes) as p, open(output_path, "w") as fout:
        for pfr_pair in tqdm(p.imap(find_paths_between_pair, data), total=nrow):
            fout.write(json.dumps(pfr_pair) + "\n")
    # with open(output_path, "w") as fout:
    #     for row in tqdm(range(nrow)):
    #         pfr_pair = find_paths_between_pair(data[row])
    #         fout.write(json.dumps(pfr_pair) + "\n")

    print(f'paths saved to {output_path}')
    print()


# def score_paths(raw_paths_path, cui_emb_path, rel_emb_path, semmed_cui_path, output_path, num_processes=1, method="triple_cls"):
#     print(f"scoring paths for {raw_paths_path}...")

#     global cui2idx, idx2cui, relation2idx, idx2relation, cui_embs, relation_embs
#     if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
#         load_resources(semmed_cui_path)
#     if cui_embs is None:
#         cui_embs = np.load(cui_emb_path)
#     if relation_embs is None:
#         relation_embs = np.load(rel_emb_path)

#     if method != "triple_cls":
#         raise NotImplementedError()

#     with open(raw_paths_path, "r") as fin:
#         data = [json.loads(line) for line in fin]
#     nrow = len(data)

#     with open(output_path, "w") as fout:
#         for row in tqdm(range(nrow)):
#             path_scores = score_paths_between_pair(data[row])
#             fout.write(json.dumps(path_scores) + "\n")

#     print(f"path scores saved to {output_path}")
#     print()

# TODO: finish this
def generate_path_and_graph_from_adj(adj_path, semmed_graph_path, output_path, graph_output_path, num_processes=1, random_state=0, dump_len=False):
    print(f'generating paths for {adj_path}...')

    random.seed(random_state)
    np.random.seed(random_state)

    global semmed
    if semmed is None:
        semmed = nx.read_gpickle(semmed_graph_path)

    with open(adj_path, "rb") as fin:
        adj_concept_pairs = pickle.load(fin)  # (adj, concepts, qm, am)
    all_len = []
    with Pool(num_processes) as p, open(output_path, 'w') as path_output, open(graph_output_path, 'w') as graph_output:
        for pfr_qa, graph, lengths in tqdm(p.imap(find_paths_from_adj_per_inst, adj_concept_pairs), total=len(adj_concept_pairs), desc='Searching for paths'):
            path_output.write(json.dumps(pfr_qa) + '\n')
            graph_output.write(json.dumps(graph) + '\n')
            all_len.append(lengths)
    if dump_len:
        with open(adj_path+'.len.pk', 'wb') as f:
            pickle.dump(all_len, f)

    print(f'paths saved to {output_path}')
    print(f'graphs saved to {graph_output_path}')
    print()


if __name__ == "__main__":
    find_paths((sys.argv[1]), (sys.argv[2]), (sys.argv[3]), (sys.argv[4]))
