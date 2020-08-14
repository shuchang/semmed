import json
import pickle as pkl
import random
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy import spatial
from tqdm import tqdm

try:
    from .semmed import relations
except ModuleNotFoundError:
    from semmed import relations


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
        res: list of all existing relations without repetition
    """
    global semmed
    rel_list = semmed[src_idx][tgt_idx]
    rel_seen = set()
    res = [r["rel"] for r in rel_list.values() if r["rel"] not in rel_seen and (rel_seen.add(r["rel"]) or True)]
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

    # Why existing src_idx or tgt_idx not in semmed_simple.nodes?
    # Because the input data is pruned using semmed cui but the nodes of semmed graph is its subset
    if src_idx not in semmed_simple.nodes() or tgt_idx not in semmed_simple.nodes():
        return

    all_path = []
    try:
        for p in nx.shortest_simple_paths(semmed_simple, source=src_idx, target=tgt_idx):
            if len(p) > 4 and len(all_path) > 5:
                break
            if len(p) >= 2: # skip paths of self loop
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
            rl.append(rel_list)

        pf_res.append({"path": p, "rel": rl})
    return pf_res


def find_paths_between_pairs(pair: list) -> list:
    """
    find paths between pairs of record_cui and hf_cui in one sample

    `param`:
        pair: list of record_cui and hf_cui pairs
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


def find_paths_from_adj_per_inst(input):
    adj, cui_idxs, record_mask, hf_mask = input
    adj = adj.toarray()
    ij, k = adj.shape

    adj = np.any(adj.reshape(ij // k, k, k), axis=0)
    simple_schema_graph = nx.from_numpy_matrix(adj)
    mapping = {i: int(c) for (i, c) in enumerate(cui_idxs)}
    simple_schema_graph = nx.relabel_nodes(simple_schema_graph, mapping)
    record_cui, hf_cui = cui_idxs[record_mask].tolist(), cui_idxs[hf_mask].tolist()
    pfr_pair = []
    lengths = []
    for single_hf_cui in hf_cui:
        for single_record_cui in record_cui:
            if single_record_cui not in simple_schema_graph.nodes() or single_hf_cui not in simple_schema_graph.nodes():
                print('pair doesn\'t exist in schema graph.')
                pf_res = None
                lengths.append([0] * 3)
            else:
                all_path = []
                try:
                    for p in nx.shortest_simple_paths(simple_schema_graph, source=single_record_cui, target=single_hf_cui):
                        if len(p) >= 4:
                            break
                        if len(p) >= 2:  # skip paths of length 1
                            all_path.append(p)
                except nx.exception.NetworkXNoPath:
                    pass

                length = [len(x) for x in all_path]
                lengths.append([length.count(2), length.count(3), length.count(4)])
                pf_res = []
                for p in all_path:
                    rl = []
                    for src in range(len(p) - 1):
                        src_cui = p[src]
                        tgt_cui = p[src + 1]
                        rel_list = get_edge(src_cui, tgt_cui)
                        rl.append(rel_list)
                    pf_res.append({"path": p, "rel": rl})
            pfr_pair.append({"record_cui": record_cui, "hf_cui": hf_cui, "pf_res": pf_res})
    g = nx.convert_node_labels_to_integers(simple_schema_graph, label_attribute='cui_idxs')

    return pfr_pair, nx.node_link_data(g), lengths


##################### path scoring #####################


def score_triple(h, t, r, flag):
    res = -10
    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t
        # result  = (cosine_sim + 1) / 2
        res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)
    return res


def score_triples(cui_idx, rel_idx, debug=False):
    """
    score triples in one path

    `params`:
        cui_idx: list of cui index in the path
        rel_idx: list of relation index lists
    `return`:
        res: score of this path
    """
    global cui_embs, relation_embs, idx2cui, idx2relation
    cui = cui_embs[cui_idx]
    relation = []
    flag = []
    for i in range(len(rel_idx)):
        embs = []
        l_flag = []

        # if 0 in rel_idx[i] and 17 not in rel_idx[i]:
        #     rel_idx[i].append(17)
        # elif 17 in rel_idx[i] and 0 not in rel_idx[i]:
        #     rel_idx[i].append(0)
        # if 15 in rel_idx[i] and 32 not in rel_idx[i]:
        #     rel_idx[i].append(32)
        # elif 32 in rel_idx[i] and 15 not in rel_idx[i]:
        #     rel_idx[i].append(15)

        for j in range(len(rel_idx[i])):
            if rel_idx[i][j] >= 33:
                embs.append(relation_embs[rel_idx[i][j] - 33])
                l_flag.append(0) # positive
            else:
                embs.append(relation_embs[rel_idx[i][j]])
                l_flag.append(1) # negative
        relation.append(embs)
        flag.append(l_flag)

    res = 1
    for i in range(cui.shape[0] - 1):
        h = cui[i]
        t = cui[i + 1]
        score = score_triple(h, t, relation[i], flag[i])
        res *= score

    # if debug:
    #     print("Num of cui:")
    #     print(len(cui_idx))
    #     to_print = ""
    #     for i in range(cui.shape[0] - 1):
    #         h = idx2cui[cui_idx[i]]
    #         to_print += h + "\t"
    #         for rel in rel_idx[i]:
    #             if rel >= 17:
    #                 # 'r-' means reverse
    #                 to_print += ("r-" + idx2relation[rel - 17] + "/  ")
    #             else:
    #                 to_print += x[rel] + "/  "
    #     to_print += idx2cui[cui_idx[-1]]
    #     print(to_print)
    #     print("Likelihood: " + str(res) + "\n")

    return res


def score_paths_between_pairs(pfr_pair):
    """
    score paths between pairs of record_cui and hf_cui in one sample

    `param`:
        pfr_pair: list of dictionaries with the form of
        {"record_cui": record_cui, "hf_cui": hf_cui, "pf_res": pf_res}
    `return`:
        pair_scores: list of path scores of paths between pairs in order
    """
    pair_scores = []
    for pf_res in pfr_pair: # single cui pair
        pair_paths = pf_res["pf_res"]
        if pair_paths is not None:
            path_scores = []
            for path in pair_paths:
                assert len(path["path"]) > 1
                score = score_triples(cui_idx=path["path"], rel_idx=path["rel"])
                path_scores.append(score)
            pair_scores.append(path_scores)
        else:
            path_scores.append(None)
    return pair_scores


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def find_paths(grounded_path, semmed_cui_path, semmed_graph_path, output_path, num_processes=8, random_state=0, debug=False):
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
        data = data[0:2]

    data = [[item["record_cui"], item["hf_cui"]] for item in data]
    nrow = len(data)

    with Pool(num_processes) as p, open(output_path, "w") as fout:
        for pfr_pair in tqdm(p.imap(find_paths_between_pairs, data), total=nrow):
            fout.write(json.dumps(pfr_pair) + "\n")
    # with open(output_path, "w") as fout:
    #     for row in tqdm(range(nrow)):
    #         pfr_pair = find_paths_between_pairs(data[row])
    #         fout.write(json.dumps(pfr_pair) + "\n")

    print(f'paths saved to {output_path}')
    print()


def score_paths(raw_paths_path, cui_emb_path, rel_emb_path, semmed_cui_path, output_path, num_processes=1, method="triple_cls"):
    print(f"scoring paths for {raw_paths_path}...")

    global cui2idx, idx2cui, relation2idx, idx2relation, cui_embs, relation_embs
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if cui_embs is None:
        cui_embs = np.load(cui_emb_path)
    if relation_embs is None:
        relation_embs = np.load(rel_emb_path)

    if method != "triple_cls":
        raise NotImplementedError()

    with open(raw_paths_path, "r") as fin:
        data = [json.loads(line) for line in fin]
    nrow = len(data)

    with Pool(num_processes) as p, open(output_path, "w") as fout:
        for path_scores in tqdm(p.imap(score_paths_between_pairs, data), total=nrow):
            fout.write(json.dumps(path_scores) + "\n")
    # with open(output_path, "w") as fout:
    #     for row in tqdm(range(nrow)):
    #         path_scores = score_paths_between_pairs(data[row])
    #         fout.write(json.dumps(path_scores) + "\n")

    print(f"path scores saved to {output_path}")
    print()


def prune_paths(raw_paths_path, path_scores_path, output_path, threshold, verbose=True):
    print(f'pruning paths for {raw_paths_path}...')
    ori_len = 0
    pruned_len = 0
    nrow = sum(1 for _ in open(raw_paths_path, 'r'))
    with open(raw_paths_path, 'r') as fin_raw, \
            open(path_scores_path, 'r') as fin_score, \
            open(output_path, 'w') as fout:
        for line_raw, line_score in tqdm(zip(fin_raw, fin_score), total=nrow):
            pairs = json.loads(line_raw)
            pairs_scores = json.loads(line_score)
            for pair, pair_scores in zip(pairs, pairs_scores):
                ori_paths = pair['pf_res']
                if ori_paths is not None:
                    pruned_paths = [p for p, s in zip(ori_paths, pair_scores) if s >= threshold]
                    ori_len += len(ori_paths)
                    pruned_len += len(pruned_paths)
                    assert len(ori_paths) >= len(pruned_paths)
                    pair['pf_res'] = pruned_paths
            fout.write(json.dumps(pairs) + '\n')

    if verbose:
        print("ori_len: {}   pruned_len: {}   keep_rate: {:.4f}".format(ori_len, pruned_len, pruned_len / ori_len))

    print(f'pruned paths saved to {output_path}')
    print()


def generate_path_and_graph_from_adj(adj_path, semmed_graph_path, output_path, graph_output_path, num_processes=4, random_state=0, dump_len=False):
    print(f'generating paths for {adj_path}...')

    random.seed(random_state)
    np.random.seed(random_state)

    global semmed
    if semmed is None:
        semmed = nx.read_gpickle(semmed_graph_path)

    with open(adj_path, "rb") as fin:
        adj_concept_pairs = pkl.load(fin)  # (adj, cui_idx, record_mask, hf_mask)
    all_len = []
    with Pool(num_processes) as p, open(output_path, 'w') as path_output, open(graph_output_path, 'w') as graph_output:
        for pfr_pair, graph, lengths in tqdm(p.imap(find_paths_from_adj_per_inst, adj_concept_pairs), total=len(adj_concept_pairs), desc='Searching for paths'):
            path_output.write(json.dumps(pfr_pair) + '\n')
            graph_output.write(json.dumps(graph) + '\n')
            all_len.append(lengths)
    if dump_len:
        with open(adj_path+'.len.pk', 'wb') as f:
            pkl.dump(all_len, f)

    print(f'paths saved to {output_path}')
    print(f'graphs saved to {graph_output_path}')
    print()


if __name__ == "__main__":
    find_paths((sys.argv[1]), (sys.argv[2]), (sys.argv[3]), (sys.argv[4]))