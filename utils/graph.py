import itertools
import json
import pickle as pkl
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

# import torch

# from .maths import *

try:
    from .semmed import relations
except ModuleNotFoundError:
    from semmed import relations


__all__ = ['generate_graph']


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
semmed_all = None
semmed_simple = None


def load_resources(semmed_cui_path):
    global cui2idx, idx2cui, relation2idx, idx2relation

    with open(semmed_cui_path, "r", encoding="utf8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}


def load_semmed(semmed_graph_path):
    global semmed, semmed_simple
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, _ in semmed.edges(data=True):
        w = 1.0 # initial weight to 1
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)


def plain_graph_generation(rcs: list, hcs: list, paths: list, rels: list) -> dict:
    """
    generate plain graph of each sample of hfdata using grounded cui pairs,
    and paths and relations between each cui pair

    `params`:
        rcs: indexs of record_cui
        hcs: indexs of hf_cui
        paths:
        rels:
    `return`:
        gobj:
    """
    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple

    graph = nx.Graph()
    for p in paths:
        for src in range(len(p) - 1):
            src_cui = p[src]
            tgt_cui = p[src + 1]
            # TODO: the weight can be computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(src_cui, tgt_cui, weight=1.0)

    for rc1, rc2 in list(itertools.combinations(rcs, 2)):
        if semmed_simple.has_edge(rc1, rc2):
            graph.add_edge(rc1, rc2, weight=1.0)

    for hc1, hc2 in list(itertools.combinations(hcs, 2)):
        if semmed_simple.has_edge(hc1, hc2):
            graph.add_edge(hc1, hc2, weight=1.0)

    if len(rcs) == 0:
        rcs.append(-1)

    if len(hcs) == 0:
        hcs.append(-1)

    if len(paths) == 0:
        for rc in rcs:
            for hc in hcs:
                graph.add_edge(rc, hc, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute="cid")
    gobj = nx.node_link_data(g)
    return gobj


def cui2adj(node_idxs):
    global idx2relation
    cui_idxs = np.array(node_idxs, dtype=np.int32)
    n_rel = len(idx2relation)
    n_node = cui_idxs.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.int8)

    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cui_idxs[s], cui_idxs[t]
            if semmed.has_edge(s_c, t_c):
                for e_attr in semmed[s_c][t_c].values():
                    if e_attr["rel"] >= 0 and e_attr["rel"] < n_rel:
                        adj[e_attr["rel"]][s][t] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cui_idxs


def cui_to_adj_matrices_2hop_all_pair(data):
    """
    """
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in nodes:
        for hf_idx in nodes:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                extra_nodes |= set(semmed_simple[record_idx]) & set(semmed_simple[hf_idx])
    extra_nodes = extra_nodes - nodes

    schema_graph = sorted(record_idxs) + sorted(hf_idxs) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    record_mask = arange < len(record_idxs)
    hf_mask = (arange >= len(record_idxs)) & (arange < (len(record_idxs) + len(hf_idxs)))
    adj, cui = cui2adj(schema_graph)
    return adj, cui, record_mask, hf_mask


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def generate_graph(grounded_path, raw_paths_path, semmed_cui_path, semmed_graph_path, output_path):
    print(f"generating schema graphs for {grounded_path} and {raw_paths_path}...")

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    nrow = sum(1 for _ in open(grounded_path, "r"))
    with open(grounded_path, "r") as fin_gr, open(raw_paths_path, "r") as fin_rp, \
         open(output_path, "w") as fout:
        for line_gr, line_rp in tqdm(zip(fin_gr, fin_rp), total=nrow):
            mcp = json.loads(line_gr) # matched cui pair?
            pfr_pair = json.loads(line_rp)

            rcs = [cui2idx[c] for c in mcp["record_cui"]]
            hcs = [cui2idx[c] for c in mcp["hf_cui"]]

            paths = []
            rel_list = []
            for pair in pfr_pair:
                if pair["pf_res"] is None: # TODO: change the find_paths
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in pair["pf_res"]]
                    cur_rels = [item["rel"] for item in pair["pf_res"]]
                paths.extend(cur_paths)
                rel_list.extend(cur_rels)

            gobj = plain_graph_generation(rcs=rcs, hcs=hcs, paths=paths, rels=rel_list)
            fout.write(json.dumps(gobj) + "\n")

    print(f"schema graphs saved to {output_path}")
    print()


def generate_adj_data_from_grounded_concepts(grounded_path, semmed_graph_path, semmed_cui_path, output_path, num_processes=8):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) cui idxs
        (3) record_mask that specifices whether a node is a record_cui
        (4) hf_mask that specifices whether a node is a hf_cui
    to the output path in python pickle format
    """
    print(f"generating adj data from {grounded_path}...")

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    data = []
    with open(grounded_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            data.append((record_idxs, hf_idxs))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(cui_to_adj_matrices_2hop_all_pair, data), total=len(data)))

    with open(output_path, "wb") as fout:
        pkl.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()
