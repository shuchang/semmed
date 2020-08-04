import itertools
import json
import sys
import pickle as pkl
from multiprocessing import Pool

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

# from .maths import *

# try:
    # from .semmed import relations
# except ModuleNotFoundError:
from .semmed import relations


__all__ = ['generate_graph']


cui2idx = None
idx2cui = None
relation2idx = None
idx2relation = None

semmed = None
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
    # cui_idxs += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cui_idxs


def cui_to_adj_matrices_2hop_all_pair(data):
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
    adj, cui_idxs = cui2adj(schema_graph)
    return adj, cui_idxs, record_mask, hf_mask


def concepts_to_adj_matrices_3hop_qa_pair(data):
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in nodes:
        for hf_idx in nodes:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                for u in semmed_simple[record_idx]:
                    for v in semmed_simple[hf_idx]:
                        if semmed_simple.has_edge(u, v):  # hf_cui is a 3-hop neighbour of record_cui
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # hf_cui is a 2-hop neighbour of record_cui
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - nodes

    schema_graph = sorted(record_idxs) + sorted(hf_idxs) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    record_mask = arange < len(record_idxs)
    hf_mask = (arange >= len(record_idxs)) & (arange < (len(record_idxs) + len(hf_idxs)))
    adj, cui_idxs = cui2adj(schema_graph)
    return adj, cui_idxs, record_mask, hf_mask


def save_nodes_of_2hop_all_pair(data):
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in hf_idxs:
        for hf_idx in hf_idxs:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                extra_nodes |= set(semmed_simple[record_idx]) & set(semmed_simple[hf_idx])
    extra_nodes = extra_nodes - nodes
    all_nodes = record_idxs | hf_idxs | extra_nodes
    return all_nodes


def save_nodes_of_3hop_all_pair(data):
    record_idxs, hf_idxs = data
    nodes = set(record_idxs) | set(hf_idxs)
    extra_nodes = set()

    for record_idx in record_idxs:
        for hf_idx in hf_idxs:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                for u in semmed_simple[record_idx]:
                    for v in semmed_simple[hf_idx]:
                        if semmed_simple.has_edge(u, v):  # hf_cui is a 3-hop neighbour of record_cui
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # hf_cui is a 2-hop neighbour of record_cui
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - nodes
    all_nodes = record_idxs | hf_idxs | extra_nodes
    return all_nodes


def save_triples(u, v):
    """
    save triples in two directions
    """
    triples = []
    # if semmed.has_edge(u, v):
        # for value in semmed[u][v].values():
        #     triples.append((u, v, value['rel']))
    # if semmed.has_edge(v, u):
        # for reverse_value in semmed[v][u].values():
        #     triples.append((v, u, reverse_value['rel']))
    for value in semmed[u][v].values():
        triples.append((u, v, value['rel']))
    for reverse_value in semmed[v][u].values():
        triples.append((v, u, reverse_value['rel']))
    return triples


def save_triples_of_3hop_all_pair(data):
    record_idxs, hf_idxs = data
    triples = []

    for record_idx in record_idxs:
        for hf_idx in hf_idxs:
            if record_idx != hf_idx and record_idx in semmed_simple.nodes and hf_idx in semmed_simple.nodes:
                for u in semmed_simple[record_idx]:
                    for v in semmed_simple[hf_idx]:
                        if semmed_simple.has_edge(u, v):
                            # hf_cui is a 3-hop neighbour of record_cui
                            triples.extend(save_triples(record_idx, u))
                            triples.extend(save_triples(u, v))
                            triples.extend(save_triples(v, hf_idx))
                        if u == v:
                            # hf_cui is a 2-hop neighbour of record_cui
                            triples.extend(save_triples(record_idx, u))
                            triples.extend(save_triples(v, hf_idx))
    return triples


def extract_triples(cui_idx: list, rel_idx: list) -> list:
    """
    extract triples in one path

    `params`:
        cui_idx: list of cui index in the path
        rel_idx: list of relation index lists
    `return`:
        triples:
    """
    triples = []
    for i in range(len(cui_idx) - 1):
        h = cui_idx[i]
        t = cui_idx[i + 1]
        if rel_idx[i] == None:
            continue
        for j in range(len(rel_idx[i])):
            triples.append((h, t, rel_idx[i][j]))
    return triples


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def generate_graph(grounded_path, pruned_paths_path, semmed_cui_path, semmed_graph_path, output_path):
    print(f"generating schema graphs for {grounded_path} and {pruned_paths_path}...")

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    nrow = sum(1 for _ in open(grounded_path, "r"))
    with open(grounded_path, "r") as fin_gr, open(pruned_paths_path, "r") as fin_rp, \
         open(output_path, "w") as fout:
        for line_gr, line_rp in tqdm(zip(fin_gr, fin_rp), total=nrow):
            mcp = json.loads(line_gr) # matched cui pair?
            pfr_pair = json.loads(line_rp)

            rcs = [cui2idx[c] for c in mcp["record_cui"]]
            hcs = [cui2idx[c] for c in mcp["hf_cui"]]

            paths = []
            rel_list = []
            for pair in pfr_pair:
                if pair["pf_res"] is None:
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
        (1) adjacency matrices (each in the form of a (R*N, N) coo sparse matrix)
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
    # res = []
    # for i in tqdm(range(0, len(qa_data))):
    #     res.append(concepts_to_adj_matrices_2hop_all_pair(qa_data[i]))

    with open(output_path, "wb") as fout:
        pkl.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


def extract_subgraph_cui(grounded_train_path, grounded_dev_path, grounded_test_path, semmed_graph_path, semmed_cui_path, output_path, num_processes=8, debug=False):
    """
    extracting all cui in the 2hop and 3hop paths of the hfdata as the subgraph cui list
    """
    print("extracting subgraph cui from grounded_path...")

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    data = []
    semmed_cui = set()
    semmed_cui_list = []

    with open(grounded_train_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))
    with open(grounded_dev_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))
    with open(grounded_test_path, "r", encoding="utf-8") as fin:
        for line in fin:
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))

    if debug:
        data = data[0:8]

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(save_nodes_of_3hop_all_pair, data), total=len(data)))

    for cui_set in res:
        semmed_cui.update(cui_set)
    semmed_cui_list = list(semmed_cui)

    with open(output_path, "w", encoding="utf-8") as fout:
        for cui in semmed_cui_list:
            fout.write(str(idx2cui[cui]) + "\n")

    print(f'extracted subgraph cui saved to {output_path}')
    print()

# Ongoing
def extract_cui_and_subgraph_from_ground(grounded_train_path, grounded_dev_path, grounded_test_path, semmed_graph_path, semmed_cui_path, output_cui_path, output_txt_path, num_processes=4, debug=False):
    """
    extracting all cui in the 2hop and 3hop paths of the hfdata as the subgraph cui list
    """
    print("extracting subgraph cui from grounded_path...")

    global cui2idx, idx2cui, relation2idx, idx2relation, semmed, semmed_simple
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)
    if any(x is None for x in [semmed, semmed_simple]):
        load_semmed(semmed_graph_path)

    data = []
    triple_list = []
    semmed_cui_list = []

    num_train = sum(1 for _ in open(grounded_train_path, "r", encoding="utf-8"))
    num_dev = sum(1 for _ in open(grounded_dev_path, "r", encoding="utf-8"))
    num_test = sum(1 for _ in open(grounded_test_path, "r", encoding="utf-8"))

    with open(grounded_train_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=num_train, desc="reading grounded train..."):
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            record_idxs = record_idxs - hf_idxs
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))
    with open(grounded_dev_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=num_dev, desc="reading grounded dev..."):
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))
    with open(grounded_test_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=num_test, desc="reading grounded test..."):
            dic = json.loads(line)
            record_idxs = set(cui2idx[c] for c in dic["record_cui"])
            hf_idxs = set(cui2idx[c] for c in dic["hf_cui"])
            if (record_idxs, hf_idxs) not in data:
                data.append((record_idxs, hf_idxs))

    if debug:
        data = data[0:2]

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(save_triples_of_3hop_all_pair, data), total=len(data)))
    # res = []
    # for line in data:
    #     res.extend(save_triples_of_3hop_all_pair(line))

    for triples in res:
        triple_list.extend(triples)
    triple_list = list(set(triple_list))
    for triple in triple_list:
        semmed_cui_list.append(idx2cui[triple[0]])
        semmed_cui_list.append(idx2cui[triple[1]])
    semmed_cui_list = list(set(semmed_cui_list))
    new_cui2idx = {c: i for i, c in enumerate(semmed_cui_list)}

    with open(output_cui_path, "w", encoding="utf-8") as fout:
        for cui in semmed_cui_list:
            fout.write(str(cui) + "\n")

    with open(output_txt_path, "w", encoding="utf-8") as fout:
        for triple in triple_list:
            fout.write(str(new_cui2idx[idx2cui[triple[0]]]) + "\t" +
            str(new_cui2idx[idx2cui[triple[1]]]) + "\t" + str(triple[2])+ "\n")

    print(f'extracted subgraph cui saved to {output_cui_path}')
    print(f'extracted subgraph saved to {output_txt_path}')
    print()


############################ The following function extracts subgraph based on path finding results ############################
def extract_subgraph_from_path(raw_paths_train_path, raw_paths_dev_path, raw_paths_test_path, semmed_cui_path, output_cui_path, output_txt_path):
    """
    extracting subgraph of SemMed by preserving all triples without repetition
    """
    print("extracting subgraph cui and subgraph from raw paths...")

    global cui2idx, idx2cui, relation2idx, idx2relation
    if any(x is None for x in [cui2idx, idx2cui, relation2idx, idx2relation]):
        load_resources(semmed_cui_path)

    triple_list = []
    semmed_cui_list = []
    nrow_train = sum(1 for _ in open(raw_paths_train_path, "r"))
    nrow_dev = sum(1 for _ in open(raw_paths_dev_path, "r"))
    nrow_test = sum(1 for _ in open(raw_paths_test_path, "r"))

    with open(raw_paths_train_path, "r") as fin:
        data = [json.loads(line) for line in fin]
        for item in tqdm(data, total=nrow_train, desc="extracting from train..."):
            for line in item:
                pair_paths = line["pf_res"]
                for path in pair_paths:
                    triples = extract_triples(cui_idx=path["path"], rel_idx=path["rel"])
                    for t in triples:
                        triple_list.append(t)

    with open(raw_paths_dev_path, "r") as fin:
        data = [json.loads(line) for line in fin]
        for item in tqdm(data, total=nrow_dev, desc="extracting from dev..."):
            for line in item:
                pair_paths = line["pf_res"]
                for path in pair_paths:
                    triples = extract_triples(cui_idx=path["path"], rel_idx=path["rel"])
                    for t in triples:
                        triple_list.append(t)

    with open(raw_paths_test_path, "r") as fin:
        data = [json.loads(line) for line in fin]
        for item in tqdm(data, total=nrow_test, desc="extracting from test..."):
            for line in item:
                pair_paths = line["pf_res"]
                for path in pair_paths:
                    triples = extract_triples(cui_idx=path["path"], rel_idx=path["rel"])
                    for t in triples:
                        triple_list.append(t)

    triple_list = list(set(triple_list))

    for triple in triple_list:
        semmed_cui_list.append(idx2cui[triple[0]])
        semmed_cui_list.append(idx2cui[triple[1]])
    semmed_cui_list = list(set(semmed_cui_list))
    new_cui2idx = {c: i for i, c in enumerate(semmed_cui_list)}

    with open(output_cui_path, "w", encoding="utf-8") as fout:
        for cui in semmed_cui_list:
            fout.write(str(cui) + "\n")

    with open(output_txt_path, "w", encoding="utf-8") as fout:
        for triple in triple_list:
            fout.write(str(new_cui2idx[idx2cui[triple[0]]]) + "\t" +
            str(new_cui2idx[idx2cui[triple[1]]]) + "\t" + str(triple[2])+ "\n")

    print(f'extracted subgraph cui saved to {output_cui_path}')
    print(f'extracted subgraph saved to {output_txt_path}')
    print()



if __name__ == "__main__":
    # generate_adj_data_from_grounded_concepts((sys.argv[1]), (sys.argv[2]), (sys.argv[3]), (sys.argv[4]))
    # extract_subgraph_cui((sys.argv[1]), (sys.argv[2]), (sys.argv[3]), (sys.argv[4]), (sys.argv[5]), (sys.argv[6]))
    extract_cui_and_subgraph_from_ground((sys.argv[1]), (sys.argv[2]), (sys.argv[3]), (sys.argv[4]), (sys.argv[5]), (sys.argv[6]), (sys.argv[7]))