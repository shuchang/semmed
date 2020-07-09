import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
# from .semmed import merged_relations
from semmed import relations
import numpy as np
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from .maths import *

__all__ = ['generate_graph']

concept2id = None
id2concept = None
relation2id = None
id2relation = None

semmed = None
semmed_all = None
semmed_simple = None


def load_resources(semmed_cui_path):
    global cui2idx, idx2cui, relation2idx, idx2relation

    with open(semmed_cui_path, "r", encoding="utf8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {w: i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}


def load_semmed(semmed_graph_path):
    """
    load the pruned or unpruned SemMed graph file

    `return`:
        semmed: multirelational directed graph of SemMed, weight of each path
        is 1.0
        semmed_simple: basic undirected graph of SemMed, weight of each edge is
        the number of paths
    """
    global semmed, semmed_simple
    semmed = nx.read_gpickle(semmed_graph_path)
    semmed_simple = nx.Graph()
    for u, v, data in semmed.edges(data=True):
        w = 1.0 # initial weight to 1
        if semmed_simple.has_edge(u, v):
            semmed_simple[u][v]['weight'] += w
        else:
            semmed_simple.add_edge(u, v, weight=w)



#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################



def generate_graph(grounded_path, pruned_paths_path, semmed_cui_path, semmed_graph_path, output_path):
