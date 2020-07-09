import networkx as nx
import json
from tqdm import tqdm

# try:
#     from .utils import check_file
# except ImportError:
#     from utils import check_file

__all__ = ["construct_graph", "merged_relations"]


# (33 pos, 31 neg): no neg_compared_with, neg_prep
relations = ['administered_to', 'affects', 'associated_with', 'augments', 'causes', 'coexists_with', 'compared_with', 'complicates',
             'converts_to', 'diagnoses', 'disrupts', 'higher_than', 'inhibits', 'isa', 'interacts_with', 'location_of', 'lower_than',
             'manifestation_of', 'measurement_of', 'measures', 'method_of', 'occurs_in', 'part_of', 'precedes', 'predisposes', 'prep',
             'prevents', 'process_of', 'produces', 'same_as', 'stimulates', 'treats', 'uses',
             'neg_administered_to', 'neg_affects', 'neg_associated_with', 'neg_augments', 'neg_causes', 'neg_coexists_with',
             'neg_complicates', 'neg_converts_to', 'neg_diagnoses', 'neg_disrupts', 'neg_higher_than', 'neg_inhibits', 'neg_isa',
             'neg_interacts_with', 'neg_location_of', 'neg_lower_than', 'neg_manifestation_of', 'neg_measurement_of', 'neg_measures',
             'neg_method_of', 'neg_occurs_in', 'neg_part_of', 'neg_precedes', 'neg_predisposes', 'neg_prevents', 'neg_process_of',
             'neg_produces', 'neg_same_as', 'neg_stimulates', 'neg_treats', 'neg_uses']


relation_groups = []


merged_relations = []

def load_merge_relation():
    # TODO: merge relation
    """
    `return`: relation_mapping: {"":}
    """


def extract_semmed_cui(semmed_csv_path, semmed_cui_path):
    # TODO: deal with some error cui and its influence on graph constructing
    """
    read the original SemMed csv file to extract all cui and store
    """
    print('extracting cui list from SemMed...')
    semmed_cui_list = []
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))
    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls == ['']:
                continue
            subj = ls[4]
            obj = ls[8]
            semmed_cui_list.append(subj)
            semmed_cui_list.append(obj)
        semmed_cui_list = list(set(semmed_cui_list))
    with open(semmed_cui_path, "w", encoding="utf-8") as fout:
        for semmed_cui in semmed_cui_list:
            fout.write(semmed_cui + "\n")

    print(f'extracted cui saved to {semmed_cui_path}')
    print()


def construct_graph(semmed_csv_path, semmed_cui_path, output_path, prune=True):
    # TODO: 1. prune 2. deal with the case that subj == obj 3. cui with | 4. cui2idx?
    """
    construct the SemMed graph file
    """
    print("generating SemMed graph file...")

    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        idx2cui = [c.strip() for c in fin]
    cui2idx = {c:i for i, c in enumerate(idx2cui)}

    idx2relation = relations
    relation2idx = {r: i for i, r in enumerate(idx2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))
    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        attrs = set()
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls == ['']:
                continue
            if ls[4] not in idx2cui or ls[8] not in idx2cui:
                continue
            rel = relation2idx[ls[3].lower()]
            subj = cui2idx[ls[4]]
            obj = cui2idx[ls[8]]
            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel)
                attrs.add((subj, obj, rel))
    nx.write_gpickle(graph, output_path)

    print(f"graph file saved to {output_path}")
    print()


if __name__ == "__main__":
    glove_init("../data/glove/glove.6B.200d.txt", "../data/glove/glove.200d", '../data/glove/tp_str_corpus.json')
