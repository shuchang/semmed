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


def extract_semmed_cui(semmed_csv_path, output_csv_path, semmed_cui_path):
    # TODO: deal with some error cui and its influence on graph constructing
    """
    read the original SemMed csv file to extract all cui and store
    """
    print('extracting cui list from SemMed...')
    semmed_cui_vocab = []
    cui_seen = set()
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))
    with open(semmed_csv_path, "r", encoding="utf-8") as fin, open(output_csv_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')
            if ls == ['']:
                continue
            subj = ls[4]
            obj = ls[8]
            if len(subj) == 8 and len(obj) == 8 and subj.startswith("C") and obj.startswith("C"):
                fout.write(line+"\n")
                for i in [subj, obj]:
                    if i not in cui_seen:
                        semmed_cui_vocab.append(i)
                        cui_seen.add(i)

    with open(semmed_cui_path, "w", encoding="utf-8") as fout:
        for semmed_cui in semmed_cui_vocab:
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
    cui2idx = {c: i for i, c in enumerate(idx2cui)}

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
            rel = relation2idx[ls[3].lower()]
            subj = cui2idx[ls[4]]
            obj = cui2idx[ls[8]]
            if subj == obj:
                continue
            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel)
                attrs.add((subj, obj, rel))
    nx.write_gpickle(graph, output_path)

    print(f"graph file saved to {output_path}")
    print()


if __name__ == "__main__":
    #extract_semmed_cui("../data/semmed/database.csv", "../data/semmed/database_pruned.csv", "../data/semmed/cui_vocab.txt")
    construct_graph("../data/semmed/database_pruned.csv","../data/semmed/cui_vocab.txt","../data/semmed/database_pruned.graph")
