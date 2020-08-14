import json
import sys

import networkx as nx
from tqdm import tqdm

# try:
#     from .utils import check_file
# except ImportError:
#     from utils import check_file

__all__ = ["construct_graph", "relations"]


# (33 pos, 31 neg): no neg_compared_with, neg_prep
relations = ['administered_to', 'affects', 'associated_with', 'augments', 'causes', 'coexists_with', 'complicates', 'converts_to',
             'diagnoses', 'disrupts', 'higher_than', 'inhibits', 'isa', 'interacts_with', 'location_of', 'lower_than', 'manifestation_of',
             'measurement_of', 'measures', 'method_of', 'occurs_in', 'part_of', 'precedes', 'predisposes', 'prevents', 'process_of',
             'produces', 'same_as', 'stimulates', 'treats', 'uses',  'compared_with', 'prep'] # negative relations are deleted


# relation_groups = []

# merged_relations = []

# def load_merge_relation():
#     # TODO: merge relation
#     """
#     `return`: relation_mapping: {"":}
#     """


def separate_semmed_cui(semmed_cui: str) -> list:
    """
    separate semmed cui with | by perserving the replace the numbers after |
    `param`:
        semmed_cui: single or multiple semmed_cui separated by |
    `return`:
        sep_cui_list: list of all separated semmed_cui
    """
    sep_cui_list = []
    sep = semmed_cui.split("|")
    first_cui = sep[0]
    sep_cui_list.append(first_cui)
    ncui = len(sep)
    for i in range(ncui - 1):
        last_digs = sep[i + 1]
        len_digs = len(last_digs)
        if len_digs < 8: # there exists some strange cui with over 7 digs
            sep_cui = first_cui[:8 - len(last_digs)] + last_digs
            sep_cui_list.append(sep_cui)
    return sep_cui_list


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def extract_semmed_cui(semmed_csv_path, semmed_cui_path):
    """
    read the original SemMed csv file to extract all cui and store
    """
    print('extracting cui list from SemMed...')

    semmed_cui_list = []
    nrow = sum(1 for _ in open(semmed_csv_path, "r", encoding="utf-8"))

    with open(semmed_csv_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=nrow):
            ls = line.strip().split(',')

            if ls[4].startswith("C"):
                subj = ls[4]
                if len(subj) != 8:
                    subj_list = separate_semmed_cui(subj)
                    semmed_cui_list.extend(subj_list)
                else:
                    semmed_cui_list.append(subj)
            if ls[8].startswith("C"):
                obj = ls[8]
                if len(obj) != 8:
                    obj_list = separate_semmed_cui(obj)
                    semmed_cui_list.extend(obj_list)
                else:
                    semmed_cui_list.append(obj)

        semmed_cui_list = list(set(semmed_cui_list))

    with open(semmed_cui_path, "w", encoding="utf-8") as fout:
        for semmed_cui in semmed_cui_list:
            fout.write(semmed_cui + "\n")

    print(f'extracted cui saved to {semmed_cui_path}')
    print()


def construct_graph(semmed_csv_path, semmed_cui_path, output_path, output_txt_path):
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
            if ls[3].lower() not in relations:
                continue
            if ls[4] == ls[8]: # delete self-loop, not useful for our task
                continue
            sent = ls[1]
            rel = relation2idx[ls[3].lower()]

            if ls[4].startswith("C") and ls[8].startswith("C"):
                if len(ls[4]) == 8 and len(ls[8]) == 8:
                    subj = cui2idx[ls[4]]
                    obj = cui2idx[ls[8]]
                    if (subj, obj, rel) not in attrs:
                        graph.add_edge(subj, obj, rel=rel, sent=sent)
                        attrs.add((subj, obj, rel))
                elif len(ls[4]) != 8 and len(ls[8]) == 8:
                    cui_list = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list]
                    obj = cui2idx[ls[8]]
                    for subj in subj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                elif len(ls[4]) == 8 and len(ls[8]) != 8:
                    cui_list = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list]
                    subj = cui2idx[ls[4]]
                    for obj in obj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                else:
                    cui_list1 = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list1]
                    cui_list2 = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list2]
                    for subj in subj_list:
                        for obj in obj_list:
                            if (subj, obj, rel) not in attrs:
                                graph.add_edge(subj, obj, rel=rel, sent=sent)
                                attrs.add((subj, obj, rel))

    nx.write_gpickle(graph, output_path)

    with open(output_txt_path, "w", encoding="utf-8") as fout:
        for triple in attrs:
            fout.write(str(triple[0]) + "\t" + str(triple[1]) + "\t" + str(triple[2])+ "\n")

    print(f"graph file saved to {output_path}")
    print(f"txt file saved to {output_txt_path}")
    print()


def construct_subgraph(semmed_csv_path, semmed_cui_path, output_graph_path, output_txt_path):
    print("generating subgraph of SemMed using newly extracted cui list...")

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
            if ls[3].lower() not in relations:
                continue
            if ls[4] not in idx2cui or ls[8] not in idx2cui:
                continue
            if ls[4] == ls[8]: # delete self-loop, not useful for our task
                continue

            sent = ls[1]
            rel = relation2idx[ls[3].lower()]

            if ls[4].startswith("C") and ls[8].startswith("C"):
                if len(ls[4]) == 8 and len(ls[8]) == 8:
                    subj = cui2idx[ls[4]]
                    obj = cui2idx[ls[8]]
                    if (subj, obj, rel) not in attrs:
                        graph.add_edge(subj, obj, rel=rel, sent=sent)
                        attrs.add((subj, obj, rel))
                elif len(ls[4]) != 8 and len(ls[8]) == 8:
                    cui_list = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list]
                    obj = cui2idx[ls[8]]
                    for subj in subj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                elif len(ls[4]) == 8 and len(ls[8]) != 8:
                    cui_list = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list]
                    subj = cui2idx[ls[4]]
                    for obj in obj_list:
                        if (subj, obj, rel) not in attrs:
                            graph.add_edge(subj, obj, rel=rel, sent=sent)
                            attrs.add((subj, obj, rel))
                else:
                    cui_list1 = separate_semmed_cui(ls[4])
                    subj_list = [cui2idx[s] for s in cui_list1]
                    cui_list2 = separate_semmed_cui(ls[8])
                    obj_list = [cui2idx[o] for o in cui_list2]
                    for subj in subj_list:
                        for obj in obj_list:
                            if (subj, obj, rel) not in attrs:
                                graph.add_edge(subj, obj, rel=rel, sent=sent)
                                attrs.add((subj, obj, rel))

    nx.write_gpickle(graph, output_graph_path)

    attrs = list(attrs)
    with open(output_txt_path, "w", encoding="utf-8") as fout:
        for triple in attrs:
            fout.write(str(triple[0]) + "\t" + str(triple[1]) + "\t" + str(triple[2])+ "\n")

    print(f"graph file saved to {output_graph_path}")
    print(f"txt file saved to {output_txt_path}")
    print()


if __name__ == "__main__":
    construct_graph((sys.argv[1]), (sys.argv[2]), (sys.argv[3]), (sys.argv[4]))
    # extract_semmed_cui((sys.argv[1]), (sys.argv[2]))
