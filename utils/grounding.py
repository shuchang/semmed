"""
Script to ground cui of medical records to that of SemMed
"""
import json
import sys
from multiprocessing import Pool

from tqdm import tqdm

__all__ = ['ground']


semmed_cui = None


def load_semmed_cui(semmed_cui_path):
    global semmed_cui
    with open(semmed_cui_path, "r", encoding="utf-8") as fin:
        semmed_cui = [l.strip() for l in fin]
    semmed_cui = [s.replace("_", "") for s in semmed_cui]


def combine_visit_cui(record_cui):
    """
    merge visits of cui into one list and delete repetition
    if nothing left, print cui not found?
    """
    combined_cui = []
    for visit_cui in record_cui:
        combined_cui += visit_cui
    combined_cui = list(set(combined_cui))
    # if combined_cui == []:
    #     print("no cui is found")
    return combined_cui


# def prune(data: list, semmed_cui) -> list:
#     """
#     prune all record cui that do not exist in SemMed cui without using multiprocessing
#     `param`:
#         data: list of dictionaries with the form of {"record_cui": record_cui, "hf_cui": hf_cui}
#         semmed_cui:
#     `return`:
#         prune_data: list of dictionaries with the form of {"record_cui": record_cui, "hf_cui": hf_cui}
#     """

#     prune_data = []
#     for item in tqdm(data, desc="grounding"):
#         prune_record_cui = []
#         record_cui = item["record_cui"]
#         for rc in record_cui:
#             if rc in semmed_cui:
#                 prune_record_cui.append(rc)

#         prune_hf_cui = []
#         hf_cui = item["hf_cui"]
#         for hc in hf_cui:
#             if hc in semmed_cui:
#                 prune_hf_cui.append(hc)

#         item["record_cui"] = prune_record_cui
#         item["hf_cui"] = prune_hf_cui
#         prune_data.append(item)
#     return prune_data


def prune(item: dict) -> dict:
    """
    prune record cui that do not exist in SemMed cui
    `param`:
        data: dict: {"record_cui": record_cui, "hf_cui": hf_cui}
    `return`:
        prune_data: dict: {"record_cui": record_cui, "hf_cui": hf_cui}
    """
    global semmed_cui

    prune_item = {}

    prune_record_cui = []
    record_cui = item["record_cui"]
    for rc in record_cui:
        if rc in semmed_cui:
            prune_record_cui.append(rc)

    prune_hf_cui = []
    hf_cui = item["hf_cui"]
    for hc in hf_cui:
        if hc in semmed_cui:
            prune_hf_cui.append(hc)

        prune_item["record_cui"] = prune_record_cui
        prune_item["hf_cui"] = prune_hf_cui
    return prune_item


def ground(medical_record_path, semmed_cui_path, output_path, num_processes=1, debug=True):
    print(f"grounding {medical_record_path} based on {semmed_cui_path}")

    global semmed_cui
    if semmed_cui is None:
        load_semmed_cui(semmed_cui_path)

    total_cui_list = []
    with open(medical_record_path, "r") as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[0:8]

    for line in lines:
        # if line == "":
        #     continue
        total_cui = {}
        j = json.loads(line)
        medical_record = j["medical_records"]
        heart_disease = j["heart_diseases"]

        combined_cui = combine_visit_cui(medical_record["record_cui"])
        total_cui["record_cui"] = combined_cui
        total_cui["hf_cui"] = heart_disease["hf_cui"]
        total_cui_list.append(total_cui)

    nrow = len(total_cui_list)
    with Pool(num_processes) as p:
        grounded_cui = list(tqdm(p.imap(prune, total_cui_list), total=nrow, desc="grounding"))

    with open(output_path, "w") as fout:
        for cui in grounded_cui:
            fout.write(json.dumps(cui) + '\n')

    print(f'grounded cui saved to {output_path}')
    print()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError("Provide three arguments")
    ground((sys.argv[1]), (sys.argv[2]), (sys.argv[3]))
