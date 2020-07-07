"""
Script to tokenize each visit (similar to word tokenize) in the
medical record (similar to sentence tokenize) and make the list
of all appeared cui
"""

import os
import json
from tqdm import tqdm

__all__ = ['tokenize_medical_record', 'make_cui_list']



def tokenize_visits(visits):
    """
    `param`:
        visits: list of V (at least 5) visits, each visit is represented by a list of cui
    `return`:
        tokens: list with size of Vxc, where c stands for the num of cui in each visit
    """
    return tokens

def tokenize_medical_record(medical_record_path, output_path):
    """
    tokenize all the medical records from the input path
    """
    nrow = sum(1 for _ in open(medical_record_path, "r"))
    with open(medical_record_path, "r") as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, total=nrow, desc="tokenizing"):
            data = json.loads(line)
            visits = data["medical_records"]["record_cui"]
            tokens = tokenize_visits(visits)
            fout.write("".join(tokens) + "\n")


def make_cui_list():
    """
    create the list of all cui appeared in the dataset
    """


if __name__ == "__main__":
    tokenize_medical_record()