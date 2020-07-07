"""
Script to ground cui of medical records to that of SemMed
"""
from multiprocessing import Pool
from tqdm import tqdm
import json



# TODO: grounding就是将病人的病史和可能得的心脏疾病一起mapping到semmed上


def load_semmed_vocab(semmed_vocab_path):
    with open(semmed_vocab_path, "r", encoding="utf-8") as fin:
        semmed_vocab = [l.strip() for l in fin]
    semmed_vocab = [s.replace("_", "") for s in semmed_vocab]
    return semmed_vocab


def ground_diseases_pair(diseases_pair):


    past_diseases, pred_diseases = diseases_pair




def match_mentioned_concepts(past_diseases, pred_diseases, num_processes):
    res = []
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_diseases_pair, zip(past_diseases, pred_diseases)), total=len(past_diseases)))
        return res

def prune(res, semmed_vocab_path):
    # reload semmed_vocab
    with open(semmed_vocab_path, "r", encoding="utf8") as fin:
        semmed_vocab_path = [l.strip() for l in fin]

    prune_res = []
    for item in tqdm(res):



def ground(medical_record_path, semmed_cui_path, output_path, num_processes=1, debug=False):
    global SEMMED_VOCAB
    if SEMMED_VOCAB is None:
        SEMMED_VOCAB = load_semmed_vocab(semmed_vocab_path)
    past_diseases = []
    heart_diseases = []
    with open(diseases_path, "r") as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[0:3]
        print(len(lines))

    for line in lines:
        if line == "":
            continue
    j = json.loads(line)
    for past_disease in j["past_diseases"]:
        past_diseases.append(past_disease["past_disease"])
    for heart_disease in j["heart_diseases"]:
        heart_diseases.append(heart_disease["heart_disease"])

    res = match_mentioned_concepts(past_diseases, heart_diseases, num_processes)
    res = prune(res, semmed_vocab_path)

    with open(output_path, "w") as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == "__main__":
    ground():