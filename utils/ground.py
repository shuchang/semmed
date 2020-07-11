import json
from tqdm import tqdm
def ground(semmed_vocab_path, hf_mapped_path, output_path):
    semmed_vocab = []
    with open(semmed_vocab_path, "r", encoding="utf-8") as fin:
        for i in fin:
            semmed_vocab.append(i.strip())
        print(semmed_vocab[4])
    with open(hf_mapped_path, "r", encoding="utf-8") as fin1, open(output_path, "w", encoding="utf-8") as fout:
        lines = [line for line in fin1]
        for line in tqdm(lines, total=len(lines)):
            j = json.loads(line)
            record_cui = j["medical_records"]["record_cui"]
            for l in record_cui:
                for cui in l:
                    if cui not in semmed_vocab:
                        l.remove(cui)
                if len(l) == 0:
                    record_cui.remove(l)
            j["medical_records"]["record_cui"] = record_cui
            fout.write(json.dumps(j) + "\n")

    print(f'grounded cui saved to {output_path}')
    print()

















if __name__ == "__main__":
    ground("../data/semmed/cui_vocab.txt", "../data/hfdata/converted/dev.jsonl", "../data/hfdata/grounded/dev_ground.jsonl")