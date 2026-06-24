from pathlib import Path

DATA_DIR = Path(r"D:\Data_training_TGMT_kaggle\data_v2_full")

chars = set()

for file in [DATA_DIR / "train.txt", DATA_DIR / "val.txt"]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                chars.update(list(parts[1]))

vocab = "".join(sorted(chars))

print("=" * 80)
print("VOCAB SIZE:", len(vocab))
print("=" * 80)
print(vocab)
print("=" * 80)

#python D:\Data_training_TGMT_kaggle\07_extract_vocab.py