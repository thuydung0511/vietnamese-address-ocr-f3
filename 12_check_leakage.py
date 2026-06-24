from pathlib import Path
import hashlib

PROJECT_DIR = Path(r"D:\Data_training_TGMT_kaggle")

TRAIN_IMG_DIR = PROJECT_DIR / "data_train"
TEST_IMG_DIR = PROJECT_DIR / "data_test_1"

TRAIN_LABEL = PROJECT_DIR / "address_train.txt"
VAL_LABEL = PROJECT_DIR / "address_val.txt"
TEST_LABEL = PROJECT_DIR / "annotation_test_1_standardized.txt"


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_labels(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                img, text = line.rstrip("\n").split("\t", 1)
                data.append((img, text.strip()))
    return data


train_names = {p.name for p in TRAIN_IMG_DIR.glob("*")}
test_names = {p.name for p in TEST_IMG_DIR.glob("*")}

same_names = train_names & test_names

print("Trùng tên ảnh:", len(same_names))
print(list(sorted(same_names))[:20])


train_hashes = {}
for p in TRAIN_IMG_DIR.glob("*"):
    train_hashes[file_hash(p)] = p.name

test_hashes = {}
for p in TEST_IMG_DIR.glob("*"):
    test_hashes[file_hash(p)] = p.name

same_hashes = set(train_hashes.keys()) & set(test_hashes.keys())

print("Trùng nội dung ảnh:", len(same_hashes))

for h in list(same_hashes)[:20]:
    print("Train:", train_hashes[h], "| Test:", test_hashes[h])


train_texts = set()
for file in [TRAIN_LABEL, VAL_LABEL]:
    for _, text in read_labels(file):
        train_texts.add(text)

test_texts = {text for _, text in read_labels(TEST_LABEL)}

same_texts = train_texts & test_texts

print("Trùng text địa chỉ:", len(same_texts))
for text in list(sorted(same_texts))[:20]:
    print(text)