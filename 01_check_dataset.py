import os

label_file = "data_v2_full/labels.txt"

count = 0
missing = 0

with open(label_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")

        if len(parts) != 2:
            continue

        img_path, text = parts

        count += 1

        if not os.path.exists(img_path):
            missing += 1

print("Tong:", count)
print("Anh loi:", missing)