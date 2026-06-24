import random

random.seed(42)

input_file = "data_v2_full/labels_clean.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

split_idx = int(len(lines) * 0.95)

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open("data_v2_full/train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("data_v2_full/val.txt", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Train:", len(train_lines))
print("Val:", len(val_lines))