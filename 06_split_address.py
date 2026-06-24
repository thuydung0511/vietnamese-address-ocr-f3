import random
from pathlib import Path

random.seed(42)

PROJECT_DIR = Path(r"D:\Data_training_TGMT_kaggle")

input_file = PROJECT_DIR / "train_standardized.txt"

train_output = PROJECT_DIR / "address_train.txt"
val_output = PROJECT_DIR / "address_val.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)

split_idx = int(len(lines) * 0.9)

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open(train_output, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(val_output, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Tổng số mẫu địa chỉ:", len(lines))
print("Train:", len(train_lines))
print("Val:", len(val_lines))
print("Đã tạo:", train_output)
print("Đã tạo:", val_output)

# python D:\Data_training_TGMT_kaggle\06_split_address.py