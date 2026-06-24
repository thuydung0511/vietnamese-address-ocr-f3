#1. Dòng có đúng format không?
#2. Có path ảnh không?
#3. Ảnh có tồn tại không?
#4. Text có rỗng không?

import os

input_file = "data_v2_full/labels.txt"
output_file = "data_v2_full/labels_clean.txt"

valid = 0
bad = 0

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line in f_in:
        line = line.strip()

        if not line:
            bad += 1
            continue

        parts = line.split("\t", 1)

        if len(parts) != 2:
            bad += 1
            continue

        img_path, text = parts
        text = text.replace("\t", " ").replace("\n", " ").strip()

        if not os.path.exists(img_path):
            bad += 1
            continue

        if text == "":
            bad += 1
            continue

        f_out.write(f"{img_path}\t{text}\n")
        valid += 1

print("Dong hop le:", valid)
print("Dong bi bo:", bad)
print("File moi:", output_file)