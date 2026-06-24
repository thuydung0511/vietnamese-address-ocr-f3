def fix_file(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []

    for line in lines:
        parts = line.strip().split("\t", 1)
        if len(parts) != 2:
            continue

        img_path, text = parts

        img_path = img_path.replace("data_v2_full/", "")
        img_path = img_path.replace("\\", "/")

        fixed_lines.append(f"{img_path}\t{text}\n")

    with open(input_path, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)

    print("Da sua:", input_path, "So dong:", len(fixed_lines))


fix_file("data_v2_full/train.txt")
fix_file("data_v2_full/val.txt")