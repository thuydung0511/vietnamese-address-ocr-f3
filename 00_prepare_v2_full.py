import pandas as pd
from PIL import Image
import io
import os

parquet_files = [
    "data_v2/train-00000-of-00003.parquet",
    "data_v2/train-00001-of-00003.parquet",
    "data_v2/train-00002-of-00003.parquet",
]

output_img_dir = "data_v2_full/images"
output_label_path = "data_v2_full/labels.txt"

os.makedirs(output_img_dir, exist_ok=True)

count = 0

with open(output_label_path, "w", encoding="utf-8") as f:
    for parquet_path in parquet_files:
        print("Dang doc:", parquet_path)
        df = pd.read_parquet(parquet_path)

        for i in range(len(df)):
            text = str(df.iloc[i]["text"]).strip()

            if text == "":
                continue

            img_bytes = df.iloc[i]["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            img_name = f"v2_{count:06d}.jpg"
            img_path = os.path.join(output_img_dir, img_name)

            img.save(img_path, quality=95)

            f.write(f"{img_path}\t{text}\n")

            count += 1

        print("Da xu ly xong:", parquet_path)

print("Tong so mau da xuat:", count)
print("Thu muc anh:", output_img_dir)
print("File nhan:", output_label_path)