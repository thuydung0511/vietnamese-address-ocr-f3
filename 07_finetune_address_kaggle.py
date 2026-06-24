# =========================================
# MỤC ĐÍCH:
# Fine-tune model VietOCR tổng quát bằng dữ liệu địa chỉ.
#
# INPUT:
# - data_train/
# - address_train.txt
# - address_val.txt
# - transformerocr.pth
#
# OUTPUT:
# - /kaggle/working/checkpoints_address/best_address_vietocr.pth
# - /kaggle/working/checkpoints_address/last_address_vietocr.pth
# =========================================

import os
from pathlib import Path

import torch
import torch.utils.data
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer


# =========================
# FIX DATALOADER CHO KAGGLE
# =========================

_original_dataloader_init = torch.utils.data.DataLoader.__init__

def _patched_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    _original_dataloader_init(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = _patched_init


# =========================
# ĐƯỜNG DẪN DATASET KAGGLE
# =========================

PROJECT_DIR = Path("/kaggle/input/datasets/phamdung0511/training-v-20/address_finetune_data")

TRAIN_FILE = PROJECT_DIR / "address_train.txt"
VAL_FILE = PROJECT_DIR / "address_val.txt"
PRETRAINED_MODEL = PROJECT_DIR / "transformerocr.pth"

CHECKPOINT_DIR = "/kaggle/working/checkpoints_address"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================
# KIỂM TRA FILE
# =========================

print("Train file:", TRAIN_FILE, TRAIN_FILE.exists())
print("Val file:", VAL_FILE, VAL_FILE.exists())
print("Pretrained model:", PRETRAINED_MODEL, PRETRAINED_MODEL.exists())
print("Data train folder:", (PROJECT_DIR / "data_train").exists())


# =========================
# TẠO VOCAB TỪ DỮ LIỆU ĐỊA CHỈ
# =========================

chars = set()

for file in [TRAIN_FILE, VAL_FILE]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                chars.update(list(parts[1]))

vocab = "".join(sorted(chars))

print("Vocab size:", len(vocab))
print("Vocab sample:", vocab[:200])


# =========================
# CẤU HÌNH VIETOCR
# =========================

config = Cfg.load_config_from_name("vgg_transformer")

config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

config["vocab"] = vocab
config["weights"] = str(PRETRAINED_MODEL)

config["dataset"]["name"] = "address_ocr"
config["dataset"]["data_root"] = str(PROJECT_DIR)
config["dataset"]["train_annotation"] = str(TRAIN_FILE)
config["dataset"]["valid_annotation"] = str(VAL_FILE)

config["trainer"]["batch_size"] = 16
config["trainer"]["print_every"] = 50
config["trainer"]["valid_every"] = 500
config["trainer"]["iters"] = 3000

config["trainer"]["num_workers"] = 0
config["dataset"]["num_workers"] = 0

config["trainer"]["export"] = os.path.join(
    CHECKPOINT_DIR,
    "best_address_vietocr.pth"
)

config["trainer"]["checkpoint"] = os.path.join(
    CHECKPOINT_DIR,
    "last_address_vietocr.pth"
)

config["optimizer"]["max_lr"] = 1e-4


print("Device:", config["device"])
print("Export:", config["trainer"]["export"])
print("Checkpoint:", config["trainer"]["checkpoint"])
print("Config OK")


# =========================
# FINE-TUNE
# =========================

trainer = Trainer(config)

print("Trainer OK")
print("Bắt đầu fine-tune địa chỉ...")

trainer.train()

print("\nFINE-TUNE DONE")
print("Best model:", config["trainer"]["export"])
print("Last checkpoint:", config["trainer"]["checkpoint"])

print("\nCác file .pth trong /kaggle/working:")
for root, dirs, files in os.walk("/kaggle/working"):
    for file in files:
        if file.endswith(".pth"):
            print(os.path.join(root, file))