# =========================================
# MỤC ĐÍCH FILE:
# Fine-tune VietOCR trên Kaggle GPU.
# File này train model và lưu checkpoint.
#
# INPUT:
# - train_aug.txt
# - valid.txt
# - data_train_aug/
#
# OUTPUT:
# - checkpoints_vietocr/best_vietocr.pth
# - checkpoints_vietocr/last.pth
#
# CHẠY TRÊN KAGGLE:
# python 02_train_vietocr_kaggle.py
# =========================================

import os
import torch
import torch.utils.data
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

_original_dataloader_init = torch.utils.data.DataLoader.__init__

def _patched_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    _original_dataloader_init(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = _patched_init

PROJECT_DIR = "/kaggle/working/Data_training_TGMT_kaggle"
CHECKPOINT_DIR = "/kaggle/working/checkpoints_vietocr"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

config = Cfg.load_config_from_name("vgg_transformer")

config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

config["dataset"]["name"] = "address_ocr"
config["dataset"]["data_root"] = PROJECT_DIR
config["dataset"]["train_annotation"] = os.path.join(PROJECT_DIR, "train_aug.txt")
config["dataset"]["valid_annotation"] = os.path.join(PROJECT_DIR, "valid.txt")

config["trainer"]["batch_size"] = 16
config["trainer"]["print_every"] = 50
config["trainer"]["valid_every"] = 500
config["trainer"]["iters"] = 15000

config["trainer"]["num_workers"] = 0
config["dataset"]["num_workers"] = 0

config["trainer"]["export"] = os.path.join(
    CHECKPOINT_DIR,
    "best_vietocr.pth"
)

config["trainer"]["checkpoint"] = os.path.join(
    CHECKPOINT_DIR,
    "last.pth"
)

print("Device:", config["device"])
print("Train annotation:", config["dataset"]["train_annotation"])
print("Valid annotation:", config["dataset"]["valid_annotation"])
print("Export:", config["trainer"]["export"])
print("Checkpoint:", config["trainer"]["checkpoint"])

trainer = Trainer(config, pretrained=True)
trainer.train()

print("\nTRAIN DONE")
print("Best model:", config["trainer"]["export"])
print("Last checkpoint:", config["trainer"]["checkpoint"])