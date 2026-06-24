from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import torch
import re
import unicodedata


PROJECT_DIR = Path(r"D:\Data_training_TGMT_kaggle")

TEST_FILE = PROJECT_DIR / "annotation_test_1_standardized.txt"
MODEL_PATH = PROJECT_DIR / "best_address_vietocr.pth"
VOCAB_FILE = PROJECT_DIR / "vocab.txt"

RESULT_TXT = PROJECT_DIR / "evaluation_address_normalized_result.txt"
RESULT_CSV = PROJECT_DIR / "evaluation_address_normalized_result.csv"


def normalize_text(s):
    s = unicodedata.normalize("NFC", str(s))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[.,;:]", "", s)
    return s


def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m]


def calc_cer(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, gt) / len(gt)


def calc_wer(pred, gt):
    pred_words = pred.strip().split()
    gt_words = gt.strip().split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein_distance(pred_words, gt_words) / len(gt_words)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = f.read()

    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = device
    config["weights"] = str(MODEL_PATH)
    config["vocab"] = vocab

    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Test file:", TEST_FILE)
    print("Vocab size:", len(vocab))

    predictor = Predictor(config)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_cer_raw = 0
    total_wer_raw = 0
    exact_raw = 0

    total_cer_norm = 0
    total_wer_norm = 0
    exact_norm = 0

    valid_count = 0
    rows = []

    for line in tqdm(lines):
        if "\t" not in line:
            continue

        img_path, gt_raw = line.split("\t", 1)
        full_img_path = PROJECT_DIR / img_path

        if not full_img_path.exists():
            print("Không tìm thấy ảnh:", full_img_path)
            continue

        img = Image.open(full_img_path).convert("RGB")

        pred_raw = predictor.predict(img).strip()
        gt_raw = gt_raw.strip()

        pred_norm = normalize_text(pred_raw)
        gt_norm = normalize_text(gt_raw)

        cer_raw = calc_cer(pred_raw, gt_raw)
        wer_raw = calc_wer(pred_raw, gt_raw)
        is_exact_raw = int(pred_raw == gt_raw)

        cer_norm = calc_cer(pred_norm, gt_norm)
        wer_norm = calc_wer(pred_norm, gt_norm)
        is_exact_norm = int(pred_norm == gt_norm)

        total_cer_raw += cer_raw
        total_wer_raw += wer_raw
        exact_raw += is_exact_raw

        total_cer_norm += cer_norm
        total_wer_norm += wer_norm
        exact_norm += is_exact_norm

        valid_count += 1

        rows.append({
            "image": img_path,
            "ground_truth_raw": gt_raw,
            "prediction_raw": pred_raw,
            "ground_truth_norm": gt_norm,
            "prediction_norm": pred_norm,
            "cer_raw": round(cer_raw, 4),
            "wer_raw": round(wer_raw, 4),
            "exact_raw": is_exact_raw,
            "cer_norm": round(cer_norm, 4),
            "wer_norm": round(wer_norm, 4),
            "exact_norm": is_exact_norm,
        })

    avg_cer_raw = total_cer_raw / valid_count if valid_count else 0
    avg_wer_raw = total_wer_raw / valid_count if valid_count else 0
    exact_rate_raw = exact_raw / valid_count if valid_count else 0

    avg_cer_norm = total_cer_norm / valid_count if valid_count else 0
    avg_wer_norm = total_wer_norm / valid_count if valid_count else 0
    exact_rate_norm = exact_norm / valid_count if valid_count else 0

    summary = (
        "KẾT QUẢ ĐÁNH GIÁ VIETOCR - RAW VS NORMALIZED\n"
        f"Số ảnh đánh giá       : {valid_count}\n\n"
        "RAW RESULT\n"
        f"CER trung bình        : {avg_cer_raw:.4f}\n"
        f"WER trung bình        : {avg_wer_raw:.4f}\n"
        f"Exact Match Rate      : {exact_rate_raw:.4f}\n\n"
        "NORMALIZED RESULT\n"
        f"CER trung bình        : {avg_cer_norm:.4f}\n"
        f"WER trung bình        : {avg_wer_norm:.4f}\n"
        f"Exact Match Rate      : {exact_rate_norm:.4f}\n"
    )

    print(summary)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    with open(RESULT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ground_truth_raw",
                "prediction_raw",
                "ground_truth_norm",
                "prediction_norm",
                "cer_raw",
                "wer_raw",
                "exact_raw",
                "cer_norm",
                "wer_norm",
                "exact_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Đã lưu:", RESULT_TXT)
    print("Đã lưu:", RESULT_CSV)


if __name__ == "__main__":
    main()