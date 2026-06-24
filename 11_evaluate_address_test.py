from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import torch


PROJECT_DIR = Path(r"D:\Data_training_TGMT_kaggle")

TEST_FILE = PROJECT_DIR / "annotation_test_1_standardized.txt"
MODEL_PATH = PROJECT_DIR / "best_address_vietocr.pth"
VOCAB_FILE = PROJECT_DIR / "vocab.txt"

RESULT_TXT = PROJECT_DIR / "evaluation_address_test_result.txt"
RESULT_CSV = PROJECT_DIR / "evaluation_address_test_result.csv"


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

    total_cer = 0
    total_wer = 0
    exact_match = 0
    valid_count = 0
    rows = []

    for line in tqdm(lines):
        if "\t" not in line:
            continue

        img_path, gt = line.split("\t", 1)
        full_img_path = PROJECT_DIR / img_path

        if not full_img_path.exists():
            print("Không tìm thấy ảnh:", full_img_path)
            continue

        img = Image.open(full_img_path).convert("RGB")
        pred = predictor.predict(img).strip()
        gt = gt.strip()

        cer = calc_cer(pred, gt)
        wer = calc_wer(pred, gt)
        is_exact = int(pred == gt)

        total_cer += cer
        total_wer += wer
        exact_match += is_exact
        valid_count += 1

        rows.append({
            "image": img_path,
            "ground_truth": gt,
            "prediction": pred,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": is_exact,
        })

    avg_cer = total_cer / valid_count if valid_count else 0
    avg_wer = total_wer / valid_count if valid_count else 0
    exact_rate = exact_match / valid_count if valid_count else 0

    summary = (
        "KẾT QUẢ ĐÁNH GIÁ VIETOCR - MODEL FINE-TUNE ĐỊA CHỈ\n"
        f"Số ảnh đánh giá  : {valid_count}\n"
        f"CER trung bình   : {avg_cer:.4f}\n"
        f"WER trung bình   : {avg_wer:.4f}\n"
        f"Exact Match Rate : {exact_rate:.4f}\n"
    )

    print(summary)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    with open(RESULT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "ground_truth", "prediction", "cer", "wer", "exact_match"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Đã lưu:", RESULT_TXT)
    print("Đã lưu:", RESULT_CSV)


if __name__ == "__main__":
    main()