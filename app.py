"""
Web demo OCR nhận dạng địa chỉ tiếng Việt viết tay - dùng VietOCR (model fine-tune)
Chạy local trên Windows, dùng Gradio.

Cách chạy:
    cd D:\\Data_training_TGMT_kaggle
    .\\venv\\Scripts\\Activate
    python app.py
"""

import os
import time

import gradio as gr
from PIL import Image

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


# ===================== CẤU HÌNH ĐƯỜNG DẪN =====================
BASE_DIR = r"D:\Data_training_TGMT_kaggle"
MODEL_PATH = os.path.join(BASE_DIR, "best_address_vietocr.pth")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.txt")

DEVICE = "cpu"  # Máy chỉ có CPU, không có GPU rời


# ===================== LOAD MODEL (chỉ load 1 lần khi khởi động) =====================
def load_predictor():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
    if not os.path.isfile(VOCAB_PATH):
        raise FileNotFoundError(f"Không tìm thấy vocab tại: {VOCAB_PATH}")

    # Đọc vocab ở BINARY mode rồi tự decode UTF-8.
    # Lý do: open(path, "r") ở Windows tự động dịch \r\n -> \n (universal newline),
    # nếu vocab gốc có \r\n được tính là token riêng thì sẽ bị mất 1 ký tự khi đọc
    # bằng text mode thông thường. Đọc binary + decode thủ công giữ đúng nguyên bản.
    with open(VOCAB_PATH, "rb") as f:
        raw_bytes = f.read()
    vocab = raw_bytes.decode("utf-8")

    print(f"Đã đọc vocab, độ dài: {len(vocab)} ký tự, repr 10 ký tự cuối: {repr(vocab[-10:])}")

    config = Cfg.load_config_from_name("vgg_transformer")
    config["weights"] = MODEL_PATH
    config["vocab"] = vocab
    config["device"] = DEVICE
    config["cnn"]["pretrained"] = False  # đang load weight đã train, không cần pretrained ImageNet lại

    predictor = Predictor(config)
    print("Model đã load xong, sẵn sàng nhận dạng.")
    return predictor


print("Đang load model VietOCR, vui lòng chờ...")
predictor = load_predictor()


# ===================== HÀM DỰ ĐOÁN =====================
def predict(image: Image.Image):
    if image is None:
        return "Vui lòng upload ảnh trước khi nhận dạng.", ""

    # Convert ảnh sang RGB trước khi predict (bắt buộc theo yêu cầu)
    image = image.convert("RGB")

    start_time = time.time()
    result = predictor.predict(image)
    elapsed = time.time() - start_time

    time_str = f"{elapsed:.3f} giây"
    return result, time_str


# ===================== GIAO DIỆN GRADIO =====================
with gr.Blocks(title="OCR Địa chỉ Tiếng Việt - VietOCR Demo") as demo:
    gr.Markdown("# 📮 Demo OCR Nhận dạng Địa chỉ Tiếng Việt Viết Tay")
    gr.Markdown(
        "Upload ảnh chứa địa chỉ viết tay, sau đó bấm **Nhận dạng** để xem kết quả OCR "
        "(model VietOCR đã fine-tune trên dữ liệu địa chỉ)."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Ảnh địa chỉ (upload tại đây)", height=400)
            predict_btn = gr.Button("🔍 Nhận dạng", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Kết quả OCR", lines=3)
            output_time = gr.Textbox(label="Thời gian dự đoán", lines=1)

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_text, output_time],
    )

    gr.Markdown(f"*Model: `best_address_vietocr.pth` | Device: `{DEVICE}`*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)