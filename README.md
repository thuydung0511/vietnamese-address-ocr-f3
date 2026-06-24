# Nhận dạng địa chỉ tiếng Việt viết tay bằng VietOCR

Đồ án môn Đồ án Thị giác máy tính — Khoa Công nghệ Thông tin, Trường Đại học Xây dựng Hà Nội.

Fine-tune mô hình VietOCR (kiến trúc `vgg_transformer`) để nhận dạng địa chỉ viết tay tiếng Việt từ ảnh, kèm web demo bằng Gradio.

## Sinh viên thực hiện

| STT | Họ và tên          | Mã sinh viên |
| --- | ------------------ | ------------ |
| 1   | Phạm Thùy Dung     | 0205568      |
| 2   | Ngô Thị Thùy Linh  | 0210168      |
| 3   | Ninh Thị Thanh Vân | 0215368      |

Giảng viên hướng dẫn: Nguyễn Đình Quý

## Tổng quan

- Mô hình gốc `transformerocr.pth` được huấn luyện trên bộ dữ liệu chữ viết tay tiếng Việt tổng quát (`Viet-Handwriting-OCR-v2`, hơn 60.000 ảnh).
- Fine-tune trên bộ dữ liệu địa chỉ viết tay chuyên biệt (2.351 ảnh) để ra mô hình `best_address_vietocr.pth`.
- Đánh giá trên tập kiểm tra độc lập `data_test_1` (549 ảnh) bằng CER, WER, Exact Match Rate.

## Kết quả chính (trên tập kiểm tra, sau chuẩn hóa văn bản)

| Mô hình           | CER    | WER    | Exact Match Rate |
| ----------------- | ------ | ------ | ---------------- |
| VietOCR gốc       | 0.0882 | 0.2466 | 13.11%           |
| VietOCR fine-tune | 0.0408 | 0.1107 | 38.62%           |

## Cấu trúc thư mục

```
.
├── 00_prepare_v2_full.py          # Chuyển đổi data_v2_full từ Parquet sang JPEG
├── 01_check_dataset.py            # Kiểm tra tính đầy đủ của data_v2_full
├── 02_clean_labels.py             # Làm sạch nhãn data_v2_full
├── 02_train_vietocr_kaggle.py     # Huấn luyện mô hình gốc transformerocr.pth
├── 06_split_address.py            # Chia train/validation cho dữ liệu địa chỉ
├── 07_extract_vocab.py            # Trích xuất vocab 406 ký tự
├── 07_finetune_address_kaggle.py  # Fine-tune ra best_address_vietocr.pth
├── 11_evaluate_address_test.py    # Đánh giá RAW trên data_test_1
├── 12_check_leakage.py            # Kiểm tra rò rỉ dữ liệu (MD5 hash)
├── 13_evaluate_address_normalized.py  # Đánh giá NORMALIZED, mô hình fine-tune
├── 14_evaluate_base_model_on_test1.py # Đánh giá NORMALIZED, mô hình gốc
├── app.py                         # Web demo Gradio
├── requirements.txt               # Danh sách thư viện cần cài
├── address_train.txt              # Nhãn tập huấn luyện (2.115 ảnh)
├── address_val.txt                # Nhãn tập validation (236 ảnh)
├── annotation_test_1_standardized.txt  # Nhãn tập kiểm tra (549 ảnh)
├── vocab.txt                      # Bộ từ vựng 406 ký tự
└── archive/                       # Code thử nghiệm/giai đoạn đầu, không dùng cho kết quả cuối
```

## Lưu ý về dữ liệu và model không có trong repo

Do giới hạn dung lượng của GitHub, các file sau **không được đính kèm** trong repo:

- `transformerocr.pth`, `best_address_vietocr.pth` (mỗi file ~145MB)
- Toàn bộ thư mục ảnh: `data_train/`, `data_test_1/`, `data_v2_full/`

Liên hệ nhóm thực hiện nếu cần các file trên để chạy lại pipeline đầy đủ.

## Cách chạy web demo

```bash
pip install vietocr==0.3.13
pip install gradio
pip install numpy==1.26.4 --force-reinstall --no-cache-dir

python app.py
```

Sau khi chạy, mở link hiển thị trên terminal (mặc định `http://127.0.0.1:7860`) bằng trình duyệt.

## Thư viện và công nghệ sử dụng

- [VietOCR](https://github.com/pbcquoc/vietocr) — thư viện OCR tiếng Việt mã nguồn mở
- PyTorch — nền tảng deep learning
- Gradio — giao diện web demo
- Kaggle Notebook (GPU Tesla T4) — môi trường huấn luyện

## Dữ liệu sử dụng

- [Viet-Handwriting-OCR-v2](https://huggingface.co/datasets/5CD-AI/Viet-Handwriting-OCR-v2) — Hugging Face
- Dataset địa chỉ viết tay — Kaggle (xem chi tiết trong báo cáo)
- ~500 mẫu địa chỉ tự thu thập bởi nhóm thực hiện

## Báo cáo đầy đủ

Xem file báo cáo PDF đính kèm để có đầy đủ phân tích, số liệu thực nghiệm và phương pháp luận chi tiết.
