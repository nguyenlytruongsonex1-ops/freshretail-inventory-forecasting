# Dự báo nhu cầu hàng tươi sống theo ngày trên FreshRetailNet-50K bằng LSTM

Đề tài môn IS6109.11.CH – Học sâu và ứng dụng trong kinh doanh.

Một mạng LSTM toàn cục dự báo doanh số 7 ngày tới cho từng sản phẩm tại từng cửa hàng, từ cửa sổ 28 ngày (doanh số, lịch, khuyến mãi, thời tiết) và các biến biết trước của 7 ngày tới. Seasonal Naive 7 ngày chỉ là mốc tham chiếu không học. Ngày hết hàng hơn 8 trong 16 giờ (6–22h) bị loại khỏi hàm mất mát.

Các nhánh:
- `main`: phần 1 (một lần chia, E1–E4).
- `full-version`: bản đầy đủ cũ (LightGBM và LSTM, khôi phục nhu cầu theo giờ, lớp tồn kho).
- `part2-rq3-rq5`: phần 2 (nhánh này), ablation tách nhóm và đánh giá độ tin cậy.

## Dữ liệu
- FreshRetailNet-50K của Dingdong-Inc: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- Giấy phép: CC BY 4.0 (theo dataset card trên Hugging Face).
- Dữ liệu không nằm trong repo. Notebook tự tải `train.parquet` và `eval.parquet` từ Hugging Face (revision cố định `08c1fab7f9257bc73679d415d65d644165d351d4`) và chỉ đọc các cột theo ngày.
- Tập con dùng trong notebook: 1,000 chuỗi lấy mẫu phân tầng (nhóm quản lý × mức hết hàng, seed 42), 97 ngày; chia theo thời gian 83 ngày train / 7 ngày validation / 7 ngày test (26/06–02/07/2024).

## Cấu trúc
```
notebooks/
  FreshRetailNet_LSTM.ipynb            notebook sạch (EDA, tiền xử lý in trước/sau từng bước, E1–E4)
  FreshRetailNet_LSTM_executed.ipynb   bản đã chạy trên CPU
  FreshRetailNet_LSTM_Part2.ipynb      phần 2: RQ3, RQ5 (sạch)
  FreshRetailNet_LSTM_Part2_executed.ipynb
reports/
  BaoCao_FreshRetailNet_LSTM.docx / .pdf         phần 1
  BaoCao_FreshRetailNet_LSTM_Part2.docx / .pdf   phần 2
results/part2/                         CSV kết quả và hình của notebook phần 2
requirements.txt
```

## Cách chạy
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter lab FreshRetailNet_LSTM.ipynb        # Run All
# hoặc: papermill FreshRetailNet_LSTM.ipynb out.ipynb -p N_SERIES 1000
```
Tham số ở ô Cấu hình: `N_SERIES`, `SEED`, `STOCKOUT_MAX_HOURS`, `WINDOW`, `HIDDEN`, `EPOCHS_MAX`, `RUN_E4`, ... Chạy không có mạng: đặt hai tệp parquet vào `notebooks/data/` và đặt `DATA_SOURCE = "local"`. Trên Colab chỉ cần `!pip install -q huggingface_hub` nếu thiếu.

Thời gian: khoảng 90 giây trên CPU (PyTorch 1 luồng) khi dữ liệu đã có trong bộ nhớ đệm, chưa tính thời gian tải dữ liệu lần đầu. Hai lần chạy liên tiếp cho kết quả giống hệt.

## Kết quả (1,000 chuỗi, một lần chạy, seed 42)
Tính trên 6,031 chuỗi – ngày test có không quá 8 giờ hết hàng; cột cuối tính trên toàn bộ 7,000 chuỗi – ngày.

| Thí nghiệm | WAPE | MAE | RMSE | WAPE (toàn bộ ngày) |
|---|---|---|---|---|
| E1 Seasonal Naive 7 ngày (tham chiếu) | 0.3859 | 0.4780 | 0.7893 | 0.4306 |
| E2 LSTM đầy đủ đặc trưng | 0.3248 | 0.4023 | 0.6553 | 0.3782 |
| E3 LSTM bỏ khuyến mãi và thời tiết | 0.3476 | 0.4306 | 0.7768 | 0.3965 |
| E4 LSTM không loại ngày hết hàng (phụ) | 0.3243 | 0.4017 | 0.6940 | 0.3737 |

LSTM giảm WAPE 15.8% so với Seasonal Naive. Bỏ khuyến mãi và thời tiết làm WAPE tăng từ 0.3248 lên 0.3476, chủ yếu ở ngày có khuyến mãi (0.3078 lên 0.3479). Đây là kết quả sơ bộ: một seed, một tuần test.

## Phần 2: ablation tách nhóm (RQ3) và độ tin cậy (RQ5)

Thay đổi so với phần 1:
- Thêm E3a (bỏ khuyến mãi: discount, activity_flag; giữ thời tiết và lịch) và E3b (bỏ thời tiết; giữ khuyến mãi và lịch); so với E2 và E3. Đo quá khớp bằng WAPE train so với validation và test.
- Chạy E1, E2, E3, E3a, E3b, E4 với 5 seed (42–46) trên 5 tuần kiểm tra trượt (29/05–02/07/2024). Mỗi tuần: huấn luyện đến tuần trước validation, 1 tuần validation cho early stopping, huấn luyện lại rồi dự báo; sửa discount, ngưỡng p99, chuẩn hóa tính lại chỉ từ ngày huấn luyện của tuần đó. Tuần cuối trùng phần 1.
- Kiểm định: Wilcoxon signed-rank trên MAE theo chuỗi (n = 1,000) với hiệu chỉnh Holm và độ lớn hiệu ứng (trung vị chênh lệch, % thay đổi, rank-biserial); phụ: Diebold–Mariano (HAC, độ trễ 6, hiệu chỉnh HLN) trên 35 ngày và Wilcoxon trên 25 lần chạy.
- Kiểm tra tất định: E2 seed 42 ở tuần cuối chạy hai lần cho dự báo giống hệt và WAPE 0.3248 trùng phần 1; chạy cả notebook hai lần cho CSV giống hệt.

Cách chạy: `jupyter lab notebooks/FreshRetailNet_LSTM_Part2.ipynb` (Run All), hoặc `papermill FreshRetailNet_LSTM_Part2.ipynb out.ipynb`. Tham số ở ô Cấu hình: `SEEDS`, `K_WINDOWS`, `MODELS`, `N_WORKERS` (mặc định 6 tiến trình, mỗi tiến trình 1 luồng torch). Thời gian đo được trên CPU 8 lõi: 14–20 phút cho 125 lần huấn luyện LSTM.

Kết quả (trung bình ± độ lệch chuẩn qua 5 seed × 5 tuần, chuỗi – ngày có không quá 8 giờ hết hàng):

| Thí nghiệm | WAPE | MAE | RMSE |
|---|---|---|---|
| E1 Seasonal Naive | 0.3941 ± 0.0130 | 0.4573 ± 0.0207 | 0.7615 ± 0.0366 |
| E2 LSTM đầy đủ | 0.3386 ± 0.0160 | 0.3927 ± 0.0188 | 0.6800 ± 0.0422 |
| E3 bỏ khuyến mãi và thời tiết | 0.3466 ± 0.0113 | 0.4025 ± 0.0247 | 0.6977 ± 0.0672 |
| E3a bỏ khuyến mãi | 0.3428 ± 0.0171 | 0.3978 ± 0.0224 | 0.6792 ± 0.0690 |
| E3b bỏ thời tiết | 0.3436 ± 0.0100 | 0.3988 ± 0.0205 | 0.7054 ± 0.0634 |
| E4 không loại ngày hết hàng | 0.3396 ± 0.0167 | 0.3942 ± 0.0253 | 0.7011 ± 0.0743 |

Wilcoxon trên MAE theo chuỗi, p sau hiệu chỉnh Holm: E2 so với E1 8.0e−101 (MAE −14.7%, rank-biserial 0.78); so với E3 2.8e−07 (−2.6%, 0.19); so với E3a 8.0e−12 (−1.4%, 0.26); so với E3b 1.00; so với E4 1.00. Ở mức ngày (Diebold–Mariano) và mức lần chạy, chỉ E2 so với E1 có ý nghĩa. Thời tiết làm khoảng cách test − train tăng nhẹ (E2 0.0371 so với E3b 0.0328).

So sánh với LightGBM, DLinear và TFT chưa làm, để lại cho hướng phát triển.
