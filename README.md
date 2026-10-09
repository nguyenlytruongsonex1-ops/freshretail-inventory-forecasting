# Dự báo nhu cầu hàng tươi sống theo ngày trên FreshRetailNet-50K bằng LSTM

Đề tài môn IS6109.11.CH – Học sâu và ứng dụng trong kinh doanh.

Một mạng LSTM toàn cục dự báo doanh số 7 ngày tới cho từng sản phẩm tại từng cửa hàng, từ cửa sổ 28 ngày (doanh số, lịch, khuyến mãi, thời tiết) và các biến biết trước của 7 ngày tới. Seasonal Naive 7 ngày chỉ là mốc tham chiếu không học. Ngày hết hàng hơn 8 trong 16 giờ (6–22h) bị loại khỏi hàm mất mát.

Bản đầy đủ trước đây (LightGBM và LSTM, khôi phục nhu cầu theo giờ, lớp tồn kho) nằm ở nhánh `full-version`.

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
reports/
  BaoCao_FreshRetailNet_LSTM.docx / .pdf
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
