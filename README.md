# Dự báo nhu cầu hàng tươi sống có xét hết hàng phục vụ tồn kho (FreshRetailNet-50K)

Đề tài môn IS6109.11.CH – Học sâu và ứng dụng trong kinh doanh.

So sánh LightGBM và LSTM trong dự báo nhu cầu 7 ngày cho từng sản phẩm tại từng cửa hàng. Doanh số bị kiểm duyệt khi hết hàng, nên đề tài thử ba cách xử lý giờ hết hàng: dùng doanh số thô, che ngày hết hàng, và khôi phục nhu cầu. Dự báo được đưa vào lớp quyết định tồn kho S = D̂ + z·σₑ với mức phục vụ 80/90/95%.

## Dữ liệu
- Bộ dữ liệu: FreshRetailNet-50K của Dingdong-Inc, https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- Giấy phép: CC BY 4.0 (theo dataset card trên Hugging Face).
- 50,000 chuỗi cửa hàng × sản phẩm, 90 ngày train và 7 ngày eval, có doanh số và trạng thái hết hàng theo giờ.
- Dữ liệu không nằm trong repo. Notebook tự tải `data/train.parquet` và `data/eval.parquet` (khoảng 115 MB) từ Hugging Face, với revision cố định `08c1fab7f9257bc73679d415d65d644165d351d4`.

## Cấu trúc
```
notebooks/
  FreshRetailNet_Inventory_Experiments.ipynb               notebook sạch, chưa chạy
  FreshRetailNet_Inventory_Experiments_executed_5000.ipynb bản đã chạy trên 5,000 chuỗi (CPU)
reports/
  BaoCao_DeXuat_FreshRetailNet_Inventory.docx / .pdf       báo cáo đề xuất
requirements.txt                                           phiên bản thư viện đã chạy kiểm chứng
```

## Cách chạy
Máy cục bộ (Python 3.10 trở lên):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab notebooks/FreshRetailNet_Inventory_Experiments.ipynb   # Run All
```
Máy có GPU NVIDIA: cài torch bản CUDA theo pytorch.org trước.

Chạy không giao diện bằng papermill:
```bash
cd notebooks
papermill FreshRetailNet_Inventory_Experiments.ipynb out_5000.ipynb -p SUBSET_N_SERIES 5000
```

Google Colab:
1. Mở notebook trên Colab. Chọn runtime T4 GPU (chỉ LSTM dùng GPU).
2. Bỏ comment ô đầu tiên: `!pip install -q lightgbm==4.6.0 psutil`.
3. Run all.

Tham số nằm ở ô CONFIG: `SUBSET_N_SERIES` (mặc định 5000, `None` là toàn bộ 50,000 chuỗi), `DEVICE`, `LSTM_EPOCHS`, `LSTM_SEEDS`, `N_JOBS`, `LGB_ORIGIN_STRIDE`.

Chạy không có mạng: đặt hai tệp parquet vào thư mục `sources/` ở gốc repo và đặt `DATA_SOURCE = "local"`.

Thời gian đo được với 5,000 chuỗi trên CPU 8 lõi: khoảng 11 phút, RAM đỉnh 4.3 GB. Toàn bộ 50,000 chuỗi chưa chạy thật. Ước tính 2–4 giờ trên CPU 8 lõi và 8–11 GB RAM. Trên Colab nên đặt `LGB_ORIGIN_STRIDE = 2`.

## Kết quả sơ bộ (tập con 5,000 chuỗi)
Đây là kết quả sơ bộ, chưa phải kết quả cuối cùng. Chạy một lần trên tập con 5,000 chuỗi lấy mẫu phân tầng, CPU, 03/10/2026. Sai số tính trên giờ còn hàng của 7 ngày kiểm tra.

| Thí nghiệm | WAPE | WPE | MAE | RMSE |
|---|---|---|---|---|
| E1 Seasonal Naive | 0.386 | −0.121 | 0.473 | 0.794 |
| E2 LightGBM, doanh số thô | 0.344 | −0.207 | 0.422 | 0.802 |
| E3 LightGBM, nhu cầu khôi phục | 0.303 | −0.068 | 0.371 | 0.619 |
| E3m LightGBM, che ngày hết hàng | 0.314 | −0.165 | 0.385 | 0.663 |
| A1 E3 bỏ thời tiết | 0.304 | −0.070 | 0.373 | 0.627 |
| A2 E3 bỏ khuyến mãi | 0.318 | −0.046 | 0.390 | 0.643 |
| E4 LSTM, doanh số thô | 0.336 | −0.141 | 0.411 | 0.722 |
| E5 LSTM, nhu cầu khôi phục | 0.323 | 0.016 | 0.396 | 0.656 |

Lớp tồn kho ở mức phục vụ 90% (chính sách z·σₑ, các ngày kiểm tra còn hàng đủ 6–22 giờ):

| Thí nghiệm | Tỷ lệ hết hàng | Fill rate | Tỷ lệ hao hụt | Chi phí newsvendor so với E1 |
|---|---|---|---|---|
| E1 | 0.073 | 0.970 | 0.707 | 1.000 |
| E2 | 0.095 | 0.955 | 0.478 | 0.910 |
| E3 | 0.059 | 0.978 | 0.583 | 0.803 |
| E3m | 0.079 | 0.974 | 0.499 | 0.756 |
| E4 | 0.078 | 0.970 | 0.556 | 0.845 |
| E5 | 0.045 | 0.983 | 0.699 | 0.871 |

Kiểm chứng khôi phục bằng che nhân tạo: WAPE giảm từ 0.315 (doanh số bị kiểm duyệt) xuống 0.191 (nhu cầu khôi phục).

## Trích dẫn dữ liệu
Y. Wang et al., "FreshRetailNet-50K: A stockout-annotated censored demand dataset for latent demand recovery and forecasting in fresh retail," arXiv:2505.16319, 2025.
