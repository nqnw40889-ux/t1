# Dự đoán hỏng hóc máy với AI4I 2020

Ứng dụng Streamlit nhận số đo vận hành và trả điểm LightGBM, ngưỡng và nhãn
cảnh báo. Ảnh động cơ trên giao diện là minh họa 3D được tạo bằng AI.

## Chạy trên Windows

Cài Python 3.12, giải nén gói rồi mở `Chay_Streamlit.bat`.
Lần đầu, chương trình cài các thư viện vào môi trường Python riêng.
Mở http://localhost:8502 nếu trình duyệt chưa mở. Nhấn Ctrl+C trong cửa sổ
Terminal để dừng. Có thể mở đúng thư mục trong VS Code và chạy:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## Đưa lên GitHub và Streamlit Community Cloud

1. Giải nén ZIP. Mở thư mục `AI4I_Dashboard` có `app.py`.
2. Tạo repository GitHub. Dùng **Add file → Upload files** và kéo các tệp,
   thư mục trong `AI4I_Dashboard` vào trang tải lên, giữ cấu trúc thư mục.
   Đặt `app.py`, `requirements.txt` và `packages.txt` ở thư mục gốc của repo,
   cạnh `data/` và `assets/`. Đưa cả `.streamlit/config.toml` lên để có giao diện
   đúng màu và cấu hình tải tệp.
3. Vào https://share.streamlit.io/ → **Create app** → chọn repo và nhánh,
   điền **Main file path** là `app.py`. Trong **Advanced settings**, chọn
   **Python 3.12**, rồi bấm **Deploy**.
4. Mở URL ứng dụng Cloud sau khi cài đặt hoàn tất. Hai phiếu mẫu PNG
   tải được trực tiếp từ mục **Dự đoán từ ảnh**. Bộ 1.500 phiếu test là
   tệp tải riêng, chỉ cần giải nén khi muốn thử nhiều ảnh.

Hướng dẫn chính thức:
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Chức năng

| Mục | Cách dùng |
|---|---|
| Dự đoán | Nạp mẫu theo UDI hoặc lấy mẫu ngẫu nhiên, chỉnh sáu thông số rồi chạy. Có kiểm tra từng trường, đối chiếu bản ghi tương tự, lưu mốc để so kịch bản, giải thích điểm, so ba mô hình và kiểm tra khi số đo dao động. Tải phiếu HTML để mở trên trình duyệt hoặc in. |
| Dự đoán từ ảnh | PNG gốc đúng mẫu cho kết quả ngay. Ảnh JPG/PNG chụp phiếu dùng OCR và cần xác nhận hoặc sửa sáu giá trị. Khi UDI và sáu giá trị khớp tập test, web hiển thị nhãn thật. |
| Dự đoán hàng loạt | Dùng dữ liệu mẫu, tải CSV/Excel, chọn sheet hoặc dán từ bảng tính. Có xem trước, chỉnh sửa, lọc cảnh báo, xếp theo số lượt kiểm tra và xuất CSV. |
| Giám sát mô phỏng | Phát lại các mẫu test, tạm dừng hoặc nhảy đến cảnh báo tiếp theo. |
| Bảo trì | Xem lịch sử cảnh báo, lập phiếu, ghi chú và cập nhật trạng thái. |
| Thử ngưỡng và kết quả kiểm thử | Minh họa tác động của ngưỡng trên validation và xem số liệu test. |

Sáu cột dùng tên như trong `data/mau_du_doan.csv`. Excel/CSV dùng dấu
chấm cho số thập phân. Khi dán từ Excel, có thể dán cả tiêu đề hoặc
dán đúng sáu cột theo thứ tự của tệp mẫu. Trình sửa trên web áp dụng
cho 200 dòng đầu. Mỗi lô tối đa 20.000 dòng.

OCR ảnh chụp dùng Tesseract tiếng Anh vì tên trường trên phiếu là
tiếng Anh. Streamlit Community Cloud cài trình đọc qua `packages.txt`.
Trên Windows, có thể thử PNG gốc ngay. Để dùng OCR cho ảnh chụp JPG,
cài Tesseract có dữ liệu tiếng Anh và thêm thư mục chứa `tesseract.exe`
vào PATH. Chụp rõ cả tên trường, giá trị và đơn vị. Mức đọc phụ thuộc
độ nét và góc chụp.

## Dữ liệu và lưu trữ

AI4I là dữ liệu tổng hợp. Cảnh báo áp dụng cho thông số trong bản ghi.
Điểm LightGBM chưa được hiệu chuẩn thành xác suất hỏng ngoài thực tế.
Mốc phát hiện tổ hợp lạ là phân vị 99 của khoảng cách đến bản ghi train
gần nhất trên validation, tính riêng theo loại sản phẩm. Mốc dùng để
gắn cảnh báo về phạm vi đầu vào.

Cảnh báo và phiếu bảo trì ghi vào SQLite tại `data/maintenance.sqlite3`.
Trên Cloud, lịch sử thuộc phiên triển khai và có thể mất khi ứng dụng
khởi động lại hoặc được triển khai lại. Tải CSV để giữ bản sao nếu cần.

Chạy lại `python train.py` để huấn luyện LightGBM, sau đó chạy
`python train_comparison.py` để cập nhật hai mô hình đối chiếu và mốc
khoảng cách. Hướng dẫn chi tiết và số liệu có trong `HUONG_DAN.txt`.

Dữ liệu: Stefan Matzka, AI4I 2020 Predictive Maintenance Dataset,
UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C,
giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
