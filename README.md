# Phân bổ ngân sách coupon cá nhân hóa: từ dự báo và nhân quả đến hành động

Tiểu luận cuối kỳ môn Phân tích dữ liệu nâng cao (261MIE401601), Thạc sĩ Thương mại điện tử, Trường Đại học Kinh tế - Luật.
Học viên: Lê Nhật Thanh.

Mã nguồn chạy chu trình tám giai đoạn trên bộ dữ liệu dunnhumby *The Complete Journey*: thiết kế dữ liệu theo thời gian, dự báo có hiệu chỉnh, giải thích mô hình, suy luận nhân quả (Double ML, AIPW, DR-learner), Predict-Then-Optimize với SPO+, quy hoạch nguyên hỗn hợp (MILP), Action Router có rào chắn và con người tham gia (HITL), và giám sát sau triển khai (PSI, Action TVD, Release Gates, Current vs Challenger).

## Nội dung

| Tệp | Mô tả |
|---|---|
| `PTDLNC_Final.py` | Mã nguồn một tệp chạy tuần tự (PyCharm, terminal) |
| `PTDLNC_Final.ipynb` | Cùng mã nguồn dạng notebook (Google Colab) |
| `outputs/locked_seal.json`, `outputs/seal_log.jsonl` | Biên bản khóa Locked Test và Out-of-Time (thời điểm, mã băm dữ liệu, mô hình, cấu hình) |
| `requirements.txt` | Phiên bản thư viện đã chạy |

Bảng và hình kết quả không được đưa lên kho này; chúng được sinh lại khi chạy mã (thư mục `outputs/tables`, `outputs/figures`).

## Chạy trên Google Colab

[![Mở trong Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lenhatthanh2302/ptdlnc-coupon-allocation/blob/main/PTDLNC_Final.ipynb)

1. Tải dữ liệu (xem bên dưới) và tải thư mục `Dataset` chứa sáu tệp CSV lên `/content/Dataset`, hoặc đặt vào `MyDrive/PTDLNC_Final/Dataset` rồi mount Google Drive.
2. Chọn *Runtime → Run all*.

## Chạy trên máy cá nhân

```bash
pip install -r requirements.txt
# đặt thư mục Dataset (sáu tệp CSV) cạnh PTDLNC_Final.py
python PTDLNC_Final.py
```

Biến môi trường `PTDLNC_RUN_STAGE` chọn chế độ chạy:

- `dev` (mặc định): chỉ dùng Train và Validation; Locked Test và Out-of-Time vẫn bị niêm phong. Thời gian chạy khoảng 2,5 đến 4 phút.
- `final`: kiểm tra biên bản khóa trong `outputs/locked_seal.json`, rồi đánh giá Locked Test và Out-of-Time. Nếu mô hình huấn luyện lại khác bản đã khóa, chương trình dừng trước khi ghi tệp.

`PTDLNC_OUT_DIR` đổi thư mục đầu ra (mặc định là `outputs` cạnh thư mục `Dataset`).

## Dữ liệu

Bộ dữ liệu dunnhumby *The Complete Journey* không được lưu lại ở đây vì dung lượng và giấy phép. Tải từ nguồn gốc: <https://www.dunnhumby.com/sourcefiles>. Sáu tệp được dùng, kèm mã băm SHA-256 đã dùng để chạy và khóa:

| Tệp | SHA-256 |
|---|---|
| `transaction_data.csv` | `3a685c0729cef664d634486189f774518b84f53cde7cbf701a5963238692b476` |
| `campaign_desc.csv` | `632535c6bcdd97b5681e84c2c0652ddc4eaca8a0f899a3bb37dae8dbdc089684` |
| `campaign_table.csv` | `a716f88953c9ddf8847e191d862e8d022afd19cf7419f52cbe6dde950a121f21` |
| `coupon_redempt.csv` | `70a98f3bfb28be0dd850f19c8d806d2a23e44014546651367b21538db6fba1f8` |
| `hh_demographic.csv` | `aeb10246d0c4f3d3433d1f441265b6c7e70b0926129b834a3b589517948b9f22` |
| `product.csv` | `7ecbcec41e0f1e5a51b43a359965cc50dc5586a2b900a378b64032750fedc949` |

## Lưu ý về tính tái lập

- Hạt giống ngẫu nhiên cố định là 2026. Dấu vân tay mô hình trong biên bản khóa tính theo dự báo trên một tập dò cố định, nên bền hơn mã băm tệp mô hình khi đổi máy; tuy vậy khác biệt lớn về phiên bản thư viện (đặc biệt scikit-learn) vẫn có thể làm lệch dự báo và khiến chế độ `final` dừng lại. Phiên bản đã chạy được ghi trong `requirements.txt`.
- Sau khi khóa, mã nguồn chỉ được sửa duy nhất đoạn đặt chú thích của hình chia đợt; mô hình, cấu hình, dữ liệu và kết quả không đổi. Vì vậy mã băm mã nguồn trong biên bản khóa khác mã băm của tệp hiện tại.
