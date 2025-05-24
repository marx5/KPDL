# Phân loại Email Spam/Ham

Dự án sử dụng Mô hình học máy có giám sát sử dụng thuật toán Naive Bayes để phân loại email thành Spam và Ham (không phải spam). Hệ thống tự động xử lý dữ liệu, huấn luyện mô hình và tạo các biểu đồ phân tích.

## Cài đặt

1. Cài đặt Python 3.7 trở lên
2. Cài đặt thư viện:
```bash
pip install pandas scikit-learn matplotlib seaborn nltk wordcloud flask
```

## Cấu trúc dự án

- `main.py`: Chạy ứng dụng web
- `train_nb.py`: Huấn luyện mô hình và tạo biểu đồ
- `utils.py`: Các hàm tiện ích
- `spam_ham_dataset.csv`: Dữ liệu huấn luyện
- `new_emails.csv`: Email mới chờ xác nhận
- `model_nb.pkl`: Mô hình đã huấn luyện
- `vectorizer.pkl`: Vectorizer TF-IDF
- `eda_images/`: Biểu đồ phân tích
- `templates/`: Giao diện web

## Sử dụng

1. Chạy ứng dụng web:
```bash
python main.py
```

2. Truy cập ứng dụng:
- Mở trình duyệt và truy cập `http://localhost:5000`
- Nhập email cần kiểm tra
- Hệ thống sẽ tự động phân loại và hiển thị kết quả

3. Tính năng tự động:
- Mô hình sẽ tự động huấn luyện lần đầu khi chạy ứng dụng
- Khi có 10 email mới được gán nhãn, hệ thống sẽ tự động huấn luyện lại
- Có thể theo dõi quá trình huấn luyện qua trang `/train`

## Biểu đồ phân tích

- Tỷ lệ Spam/Ham
- Phân bố độ dài email (ký tự, từ, câu)
- Top từ phổ biến trong Spam/Ham
- Đánh giá hiệu suất mô hình
- Ma trận nhầm lẫn

## Ghi chú

- Lỗi được ghi vào `retrain_log.txt`
- Mô hình được cập nhật tự động khi có đủ 10 email mới có nhãn