# Phân loại Email Spam/Ham

## Mô tả dự án
Dự án này sử dụng mô hình Naive Bayes để phân loại email thành hai nhóm: **Spam** và **Ham** (không phải spam). Dữ liệu được xử lý, huấn luyện và đánh giá tự động, đồng thời sinh ra các biểu đồ phân tích dữ liệu (EDA) giúp trực quan hóa đặc trưng của email.

---

## Cấu trúc thư mục
- `main.py`: File chính để chạy ứng dụng.
- `train_nb.py`: Script huấn luyện mô hình, sinh biểu đồ EDA, lưu model và vectorizer.
- `utils.py`: Lưu email mới, v.v.
- `spam_ham_dataset.csv`: Dataset gốc, chứa các email đã gán nhãn.
- `new_emails.csv`: Lưu các email mới chờ xác nhận nhãn để bổ sung vào dataset.
- `model_nb.pkl`: File lưu model Naive Bayes đã huấn luyện.
- `vectorizer.pkl`: File lưu vectorizer (TF-IDF) đã huấn luyện.
- `eda_images/`: Thư mục chứa các ảnh biểu đồ EDA và kết quả huấn luyện.
- `templates/`: Thư mục giao diện.

---

## Hướng dẫn cài đặt
1. **Cài đặt Python >= 3.7**
2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install pandas scikit-learn matplotlib seaborn nltk wordcloud
   ```

## Hướng dẫn sử dụng
### 1. Huấn luyện mô hình
Chạy script huấn luyện:
```bash
python train_nb.py
```
- Model và vectorizer mới sẽ được lưu vào `model_nb.pkl` và `vectorizer.pkl`.
- Các biểu đồ EDA sẽ được sinh ra trong thư mục `eda_images/`.

### 2. Thêm email mới
- Sử dụng hàm `save_new_email` trong `utils.py` để thêm email mới vào `new_emails.csv`.
- Sau khi gán nhãn cho email mới, chạy lại script huấn luyện để cập nhật model.

---

## Ý nghĩa các biểu đồ EDA
- **pie_ham_spam.png**: Biểu đồ tròn tỷ lệ email ham/spam.
- **hist_characters.png**: Phân bố số ký tự giữa ham và spam.
- **hist_words.png**: Phân bố số từ giữa ham và spam.
- **hist_sents.png**: Phân bố số câu giữa ham và spam.
- **top_spam_words.png**: Top 50 từ phổ biến nhất trong spam.
- **top_ham_words.png**: Top 50 từ phổ biến nhất trong ham.
- **cv_train_test_compare.png**: So sánh accuracy giữa train (cross-validation) và test.
- **cv_metric_mean_std.png**: Trung bình các chỉ số (accuracy, precision, recall, f1) trên 10-fold cross-validation.
- **confusion_matrix.png**: Ma trận nhầm lẫn trên tập test.

---

## Ghi chú
- Mọi lỗi sẽ được ghi vào file `retrain_log.txt`.

---