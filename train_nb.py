import pandas as pd
import pickle
import datetime
import os
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import time
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Đảm bảo đã tải stopwords, punkt
nltk.download('punkt')
nltk.download('stopwords')

start_time = time.time()
log_file = "retrain_log.txt"

IMG_DIR = "eda_images"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

def log(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(f"{timestamp} {message}\n")
    print(f"{timestamp} {message}", flush=True)

def transformed_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    lst = []
    for i in text:
        if i.isalnum():
            lst.append(i)
    text = lst[:]
    lst.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            lst.append(i)
    text = lst[:]
    lst.clear()
    for i in text:
        lst.append(ps.stem(i))
    return " ".join(lst)

try:
    log("Đang khởi động huấn luyện lại mô hình...")

    # Đọc dữ liệu gốc
    df = pd.read_csv('spam_ham_dataset.csv')
    if df.columns[0] == '':
        df = df.drop(df.columns[0], axis=1)
    df = df[['label', 'text']].drop_duplicates().dropna()
    log(f"Đọc {df.shape[0]} mẫu từ spam_ham_dataset.csv.")

    # Đọc thêm dữ liệu mới (nếu có)
    added_new = 0
    df_new = None
    try:
        df_new = pd.read_csv('new_emails.csv')
        if 'label' in df_new.columns and 'text' in df_new.columns and 'label_num' in df_new.columns:
            df_new = df_new[df_new['label'].notnull() & (df_new['label'] != '')]
            if not df_new.empty:
                df_new = df_new[['label', 'text', 'label_num']]
                added_new = df_new.shape[0]
                log(f"Đã tìm thấy {added_new} mail mới từ new_emails.csv để huấn luyện lại.")
            else:
                df_new = None
                log("new_emails.csv không có mail mới có nhãn thực.")
        else:
            df_new = None
            log("new_emails.csv không có nhãn thực hoặc không có cột text.")
    except Exception as e:
        log(f"Không có new_emails.csv hoặc lỗi khi đọc file: {str(e)}")
        df_new = None

    # Kiểm tra sự tồn tại của model cũ
    has_old_model = os.path.isfile('model_nb.pkl') and os.path.isfile('vectorizer.pkl')

    # Điều kiện dừng
    if (df_new is None or added_new == 0) and has_old_model:
        log("Không có mail mới và đã có model. Dừng huấn luyện lại, giữ nguyên model cũ.")
        exit(0)

    # Nếu có mail mới thì gộp vào dữ liệu train
    if df_new is not None and added_new > 0:
        df = pd.concat([df, df_new], ignore_index=True)
    counts = df['label'].value_counts()
    log(f"Tổng mẫu sau gộp: {df.shape[0]} | Ham: {counts.get('ham', 0)} | Spam: {counts.get('spam', 0)}")

    # Mã hóa nhãn
    log("Đang mã hóa nhãn...")
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])
    log("Đã mã hóa nhãn.")

    # Loại bỏ dòng rỗng text (nếu có)
    df = df[df['text'].notnull()]
    df = df[df['text'].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    # Tiền xử lý chuẩn: tạo transformed_text
    log("Đang tiền xử lý dữ liệu văn bản bằng transformed_text...")
    df['transformed_text'] = df['text'].apply(transformed_text)

    # ==== VẼ BIỂU ĐỒ EDA VÀ LƯU ẢNH ====
    # Biểu đồ tròn tỷ lệ ham/spam
    plt.figure(figsize=(6, 6))
    counts_pie = df['label'].value_counts()
    labels = ['ham', 'spam'] if 0 in counts_pie.index and 1 in counts_pie.index else counts_pie.index
    plt.pie(counts_pie, labels=labels, autopct='%.2f')
    plt.title('Tỷ lệ Ham vs Spam')
    plt.savefig(f'{IMG_DIR}/pie_ham_spam.png')
    plt.close()

    # Histogram số ký tự
    df['num_characters'] = df['text'].apply(len)
    plt.figure(figsize=(12, 4))
    sns.histplot(df[df['label'] == 0]['num_characters'], label='ham', kde=False)
    sns.histplot(df[df['label'] == 1]['num_characters'], label='spam', kde=False, color='red')
    plt.title('Phân bố số ký tự giữa Ham và Spam')
    plt.xlabel('Số ký tự')
    plt.ylabel('Số lượng')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/hist_characters.png')
    plt.close()

    # Histogram số từ
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    plt.figure(figsize=(12, 4))
    sns.histplot(df[df['label'] == 0]['num_words'], label='ham', kde=False)
    sns.histplot(df[df['label'] == 1]['num_words'], label='spam', kde=False, color='red')
    plt.title('Phân bố số từ giữa Ham và Spam')
    plt.xlabel('Số từ')
    plt.ylabel('Số lượng')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/hist_words.png')
    plt.close()

    # Histogram số câu
    df['num_sents'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    plt.figure(figsize=(12, 4))
    sns.histplot(df[df['label'] == 0]['num_sents'], label='ham', kde=False)
    sns.histplot(df[df['label'] == 1]['num_sents'], label='spam', kde=False, color='red')
    plt.title('Phân bố số câu giữa Ham và Spam')
    plt.xlabel('Số câu')
    plt.ylabel('Số lượng')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/hist_sents.png')
    plt.close()

    # Bar chart 50 từ phổ biến nhất trong spam
    spam_words = []
    for text in df[df['label'] == 1]['transformed_text']:
        spam_words += text.split()
    data_spam_word = pd.DataFrame(Counter(spam_words).most_common(50), columns=['Word', 'Frequency'])
    plt.figure(figsize=(14, 6))
    sns.barplot(data=data_spam_word, x='Word', y='Frequency', palette='Reds_r')
    plt.xticks(rotation=90)
    plt.title('Top 50 từ phổ biến nhất trong Spam')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/top_spam_words.png')
    plt.close()

    # Bar chart 50 từ phổ biến nhất trong ham
    ham_words = []
    for text in df[df['label'] == 0]['transformed_text']:
        ham_words += text.split()
    data_ham_word = pd.DataFrame(Counter(ham_words).most_common(50), columns=['Word', 'Frequency'])
    plt.figure(figsize=(14, 6))
    sns.barplot(data=data_ham_word, x='Word', y='Frequency', palette='Blues')
    plt.xticks(rotation=90)
    plt.title('Top 50 từ phổ biến nhất trong Ham')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/top_ham_words.png')
    plt.close()

    # Vector hóa trên transformed_text
    log("Đang vector hóa dữ liệu (trên transformed_text)...")
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['label'].values
    log(f"Đã vector hóa dữ liệu. Kích thước X: {X.shape}")

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # Đánh giá cross-validation trên train (10-fold)
    log("Bắt đầu đánh giá mô hình bằng cross-validation (10-fold) trên tập train...")
    nb_clf = MultinomialNB()
    cv_results = cross_validate(
        nb_clf,
        X_train,
        y_train,
        cv=10,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_estimator=True
    )
    log(f"Accuracy (train, 10-fold): {cv_results['test_accuracy']}")
    log(f"Precision (train, 10-fold): {cv_results['test_precision']}")
    log(f"Recall (train, 10-fold): {cv_results['test_recall']}")
    log(f"F1 (train, 10-fold): {cv_results['test_f1']}")
    log("Trung bình các chỉ số trên train (10-fold):")
    log(f"Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    log(f"Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
    log(f"Recall: {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
    log(f"F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

    arr_nb = np.array([model.score(X_test, y_test) for model in cv_results['estimator']])
    log("Score của từng estimator (các mô hình train ở mỗi fold) trên tập test:")
    log(str(arr_nb))
    log(f"Score trung bình trên test: {arr_nb.mean():.4f} ± {arr_nb.std():.4f}")

    # --- Biểu đồ cross-validation ---
    # Bar so sánh accuracy estimator trên train và test
    bar_width = 0.35
    indices = np.arange(len(cv_results['test_accuracy']))
    plt.figure(figsize=(10, 6))
    plt.bar(indices, cv_results['test_accuracy'], width=bar_width, label='Train (cross-validation)')
    plt.bar(indices + bar_width, arr_nb, width=bar_width, label='Test (estimator)', color='orange')
    plt.xlabel('Estimator (Fold)')
    plt.ylabel('Accuracy')
    plt.title('So sánh Accuracy: Train (CV) vs Test theo từng Estimator')
    plt.xticks(indices + bar_width/2, [f'{i+1}' for i in range(len(cv_results['test_accuracy']))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/cv_train_test_compare.png")
    plt.close()

    # Bar các chỉ số trung bình (mean ± std) trên 10-fold
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    means = [
        cv_results['test_accuracy'].mean(),
        cv_results['test_precision'].mean(),
        cv_results['test_recall'].mean(),
        cv_results['test_f1'].mean()
    ]
    stds = [
        cv_results['test_accuracy'].std(),
        cv_results['test_precision'].std(),
        cv_results['test_recall'].std(),
        cv_results['test_f1'].std()
    ]
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, means, yerr=stds, capsize=8, color=['blue', 'green', 'orange', 'red'])
    plt.title('Trung bình các chỉ số trên 10-fold Cross-Validation (Train)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/cv_metric_mean_std.png")
    plt.close()

    # Huấn luyện mô hình cuối cùng trên tập train đầy đủ
    log("Đang huấn luyện mô hình cuối cùng trên tập train đầy đủ...")
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    log(f"Hiệu suất (test 30%): Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    log("Classification report:\n" + classification_report(y_test, y_pred))

    # Lưu model và vectorizer
    log("Đang lưu model và vectorizer...")
    pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
    pickle.dump(nb_clf, open('model_nb.pkl', 'wb'))
    end_time = time.time()
    log(f"Đã retrain và lưu model mới! Thời gian: {end_time - start_time:.2f} giây")

    # Chuyển mail mới sang file gốc, reset file new_emails.csv
    if df_new is not None and added_new > 0:
        log("Đang chuyển toàn bộ mail mới sang spam_ham_dataset.csv...")
        df_new_to_save = df_new.copy()
        df_old_full = pd.read_csv('spam_ham_dataset.csv')
        start_idx = 1
        if df_old_full.columns[0] == '':
            start_idx = int(df_old_full.iloc[-1, 0]) + 1 if df_old_full.shape[0] > 0 else 1
        else:
            start_idx = df_old_full.shape[0] + 1
        df_new_to_save.insert(0, '', range(start_idx, start_idx + df_new_to_save.shape[0]))
        with open('spam_ham_dataset.csv', 'a', encoding='utf-8', newline='') as f:
            df_new_to_save.to_csv(f, header=False, index=False)
        log("Đã chuyển toàn bộ mail mới sang spam_ham_dataset.csv.")

        with open('new_emails.csv', 'w', encoding='utf-8', newline='') as f:
            f.write(',"label","text","label_num"\n')
        log("Đã reset file new_emails.csv (chỉ còn header).")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix trên tập Test')
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/confusion_matrix.png")
    plt.close()

except Exception as e:
    log(f"TRAINING FAILED: {str(e)}")
    log(f"TRAINING TIME: {time.time() - start_time:.2f} giây")
    log("Đã xảy ra lỗi trong quá trình huấn luyện. Vui lòng kiểm tra log_file để biết thêm chi tiết.")
