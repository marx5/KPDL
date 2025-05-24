from flask import Flask, render_template, request, session, Response
import pickle
from utils import save_new_email
import os
import subprocess
import sys

app = Flask(__name__)

def load_model():
    vectorizer_path = 'vectorizer.pkl'
    model_path = 'model_nb.pkl'

    # Kiểm tra sự tồn tại của file model
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print("[INFO] Model files not found. Initial training is required.")
        # Chạy script train_nb.py để huấn luyện mô hình ban đầu
        try:
            print("[INFO] Running initial training (train_nb.py)... This may take a moment.")
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            # Sử dụng subprocess.run để chạy đồng bộ và đợi kết thúc
            result = subprocess.run(
                [sys.executable, 'train_nb.py'],
                capture_output=True, # Capture stdout/stderr
                text=True, # Decode output as text
                encoding='utf-8',
                env=env
            )
            print("[INFO] train_nb.py finished. Output:")
            print(result.stdout)
            if result.returncode != 0:
                print(f"[ERROR] train_nb.py failed with return code {result.returncode}. Stderr:\n{result.stderr}")
                raise RuntimeError("Initial training failed.")
            print("[INFO] Initial training completed successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to run initial training script: {e}")
            raise RuntimeError("Failed during initial model training.") from e

    # Sau khi đảm bảo file tồn tại (hoặc đã được tạo ra), tiến hành load model
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            nb_clf = pickle.load(f)
        print("[INFO] Model and vectorizer loaded successfully.")
        return vectorizer, nb_clf
    except Exception as e:
        print(f"[ERROR] Failed to load model or vectorizer after training attempt: {e}")
        # Nếu vẫn lỗi sau khi cố gắng train, có vấn đề nghiêm trọng
        raise RuntimeError("Failed to load model after training attempt.") from e

# Gọi load_model ngay khi khởi động app. Nếu file không có, nó sẽ tự động train.
vectorizer, nb_clf = load_model()

@app.route('/', methods=['GET', 'POST'])
def main_function():
    if request.method == "POST":
        email_text = request.form['email']
        X_input = vectorizer.transform([email_text])
        prediction = int(nb_clf.predict(X_input)[0])

        if 'save_train' in request.form:
            label = request.form['label']
            save_new_email(email_text, prediction, label)
            return render_template("index.html",
                                   saved=True,
                                   email_text=email_text,
                                   prediction=prediction,
                                   label=label)
        else:
            return render_template("index.html",
                                   email_text=email_text,
                                   prediction=prediction)
    else:
        return render_template("index.html")


@app.route('/train', methods=['GET'])
def retrain():
    return render_template("train.html")

@app.route('/train-log', methods=['POST'])
def train_log():
    def generate():
        # Xóa log cũ nếu có
        try:
            if os.path.exists('retrain_log.txt'):
                os.remove('retrain_log.txt')
        except: pass

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            ['python', 'train_nb.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8', 
            env=env
        )
        for line in iter(process.stdout.readline, ''):
            yield line
        process.stdout.close()
        process.wait()
    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
