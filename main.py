from flask import Flask, render_template, request, session, Response
import pickle
from utils import save_new_email
import os
import subprocess

app = Flask(__name__)

def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model_nb.pkl', 'rb') as f:
        nb_clf = pickle.load(f)
    return vectorizer, nb_clf

vectorizer, nb_clf = load_model()

@app.route('/', methods=['GET', 'POST'])
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
