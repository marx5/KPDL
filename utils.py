import pandas as pd
import os
import csv


def save_new_email(email_text, prediction=None, label=None):
    save_path = "new_emails.csv"
    # Xác định nhãn số
    if label is not None and label != "":
        label_str = label.strip().lower()
        label_num = 0 if label_str == 'ham' else 1
    else:
        label_num = prediction if prediction is not None else ""
    columns = ['', 'label', 'text', 'label_num']

    file_exists = os.path.isfile(save_path)
    file_not_empty = file_exists and os.path.getsize(save_path) > 0

    if file_not_empty:
        try:
            df_old = pd.read_csv(save_path)
            new_index = int(df_old.iloc[-1, 0]) + 1 if df_old.shape[0] > 0 else 1
        except Exception:
            new_index = 1
    else:
        new_index = 1

    row = {
        '': new_index,
        'label': label if label else "",
        'text': email_text,
        'label_num': label_num
    }
    df_new = pd.DataFrame([row], columns=columns)

    if not file_not_empty:
        df_new.to_csv(save_path, mode='w', header=columns, index=False,
                      quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
    else:
        df_new.to_csv(save_path, mode='a', header=False, index=False,
                      quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')

    # --- Thêm logic kiểm tra số lượng và trigger train --- START
    try:
        # Đọc lại file sau khi ghi để đếm số dòng
        df_current_new = pd.read_csv(save_path)
        # Trừ đi 1 vì dòng header
        num_new_emails = df_current_new.shape[0] - 1
        # Chỉ đếm các dòng có nhãn thực
        num_new_emails_labelled = df_current_new[df_current_new['label'].notnull() & (df_current_new['label'] != '')].shape[0]

        TRAIN_THRESHOLD = 10

        if num_new_emails_labelled >= TRAIN_THRESHOLD:
            print("\n[AUTO-TRAIN] Đã đạt ngưỡng 10 email mới có nhãn thực. Đang kích hoạt huấn luyện lại...\n", flush=True)
            import subprocess
            import sys
            import os

            # Chạy train_nb.py trong subprocess (không chặn)
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                # Sử dụng sys.executable để đảm bảo dùng đúng interpreter
                subprocess.Popen(
                    [sys.executable, 'train_nb.py'],
                    stdout=subprocess.PIPE, # Có thể đổi thành subprocess.DEVNULL nếu không muốn output
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding='utf-8', 
                    env=env
                )
                print("[AUTO-TRAIN] Đã gửi lệnh huấn luyện lại.", flush=True)
            except Exception as e:
                print(f"[AUTO-TRAIN] Lỗi khi gọi subprocess train_nb.py: {str(e)}", flush=True)

    except Exception as e:
        print(f"[AUTO-TRAIN] Lỗi trong logic kiểm tra và trigger train: {str(e)}", flush=True)
    # --- Thêm logic kiểm tra số lượng và trigger train --- END
