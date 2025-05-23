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
