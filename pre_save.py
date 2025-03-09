import os
import shutil
import pickle

def save_script_copy(script_path, dest_folder):
    """
    Копирует файл скрипта в указанную папку.
    """
    try:
        shutil.copy(script_path, os.path.join(dest_folder, os.path.basename(script_path)))
        print(f"Сохранён скрипт: {script_path}")
    except Exception as e:
        print(f"Ошибка при копировании скрипта {script_path}: {e}")

def save_model_script(model_script_path, dest_folder):
    """
    Копирует файл модели в указанную папку.
    """
    if os.path.exists(model_script_path):
        try:
            shutil.copy(model_script_path, os.path.join(dest_folder, os.path.basename(model_script_path)))
            print(f"Сохранён скрипт модели: {model_script_path}")
        except Exception as e:
            print(f"Ошибка при копировании скрипта модели {model_script_path}: {e}")
    else:
        print(f"Скрипт модели не найден: {model_script_path}")

def save_scaler(scaler, dest_folder, filename="scaler.pkl"):
    """
    Сохраняет объект скейлера в указанный файл в папке dest_folder.
    """
    try:
        scaler_path = os.path.join(dest_folder, filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print("Scaler сохранён в файл:", scaler_path)
    except Exception as e:
        print("Ошибка при сохранении скейлера:", e)
