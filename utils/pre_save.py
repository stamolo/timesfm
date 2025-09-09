import os
import shutil
import pickle
import logging

logger = logging.getLogger(__name__)

def save_script_copy(script_path: str, dest_folder: str) -> None:
    """
    Копирует файл скрипта в указанную папку.
    """
    try:
        shutil.copy(script_path, os.path.join(dest_folder, os.path.basename(script_path)))
        logger.info("Сохранён скрипт: %s", script_path)
    except Exception as e:
        logger.error("Ошибка при копировании скрипта %s: %s", script_path, e)

def save_model_script(model_script_path: str, dest_folder: str) -> None:
    """
    Копирует файл модели в указанную папку.
    """
    if os.path.exists(model_script_path):
        try:
            shutil.copy(model_script_path, os.path.join(dest_folder, os.path.basename(model_script_path)))
            logger.info("Сохранён скрипт модели: %s", model_script_path)
        except Exception as e:
            logger.error("Ошибка при копировании скрипта модели %s: %s", model_script_path, e)
    else:
        logger.error("Скрипт модели не найден: %s", model_script_path)

def save_scaler(scaler, dest_folder: str, filename: str = "scaler.pkl") -> None:
    """
    Сохраняет объект скейлера в указанный файл в папке dest_folder.
    """
    try:
        scaler_path = os.path.join(dest_folder, filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler сохранён в файл: %s", scaler_path)
    except Exception as e:
        logger.error("Ошибка при сохранении скейлера: %s", e)
