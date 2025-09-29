import os
import subprocess
import pandas as pd
import logging
import sys
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_3():
    """
    Шаг 3: Распознавание признака на данных с отступами и очистка результата.
    Возвращает True в случае успеха и False в случае ошибки.
    """
    logger.info("---[ Шаг 3: Распознавание нового признака ]---")
    try:
        step_1_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_1_OUTPUT_FILE'])
        padded_data_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_2_OUTPUT_FILE'])

        if not os.path.exists(padded_data_path):
            logger.error(f"Файл {padded_data_path} не найден. Запустите Шаг 2.")
            return False

        df_original = pd.read_csv(step_1_path, sep=';', decimal=',', encoding='utf-8')
        df_padded = pd.read_csv(padded_data_path, sep=';', decimal=',', encoding='utf-8')

        temp_input_for_pred_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], 'temp_for_prediction.csv')
        df_for_pred = df_padded[PIPELINE_CONFIG['PREDICTION_INPUT_COLUMNS']]
        df_for_pred.to_csv(temp_input_for_pred_path, header=True, index=False, sep=';', decimal=',',
                           encoding='utf-8-sig')
        logger.info(f"Подготовлен временный файл для предсказания: {temp_input_for_pred_path}")

        predicted_output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], 'step_2_predicted.csv')

        command = [
            sys.executable, 'pred256_3.py',
            '--csv', temp_input_for_pred_path,
            '--output', predicted_output_path,
            '--model', PIPELINE_CONFIG['MODEL_PATH']
        ]
        logger.info(f"Запуск команды: {' '.join(command)}")

        # --- ИСПРАВЛЕНИЕ ---
        # Заменена кодировка 'utf-8' на 'cp1251' и добавлен параметр errors='ignore'.
        # Это необходимо для корректного чтения вывода дочернего процесса в среде Windows,
        # где консоль по умолчанию использует другую кодировку (часто cp1251 или cp866).
        # 'errors=ignore' предотвратит падение, если в выводе встретятся нераспознанные символы.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding='cp1251', errors='ignore')
        stdout, stderr = process.communicate()

        logger.info("---[ Вывод скрипта pred256_3.py ]---")
        if stdout: logger.info(stdout)
        if stderr: logger.error(stderr)

        if process.returncode != 0:
            raise Exception(f"Скрипт pred256_3.py завершился с ошибкой. Код: {process.returncode}")

        logger.info(f"Скрипт предсказания успешно выполнен. Результат в: {predicted_output_path}")

        df_predicted = pd.read_csv(predicted_output_path, sep=';', decimal=',')

        if 'predicted_label' in df_predicted.columns:
            padding_size = PIPELINE_CONFIG['PADDING_SIZE']
            num_original_rows = len(df_original)

            predicted_labels = df_predicted['predicted_label'].iloc[
                               padding_size: padding_size + num_original_rows].reset_index(drop=True)
            logger.info(f"Предсказания для отступов ({padding_size} в начале и {padding_size} в конце) удалены.")

            new_column_name = "клинья_0123"
            df_original[new_column_name] = predicted_labels

            step_3_output_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_3_OUTPUT_FILE'])
            df_original.to_csv(step_3_output_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
            logger.info(f"Новый признак '{new_column_name}' добавлен. Результат сохранен в: {step_3_output_path}")
        else:
            logger.error("Столбец 'predicted_label' не найден в файле с предсказаниями.")
            return False

        os.remove(temp_input_for_pred_path)
        os.remove(predicted_output_path)

        logging.info("Временные файлы удалены.")
        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 3: {e}", exc_info=True)
        return False
