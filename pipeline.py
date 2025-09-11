import logging
import os
from config import PIPELINE_CONFIG

# Импортируем функции для выполнения каждого шага из соответствующих модулей
from steps.step_1_extraction import run_step_1
from steps.step_2_padding import run_step_2
from steps.step_3_prediction import run_step_3
from steps.step_4_remapping import run_step_4
from steps.step_5_tool_depth import run_step_5
# --- ИЗМЕНЕННЫЙ ИМПОРТ ---
from steps.step_6_block_average import run_step_6

# Настройка логирования для отслеживания всего процесса
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("pipeline.log", mode='w', encoding='utf-8')  # Вывод в файл
    ]
)


def create_output_directory():
    """Создает выходную директорию, если она не существует."""
    os.makedirs(PIPELINE_CONFIG['OUTPUT_DIR'], exist_ok=True)
    logging.info(f"Директория '{PIPELINE_CONFIG['OUTPUT_DIR']}' готова.")


def main():
    """
    Главная функция-оркестратор для последовательного запуска шагов пайплайна.
    """
    logging.info("======[ Запуск пайплайна ]======")
    create_output_directory()

    # Определяем все шаги пайплайна в виде списка функций и их названий.
    pipeline_steps = [
        (run_step_1, "Шаг 1: Выгрузка данных"),
        (run_step_2, "Шаг 2: Добавление отступов"),
        (run_step_3, "Шаг 3: Предсказание состояний"),
        (run_step_4, "Шаг 4: Перекодирование состояний"),
        (run_step_5, "Шаг 5: Расчет глубины инструмента"),
        # --- ИЗМЕНЕННОЕ ОПИСАНИЕ ---
        (run_step_6, "Шаг 6: Расчет среднего веса по блокам")
    ]

    # Последовательно выполняем каждый шаг в цикле.
    for step_function, step_name in pipeline_steps:
        try:
            success = step_function()
            if not success:
                logging.error(f"Пайплайн остановлен из-за ошибки на этапе: '{step_name}'")
                # Прерываем цикл, если шаг завершился неудачно.
                break
        except Exception as e:
            logging.error(f"Пайплайн остановлен из-за критической ошибки на этапе '{step_name}': {e}", exc_info=True)
            # Прерываем цикл при любом непредвиденном исключении.
            break
    else:
        # Этот блок 'else' относится к циклу 'for'.
        # Он выполнится только в том случае, если цикл завершился естественно (без 'break').
        logging.info("======[ Пайплайн успешно завершен ]======")


if __name__ == "__main__":
    main()

