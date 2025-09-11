import logging
import os
from config import PIPELINE_CONFIG

# Импортируем функции для выполнения каждого шага
from steps.step_1_extraction import run_step_1
from steps.step_2_padding import run_step_2
from steps.step_3_prediction import run_step_3
from steps.step_4_remapping import run_step_4
from steps.step_5_tool_depth import run_step_5
from steps.step_6_block_average import run_step_6
from steps.step_7_advanced_reset import run_step_7
from steps.step_8_bottom_hole_depth import run_step_8

# Настройка логирования для отслеживания всего процесса
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("pipeline.log", mode='w', encoding='utf-8')  # Вывод в файл
    ]
)


def main():
    """
    Главная функция-оркестратор для последовательного запуска шагов пайплайна.
    """
    logging.info("======[ Запуск пайплайна ]======")

    pipeline_steps = [
        ("Шаг 1: Выгрузка данных", run_step_1),
        ("Шаг 2: Добавление отступов", run_step_2),
        ("Шаг 3: Распознавание признака", run_step_3),
        ("Шаг 4: Перекодирование состояний", run_step_4),
        ("Шаг 5: Расчет начальной глубины", run_step_5),
        ("Шаг 6: Расчет среднего веса по блокам", run_step_6),
        ("Шаг 7: Продвинутый сброс глубины", run_step_7),
        ("Шаг 8: Расчет глубины забоя", run_step_8)
    ]

    try:
        os.makedirs(PIPELINE_CONFIG['OUTPUT_DIR'], exist_ok=True)
        logging.info(f"Директория '{PIPELINE_CONFIG['OUTPUT_DIR']}' готова.")

        for name, step_function in pipeline_steps:
            logging.info(f"---[ Запуск: {name} ]---")
            if not step_function():
                logging.error(f"Пайплайн остановлен из-за ошибки на этапе: {name}.")
                return  # Прерываем выполнение
            logging.info(f"---[ Успешно завершен: {name} ]---")

        logging.info("======[ Пайплайн успешно завершен ]======")

    except Exception as e:
        logging.error(f"Критическая ошибка в пайплайне: {e}", exc_info=True)


if __name__ == "__main__":
    main()

