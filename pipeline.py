import logging
import os
from config import PIPELINE_CONFIG
from utils_pipline import db_handler

# Импортируем функции для выполнения каждого шага
from steps.step_1_extraction import run_step_1
from steps.step_2_padding import run_step_2
from steps.step_3_prediction import run_step_3
from steps.step_4_remapping import run_step_4
from steps.step_5_tool_depth import run_step_5
from steps.step_6_block_average import run_step_6
from steps.step_7_advanced_reset import run_step_7
from steps.step_8_bottom_hole_depth import run_step_8
from steps.step_9_derivatives import run_step_9
from steps.step_10_above_bottom_hole import run_step_10
from steps.step_11.main import run_step_11
from steps.step_12_plotting import run_step_12

# Настройка логирования для отслеживания всего процесса
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        ("Шаг 8: Расчет глубины забоя", run_step_8),
        ("Шаг 9: Расчет производных", run_step_9),
        ("Шаг 10: Расчет 'Над забоем' и запись в БД", run_step_10),
        ("Шаг 11: Поиск аномалий и запись результатов", run_step_11),
        ("Шаг 12: Построение итоговых графиков из БД", run_step_12)
    ]

    try:
        os.makedirs(PIPELINE_CONFIG['OUTPUT_DIR'], exist_ok=True)
        logging.info(f"Директория '{PIPELINE_CONFIG['OUTPUT_DIR']}' готова.")

        # --- Инициализация схемы в БД перед запуском шагов ---
        logging.info("Инициализация схемы в базе данных...")
        db_handler.setup_schema()
        # ----------------------------------------------------------------

        start_step = PIPELINE_CONFIG.get("START_PIPELINE_FROM_STEP", 1)
        logging.info(f"Пайплайн будет запущен с Шага {start_step}.")

        # Преобразуем список шагов в словарь для удобного доступа
        steps_map = {int(s[0].split(':')[0].split(' ')[1]): (s[0], s[1]) for s in pipeline_steps}

        for step_number in sorted(steps_map.keys()):
            if step_number < start_step:
                logging.info(
                    f"---[ Пропущен: {steps_map[step_number][0]} (согласно настройке START_PIPELINE_FROM_STEP={start_step}) ]---")
                continue

            name, step_function = steps_map[step_number]

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

