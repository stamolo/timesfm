import os
import pandas as pd
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from config import PIPELINE_CONFIG
from matplotlib.lines import Line2D
import multiprocessing
import matplotlib.cm as cm

# Используем бэкенд, который не требует GUI. Важно для мультипроцессинга.
matplotlib.use('Agg')
logger = logging.getLogger(__name__)

# Глобальная переменная для хранения DataFrame в каждом рабочем процессе
worker_df = None


def init_worker(df_to_share):
    """
    Функция-инициализатор для каждого процесса в пуле.
    Копирует DataFrame в глобальную переменную этого процесса ОДИН РАЗ.
    """
    global worker_df
    worker_df = df_to_share


def _generate_plot_for_chunk(args):
    """
    Вспомогательная функция для генерации и сохранения одного графика.
    Использует глобальный worker_df, чтобы не передавать данные с каждой задачей.
    """
    global worker_df
    try:
        # 1. Распаковка аргументов.
        i, total_chunks, start_chunk_time, end_chunk_time, plot_settings, \
            base_plot_filename, output_dir, pressure_limits, rpm_limits, \
            show_contrib_plot, feature_cols = args

        # 2. Нарезка данных из ГЛОБАЛЬНОГО DataFrame этого процесса.
        df_chunk = worker_df[(worker_df.index >= start_chunk_time) & (worker_df.index < end_chunk_time)]

        if df_chunk.empty:
            return {'start_time': start_chunk_time, 'end_time': end_chunk_time, 'reason': 'no_data'}

        # --- 3. Построение графика для текущего интервала ---
        plt.style.use('seaborn-v0_8-whitegrid')

        if show_contrib_plot:
            fig, axes = plt.subplots(1, 5, figsize=(35, 14), sharey=True,
                                     gridspec_kw={'width_ratios': [1, 1.2, 1, 1.2, 1.2]})
            ax1, ax2, ax3, ax4, ax5 = axes
        else:
            fig, axes = plt.subplots(1, 4, figsize=(28, 14), sharey=True)
            ax1, ax2, ax3, ax4 = axes

        main_ax = ax1
        if len(df_chunk.index.unique()) > 1:
            main_ax.set_ylim(df_chunk.index.min(), df_chunk.index.max())
        main_ax.invert_yaxis()
        main_ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        main_ax.set_ylabel('Время (чч:мм)', fontsize=12)

        # --- Извлечение параметров и цветов ---
        colors = plot_settings.get('colors', {})
        bit_depth_col = plot_settings.get('bit_depth_col')
        bhd_col = plot_settings.get('bhd_col')
        hookload_col = plot_settings.get('hookload_col')
        predicted_hookload_col = plot_settings.get('predicted_hookload_col')
        avg_hookload_col = plot_settings.get('avg_hookload_col')
        slips_col = plot_settings.get('slips_col')
        pressure_col = plot_settings.get('pressure_col')
        rpm_col = plot_settings.get('rpm_col')
        hook_height_col = plot_settings.get('hook_height_col')
        training_flag_col = plot_settings.get('training_flag_col')

        # --- Панель 1: Положение долота и забоя ---
        ax1.set_xlabel('Глубина долота, м', fontsize=12, color=colors.get('bit_depth'))
        ax1.plot(df_chunk[bit_depth_col], df_chunk.index, label='Глубина долота', color=colors.get('bit_depth'))
        ax1.tick_params(axis='x', labelcolor=colors.get('bit_depth'))
        ax1_twin = ax1.twiny()
        ax1_twin.set_xlabel('Глубина забоя, м', fontsize=12, color=colors.get('bhd'))
        ax1_twin.plot(df_chunk[bhd_col], df_chunk.index, color=colors.get('bhd'), linestyle='--')
        ax1_twin.tick_params(axis='x', labelcolor=colors.get('bhd'))

        # --- Панель 2: Анализ веса и состояние клиньев ---
        ax2.set_xlabel('Вес на крюке, т', fontsize=12)
        ax2.plot(df_chunk[hookload_col], df_chunk.index, label='Вес на крюке', color=colors.get('hookload'))
        ax2.plot(df_chunk[predicted_hookload_col], df_chunk.index, label='Predicted вес',
                 color=colors.get('predicted_hookload'))
        ax2.plot(df_chunk[avg_hookload_col], df_chunk.index, label='Средний вес по блоку',
                 color=colors.get('avg_hookload'))
        anomalies = df_chunk[df_chunk['is_anomaly'] == 1]
        if not anomalies.empty:
            ax2.scatter(anomalies[hookload_col], anomalies.index, label='Аномалии',
                        color=colors.get('anomaly'), s=plot_settings.get('anomaly_marker_s', 15),
                        alpha=plot_settings.get('anomaly_marker_alpha', 0.6), zorder=5)
        ax2.legend(loc='upper left')

        ax2_twin = ax2.twiny()
        ax2_twin.set_xlabel('Состояние клиньев', fontsize=12, color=colors.get('slips'))
        ax2_twin.plot(df_chunk[slips_col], df_chunk.index, color=colors.get('slips'), linestyle=':')
        ax2_twin.tick_params(axis='x', labelcolor=colors.get('slips'))

        # --- Панель 3: Положение талевого блока ---
        ax3.set_xlabel('Высота, м', fontsize=12)
        trained_height = df_chunk[hook_height_col].copy()
        trained_height[df_chunk[training_flag_col] == 0] = np.nan
        ax3.plot(df_chunk[hook_height_col], df_chunk.index, color=colors.get('hook_height_not_trained'), zorder=1)
        ax3.plot(trained_height, df_chunk.index, color=colors.get('hook_height_trained'), zorder=2, linewidth=2)

        if plot_settings.get('fixed_hook_height_scale'):
            ax3.set_xlim(plot_settings.get('hook_height_min', 0), plot_settings.get('hook_height_max', 35))

        legend_elements = [
            Line2D([0], [0], color=colors.get('hook_height_not_trained'), lw=2, label='Высота (не училась)'),
            Line2D([0], [0], color=colors.get('hook_height_trained'), lw=2, label='Высота (училась)')]
        ax3.legend(handles=legend_elements)

        # --- Панель 4: Мех. и гидрав. параметры ---
        ax4.set_xlabel('Давление, атм', fontsize=12, color=colors.get('pressure'))
        ax4.plot(df_chunk[pressure_col], df_chunk.index, color=colors.get('pressure'))
        ax4.tick_params(axis='x', labelcolor=colors.get('pressure'))
        if pressure_limits:
            ax4.set_xlim(pressure_limits)
        ax4_twin = ax4.twiny()
        ax4_twin.set_xlabel('Обороты, об/мин', fontsize=12, color=colors.get('rpm'))
        ax4_twin.plot(df_chunk[rpm_col], df_chunk.index, color=colors.get('rpm'), linestyle=':')
        ax4_twin.tick_params(axis='x', labelcolor=colors.get('rpm'))
        if rpm_limits:
            ax4_twin.set_xlim(rpm_limits)

        # --- Панель 5: Вклад признаков в предсказание ---
        if show_contrib_plot:
            ax5.set_xlabel('Вклад в расчетный вес, т', fontsize=12)

            if feature_cols:
                contribution_cols = [f'contribution_{f}' for f in feature_cols]
                intercept_col = 'contribution_intercept'
                all_contrib_cols = [intercept_col] + contribution_cols

                # Список всех столбцов, необходимых для этого графика
                cols_for_plot5 = all_contrib_cols + [predicted_hookload_col]

                if all(col in df_chunk.columns for col in all_contrib_cols):

                    # ИСПРАВЛЕНИЕ: Создаем один DataFrame для графика и удаляем пропуски из всех
                    # нужных столбцов ОДНОВРЕМЕННО. Это гарантирует совпадение размерностей.
                    plot_data = df_chunk[cols_for_plot5].dropna()

                    if not plot_data.empty:
                        y = plot_data.index
                        left_boundary = np.zeros(len(plot_data))

                        plot_colors = cm.get_cmap('viridis', len(all_contrib_cols))

                        # Используем 'plot_idx', чтобы не конфликтовать с внешней переменной 'i'
                        for plot_idx, col in enumerate(all_contrib_cols):
                            right_boundary = left_boundary + plot_data[col].values
                            label_name = col.replace('contribution_', '').replace('_', ' ')
                            ax5.fill_betweenx(y, left_boundary, right_boundary,
                                              label=label_name, color=plot_colors(plot_idx), alpha=0.8)
                            left_boundary = right_boundary

                        # Строим итоговую линию по отфильтрованным данным
                        ax5.plot(plot_data[predicted_hookload_col], y, color='red',
                                 linestyle='--', linewidth=1.5, label='Итог (Predicted)')

                        ax5.legend(loc='upper left', fontsize='small')
                        ax5.grid(True, which='both', linestyle='--', linewidth=0.5)
                else:
                    logger.warning(
                        f"График вклада не будет построен для чанка {i + 1}: отсутствуют необходимые столбцы.")
                    ax5.text(0.5, 0.5, 'Нет данных для\nрасчета вклада',
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax5.transAxes, fontsize=12, color='red')

        # --- 4. Общий заголовок и сохранение ---
        date_str = start_chunk_time.strftime('%Y-%m-%d')
        time_interval_str = f"{start_chunk_time.strftime('%H-%M')} - {end_chunk_time.strftime('%H-%M')}"
        # Заменяем символ '|' на запятую для большей надежности при сохранении
        title_text = f'Комплексный анализ буровых параметров (Часть {i + 1}/{total_chunks})\nДата: {date_str}, Время: {time_interval_str}'
        fig.suptitle(title_text, fontsize=16)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{base_plot_filename}_{i}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return None
    except Exception as e:
        logger.error(f"Ошибка при генерации графика для интервала {start_chunk_time} - {end_chunk_time}: {e}",
                     exc_info=True)
        return {'start_time': start_chunk_time, 'end_time': end_chunk_time, 'reason': f'error: {e}'}


def run_step_12():
    """
    Шаг 12: Построение комплексных графиков для визуального анализа с использованием
    параллельной обработки для ускорения.
    """
    logger.info("---[ Шаг 12: Построение итоговых графиков (параллельный режим) ]---")
    try:
        # --- 1. Загрузка данных и параметров ---
        input_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_12_INPUT_FILE'])
        if not os.path.exists(input_path):
            logger.error(f"Файл {input_path} не найден. Запустите Шаг 11.")
            return False

        df = pd.read_csv(input_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {input_path}, содержащий {len(df)} строк.")

        # --- 2. Предварительная обработка данных ---
        time_col = PIPELINE_CONFIG.get('SORT_COLUMN')
        df[time_col] = pd.to_datetime(df[time_col].astype(str).str.replace(',', '.'))
        df.set_index(time_col, inplace=True)

        plot_settings = PIPELINE_CONFIG.get('STEP_12_PLOT_SETTINGS', {})
        pressure_col = plot_settings.get('pressure_col', 'Давление_на_входе_18')
        rpm_col = plot_settings.get('rpm_col', 'Обороты_ротора_72')
        pressure_limits, rpm_limits = None, None

        if plot_settings.get('fixed_pressure_rpm_scale', False):
            percentile = plot_settings.get('scale_percentile', 95.0)
            if pressure_col in df.columns:
                p_upper = df[pressure_col].quantile(percentile / 100.0)
                pressure_limits = (-5, p_upper * 1.05)
            if rpm_col in df.columns:
                r_upper = df[rpm_col].quantile(percentile / 100.0)
                rpm_limits = (-5, r_upper * 1.05)

        # --- 3. Разбивка на временные интервалы ---
        chunk_minutes = PIPELINE_CONFIG.get('STEP_12_CHUNK_MINUTES', 120)
        start_time = df.index.min().floor('h')
        end_time = df.index.max().ceil('h')
        time_chunks = pd.date_range(start=start_time, end=end_time, freq=f'{chunk_minutes}min')

        # --- 4. Подготовка задач для параллельной обработки ---
        tasks = []
        base_plot_filename = PIPELINE_CONFIG.get('STEP_12_PLOT_FILE_PREFIX', 'plot')
        total_chunks = len(time_chunks) - 1

        show_contrib_plot = PIPELINE_CONFIG.get('STEP_12_SHOW_CONTRIBUTION_PLOT', False)
        feature_cols_for_plot = PIPELINE_CONFIG.get('STEP_11_FEATURE_COLUMNS', []) if show_contrib_plot else []
        if show_contrib_plot:
            logger.info("Включена опция построения графика вклада признаков.")

        for i in range(total_chunks):
            start_chunk_time = time_chunks[i]
            end_chunk_time = time_chunks[i + 1]

            task_args = (
                i, total_chunks, start_chunk_time, end_chunk_time,
                plot_settings, base_plot_filename, PIPELINE_CONFIG['OUTPUT_DIR'],
                pressure_limits, rpm_limits,
                show_contrib_plot, feature_cols_for_plot
            )
            tasks.append(task_args)
        logger.info(f"Подготовлено {len(tasks)} задач.")

        # --- 5. Параллельное создание графиков ---
        num_processes = min(multiprocessing.cpu_count() - 1, len(tasks), 8)
        logger.info(f"Запуск параллельной генерации {len(tasks)} графиков на {num_processes} процессах...")
        with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(df,)) as pool:
            results = list(tqdm(pool.imap(_generate_plot_for_chunk, tasks), total=len(tasks), desc="Создание графиков"))

        # --- 6. Сбор информации о пропущенных интервалах ---
        skipped_chunks = [res for res in results if res is not None]
        if skipped_chunks:
            report_df = pd.DataFrame(skipped_chunks)
            report_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'],
                                       PIPELINE_CONFIG['STEP_12_SKIPPED_CHUNKS_REPORT_FILE'])
            report_df.to_csv(report_path, index=False, sep=';', encoding='utf-8-sig')
            logger.info(f"Отчет о {len(skipped_chunks)} пропущенных/ошибочных интервалах сохранен в: {report_path}")

        return True

    except Exception as e:
        logger.error(f"Критическая ошибка на Шаге 12: {e}", exc_info=True)
        return False

