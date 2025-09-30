import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from config import PIPELINE_CONFIG
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def run_step_12():
    """
    Шаг 12: Построение комплексных графиков для визуального анализа.
    """
    logger.info("---[ Шаг 12: Построение итоговых графиков ]---")
    try:
        # --- 1. Загрузка данных и параметров ---
        input_filename = PIPELINE_CONFIG['STEP_12_INPUT_FILE']
        input_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], input_filename)
        if not os.path.exists(input_path):
            logger.error(f"Файл {input_path} не найден. Запустите Шаг 11.")
            return False

        df = pd.read_csv(input_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {input_path}, содержащий {len(df)} строк.")

        # --- 2. Получение имен столбцов и настроек из конфига ---
        plot_settings = PIPELINE_CONFIG.get('STEP_12_PLOT_SETTINGS', {})
        time_col = PIPELINE_CONFIG.get('SORT_COLUMN')

        bit_depth_col = plot_settings.get('bit_depth_col', 'Глубина_долота_35')
        bhd_col = plot_settings.get('bhd_col', 'Глубина_забоя_36')
        hookload_col = plot_settings.get('hookload_col', 'Вес_на_крюке_28')
        predicted_hookload_col = plot_settings.get('predicted_hookload_col', 'predicted_weight')
        avg_hookload_col = plot_settings.get('avg_hookload_col', 'средний_вес_по_блоку')
        slips_col = plot_settings.get('slips_col', 'клинья_0123')
        pressure_col = plot_settings.get('pressure_col', 'Давление_на_входе_18')
        rpm_col = plot_settings.get('rpm_col', 'Обороты_ротора_72')
        hook_height_col = plot_settings.get('hook_height_col', 'Высота_крюка_103')
        training_flag_col = plot_settings.get('training_flag_col', 'Модель_обучалась_флаг')

        colors = plot_settings.get('colors', {})
        anomaly_s = plot_settings.get('anomaly_marker_s', 15)
        anomaly_alpha = plot_settings.get('anomaly_marker_alpha', 0.6)

        # --- 3. Подготовка данных для построения графиков ---
        if time_col not in df.columns:
            logger.error(f"Временной столбец '{time_col}' не найден.")
            return False
        df[time_col] = pd.to_datetime(df[time_col].astype(str).str.replace(',', '.'))
        df.set_index(time_col, inplace=True)

        # --- 4. Разбивка на временные интервалы ---
        chunk_minutes = PIPELINE_CONFIG.get('STEP_12_CHUNK_MINUTES', 120)
        start_time = df.index.min().floor('h')
        end_time = df.index.max().ceil('h')
        time_chunks = pd.date_range(start=start_time, end=end_time, freq=f'{chunk_minutes}min')

        skipped_chunks = []
        base_plot_filename = PIPELINE_CONFIG.get('STEP_12_PLOT_FILE_PREFIX', 'plot')

        for i in tqdm(range(len(time_chunks) - 1), desc="Создание графиков"):
            start_chunk_time = time_chunks[i]
            end_chunk_time = time_chunks[i + 1]
            df_chunk = df[(df.index >= start_chunk_time) & (df.index < end_chunk_time)]

            if df_chunk.empty:
                skipped_chunks.append({'start_time': start_chunk_time, 'end_time': end_chunk_time, 'reason': 'no_data'})
                continue

            # --- 5. Построение графика для текущего интервала ---
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 14), sharey=True)

            main_ax = ax1
            if len(df_chunk.index.unique()) > 1:
                main_ax.set_ylim(df_chunk.index.min(), df_chunk.index.max())
            main_ax.invert_yaxis()
            main_ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            main_ax.set_ylabel('Время (чч:мм)', fontsize=12)

            # --- Панель 1: Положение долота ---
            ax1.set_xlabel('Глубина, м', fontsize=12)
            ax1.plot(df_chunk[bit_depth_col], df_chunk.index, label='Глубина долота', color=colors.get('bit_depth'))
            ax1.plot(df_chunk[bhd_col], df_chunk.index, label='Глубина забоя', color=colors.get('bhd'), linestyle='--')
            ax1.legend()

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
                            color=colors.get('anomaly'), s=anomaly_s, alpha=anomaly_alpha, zorder=5)
            ax2.legend(loc='upper left')

            ax2_twin = ax2.twiny()
            ax2_twin.set_xlabel('Состояние клиньев', fontsize=12, color=colors.get('slips'))
            ax2_twin.plot(df_chunk[slips_col], df_chunk.index, color=colors.get('slips'), linestyle=':')
            ax2_twin.tick_params(axis='x', labelcolor=colors.get('slips'))

            # --- Панель 3: Положение талевого блока ---
            ax3.set_xlabel('Высота, м', fontsize=12)

            # Создаем серию с NaNs там, где модель не училась
            trained_height = df_chunk[hook_height_col].copy()
            trained_height[df_chunk[training_flag_col] == 0] = np.nan

            # Рисуем серую линию как подложку
            ax3.plot(df_chunk[hook_height_col], df_chunk.index, color=colors.get('hook_height_not_trained'), zorder=1)
            # Поверх рисуем черную линию, которая будет прерываться на NaNs
            ax3.plot(trained_height, df_chunk.index, color=colors.get('hook_height_trained'), zorder=2, linewidth=2)

            # Создаем кастомную легенду
            legend_elements = [
                Line2D([0], [0], color=colors.get('hook_height_not_trained'), lw=2, label='Высота (не училась)'),
                Line2D([0], [0], color=colors.get('hook_height_trained'), lw=2, label='Высота (училась)')]
            ax3.legend(handles=legend_elements)

            # --- Панель 4: Мех. и гидрав. параметры ---
            ax4.set_xlabel('Давление, атм', fontsize=12, color=colors.get('pressure'))
            ax4.plot(df_chunk[pressure_col], df_chunk.index, color=colors.get('pressure'))
            ax4.tick_params(axis='x', labelcolor=colors.get('pressure'))
            ax4_twin = ax4.twiny()
            ax4_twin.set_xlabel('Обороты, об/мин', fontsize=12, color=colors.get('rpm'))
            ax4_twin.plot(df_chunk[rpm_col], df_chunk.index, color=colors.get('rpm'), linestyle=':')
            ax4_twin.tick_params(axis='x', labelcolor=colors.get('rpm'))

            # --- Общий заголовок ---
            date_str = start_chunk_time.strftime('%Y-%m-%d')
            time_interval_str = f"{start_chunk_time.strftime('%H:%M')} - {end_chunk_time.strftime('%H:%M')}"
            fig.suptitle(
                f'Комплексный анализ буровых параметров (Часть {i + 1}/{len(time_chunks) - 1})\nДата: {date_str} | Время: {time_interval_str}',
                fontsize=16)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            # --- 6. Сохранение графика ---
            plot_filename = f"{base_plot_filename}_{i}.png"
            plot_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- 7. Сохранение отчета о пропущенных интервалах ---
        if skipped_chunks:
            report_df = pd.DataFrame(skipped_chunks)
            report_filename = PIPELINE_CONFIG['STEP_12_SKIPPED_CHUNKS_REPORT_FILE']
            report_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], report_filename)
            report_df.to_csv(report_path, index=False, sep=';', encoding='utf-8-sig')
            logger.info(f"Отчет о пропущенных интервалах сохранен в: {report_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 12: {e}", exc_info=True)
        return False

