import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


def run_step_12():
    """
    Шаг 12: Построение итоговых комплексных графиков.
    """
    logger.info("---[ Шаг 12: Построение итоговых графиков (ось Y - время) ]---")
    try:
        # 1. Загрузка данных и параметров
        step_11_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], PIPELINE_CONFIG['STEP_11_OUTPUT_FILE'])
        if not os.path.exists(step_11_path):
            logger.error(f"Файл {step_11_path} не найден. Запустите Шаг 11.")
            return False

        df = pd.read_csv(step_11_path, sep=';', decimal=',', encoding='utf-8')
        logger.info(f"Загружен файл {step_11_path}, содержащий {len(df)} строк.")

        # 2. Получение имен столбцов и настроек графиков из конфига
        time_col = PIPELINE_CONFIG.get('SORT_COLUMN')
        final_bhd_col = PIPELINE_CONFIG['STEP_8_OUTPUT_BHD_COLUMN']
        bit_depth_col = PIPELINE_CONFIG['STEP_9_BIT_DEPTH_COLUMN']
        hookload_col = PIPELINE_CONFIG['STEP_11_TARGET_COLUMN']
        hook_height_col = PIPELINE_CONFIG['STEP_9_HOOK_HEIGHT_COLUMN']
        pressure_col = "Давление_на_входе_18"
        rpm_col = "Обороты_ротора_72"
        avg_weight_col = PIPELINE_CONFIG['STEP_6_OUTPUT_AVG_WEIGHT_COLUMN']
        slips_0123_col = PIPELINE_CONFIG['STEP_4_INPUT_COLUMN']

        plot_settings = PIPELINE_CONFIG.get('STEP_12_PLOT_SETTINGS', {})

        if time_col not in df.columns:
            logger.error(f"Столбец времени '{time_col}' не найден.")
            return False
        if df[time_col].dtype == 'object':
            df[time_col] = pd.to_datetime(df[time_col].str.replace(',', '.'))

        # 3. Построение графиков по временным интервалам
        minutes_per_chunk = PIPELINE_CONFIG.get('STEP_12_CHUNK_MINUTES', 120)
        time_delta = pd.Timedelta(minutes=minutes_per_chunk)
        overall_start_time = df[time_col].min()
        overall_end_time = df[time_col].max()

        if pd.isna(overall_start_time):
            logger.warning("Нет данных о времени для построения графиков. Шаг 12 пропущен.")
            return True

        anchor_time = overall_start_time.floor('H')

        num_chunks = int(np.ceil((overall_end_time - anchor_time) / time_delta)) if not pd.isna(overall_end_time) else 0
        if num_chunks == 0 and not df.empty:
            num_chunks = 1

        logger.info(
            f"Данные будут разбиты на {num_chunks} графиков по {minutes_per_chunk} минут каждый (с выравниванием по часу).")

        base_plot_filename = PIPELINE_CONFIG['STEP_12_PLOT_FILE']
        filename, file_extension = os.path.splitext(base_plot_filename)

        for i in range(num_chunks):
            chunk_start_time = anchor_time + i * time_delta
            chunk_end_time = chunk_start_time + time_delta

            df_chunk = df[(df[time_col] >= chunk_start_time) & (df[time_col] < chunk_end_time)]

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(28, 16), sharey=True)

            if df_chunk.empty:
                logger.info(f"Создание ПУСТОГО графика для части {i + 1}/{num_chunks} (нет данных в интервале)...")
                start_depth, end_depth = np.nan, np.nan
                time_info_string = f"Нет данных в интервале: {chunk_start_time.strftime('%Y-%m-%d %H:%M')} - {chunk_end_time.strftime('%Y-%m-%d %H:%M')}"

                # Настраиваем пустые панели
                ax1.set_title('Положение долота')
                ax2.set_title('Анализ веса и состояния клиньев')
                ax3.set_title('Мех. и гидравлика')
                ax4.set_title('Положение талевого блока')

                for ax in [ax1, ax2, ax3, ax4]:
                    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

            else:
                logger.info(
                    f"Создание графика для части {i + 1}/{num_chunks} (время с {chunk_start_time.strftime('%Y-%m-%d %H:%M')} по {chunk_end_time.strftime('%Y-%m-%d %H:%M')})...")

                y_data = df_chunk[time_col]

                # --- Панель 1: Глубины ---
                ax1.plot(df_chunk[bit_depth_col], y_data, label='Глубина долота',
                         color=plot_settings.get('bit_depth', {}).get('color', 'green'))
                ax1.plot(df_chunk[final_bhd_col], y_data, label='Глубина забоя',
                         color=plot_settings.get('final_bhd', {}).get('color', 'black'), linestyle='--')
                ax1.set_xlabel('Глубина, м')
                ax1.set_title('Положение долота')
                ax1.legend()
                ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

                # --- Панель 2: Вес на крюке, аномалии и состояние клиньев ---
                ax2.plot(df_chunk[hookload_col], y_data, label='Факт. вес на крюке',
                         color=plot_settings.get('hookload', {}).get('color', 'royalblue'), linewidth=1)
                ax2.plot(df_chunk['predicted_weight'], y_data, label='Модельный вес',
                         color=plot_settings.get('predicted_weight', {}).get('color', 'skyblue'), linestyle='--',
                         linewidth=1)
                ax2.plot(df_chunk[avg_weight_col], y_data, label='Средний вес (блок)',
                         color=plot_settings.get('avg_weight', {}).get('color', 'orange'), linestyle='-', linewidth=2,
                         drawstyle='steps-post')

                anomalies = df_chunk[df_chunk['is_anomaly'] == 1]
                if not anomalies.empty:
                    anomaly_style = plot_settings.get('anomaly', {})
                    ax2.scatter(anomalies[hookload_col], anomalies[time_col], color=anomaly_style.get('color', 'red'),
                                s=anomaly_style.get('size', 25), zorder=5, label='Аномалия',
                                alpha=anomaly_style.get('alpha', 0.6))

                ax2.set_xlabel('Вес на крюке, т', color=plot_settings.get('hookload', {}).get('color', 'royalblue'))
                ax2.tick_params(axis='x', labelcolor=plot_settings.get('hookload', {}).get('color', 'royalblue'))
                ax2.set_title('Анализ веса и состояния клиньев')
                ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

                ax2_twin = ax2.twiny()
                slips_style = plot_settings.get('slips_0123', {})
                ax2_twin.plot(df_chunk[slips_0123_col], y_data, label='Клинья (0123)',
                              color=slips_style.get('color', 'black'), drawstyle='steps-post',
                              alpha=slips_style.get('alpha', 0.7))
                ax2_twin.set_xlabel('Состояние клиньев', color=slips_style.get('color', 'black'))
                ax2_twin.tick_params(axis='x', labelcolor=slips_style.get('color', 'black'))
                ax2_twin.set_xticks([0, 1, 2, 3])
                ax2_twin.set_xlim(-0.5, 3.5)

                lines, labels = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right')

                # --- Панель 3: Давление и обороты ---
                ax3_twin = ax3.twiny()
                pressure_style = plot_settings.get('pressure', {})
                rpm_style = plot_settings.get('rpm', {})
                ax3.plot(df_chunk[pressure_col], y_data, label='Давление на входе',
                         color=pressure_style.get('color', 'firebrick'))
                ax3_twin.plot(df_chunk[rpm_col], y_data, label='Обороты ротора', color=rpm_style.get('color', 'purple'),
                              linestyle=':')

                ax3.set_xlabel('Давление, атм', color=pressure_style.get('color', 'firebrick'))
                ax3_twin.set_xlabel('Обороты, об/мин', color=rpm_style.get('color', 'purple'))
                ax3.tick_params(axis='x', labelcolor=pressure_style.get('color', 'firebrick'))
                ax3_twin.tick_params(axis='x', labelcolor=rpm_style.get('color', 'purple'))
                ax3.set_title('Мех. и гидравлика')
                ax3.grid(True, which='both', linestyle=':', linewidth=0.5)
                lines3, labels3 = ax3.get_legend_handles_labels()
                lines3_twin, labels3_twin = ax3_twin.get_legend_handles_labels()
                ax3_twin.legend(lines3 + lines3_twin, labels3 + labels3_twin, loc='upper right')

                # --- Панель 4: Высота крюка ---
                ax4.plot(df_chunk[hook_height_col], y_data, label='Высота крюка',
                         color=plot_settings.get('hook_height', {}).get('color', 'darkcyan'))
                ax4.set_xlabel('Высота, м')
                ax4.set_title('Положение талевого блока')
                ax4.legend()
                ax4.grid(True, which='both', linestyle=':', linewidth=0.5)

                start_depth = df_chunk[final_bhd_col].min()
                end_depth = df_chunk[final_bhd_col].max()

                start_time_obj = df_chunk[time_col].min()
                end_time_obj = df_chunk[time_col].max()
                start_date = start_time_obj.date()
                end_date = end_time_obj.date()

                if start_date == end_date:
                    time_info_string = f"Дата: {start_date.strftime('%Y-%m-%d')} | Время: {start_time_obj.strftime('%H:%M')} - {end_time_obj.strftime('%H:%M')}"
                else:
                    time_info_string = f"Временной интервал: {start_time_obj.strftime('%Y-%m-%d %H:%M')} - {end_time_obj.strftime('%Y-%m-%d %H:%M')}"

            # --- Общие настройки для всех графиков (пустых и с данными) ---
            ax1.set_ylabel('Время (ЧЧ:ММ)')
            ax1.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.set_ylim(chunk_end_time, chunk_start_time)  # Установка границ и инвертирование оси Y

            fig.suptitle(
                f'Комплексный анализ буровых параметров (Часть {i + 1}/{num_chunks})\n'
                f'Интервал глубин: {start_depth:.2f} - {end_depth:.2f} м\n'
                f'{time_info_string}',
                fontsize=16
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            chunk_plot_filename = f"{filename}_{i + 1}{file_extension}"
            plot_path = os.path.join(PIPELINE_CONFIG['OUTPUT_DIR'], chunk_plot_filename)

            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"График {i + 1} сохранен в: {plot_path}")

        return True

    except Exception as e:
        logger.error(f"Ошибка на Шаге 12: {e}", exc_info=True)
        return False

