import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import os
from typing import Optional, Dict, List, Tuple
from torch.utils.data import TensorDataset, DataLoader
from .data_utils import load_csv_data

logger = logging.getLogger(__name__)


def plot_initial_dataset_check(csv_path: str, config: Dict, output_folder: str):
    """
    Загружает исходный CSV-файл и строит график для каждого признака и цели
    на отдельных подграфиках для проверки данных перед обучением.
    """
    logger.info(f"Проверка исходных данных: построение графика из {csv_path}")
    try:
        data = load_csv_data(
            csv_path,
            sep=config["CSV_SETTINGS"]["sep"],
            decimal=config["CSV_SETTINGS"]["decimal"]
        )

        target_column_idx = config.get("TARGET_COLUMN", -1)
        if target_column_idx < 0:
            target_column_idx = data.shape[1] + target_column_idx

        feature_column_indices = config.get("FEATURE_COLUMNS")
        if not feature_column_indices:
            # Если признаки не указаны, берем все столбцы кроме целевого
            feature_column_indices = [i for i in range(data.shape[1]) if i != target_column_idx]

        # Собираем все столбцы для отрисовки
        columns_to_plot = sorted(list(set(feature_column_indices + [target_column_idx])))

        num_plots = len(columns_to_plot)
        fig, axes = plt.subplots(num_plots, 1, figsize=(20, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]  # make it iterable

        fig.suptitle('Проверка исходного обучающего датасета', fontsize=16)

        for ax, col_idx in zip(axes, columns_to_plot):
            ax.plot(data[:, col_idx], linewidth=1)

            plot_title = f"Столбец {col_idx}"
            if col_idx == target_column_idx:
                plot_title += " (Цель)"
            else:
                plot_title += " (Признак)"

            ax.set_title(plot_title)
            ax.grid(True)

        axes[-1].set_xlabel("Временные отсчеты")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(output_folder, "initial_train_dataset_check.png")
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"График для проверки исходных данных сохранен в: {save_path}")

    except Exception as e:
        logger.error(f"Ошибка при построении графика для проверки данных: {e}", exc_info=True)


def get_all_preds_and_trues(
        model: torch.nn.Module,
        dataset: TensorDataset,
        device: torch.device,
        batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогоняет весь датасет через модель и собирает все предсказания,
    истинные значения и исходные признаки в виде единых временных рядов.
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []
    all_features: List[np.ndarray] = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred_batch = model(X_batch)

            all_preds.append(y_pred_batch.squeeze().cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
            all_features.append(X_batch.cpu().numpy())

    if not all_preds:
        return np.array([]), np.array([]), np.array([])

    full_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    full_trues = np.concatenate([t.reshape(-1) for t in all_trues])

    num_features = all_features[0].shape[2]
    full_features = np.concatenate(all_features).reshape(-1, num_features)

    return full_preds, full_trues, full_features


def plot_regression_results(
        model: torch.nn.Module,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        scaler: Optional,
        device: torch.device,
        config: Optional[Dict] = None
) -> plt.Figure:
    """
    Строит 2x2 график с результатами регрессии.
    - Для обучающего набора предсказания накладываются на исходный временной ряд.
    - Для валидационного набора используется "склеенный" вид (т.к. окна не пересекаются).
    """
    if config is None:
        config = {"FEATURE_COLUMNS": [0, 1], "BATCH_SIZE": 64}
        logger.debug("Конфигурация не передана, использованы значения по умолчанию.")

    model.eval()
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=False)
    fig.suptitle('Результаты на полных наборах данных', fontsize=16)

    datasets_info = {
        "Обучающий набор": (train_dataset, True),
        "Валидационный набор": (val_dataset, False)
    }

    feature_indices = config.get("FEATURE_COLUMNS", [0, 1])
    batch_size = config.get("BATCH_SIZE", 64)

    for col_idx, (title, (dataset, reconstruct)) in enumerate(datasets_info.items()):

        if len(dataset) == 0:
            logger.warning(f"Датасет '{title}' пуст, пропуск отрисовки.")
            for row in range(2):
                axes[row, col_idx].text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                axes[row, col_idx].set_title(title)
            continue

        y_pred_stitched, y_true_stitched, features_stitched = get_all_preds_and_trues(model, dataset, device,
                                                                                      batch_size)

        y_true_to_plot, y_pred_to_plot, features_to_plot = y_true_stitched, y_pred_stitched, features_stitched
        original_features_for_plot = None

        if reconstruct:
            try:
                csv_path = config["TRAIN_CSV"] if title == "Обучающий набор" else config["VAL_TEST_CSV"]
                original_data = load_csv_data(csv_path, sep=config["CSV_SETTINGS"]["sep"],
                                              decimal=config["CSV_SETTINGS"]["decimal"])

                target_column_idx = config.get("TARGET_COLUMN", -1)
                if target_column_idx < 0:
                    target_column_idx = original_data.shape[1] + target_column_idx

                y_true_to_plot = original_data[:, target_column_idx]
                original_features_for_plot = original_data  # Сохраняем для нижнего графика

                # Реконструируем предсказания на временной оси исходных данных
                y_pred_reconstructed = np.full(y_true_to_plot.shape, np.nan)
                output_len = config["OUTPUT_LENGTH"]
                input_len = config["INPUT_LENGTH"]
                step = config["WINDOW_STEP"]
                center_offset = (input_len - output_len) // 2

                y_pred_windows = y_pred_stitched.reshape(-1, output_len)
                num_windows = y_pred_windows.shape[0]

                for i in range(num_windows):
                    window_start_idx = i * step
                    pred_placement_start_idx = window_start_idx + center_offset
                    pred_placement_end_idx = pred_placement_start_idx + output_len
                    pred_window = y_pred_windows[i]

                    if pred_placement_start_idx >= len(y_pred_reconstructed): continue

                    actual_end_idx = min(pred_placement_end_idx, len(y_pred_reconstructed))
                    len_to_place = actual_end_idx - pred_placement_start_idx

                    y_pred_reconstructed[pred_placement_start_idx:actual_end_idx] = pred_window[:len_to_place]

                y_pred_to_plot = y_pred_reconstructed

            except Exception as e:
                logger.error(f"Не удалось реконструировать график для '{title}': {e}. Отображается 'склеенный' вид.")

        time_axis = np.arange(len(y_true_to_plot))
        plot_limit = min(len(y_true_to_plot), 150000)

        # === Верхний график: Предсказания ===
        ax1 = axes[0, col_idx]
        ax1.plot(time_axis[:plot_limit], y_true_to_plot[:plot_limit], 'g-', lw=2, alpha=0.8, label="Истинное значение")
        ax1.plot(time_axis[:plot_limit], y_pred_to_plot[:plot_limit], 'r--', lw=1.5, alpha=0.8,
                 label="Предсказанное значение")
        ax1.set_title(f"{title}: Предсказание (первые {plot_limit} точек)")
        ax1.legend()
        ax1.grid(True)

        # === Нижний график: Признаки ===
        ax2 = axes[1, col_idx]
        ax2_twin = ax2.twinx()

        # Для нижнего графика используем оригинальные признаки, если они были загружены
        data_for_features = original_features_for_plot if original_features_for_plot is not None else features_to_plot

        # Логика денормализации
        if scaler is not None:
            # Денормализуем только выбранные признаки
            if data_for_features.shape[1] == scaler.n_features_in_:
                features_denorm = scaler.inverse_transform(data_for_features)
            else:  # Если количество столбцов не совпадает (например, только признаки, без цели)
                placeholder = np.zeros((data_for_features.shape[0], scaler.n_features_in_))
                placeholder[:, feature_indices] = data_for_features[:, :len(feature_indices)]
                features_denorm = scaler.inverse_transform(placeholder)

            feature_0 = features_denorm[:, feature_indices[0]]
            feature_1 = features_denorm[:, feature_indices[1]]
            label_0, label_1 = f"Признак {feature_indices[0]} (денорм.)", f"Признак {feature_indices[1]} (денорм.)"

        elif data_for_features.shape[1] >= max(feature_indices) + 1:
            feature_0 = data_for_features[:, feature_indices[0]]
            feature_1 = data_for_features[:, feature_indices[1]]
            label_0, label_1 = f"Признак {feature_indices[0]}", f"Признак {feature_indices[1]}"
        else:
            ax2.set_title(f"{title}: Недостаточно признаков")
            ax2.text(0.5, 0.5, f'Нужно >= {max(feature_indices) + 1} признаков', ha='center', va='center')
            continue

        p1 = ax2.plot(time_axis[:plot_limit], feature_0[:plot_limit], color='blue', alpha=0.7, label=label_0)
        ax2.set_ylabel(label_0, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        p2 = ax2_twin.plot(time_axis[:plot_limit], feature_1[:plot_limit], color='orange', alpha=0.7, label=label_1)
        ax2_twin.set_ylabel(label_1, color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')

        ax2.set_title(f"{title}: Признаки {feature_indices[0]} и {feature_indices[1]} (первые {plot_limit} точек)")
        ax2.set_xlabel("Время")
        ax2.grid(True)
        ax2.sharex(ax1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    logger.info("Построение сводного графика для полных датасетов завершено.")
    return fig

