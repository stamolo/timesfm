import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def plot_classification_results_with_labels(
        model: torch.nn.Module,
        dataset,
        scaler: Optional,
        device: torch.device,
        num_examples: int = 1,
        config: Optional[Dict] = None
) -> plt.Figure:
    """
    Строит график с результатами классификации.

    :param model: Обученная модель.
    :param dataset: Датасет (TensorDataset), из которого выбирается несколько примеров.
    :param scaler: Скейлер для денормализации, если он используется.
    :param device: Устройство, на котором выполняется модель.
    :param num_examples: Количество примеров для построения графиков.
    :param config: Конфигурация с параметрами INPUT_LENGTH, OUTPUT_LENGTH и NUM_CLASSES.
    :return: Фигура matplotlib.
    """
    if config is None:
        config = {"INPUT_LENGTH": 1200, "OUTPUT_LENGTH": 1100, "NUM_CLASSES": 4}
        logger.debug("Конфигурация не передана, использованы значения по умолчанию.")

    model.eval()
    fig, axes = plt.subplots(1, num_examples, figsize=(18, 5), sharex=True)
    if num_examples == 1:
        axes = [axes]

    try:
        indices = np.random.choice(len(dataset), num_examples, replace=False)
    except Exception as e:
        logger.error("Ошибка при выборе примеров из датасета: %s", e)
        raise

    for ax, idx in zip(axes, indices):
        try:
            X, y_true = dataset[idx]
        except Exception as e:
            logger.error("Ошибка получения примера с индексом %s: %s", idx, e)
            continue

        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        y_true_np = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.asarray(y_true)
        X_input = torch.tensor(X_np).unsqueeze(0).to(device) if X_np.ndim == 2 else torch.tensor(X_np).to(device)
        time_axis_full = np.arange(config["INPUT_LENGTH"])
        center_start = (config["INPUT_LENGTH"] - config["OUTPUT_LENGTH"]) // 2
        center_end = center_start + config["OUTPUT_LENGTH"]

        logits = model(X_input)
        probs = torch.softmax(logits, dim=2)
        y_pred = torch.argmax(probs, dim=2)[0].detach().cpu().numpy()

        if scaler is not None:
            X_denorm = scaler.inverse_transform(X_np)
            feature_to_plot = X_denorm[:, 0]
            title = "Feature 0 (denormalized)"
        else:
            feature_to_plot = X_np[:, 0]
            title = "Feature 0"

        ax.plot(time_axis_full, feature_to_plot, 'b-', lw=2, label="Feature 0")
        ax.set_xlabel("Time")
        ax.set_title(title)

        ax2 = ax.twinx()
        ax2.set_ylim(-0.5, config["NUM_CLASSES"] - 0.5)
        ax2.set_yticks(np.arange(config["NUM_CLASSES"]))
        ax2.set_ylabel("Класс")
        ax2.plot(time_axis_full, y_true_np, 'g--', lw=2, label="Истинный")
        ax2.plot(np.arange(center_start, center_end), y_pred, 'r-', lw=2, label="Предсказанный")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    logger.info("Построение графика завершено.")
    return fig
