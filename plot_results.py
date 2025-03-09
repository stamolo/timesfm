import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_classification_results_with_labels(model, dataset, scaler, device, num_examples=1, config=None):
    """
    Строит график с результатами классификации.

    :param model: обученная модель.
    :param dataset: датасет (TensorDataset), из которого выбирается несколько примеров.
    :param scaler: скейлер для денормализации, если он используется.
    :param device: устройство, на котором выполняется модель.
    :param num_examples: количество примеров для построения графиков.
    :param config: конфигурация с параметрами INPUT_LENGTH и OUTPUT_LENGTH.
    :return: фигура matplotlib.
    """
    # Если конфигурация не передана, задаем значения по умолчанию
    if config is None:
        config = {"INPUT_LENGTH": 1200, "OUTPUT_LENGTH": 1100, "NUM_CLASSES": 4}

    model.eval()
    fig, axes = plt.subplots(1, num_examples, figsize=(18, 5), sharex=True)
    if num_examples == 1:
        axes = [axes]

    indices = np.random.choice(len(dataset), num_examples, replace=False)

    for ax, idx in zip(axes, indices):
        X, y_true = dataset[idx]
        X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        y_true_np = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.asarray(y_true)
        # Если данные двумерные, добавляем batch-измерение
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

        ax.plot(np.asarray(time_axis_full), np.asarray(feature_to_plot), 'b-', lw=2, label="Feature 0")
        ax.set_xlabel("Time")
        ax.set_title(title)

        ax2 = ax.twinx()
        ax2.set_ylim(-0.5, config["NUM_CLASSES"] - 0.5)
        ax2.set_yticks(np.arange(config["NUM_CLASSES"]))
        ax2.set_ylabel("Класс")
        ax2.plot(np.asarray(time_axis_full), np.asarray(y_true_np), 'g--', lw=2, label="Истинный")
        ax2.plot(np.arange(center_start, center_end), np.asarray(y_pred), 'r-', lw=2, label="Предсказанный")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    return fig
