import numpy as np
import torch


def calculate_class_weights(labels, num_classes, device=None, epsilon=1e-6):
    """
    Вычисляет веса классов на основе массива меток.

    :param labels: NumPy-массив с метками.
    :param num_classes: Общее количество классов.
    :param device: Если задан, возвращает тензор на данном устройстве.
    :param epsilon: Малая константа для предотвращения деления на ноль.
    :return: Тензор весов классов.
    """
    flat_labels = labels.flatten()
    counts = np.bincount(flat_labels, minlength=num_classes).astype(np.float32)
    total = flat_labels.shape[0]
    weights = total / (num_classes * (counts + epsilon))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weights_tensor = weights_tensor.to(device)
    return weights_tensor
