import numpy as np
import logging

logger = logging.getLogger(__name__)

def print_label_statistics(labels: np.ndarray, num_classes: int) -> None:
    """
    Выводит статистику по меткам:
    - Общее количество примеров
    - Количество примеров для каждого класса
    """
    flat_labels = labels.flatten()
    counts = np.bincount(flat_labels, minlength=num_classes).astype(np.float32)
    total = flat_labels.shape[0]
    logger.info("Общее количество примеров: %d", total)
    for i, count in enumerate(counts):
        logger.info("Класс %d: %d примеров", i, int(count))
