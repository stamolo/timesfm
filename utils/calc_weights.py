import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def calculate_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: Optional[torch.device] = None,
    epsilon: float = 1e-6
) -> torch.Tensor:
    logger.info("Вычисление весов классов для %d классов", num_classes)
    flat_labels = labels.flatten()
    counts = np.bincount(flat_labels, minlength=num_classes).astype(np.float32)
    total = flat_labels.shape[0]
    logger.info("Общее количество примеров: %d, распределение: %s", total, counts)
    weights = total / (num_classes * (counts + epsilon))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weights_tensor = weights_tensor.to(device)
    logger.info("Веса классов вычислены: %s", weights_tensor)
    return weights_tensor
