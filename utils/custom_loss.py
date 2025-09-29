import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """
    Реализация Focal Loss.
    Отлично подходит для задач с сильным дисбалансом классов.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: логиты модели shape (B, T, C) или (BT, C)
            targets: истинные метки shape (B, T) или (BT)
        """
        # Преобразуем в 2D, если необходимо
        if inputs.ndim > 2:
            inputs = inputs.reshape(-1, inputs.size(-1))
        if targets.ndim > 1:
            targets = targets.reshape(-1)

        # Преобразуем логиты в вероятности с помощью softmax
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Основная формула Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            # Применяем веса классов
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemporalWeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss с динамическими весами, которые увеличиваются вокруг событий.
    Это заставляет модель обращать больше внимания на временной контекст событий.
    """

    def __init__(
            self,
            num_classes: int,
            alpha: Optional[torch.Tensor] = None,
            peak_weight: float = 5.0,
            window_size: int = 10,
            base_weight: float = 1.0
    ):
        super(TemporalWeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Веса для классов
        self.peak_weight = peak_weight
        self.window_size = window_size
        self.base_weight = base_weight

        # Создаем линейно убывающее окно
        self.window = self._create_window()

    def _create_window(self) -> np.ndarray:
        """Создает симметричное окно с линейным затуханием от пика."""
        return np.linspace(self.peak_weight, self.base_weight, self.window_size + 1, endpoint=True)[1:]

    def _create_temporal_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Создает тензор весов для каждого элемента в батче на основе его временного положения
        относительно событий.
        """
        batch_size, seq_len = targets.shape
        # Изначально все веса базовые
        weights = torch.full_like(targets, self.base_weight, dtype=torch.float32)

        # Переводим на CPU для numpy операций, так как это проще
        targets_np = targets.cpu().numpy()

        for i in range(batch_size):
            # Находим индексы, где есть события (класс не 0)
            event_indices = np.where(targets_np[i] > 0)[0]

            # Если событий нет, веса остаются базовыми
            if len(event_indices) == 0:
                continue

            # Применяем окно вокруг каждого события
            for idx in event_indices:
                # Устанавливаем пиковый вес в точке события
                weights[i, idx] = max(weights[i, idx].item(), self.peak_weight)

                # === ИСПРАВЛЕНИЕ НАЧАЛО ===
                # Применяем левую часть окна
                left_start = max(0, idx - self.window_size)
                if left_start < idx:  # Убедимся, что срез не пустой
                    actual_len = idx - left_start
                    window_part = self.window[-actual_len:]
                    weights_slice = weights[i, left_start:idx]
                    window_tensor = torch.from_numpy(window_part).float().to(weights.device)
                    weights[i, left_start:idx] = torch.max(weights_slice, window_tensor)

                # Применяем правую часть окна
                right_end = min(seq_len, idx + self.window_size + 1)
                if idx + 1 < right_end:  # Убедимся, что срез не пустой
                    actual_len = right_end - (idx + 1)
                    window_part = self.window[:actual_len]
                    weights_slice = weights[i, idx + 1:right_end]
                    # .copy() важен для избежания проблем с памятью у перевернутых numpy массивов
                    window_tensor = torch.from_numpy(np.flip(window_part).copy()).float().to(weights.device)
                    weights[i, idx + 1:right_end] = torch.max(weights_slice, window_tensor)
                # === ИСПРАВЛЕНИЕ КОНЕЦ ===

        return weights.to(targets.device)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Логиты модели. Shape: (B, T, C)
            targets (torch.Tensor): Истинные метки. Shape: (B, T)
        """
        # 1. Рассчитываем стандартный CrossEntropy, но без редукции
        # B, T, C -> BT, C
        # B, T -> BT
        loss_no_reduction = F.cross_entropy(
            inputs.reshape(-1, self.num_classes),
            targets.reshape(-1),
            weight=self.alpha,
            reduction='none'
        )

        # 2. Создаем временные веса
        temporal_weights = self._create_temporal_weights(targets)

        # 3. Применяем веса к loss
        # BT -> B, T
        loss_no_reduction = loss_no_reduction.view_as(targets)
        weighted_loss = loss_no_reduction * temporal_weights

        # 4. Усредняем
        return weighted_loss.mean()

